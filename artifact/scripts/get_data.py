"""Fetch / generate the public datasets.

    python scripts/get_data.py clickbench --partitions 8
    python scripts/get_data.py tpch --sf 1
    python scripts/get_data.py tpcds --sf 1 --table store_sales
    python scripts/get_data.py status

Every dataset here is public and needs no annotation, credentials or licence
click-through.  Downloads resume, so an interrupted run is safe to repeat.

Sizing note: each ClickBench partition is ~117 MB / ~1 M rows, and the full set
is 100 partitions (~11.7 GB).  Budget roughly 3x the source size on disk, since
the benchmark writes the same data once per format.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colbench import paths  # noqa: E402

# Sources are read during materialisation, outside the timer.
DATA = paths.data_dir()

CLICKBENCH_PART = (
    "https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_{i}.parquet"
)
CLICKBENCH_N_PARTITIONS = 100


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def download(url: str, dest: Path, *, chunk: int = 1 << 20) -> Path:
    """Resumable download.  A ``.part`` file is promoted only once complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  have  {dest.name} ({_human(dest.stat().st_size)})")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}

    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        if have and resp.status_code == 416:
            part.rename(dest)
            return dest
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) + have

        mode = "ab" if have else "wb"
        done = have
        t0 = last = time.time()
        with open(part, mode) as fh:
            for block in resp.iter_content(chunk_size=chunk):
                fh.write(block)
                done += len(block)
                if time.time() - last > 2:
                    pct = f"{done / total:6.1%}" if total else "  ?  "
                    rate = done / max(1e-9, time.time() - t0)
                    print(f"\r  {dest.name}: {pct} {_human(done)} "
                          f"@ {_human(rate)}/s", end="", flush=True)
                    last = time.time()
    print(f"\r  got   {dest.name} ({_human(part.stat().st_size)})" + " " * 20)
    part.rename(dest)
    return dest


def cmd_clickbench(args) -> None:
    n = args.partitions
    if not 1 <= n <= CLICKBENCH_N_PARTITIONS:
        raise SystemExit(f"--partitions must be 1..{CLICKBENCH_N_PARTITIONS}")

    raw = DATA / "raw" / "clickbench"
    need_gb = n * 0.117 * 2
    free = _free_gb(DATA)
    print(f"clickbench: {n} partitions, need ~{need_gb:.1f} GB, free {free:.1f} GB")
    if free < need_gb * 1.2:
        raise SystemExit("not enough free space; lower --partitions")

    parts = [download(CLICKBENCH_PART.format(i=i), raw / f"hits_{i}.parquet")
             for i in range(n)]

    out = DATA / f"clickbench_{n}p.parquet"
    if out.exists() and not args.force:
        print(f"merged file already present: {out} ({_human(out.stat().st_size)})")
        return

    print(f"merging {len(parts)} partitions -> {out.name}")
    _merge_parquet(parts, out, row_group_rows=args.row_group_rows)
    _report(out)


def _merge_parquet(parts: list[Path], out: Path, *, row_group_rows: int) -> None:
    """Concatenate Parquet files streaming, with an explicit row-group size.

    The source's own row-group layout must not leak into the experiment: the
    merged file is the *input*, and every format is written from it afresh with
    the chunk size the matrix asks for.
    """
    first = pq.ParquetFile(str(parts[0]))
    schema = first.schema_arrow
    tmp = out.with_suffix(".tmp")
    writer = pq.ParquetWriter(str(tmp), schema, compression="zstd")
    rows = 0
    try:
        for p in parts:
            pf = pq.ParquetFile(str(p))
            if pf.schema_arrow != schema:
                raise SystemExit(f"schema mismatch in {p.name}; refusing to merge")
            for batch in pf.iter_batches(batch_size=row_group_rows):
                writer.write_table(pa.Table.from_batches([batch], schema=schema))
                rows += batch.num_rows
            print(f"\r  merged {rows:,} rows", end="", flush=True)
    finally:
        writer.close()
    print()
    tmp.replace(out)


def cmd_tpch(args) -> None:
    import duckdb

    sf = args.sf
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"tpch_sf{sf}_lineitem.parquet"
    if out.exists() and not args.force:
        print(f"already present: {out} ({_human(out.stat().st_size)})")
        return

    print(f"tpch: generating SF{sf} with duckdb (in-process, no dbgen needed)")
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf});")
    tmp = out.with_suffix(".tmp")
    con.execute(
        f"COPY lineitem TO '{tmp.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880);"
    )
    con.close()
    tmp.replace(out)
    _report(out)


def cmd_tpcds(args) -> None:
    import duckdb

    sf = args.sf
    table = args.table
    if not table.replace("_", "").isalnum():
        raise SystemExit(f"unsafe TPC-DS table name: {table!r}")

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"tpcds_sf{sf}_{table}.parquet"
    if out.exists() and not args.force:
        print(f"already present: {out} ({_human(out.stat().st_size)})")
        return

    print(f"tpcds: generating SF{sf} {table} with duckdb")
    con = duckdb.connect()
    con.execute("INSTALL tpcds; LOAD tpcds;")
    con.execute(f"CALL dsdgen(sf={sf});")
    tmp = out.with_suffix(".tmp")
    con.execute(
        f"COPY {table} TO '{tmp.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880);"
    )
    con.close()
    tmp.replace(out)
    _report(out)


def _report(path: Path) -> None:
    pf = pq.ParquetFile(str(path))
    md = pf.metadata
    print(f"  {path.name}: {md.num_rows:,} rows x {md.num_columns} cols, "
          f"{md.num_row_groups} row groups, {_human(path.stat().st_size)}")


def cmd_status(args) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"data dir: {DATA}   free: {_free_gb(DATA):.1f} GB")
    files = sorted(DATA.glob("*.parquet"))
    if not files:
        print("  (no prepared datasets yet)")
    for f in files:
        try:
            _report(f)
        except Exception as exc:
            print(f"  {f.name}: UNREADABLE ({exc})")
    raw = DATA / "raw"
    if raw.exists():
        n = len(list(raw.rglob("*.parquet")))
        size = sum(p.stat().st_size for p in raw.rglob("*") if p.is_file())
        print(f"  raw/: {n} source files, {_human(size)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    cb = sub.add_parser("clickbench", help="download ClickBench hits partitions")
    cb.add_argument("--partitions", type=int, default=8)
    cb.add_argument("--row-group-rows", type=int, default=122_880)
    cb.add_argument("--force", action="store_true")
    cb.set_defaults(func=cmd_clickbench)

    th = sub.add_parser("tpch", help="generate TPC-H lineitem via duckdb")
    th.add_argument("--sf", type=int, default=1)
    th.add_argument("--force", action="store_true")
    th.set_defaults(func=cmd_tpch)

    td = sub.add_parser("tpcds", help="generate one TPC-DS table via duckdb")
    td.add_argument("--sf", type=int, default=1)
    td.add_argument("--table", default="store_sales")
    td.add_argument("--force", action="store_true")
    td.set_defaults(func=cmd_tpcds)

    st = sub.add_parser("status", help="show prepared datasets")
    st.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
