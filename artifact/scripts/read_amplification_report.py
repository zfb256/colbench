#!/usr/bin/env python3
"""Generate Parquet read-amplification estimates for the completed T1 run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colbench import access, paths, runner  # noqa: E402
from colbench.amplification import ParquetLayout, estimate_parquet_access  # noqa: E402
from colbench.formats import ParquetAdapter  # noqa: E402


CHUNKS = [1024, 8192, 32768, 131072, 524288, 1048576]


def pick_columns(ds: runner.Dataset, fraction: float) -> list[str]:
    names = ds.column_names()
    k = max(1, round(len(names) * fraction))
    step = max(1, len(names) // k)
    return names[::step][:k]


def t1_accesses(ds: runner.Dataset) -> list[access.Access]:
    cols = pick_columns(ds, 0.1)
    mid = access.window_sweep(ds.n_rows(), 4096, n_points=5)[2]
    return [
        access.full_scan(batch_rows=65_536),
        access.projection_scan(cols, batch_rows=65_536),
        access.shuffle_read(window=mid, batch_rows=4096, n_batches=64, columns=cols, seed=31),
    ]


def measured_rows(result_paths: list[Path]) -> dict[tuple[int, str], dict]:
    out = {}
    for path in result_paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if (
                    row.get("format") == "parquet"
                    and row.get("cache_mode") == "cold"
                    and row.get("threads") == 1
                ):
                    out[(int(row["chunk_rows"]), row["access"]["label"])] = row
    return out


def materialize_parquet(ds: runner.Dataset, chunk: int, compression: str) -> Path:
    path = paths.work_dir() / f"{ds.name}__parquet__c{chunk}.parquet"
    if path.exists():
        return path
    ParquetAdapter().write(ds.reader(), path, chunk_rows=chunk, compression=compression)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="clickbench_32p")
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--keep-layouts", action="store_true")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("results", nargs="+", type=Path,
                    help="explicit qualified-run JSONL files")
    args = ap.parse_args(argv)

    source = paths.data_dir() / f"{args.dataset}.parquet"
    ds = runner.Dataset(args.dataset, source=source)
    result_paths = args.results
    measured = measured_rows(result_paths)
    accesses = t1_accesses(ds)

    expected = {(chunk, a.label) for chunk in CHUNKS for a in accesses}
    missing = sorted(expected - measured.keys())
    if missing:
        raise SystemExit(
            f"qualified inputs are incomplete: missing {len(missing)} "
            f"Parquet cold cells; first missing cell is {missing[0]}"
        )

    rows = []
    for chunk in CHUNKS:
        layout_path = materialize_parquet(ds, chunk, args.compression)
        layout = ParquetLayout.from_file(layout_path)
        try:
            for a in accesses:
                est = estimate_parquet_access(layout, a)
                m = measured.get((chunk, a.label), {})
                measured_bytes = m.get("read_bytes_median")
                rows.append({
                    "dataset": args.dataset,
                    "format": "parquet",
                    "chunk_rows": chunk,
                    "access_label": a.label,
                    "access_kind": a.kind,
                    "rows_requested": est.rows_requested,
                    "bytes_useful": est.bytes_useful,
                    "predicted_bytes_access_union": est.bytes_required_access_union,
                    "predicted_bytes_per_call": est.bytes_required_per_call,
                    "amplification_access_union": est.amp_access_union,
                    "amplification_per_call": est.amp_per_call,
                    "measured_read_bytes_median": measured_bytes,
                    "measured_wall_s_median": m.get("wall_s_median"),
                    "measured_file_size_bytes": m.get("file_size_bytes"),
                    "prediction_error_union": (
                        (est.bytes_required_access_union - measured_bytes) / measured_bytes
                        if measured_bytes else None
                    ),
                    "prediction_error_per_call": (
                        (est.bytes_required_per_call - measured_bytes) / measured_bytes
                        if measured_bytes else None
                    ),
                })
        finally:
            if not args.keep_layouts:
                layout_path.unlink(missing_ok=True)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
