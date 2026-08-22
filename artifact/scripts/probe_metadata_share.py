#!/usr/bin/env python
"""How much of a shuffle's wall time is repeated Parquet footer parsing?

``ParquetAdapter.take()`` under the per-call convention constructs a fresh
``ParquetFile`` on every call, so a 64-batch shuffle deserialises the file
footer 64 times.  Footer cost scales with row groups x columns, which is exactly
what the layout sweep varies -- so a sweep that looks like a layout effect can
be substantially a harness effect.

Two parts, with very different costs:

  [A]/[A2]  DEFAULT.  Footer only: opens files and reads metadata, never row
            data.  Seconds, negligible I/O, safe to run beside a measurement.
            Fits a per-entry parse rate and projects it onto ClickBench.

  [B]       OPT-IN via --direct.  Runs a real 64-batch shuffle twice, once
            per-call and once hoisted.  This READS AND DECOMPRESSES COLUMN
            DATA and can take minutes on a wide table.  Do NOT run it beside a
            timed measurement.

Run:
  python scripts/probe_metadata_share.py <dir-with-parquet-layouts> [more dirs]
  python scripts/probe_metadata_share.py <dir> --direct
  python scripts/probe_metadata_share.py <dir> --direct --window 362039 \\
         --columns WatchID,CounterClass,ResolutionWidth
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colbench import access as acc, measure  # noqa: E402

CLICKBENCH_ROWS = 32_000_000
CLICKBENCH_COLS = 105
BATCHES = 64
BATCH_ROWS = 4096
#: The paper's ClickBench-32p shuffle window, round(sqrt(4096 * 32e6)).  Taking
#: int() of the square root gives 362,038 and a different access plan.
PAPER_WINDOW = 362_039
#: The ten columns the paper projects -- not the first ten of the schema.
PAPER_COLUMNS = ["WatchID", "CounterClass", "ResolutionWidth", "CookieEnable",
                 "AdvEngineID", "PageCharset", "IsParameter", "RemoteIP",
                 "DNSTiming", "ParamCurrencyID"]


def groups_for(n_rows: int, chunk: int) -> int:
    """Row groups a writer emits for ``n_rows`` at ``chunk``.

    The trailing partial group counts: floor division silently drops it and
    undercounts every chunk size that does not divide the table evenly.
    """
    return max(1, math.ceil(n_rows / chunk))


def parse_rate(files: list[Path]) -> tuple[float, float] | None:
    print("\n[A] footer parse cost vs (row groups x columns)  -- footer only")
    print("  %-34s %9s %6s %11s %13s"
          % ("file", "groups", "cols", "entries", "parse_s"))
    pts = []
    for f in files:
        try:
            md = pq.ParquetFile(str(f)).metadata
        except Exception as exc:
            print("  %-34s skipped (%s)" % (f.name[:34], type(exc).__name__))
            continue
        entries = md.num_row_groups * md.num_columns
        if entries < 100:
            continue
        pq.ParquetFile(str(f))       # warm: exclude the first cold footer read
        t0 = time.perf_counter()
        for _ in range(5):
            pq.ParquetFile(str(f)).metadata.num_row_groups
        dt = (time.perf_counter() - t0) / 5
        pts.append((entries, dt))
        print("  %-34s %9d %6d %11d %13.4f"
              % (f.name[:34], md.num_row_groups, md.num_columns, entries, dt))
    if len(pts) < 2:
        print("  need at least two layouts to separate per-entry cost from"
              " fixed file-open overhead")
        return None
    # Least squares, not a median of ratios: small files carry a fixed open cost
    # that a ratio charges to their few entries and so biases the estimate.
    n = len(pts)
    sx = sum(e for e, _ in pts)
    sy = sum(t for _, t in pts)
    sxx = sum(e * e for e, _ in pts)
    sxy = sum(e * t for e, t in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        print("  layouts are all the same size; cannot fit")
        return None
    rate = (n * sxy - sx * sy) / denom
    intercept = (sy - rate * sx) / n
    print("  fitted %.3f us per metadata entry (fixed overhead %.4f s)"
          % (rate * 1e6, intercept))
    if abs(intercept) > 0.05:
        print("  WARNING: large intercept -- cost is not dominated by entries")
    return rate, intercept


def project(rate: float, intercept: float) -> None:
    print("\n[A2] projection onto ClickBench-32p Parquet layouts")
    print("  %9s %9s %11s %10s %14s"
          % ("chunk", "groups", "entries", "parse_s", "x64 batches"))
    for chunk in (1_024, 8_192, 32_768, 131_072, 524_288, 1_048_576):
        groups = groups_for(CLICKBENCH_ROWS, chunk)
        entries = groups * CLICKBENCH_COLS
        one = intercept + rate * entries
        print("  %9d %9d %11d %10.3f %14.1f"
              % (chunk, groups, entries, one, one * BATCHES))
    print("  (compare the x64 column against the measured shuffle medians;")
    print("   the difference is what is left for I/O and decoding)")


def direct(path: Path, window: int | None, columns: list[str] | None) -> bool:
    print(f"\n[B] direct per-call vs hoisted on {path.name}")
    print("  NOTE: this reads and decompresses real column data")
    pf = pq.ParquetFile(str(path))
    nrows = pf.metadata.num_rows
    names = list(pf.schema_arrow.names)

    if columns:
        missing = [c for c in columns if c not in names]
        if missing:
            print("  requested columns absent from this file: %s" % missing)
            return False
        cols = columns
    elif all(c in names for c in PAPER_COLUMNS):
        cols = PAPER_COLUMNS
        print("  using the paper's ten projected columns")
    else:
        cols = names[: min(10, len(names))]
        print("  WARNING: paper columns absent; using the first %d columns,"
              " so absolute times are not comparable with the paper" % len(cols))

    if window is None:
        window = (PAPER_WINDOW if nrows == CLICKBENCH_ROWS
                  else round(math.sqrt(BATCH_ROWS * nrows)))
    print("  window=%d  batches=%d x %d rows  columns=%d"
          % (window, BATCHES, BATCH_ROWS, len(cols)))
    batches = acc.shuffle_batches(nrows, batch_rows=BATCH_ROWS, window=window,
                                  n_batches=BATCHES, seed=31)

    def open_file() -> tuple[pq.ParquetFile, np.ndarray]:
        f = pq.ParquetFile(str(path))
        md = f.metadata
        starts, off = [], 0
        for i in range(md.num_row_groups):
            starts.append(off)
            off += md.row_group(i).num_rows
        return f, np.asarray(starts, dtype=np.int64)

    def take(f: pq.ParquetFile, starts: np.ndarray, idx: np.ndarray) -> None:
        groups = np.searchsorted(starts, idx, side="right") - 1
        order = np.argsort(groups, kind="stable")
        tables = []
        for group in np.unique(groups):
            mask = groups[order] == group
            local = idx[order][mask] - starts[group]
            tables.append(f.read_row_group(int(group), columns=cols).take(pa.array(local)))
        inverse = np.argsort(order, kind="stable")
        pa.concat_tables(tables).take(pa.array(inverse))

    def per_call() -> None:
        for idx in batches:
            take(*open_file(), idx)

    def hoisted() -> None:
        f, starts = open_file()
        for idx in batches:
            take(f, starts, idx)

    hoisted()      # warm the page cache so this measures CPU, not storage
    per_call()     # warm the other path too, so neither ordering wins
    t0 = time.perf_counter(); per_call(); t_call = time.perf_counter() - t0
    t0 = time.perf_counter(); hoisted(); t_hoist = time.perf_counter() - t0
    share = (t_call - t_hoist) / t_call * 100 if t_call else 0
    print("  per-call %.3f s | hoisted %.3f s | metadata share %.1f%%"
          % (t_call, t_hoist, share))
    return True


def _entries(f: Path) -> int:
    try:
        md = pq.ParquetFile(str(f)).metadata
        return md.num_row_groups * md.num_columns
    except Exception:
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--window", type=int)
    ap.add_argument("--columns", help="comma-separated projection")
    ap.add_argument("--direct-file",
                    help="run [B] on this layout instead of the widest one, so"
                         " two hosts can be compared on the same file")
    ap.add_argument("--threads", type=int, default=1,
                    help="Arrow CPU and I/O threads for [B] (default: 1,"
                         " matching the paper timing contract)")
    args = ap.parse_args(argv)
    if args.threads < 1:
        ap.error("--threads must be at least 1")

    files: list[Path] = []
    for d in args.paths:
        p = Path(d)
        files += sorted(p.rglob("*.parquet")) if p.is_dir() else [p]
    files = [f for f in files if f.is_file()]
    if not files:
        print("no .parquet files found")
        return 2
    print(f"pyarrow {pa.__version__}; {len(files)} candidate file(s)")

    fit = parse_rate(files)
    if fit is not None:
        project(*fit)
    elif not args.direct:
        return 1
    else:
        # A single file cannot separate per-entry cost from fixed overhead, but
        # it is still a valid target for the direct comparison.
        print("  [A2] skipped: no rate to project")

    if not args.direct:
        print("\n[B] skipped: pass --direct to run it. It reads real column"
              " data and must not run beside a timed measurement.")
        return 0

    if args.direct_file:
        target = Path(args.direct_file)
        if not target.is_file():
            print("\n[B] --direct-file not found: %s" % target)
            return 2
    else:
        target = max(files, key=_entries)
    if _entries(target) < 10_000:
        print("\n[B] skipped: no layout with enough row groups to be informative")
        return 0
    columns = ([c for c in args.columns.split(",") if c]
               if args.columns else None)
    print("  enforcing Arrow CPU/I/O threads=%d" % args.threads)
    with measure.threads(args.threads):
        ok = direct(target, args.window, columns)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
