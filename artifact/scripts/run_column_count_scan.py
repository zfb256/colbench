#!/usr/bin/env python3
"""Run a narrow projection-width scan for the shuffle access."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colbench import access, cache, paths, runner, storage  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="tpch_sf1_lineitem")
    ap.add_argument("--chunk-rows", type=int, default=131_072)
    ap.add_argument("--widths", default="1,2,4,8,16")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", default=str(paths.results_dir() / "column_count_scan_tpch.jsonl"))
    args = ap.parse_args(argv)

    ds = runner.Dataset(args.dataset, source=paths.data_dir() / f"{args.dataset}.parquet")
    names = ds.column_names()
    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    if any(width < 1 or width > len(names) for width in widths):
        raise SystemExit(f"widths must be within 1..{len(names)}")

    dev = paths.describe_mount(paths.work_dir())
    gate_path = paths.results_dir() / "gate_report.json"
    gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
    problems = storage.validate_gate_report(gate_report, paths.work_dir(), dev)
    if problems and os.environ.get("COLBENCH_ALLOW_UNQUALIFIED_STORAGE") != "1":
        detail = "\n".join(f"  - {problem}" for problem in problems)
        raise SystemExit(f"storage gate does not authorize this run:\n{detail}")
    if problems:
        detail = "\n".join(f"  - {problem}" for problem in problems)
        print(
            "\nWARNING: running with an explicit unqualified-storage override.\n"
            f"{detail}\n"
            "Record this run as provider virtual block storage, not NVMe/local SSD."
        )

    window = access.window_sweep(ds.n_rows(), 4096, n_points=5)[2]
    accesses = [
        access.shuffle_read(
            window=window,
            batch_rows=4096,
            n_batches=64,
            columns=names[:width],
            seed=31,
        )
        for width in widths
    ]

    modes = [cache.CacheMode.WARM]
    r = runner.Runner(paths.work_dir(), Path(args.out))
    if r.eviction and r.eviction.supported:
        modes.insert(0, cache.CacheMode.COLD)
    else:
        print("NOTE: running WARM-cache only; this host cannot evict.")

    matrix = runner.Matrix(
        datasets=[ds],
        formats=["parquet"],
        chunk_rows=[args.chunk_rows],
        accesses=accesses,
        cache_modes=modes,
        threads=[1],
        compression="zstd",
        repeats=args.repeats,
        warmup=1,
    )
    records = r.run(matrix)
    print(f"\nwrote {len(records)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
