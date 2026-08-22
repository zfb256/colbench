#!/usr/bin/env python3
"""Bootstrap uncertainty from raw repetitions in new T1 records."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbench.analyze import bootstrap_chunk_uncertainty


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    samples = {}
    for pattern in args.results:
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    if (row["format"], row["cache_mode"], row["access"]["kind"]) != (
                        "parquet", "cold", "shuffle"
                    ):
                        continue
                    raw = row.get("samples")
                    if not raw:
                        raise SystemExit(f"{path}: record predates raw-sample schema")
                    samples[row["chunk_rows"]] = [sample["wall_s"] for sample in raw]

    report = {
        "bootstrap_draws": args.draws,
        "bootstrap_seed": args.seed,
        "cells": bootstrap_chunk_uncertainty(samples, n_boot=args.draws, seed=args.seed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for cell in report["cells"]:
        lo, hi = cell["median_ci95_s"]
        print(
            f"{cell['chunk_rows']:>8}: {cell['median_s']:.3f}s "
            f"CI [{lo:.3f}, {hi:.3f}], P(best)={cell['probability_best']:.3f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
