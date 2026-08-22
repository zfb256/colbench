#!/usr/bin/env python3
"""Measure how representative the recorded shuffle seed is analytically."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbench import access
from colbench.amplification import row_group_touches


def _shuffle_record(path: str) -> dict | None:
    with open(path, encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if (row["format"], row["cache_mode"], row["access"]["kind"]) == (
                "parquet", "cold", "shuffle"
            ):
                return row
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-glob", default="results/fig4_clickbench_32p_c*.jsonl")
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("results/seed_sensitivity.json"))
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(args.results_glob)):
        row = _shuffle_record(path)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit(f"no cold Parquet shuffle rows matched {args.results_glob!r}")

    spec = rows[0]["access"]
    values = {row["chunk_rows"]: [] for row in rows}
    for seed in range(args.seed_count):
        plans = access.shuffle_batches(
            rows[0]["dataset"]["n_rows"],
            batch_rows=spec["batch_rows"],
            window=spec["window"],
            n_batches=spec["n_batches"],
            seed=seed,
        )
        for chunk_rows in values:
            per_call, union = row_group_touches(plans, chunk_rows)
            values[chunk_rows].append({
                "seed": seed, "per_call_groups": per_call, "union_groups": union
            })

    report = {"seed_count": args.seed_count, "chunks": []}
    recorded_seed = spec["seed"]
    for chunk_rows, chunk_values in sorted(values.items()):
        recorded = chunk_values[recorded_seed]
        median = statistics.median(v["per_call_groups"] for v in chunk_values)
        union_median = statistics.median(v["union_groups"] for v in chunk_values)
        report["chunks"].append({
            "chunk_rows": chunk_rows,
            "recorded_seed": recorded_seed,
            "recorded_per_call_groups": recorded["per_call_groups"],
            "per_call_groups_median": median,
            "recorded_vs_median_pct": 100 * (recorded["per_call_groups"] / median - 1),
            "recorded_union_groups": recorded["union_groups"],
            "union_groups_median": union_median,
            "recorded_union_vs_median_pct": 100 * (
                recorded["union_groups"] / union_median - 1
            ),
            "per_seed": chunk_values,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
