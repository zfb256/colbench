"""Held-out validation of the two-term byte model.

The column-chunk union predicts the bytes a layout forces a reader to fetch.
Adding one calibrated term for per-projected-column-row-group fixed cost gives

    bytes = sum_{g in G(I)} sum_{c in C} S(g,c)  +  k * |G(I)| * |C|

with a single free parameter k. This script calibrates k on a subset of chunk
sizes and predicts the rest, so the reported accuracy is out-of-sample rather
than a fit quality. It also checks whether the model reproduces the chunk size
at which measured bytes are minimised.

    python scripts/validate_two_term_model.py
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fit_group_overhead import (  # noqa: E402
    CHUNKS, PROJECTED_COLUMNS, load_amplification, touched_row_groups,
)

# The constant-per-group term is fitted only below this boundary.
VALID_MAX_CHUNK = 131_072


def main() -> int:
    amp = load_amplification()
    data = []
    for chunk in CHUNKS:
        data.append({
            "chunk": chunk,
            "groups": touched_row_groups(chunk),
            "projected_columns": PROJECTED_COLUMNS,
            "union": amp[chunk]["predicted"],
            "measured": amp[chunk]["measured"],
        })

    inside = [d for d in data if d["chunk"] <= VALID_MAX_CHUNK]
    outside = [d for d in data if d["chunk"] > VALID_MAX_CHUNK]

    print("=== leave-two-out calibration inside the valid regime ===")
    print(f"chunks used: {[d['chunk'] for d in inside]}\n")
    print(f"{'calibrated on':>22s} {'k (KB/col-group)':>17s} "
          f"{'held-out chunk':>15s} {'pred MB':>9s} {'meas MB':>9s} {'err':>7s}")

    errors = []
    for train in combinations(inside, 2):
        test = [d for d in inside if d not in train]
        k = float(np.mean([
            (d["measured"] - d["union"]) / (d["groups"] * d["projected_columns"])
            for d in train
        ]))
        names = ",".join(f"{d['chunk']:,}" for d in train)
        for d in test:
            pred = d["union"] + k * d["groups"] * d["projected_columns"]
            err = (pred - d["measured"]) / d["measured"]
            errors.append(abs(err))
            print(f"{names:>22s} {k / 1e3:13.1f} {d['chunk']:>15,d} "
                  f"{pred / 1e6:9.1f} {d['measured'] / 1e6:9.1f} {err:+6.1%}")

    errors = np.array(errors)
    print(f"\n  held-out absolute error: median {np.median(errors):.1%}, "
          f"max {errors.max():.1%}, over {len(errors)} predictions")

    k_all = float(np.mean([
        (d["measured"] - d["union"]) / (d["groups"] * d["projected_columns"])
        for d in inside
    ]))
    print(f"\n=== single k calibrated on all four in-regime chunks: "
          f"{k_all / 1e3:.2f} KB/col-group ===")
    print(f"{'chunk':>10s} {'union MB':>9s} {'+k*|G|*|C|':>13s} {'pred MB':>9s} "
          f"{'meas MB':>9s} {'err':>7s}")
    for d in data:
        pred = d["union"] + k_all * d["groups"] * d["projected_columns"]
        err = (pred - d["measured"]) / d["measured"]
        tag = "" if d["chunk"] <= VALID_MAX_CHUNK else "   <- outside regime"
        print(f"{d['chunk']:>10,d} {d['union'] / 1e6:9.1f} "
              f"{k_all * d['groups'] * d['projected_columns'] / 1e6:13.1f} {pred / 1e6:9.1f} "
              f"{d['measured'] / 1e6:9.1f} {err:+6.1%}{tag}")

    print("\n=== does the model locate the byte optimum? ===")
    pred_min = min(data, key=lambda d: d["union"] + k_all * d["groups"] * d["projected_columns"])
    meas_min = min(data, key=lambda d: d["measured"])
    print(f"  model argmin  : {pred_min['chunk']:,} rows")
    print(f"  measured argmin: {meas_min['chunk']:,} rows")
    print("  " + ("match" if pred_min["chunk"] == meas_min["chunk"] else "MISMATCH"))

    print("\n=== why the model fails above 131,072 rows ===")
    for d in outside:
        print(f"  chunk {d['chunk']:>9,d}: union {d['union'] / 1e6:6.1f} MB, "
              f"measured {d['measured'] / 1e6:6.1f} MB, only {d['groups']} groups "
              f"touched by {64} batches")
    print("  With fewer groups than batches, several batches fall in the same")
    print("  group, so measurement sits between the union bound and a per-call")
    print("  sum. Partial reuse, not per-group cost, drives the gap there.")

    out = ROOT / "results" / "two_term_model_validation.json"
    out.write_text(json.dumps({
        "model": "bytes = union(G(I),C) + k * |G(I)| * |C|",
        "valid_max_chunk": VALID_MAX_CHUNK,
        "k_bytes_per_column_group": k_all,
        "projected_columns": PROJECTED_COLUMNS,
        "held_out_abs_error_median": float(np.median(errors)),
        "held_out_abs_error_max": float(errors.max()),
        "n_held_out_predictions": int(len(errors)),
        "model_argmin_chunk": pred_min["chunk"],
        "measured_argmin_chunk": meas_min["chunk"],
        "per_chunk": [
            {**d, "predicted": d["union"] + k_all * d["groups"] * d["projected_columns"]} for d in data
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
