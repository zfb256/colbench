"""Quantify the per-column-row-group overhead left unexplained by the byte model.

The column-chunk model predicts the bytes a layout forces a reader to fetch. It
tracks the measurement at mid and large row groups and underpredicts sharply at
small ones. This script tests whether the shortfall is proportional to the
number of projected column chunks the access touches, which is what a
per-column-row-group fixed cost would predict:

    measured = predicted_union + a + b * |G(I)| * |C|

A good fit turns "reader-side fixed overhead" from a qualitative statement into
a coefficient with units of bytes per projected column chunk.

    python scripts/fit_group_overhead.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colbench import access as access_mod  # noqa: E402

N_ROWS = 32_000_000
WINDOW = 362_039
BATCH_ROWS = 4_096
N_BATCHES = 64
SEED = 31
PROJECTED_COLUMNS = 10
CHUNKS = (1024, 8192, 32768, 131072, 524288, 1048576)


def touched_row_groups(chunk_rows: int) -> int:
    """Distinct row groups the T1 shuffle plan intersects at this chunk size.

    Row group boundaries are implied by the writer: group g covers rows
    [g*chunk, (g+1)*chunk), so no file access is needed to map an index.
    """
    batches = access_mod.shuffle_batches(
        N_ROWS, batch_rows=BATCH_ROWS, window=WINDOW,
        n_batches=N_BATCHES, seed=SEED,
    )
    groups: set[int] = set()
    for idx in batches:
        groups.update((np.asarray(idx) // chunk_rows).tolist())
    return len(groups)


def load_amplification() -> dict[int, dict]:
    """Measured and predicted shuffle bytes per chunk size."""
    path = ROOT / "results" / "read_amplification_t1.csv"
    out: dict[int, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["format"] != "parquet" or row["access_kind"] != "shuffle":
                continue
            out[int(row["chunk_rows"])] = {
                "measured": float(row["measured_read_bytes_median"]),
                "predicted": float(row["predicted_bytes_access_union"]),
            }
    return out


def main() -> int:
    amp = load_amplification()
    missing = [c for c in CHUNKS if c not in amp]
    if missing:
        raise SystemExit(f"no shuffle amplification rows for chunks {missing}")

    rows = []
    for chunk in CHUNKS:
        g = touched_row_groups(chunk)
        measured = amp[chunk]["measured"]
        predicted = amp[chunk]["predicted"]
        rows.append({
            "chunk_rows": chunk,
            "row_groups_in_file": N_ROWS // chunk,
            "row_groups_touched": g,
            "predicted_MB": predicted / 1e6,
            "measured_MB": measured / 1e6,
            "residual_MB": (measured - predicted) / 1e6,
            "residual_per_group_KB": (measured - predicted) / g / 1e3,
            "residual_per_column_group_KB": (
                (measured - predicted) / (g * PROJECTED_COLUMNS) / 1e3
            ),
        })

    print(f"{'chunk':>9s} {'groups/file':>12s} {'touched':>8s} "
          f"{'pred MB':>9s} {'meas MB':>9s} {'resid MB':>9s} {'resid/col-group KB':>19s}")
    for r in rows:
        print(f"{r['chunk_rows']:>9,d} {r['row_groups_in_file']:>12,d} "
              f"{r['row_groups_touched']:>8,d} {r['predicted_MB']:>9.1f} "
              f"{r['measured_MB']:>9.1f} {r['residual_MB']:>9.1f} "
              f"{r['residual_per_column_group_KB']:>19.2f}")

    # The 1,024-row case contributes 15,721 of 18,464 touched groups.
    g = np.array([r["row_groups_touched"] for r in rows], dtype=float)
    resid = np.array([r["residual_MB"] for r in rows], dtype=float) * 1e6
    A = np.vstack([np.ones_like(g), g]).T
    (a_all, b_all), *_ = np.linalg.lstsq(A, resid, rcond=None)
    pred_all = a_all + b_all * g
    ss_res = float(((resid - pred_all) ** 2).sum())
    ss_tot = float(((resid - resid.mean()) ** 2).sum())
    r2_all = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\nglobal straight-line fit over all six points")
    print(f"  b  = {b_all / PROJECTED_COLUMNS / 1e3:7.2f} KB per projected column chunk,  R2 = {r2_all:.4f}")
    print("  REJECTED. The 1,024-row point supplies 85% of the touched groups, so")
    print("  R2 measures leverage, not fit. Worst relative error inside the sweep:")
    worst = max(rows, key=lambda r: abs(r["residual_MB"]
                                        - (a_all + b_all * r["row_groups_touched"]) / 1e6))
    wf = (a_all + b_all * worst["row_groups_touched"]) / 1e6
    print(f"    chunk {worst['chunk_rows']:,d}: residual {worst['residual_MB']:.1f} MB, "
          f"fitted {wf:.1f} MB ({wf / max(worst['residual_MB'], 1e-9):.1f}x over)")

    # Fit the range where per-group cost dominates separately.
    SMALL = [r for r in rows if r["chunk_rows"] <= 131_072]
    LARGE = [r for r in rows if r["chunk_rows"] > 131_072]
    per_small = np.array([r["residual_per_group_KB"] for r in SMALL])
    per_large = np.array([r["residual_per_group_KB"] for r in LARGE])

    per_col_small = per_small / PROJECTED_COLUMNS
    per_col_large = per_large / PROJECTED_COLUMNS

    print(f"\nper-column-row-group excess, chunks 1,024 to 131,072")
    print(f"  mean   {per_col_small.mean():7.2f} KB per projected column chunk")
    print(f"  range  {per_col_small.min():7.2f} to {per_col_small.max():.2f} KB "
          f"({(per_small.max() / per_small.min() - 1) * 100:.1f}% spread "
          f"over a {131072 // 1024}x change in chunk size)")
    print(f"\nper-column-row-group excess, chunks 524,288 and 1,048,576")
    print(f"  {per_col_large.min():.0f} to {per_col_large.max():.0f} KB per projected column chunk, "
          f"{per_col_large.mean() / per_col_small.mean():.0f}x larger")
    print("  The constant does not extend to this regime, where few groups are")
    print("  touched and layout-induced bytes rather than per-group cost dominate.")

    out = ROOT / "results" / "group_overhead_fit.json"
    out.write_text(json.dumps({
        "quantity": "measured cold bytes minus column-chunk union prediction",
        "projected_columns": PROJECTED_COLUMNS,
        "per_column_group_excess_KB_small_chunks": {
            "chunks": [r["chunk_rows"] for r in SMALL],
            "values": [r["residual_per_column_group_KB"] for r in SMALL],
            "mean": float(per_col_small.mean()),
            "spread_pct": float((per_small.max() / per_small.min() - 1) * 100),
        },
        "per_column_group_excess_KB_large_chunks": {
            "chunks": [r["chunk_rows"] for r in LARGE],
            "values": [r["residual_per_column_group_KB"] for r in LARGE],
        },
        "global_linear_fit_rejected": {
            "b_bytes_per_column_group": b_all / PROJECTED_COLUMNS, "r_squared": r2_all,
            "reason": "high-leverage 1,024-row point supplies most touched groups",
        },
        "seed": SEED, "window": WINDOW, "batch_rows": BATCH_ROWS,
        "n_batches": N_BATCHES, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
