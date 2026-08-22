#!/usr/bin/env python3
"""Screen TPC-H lineitem for the two-term Parquet byte model."""

from __future__ import annotations

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
K_BYTES_PER_COLUMN_GROUP = 6_210


def pick_columns(ds: runner.Dataset) -> list[str]:
    names = ds.column_names()
    k = max(1, round(len(names) * 0.1))
    step = max(1, len(names) // k)
    return names[::step][:k]


def main() -> int:
    ds = runner.Dataset("tpch_sf1_lineitem", source=paths.data_dir() / "tpch_sf1_lineitem.parquet")
    cols = pick_columns(ds)
    window = access.window_sweep(ds.n_rows(), 4096, n_points=5)[2]
    acc = access.shuffle_read(window=window, batch_rows=4096, n_batches=64, columns=cols, seed=31)
    adapter = ParquetAdapter()

    rows = []
    for chunk in CHUNKS:
        path = paths.work_dir() / f"{ds.name}__parquet__c{chunk}.parquet"
        adapter.write(ds.reader(), path, chunk_rows=chunk, compression="zstd")
        try:
            layout = ParquetLayout.from_file(path)
            plans = access.plan(acc, layout.n_rows)
            union_groups: set[int] = set()
            per_call_groups = 0
            for idx in plans or []:
                groups = layout.row_groups_for_rows(idx)
                union_groups.update(int(g) for g in groups)
                per_call_groups += len(groups)
            est = estimate_parquet_access(layout, acc)
            rows.append({
                "dataset": ds.name,
                "access_label": acc.label,
                "columns": ",".join(cols),
                "chunk_rows": chunk,
                "union_groups": len(union_groups),
                "per_call_groups": per_call_groups,
                "projected_columns": len(cols),
                "union_mb": est.bytes_required_access_union / 1e6,
                "per_call_mb": est.bytes_required_per_call / 1e6,
                "two_term_union_mb": (
                    est.bytes_required_access_union
                    + K_BYTES_PER_COLUMN_GROUP * len(cols) * len(union_groups)
                ) / 1e6,
                "two_term_per_call_mb": (
                    est.bytes_required_per_call
                    + K_BYTES_PER_COLUMN_GROUP * len(cols) * per_call_groups
                ) / 1e6,
            })
        finally:
            path.unlink(missing_ok=True)

    out_csv = paths.results_dir() / "tpch_model_screen.csv"
    out_json = paths.results_dir() / "tpch_model_screen.json"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    best_union = min(rows, key=lambda r: r["two_term_union_mb"])
    worst_union = max(rows, key=lambda r: r["two_term_union_mb"])
    best_call = min(rows, key=lambda r: r["two_term_per_call_mb"])
    worst_call = max(rows, key=lambda r: r["two_term_per_call_mb"])
    print(f"wrote {out_csv}")
    print(f"wrote {out_json}")
    print(
        "two-term union: "
        f"{worst_union['two_term_union_mb'] / best_union['two_term_union_mb']:.2f}x, "
        f"best chunk {best_union['chunk_rows']}"
    )
    print(
        "two-term per-call: "
        f"{worst_call['two_term_per_call_mb'] / best_call['two_term_per_call_mb']:.2f}x, "
        f"best chunk {best_call['chunk_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
