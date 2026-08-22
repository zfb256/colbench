#!/usr/bin/env python3
"""Summarize auxiliary cross-dataset checks and projection-width scans."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_chunk(path: Path, dataset: str) -> list[dict]:
    rows = load_jsonl(path)
    out = []
    for cache in ("cold", "warm"):
        for label in sorted({r["access"]["label"] for r in rows}):
            xs = [r for r in rows if r["cache_mode"] == cache and r["access"]["label"] == label]
            best = min(xs, key=lambda r: r["wall_s_median"])
            worst = max(xs, key=lambda r: r["wall_s_median"])
            default = next(r for r in xs if r["chunk_rows"] == 1_048_576)
            out.append({
                "dataset": dataset,
                "cache": cache,
                "access_label": label,
                "swing": worst["wall_s_median"] / best["wall_s_median"],
                "best_chunk_rows": best["chunk_rows"],
                "best_wall_s": best["wall_s_median"],
                "default_wall_s": default["wall_s_median"],
                "default_over_best": default["wall_s_median"] / best["wall_s_median"],
                "n_chunks": len(xs),
            })
    return out


def summarize_tpch_chunk() -> list[dict]:
    return summarize_chunk(RESULTS / "fig4_tpch_sf1_lineitem_parquet.jsonl", "tpch_sf1_lineitem")


def summarize_tpcds_chunk() -> list[dict]:
    return summarize_chunk(RESULTS / "fig4_tpcds_sf1_store_sales_parquet.jsonl", "tpcds_sf1_store_sales")


def summarize_column_count() -> list[dict]:
    rows = []
    for path in (
        RESULTS / "column_count_scan_tpch.jsonl",
        RESULTS / "column_count_scan_tpch_c8192.jsonl",
    ):
        if path.exists():
            rows.extend(load_jsonl(path))
    out = []
    for r in rows:
        label = r["access"]["label"]
        match = re.search(r"cols=(\d+)", label)
        out.append({
            "dataset": "tpch_sf1_lineitem",
            "cache": r["cache_mode"],
            "projected_columns": int(match.group(1)) if match else None,
            "chunk_rows": r["chunk_rows"],
            "wall_s": r["wall_s_median"],
            "read_mb": r["read_bytes_median"] / 1e6,
            "rel_iqr": r["wall_s_rel_iqr"],
        })
    return sorted(out, key=lambda r: (r["chunk_rows"], r["cache"], r["projected_columns"]))


def main() -> int:
    tpch = summarize_tpch_chunk()
    tpcds = summarize_tpcds_chunk()
    cols = summarize_column_count()
    write_rows(RESULTS / "tpch_chunk_summary.csv", tpch)
    write_rows(RESULTS / "tpcds_chunk_summary.csv", tpcds)
    write_rows(RESULTS / "column_count_scan_tpch_summary.csv", cols)
    print("TPC-H chunk summary")
    for row in tpch:
        if row["access_label"].startswith("A3") and row["cache"] == "cold":
            print(
                f"  cold shuffle: {row['swing']:.2f}x swing, "
                f"best {row['best_chunk_rows']}, "
                f"default/best {row['default_over_best']:.2f}x"
            )
    print("TPC-DS chunk summary")
    for row in tpcds:
        if row["access_label"].startswith("A3") and row["cache"] == "cold":
            print(
                f"  cold shuffle: {row['swing']:.2f}x swing, "
                f"best {row['best_chunk_rows']}, "
                f"default/best {row['default_over_best']:.2f}x"
            )
    print("Column-count scan, cold")
    for row in cols:
        if row["cache"] == "cold":
            print(
                f"  chunk={row['chunk_rows']:6d} cols={row['projected_columns']:2d}: "
                f"{row['wall_s']:.3f}s, {row['read_mb']:.1f} MB"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
