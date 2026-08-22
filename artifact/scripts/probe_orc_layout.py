#!/usr/bin/env python3
"""Verify that PyArrow's byte-denominated ORC control changes stripe layout."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import orc

DEFAULT_STRIPE_SIZES = (64 << 10, 256 << 10, 1 << 20, 4 << 20)


def probe(rows: int, seed: int, stripe_sizes=DEFAULT_STRIPE_SIZES) -> dict:
    rng = np.random.default_rng(seed)
    table = pa.table({
        "id": np.arange(rows, dtype=np.int64),
        "x": rng.integers(0, 2**63 - 1, size=rows, dtype=np.int64),
        "y": rng.integers(0, 2**31 - 1, size=rows, dtype=np.int32),
    })
    results = []
    with tempfile.TemporaryDirectory(prefix="colbench-orc-") as tmp:
        for stripe_size in stripe_sizes:
            path = Path(tmp) / f"stripe-{stripe_size}.orc"
            orc.write_table(
                table, path, stripe_size=stripe_size,
                row_index_stride=10_000, compression="uncompressed",
            )
            # Read through an explicitly closed handle. ORCFile keeps the
            # file open, which is harmless on POSIX but leaves the temporary
            # directory unremovable on Windows.
            with path.open("rb") as handle:
                source = orc.ORCFile(handle)
                nstripes = source.nstripes
                stripe_rows = [
                    source.read_stripe(index).num_rows
                    for index in range(nstripes)
                ]
                row_index_stride = source.row_index_stride
            results.append({
                "requested_stripe_size_bytes": stripe_size,
                "file_size_bytes": path.stat().st_size,
                "nstripes": nstripes,
                "stripe_rows_histogram": {
                    str(row_count): count
                    for row_count, count in sorted(Counter(stripe_rows).items())
                },
                "rows_accounted": sum(stripe_rows),
                "row_index_stride": row_index_stride,
            })

    counts = [result["nstripes"] for result in results]
    return {
        "pyarrow_version": pa.__version__,
        "rows": rows,
        "columns": table.num_columns,
        "seed": seed,
        "compression": "uncompressed",
        "requested_row_index_stride": 10_000,
        "stripe_layout_responds": len(set(counts)) > 1,
        "stripe_counts_nonincreasing": counts == sorted(counts, reverse=True),
        "row_index_stride_preserved": all(
            result["row_index_stride"] == 10_000 for result in results
        ),
        "settings": results,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--out", type=Path, default=Path("results/orc_layout_probe.json")
    )
    args = parser.parse_args(argv)

    report = probe(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0 if (
        report["stripe_layout_responds"]
        and report["stripe_counts_nonincreasing"]
        and report["row_index_stride_preserved"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
