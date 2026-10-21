#!/usr/bin/env python3
"Measure PyArrow ORC's default 64 MiB stripe layout on a real dataset."

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.orc as orc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colbench.formats import canonical_reader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    args = parser.parse_args()

    source = args.source.resolve()
    reader = pq.ParquetFile(source).iter_batches(batch_size=args.batch_rows)
    schema = pq.ParquetFile(source).schema_arrow
    canonical, changes = canonical_reader(
        pa.RecordBatchReader.from_batches(schema, reader)
    )

    with tempfile.TemporaryDirectory(
        prefix="orc-default-layout-", dir=str(ROOT / "results")
    ) as tmp:
        target = Path(tmp) / "default.orc"
        writer = orc.ORCWriter(target, compression="zstd")
        try:
            for batch in canonical:
                writer.write(pa.Table.from_batches([batch], schema=canonical.schema))
        finally:
            writer.close()

        f = orc.ORCFile(target)
        first = f.schema.names[0]
        stripe_rows = [f.read_stripe(i, columns=[first]).num_rows
                       for i in range(f.nstripes)]
        full = stripe_rows[:-1] if len(stripe_rows) > 1 else stripe_rows
        result = {
            "source": str(source),
            "pyarrow": pa.__version__,
            "compression": "zstd",
            "stripe_size_argument": "omitted",
            "library_default_stripe_size_bytes": 64 << 20,
            "canonical_schema_changes": [
                {"column": c, "from": before, "to": after}
                for c, before, after in changes
            ],
            "n_rows": sum(stripe_rows),
            "n_stripes": len(stripe_rows),
            "median_rows_all": statistics.median(stripe_rows),
            "median_rows_full_stripes": statistics.median(full),
            "min_rows": min(stripe_rows),
            "max_rows": max(stripe_rows),
            "stripe_rows": stripe_rows,
            "file_size_bytes": target.stat().st_size,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
