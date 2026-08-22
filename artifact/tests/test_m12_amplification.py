from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbench import access
from colbench.amplification import ParquetLayout, estimate_parquet_access, row_group_touches


def write_small_parquet(path: Path) -> None:
    table = pa.table({
        "a": pa.array(range(12), type=pa.int64()),
        "b": pa.array([x * 10 for x in range(12)], type=pa.int64()),
    })
    writer = pq.ParquetWriter(str(path), table.schema, compression="zstd")
    try:
        for start in range(0, 12, 4):
            writer.write_table(table.slice(start, 4))
    finally:
        writer.close()


def test_scan_counts_selected_column_chunks(tmp_path):
    path = tmp_path / "tiny.parquet"
    write_small_parquet(path)
    layout = ParquetLayout.from_file(path)
    est = estimate_parquet_access(layout, access.projection_scan(["a"]))

    expected = sum(layout.compressed_sizes[:, layout.column_indices(["a"])].ravel())
    assert est.bytes_required_access_union == expected
    assert est.bytes_required_per_call == expected
    assert est.rows_requested == 12
    assert est.amp_access_union == 1.0


def test_shuffle_reports_union_and_per_call_bytes(tmp_path):
    path = tmp_path / "tiny.parquet"
    write_small_parquet(path)
    layout = ParquetLayout.from_file(path)
    a = access.shuffle_read(window=4, batch_rows=2, n_batches=2, columns=["a"], seed=3)
    est = estimate_parquet_access(layout, a)

    assert est.rows_requested == 4
    assert est.bytes_required_per_call >= est.bytes_required_access_union
    assert est.amp_per_call >= est.amp_access_union
    assert est.bytes_useful > 0


def test_row_group_touches_counts_repeated_groups_once_per_call():
    plans = [pa.array([0, 1, 12]).to_numpy(), pa.array([2, 13]).to_numpy()]
    assert row_group_touches(plans, chunk_rows=10) == (4, 2)
