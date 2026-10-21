'Synthetic table generation for the controlled sweeps.'

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pyarrow as pa


@dataclass(frozen=True)
class TableSpec:
    'Declarative description of a synthetic table.'

    n_rows: int
    n_float: int = 8
    n_int: int = 8
    n_str: int = 4
    str_cardinality: int = 64
    int_range: int = 1_000
    null_rate: float = 0.0
    float_decimals: int | None = None
    seed: int = 0
    batch_rows: int = 65_536

    @property
    def n_cols(self) -> int:
        return 1 + self.n_float + self.n_int + self.n_str

    def column_names(self) -> list[str]:
        names = ["row_id"]
        names += [f"f{i}" for i in range(self.n_float)]
        names += [f"i{i}" for i in range(self.n_int)]
        names += [f"s{i}" for i in range(self.n_str)]
        return names

    def schema(self) -> pa.Schema:
        fields = [pa.field("row_id", pa.int64(), nullable=False)]
        fields += [pa.field(f"f{i}", pa.float64()) for i in range(self.n_float)]
        fields += [pa.field(f"i{i}", pa.int64()) for i in range(self.n_int)]
        fields += [pa.field(f"s{i}", pa.string()) for i in range(self.n_str)]
        return pa.schema(fields)


def _string_pool(cardinality: int, rng: np.random.Generator) -> np.ndarray:
    return np.array([f"val_{i:08d}" for i in range(cardinality)], dtype=object)


def iter_batches(spec: TableSpec) -> Iterator[pa.RecordBatch]:
    schema = spec.schema()
    rng = np.random.default_rng(spec.seed)
    pool = _string_pool(spec.str_cardinality, rng) if spec.n_str else None

    produced = 0
    while produced < spec.n_rows:
        n = min(spec.batch_rows, spec.n_rows - produced)
        arrays: list[pa.Array] = [
            pa.array(np.arange(produced, produced + n, dtype=np.int64), type=pa.int64())
        ]

        for _ in range(spec.n_float):
            vals = rng.random(n)
            if spec.float_decimals is not None:
                vals = np.round(vals, spec.float_decimals)
            arrays.append(_maybe_null(vals, spec.null_rate, rng, pa.float64()))

        for _ in range(spec.n_int):
            vals = rng.integers(0, spec.int_range, size=n, dtype=np.int64)
            arrays.append(_maybe_null(vals, spec.null_rate, rng, pa.int64()))

        for _ in range(spec.n_str):
            vals = pool[rng.integers(0, len(pool), size=n)]
            arrays.append(_maybe_null(vals, spec.null_rate, rng, pa.string()))

        yield pa.RecordBatch.from_arrays(arrays, schema=schema)
        produced += n


def _maybe_null(values, null_rate: float, rng, dtype) -> pa.Array:
    if null_rate <= 0:
        return pa.array(values, type=dtype)
    mask = rng.random(len(values)) < null_rate
    return pa.array(values, type=dtype, mask=mask)


def reader(spec: TableSpec) -> pa.RecordBatchReader:
    "A :class:`pyarrow.RecordBatchReader` -- what every adapter's ``write`` consumes."
    return pa.RecordBatchReader.from_batches(spec.schema(), iter_batches(spec))


def wide(n_rows: int, n_cols: int, seed: int = 0) -> TableSpec:
    """A wide, ML-feature-table-shaped spec with ``n_cols`` total columns."""
    if n_cols < 4:
        raise ValueError("n_cols must be >= 4")
    payload = n_cols - 1
    n_float = payload // 2
    n_int = payload - n_float
    return TableSpec(n_rows=n_rows, n_float=n_float, n_int=n_int, n_str=0, seed=seed)


PRESETS: dict[str, TableSpec] = {
    "tiny": TableSpec(n_rows=50_000, batch_rows=8_192, seed=1),
    "smoke": TableSpec(n_rows=1_000_000, seed=2),
    "incompressible": TableSpec(n_rows=1_000_000, n_str=0, int_range=2**62, seed=3),
    "compressible": TableSpec(
        n_rows=1_000_000, str_cardinality=8, int_range=16, float_decimals=2, seed=4
    ),
}
