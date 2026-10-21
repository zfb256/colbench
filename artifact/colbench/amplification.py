'Parquet read-amplification accounting.'

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import access as acc_mod


@dataclass(frozen=True)
class ParquetLayout:
    path: Path
    n_rows: int
    column_names: tuple[str, ...]
    row_group_starts: np.ndarray
    row_group_rows: np.ndarray
    compressed_sizes: np.ndarray

    @classmethod
    def from_file(cls, path: Path) -> "ParquetLayout":
        import pyarrow.parquet as pq

        path = Path(path)
        pf = pq.ParquetFile(str(path))
        md = pf.metadata
        names = tuple(pf.schema_arrow.names)
        starts = np.empty(md.num_row_groups, dtype=np.int64)
        rows = np.empty(md.num_row_groups, dtype=np.int64)
        sizes = np.zeros((md.num_row_groups, len(names)), dtype=np.int64)
        offset = 0
        for rg_i in range(md.num_row_groups):
            rg = md.row_group(rg_i)
            starts[rg_i] = offset
            rows[rg_i] = rg.num_rows
            offset += rg.num_rows
            for col_i in range(rg.num_columns):
                sizes[rg_i, col_i] = rg.column(col_i).total_compressed_size
        return cls(path, offset, names, starts, rows, sizes)

    def column_indices(self, columns: Sequence[str] | None) -> np.ndarray:
        if columns is None:
            return np.arange(len(self.column_names), dtype=np.int64)
        lookup = {name: i for i, name in enumerate(self.column_names)}
        return np.array([lookup[c] for c in columns], dtype=np.int64)

    def row_groups_for_rows(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            return np.array([], dtype=np.int64)
        if indices.min() < 0 or indices.max() >= self.n_rows:
            raise ValueError("row index out of bounds")
        return np.unique(np.searchsorted(self.row_group_starts, indices, side="right") - 1)

    def bytes_for_groups(self, groups: Iterable[int], columns: Sequence[str] | None) -> int:
        groups = np.fromiter((int(g) for g in groups), dtype=np.int64)
        if groups.size == 0:
            return 0
        cols = self.column_indices(columns)
        return int(self.compressed_sizes[np.ix_(groups, cols)].sum())

    def average_bytes_per_row(self, columns: Sequence[str] | None) -> float:
        cols = self.column_indices(columns)
        return float(self.compressed_sizes[:, cols].sum() / self.n_rows)


@dataclass(frozen=True)
class AmplificationEstimate:
    access_label: str
    rows_requested: int
    bytes_required_access_union: int
    bytes_required_per_call: int
    bytes_useful: float

    @property
    def amp_access_union(self) -> float:
        return self.bytes_required_access_union / self.bytes_useful if self.bytes_useful else 0.0

    @property
    def amp_per_call(self) -> float:
        return self.bytes_required_per_call / self.bytes_useful if self.bytes_useful else 0.0


def row_group_touches(plans: Sequence[np.ndarray], chunk_rows: int) -> tuple[int, int]:
    """Return per-call and union row-group counts for an exact-size layout."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    groups = [np.unique(np.asarray(indices, dtype=np.int64) // chunk_rows) for indices in plans]
    per_call = sum(len(touched) for touched in groups)
    union = len(np.unique(np.concatenate(groups))) if groups else 0
    return per_call, union


def estimate_parquet_access(layout: ParquetLayout, access: acc_mod.Access) -> AmplificationEstimate:
    columns = list(access.columns) if access.columns else None
    if access.kind == "scan":
        groups = np.arange(len(layout.row_group_rows), dtype=np.int64)
        rows_requested = layout.n_rows
        required = layout.bytes_for_groups(groups, columns)
        useful = rows_requested * layout.average_bytes_per_row(columns)
        return AmplificationEstimate(access.label, rows_requested, required, required, useful)

    if access.kind not in ("shuffle", "point"):
        raise NotImplementedError(f"analytical estimate for {access.kind!r} is not implemented")

    plans = acc_mod.plan(access, layout.n_rows)
    if plans is None:
        raise ValueError(f"{access.kind} did not produce row index plans")

    all_groups: set[int] = set()
    per_call = 0
    rows_requested = 0
    for idx in plans:
        groups = layout.row_groups_for_rows(idx)
        all_groups.update(int(g) for g in groups)
        per_call += layout.bytes_for_groups(groups, columns)
        rows_requested += int(len(idx))

    union = layout.bytes_for_groups(sorted(all_groups), columns)
    useful = rows_requested * layout.average_bytes_per_row(columns)
    return AmplificationEstimate(access.label, rows_requested, union, per_call, useful)
