"""Access patterns A1-A5.

A3 models locality between sequential scans and fully random point access.

We parameterise that middle ground with a single knob, the **shuffle window**
``W``.  To build one batch of ``B`` rows we pick a random window start ``s`` and
draw ``B`` row ids from ``[s, s + W)``:

* ``W == B``  -> the batch is one contiguous run of rows: sequential I/O
* ``W == N``  -> the batch is spread over the whole dataset: random I/O

Sweeping ``W`` from ``B`` to ``N`` tests how layout sensitivity changes with
locality.

Note that all formats receive *identical* index arrays in *identical* order.
Whether a format sorts them internally before doing I/O is its own design
decision and part of what is being measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .formats import FormatAdapter, Predicate


def _sample_without_replacement(hi: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """``k`` distinct values from ``[0, hi)``.

    Floyd's algorithm for the sparse case keeps this O(k) instead of O(hi);
    a window of 10^8 must not cost a 10^8-element permutation.
    """
    if k > hi:
        raise ValueError(f"cannot draw {k} distinct values from {hi}")
    if hi <= 4 * k:
        return rng.choice(hi, size=k, replace=False).astype(np.int64)

    chosen: set[int] = set()
    for j in range(hi - k, hi):
        t = int(rng.integers(0, j + 1))
        chosen.add(j if t in chosen else t)
    out = np.fromiter(chosen, dtype=np.int64, count=k)
    rng.shuffle(out)
    return out


def shuffle_batches(
    n_rows: int,
    *,
    batch_rows: int,
    window: int,
    n_batches: int,
    seed: int = 0,
) -> list[np.ndarray]:
    """Row ids for ``n_batches`` batches at shuffle locality ``window``."""
    if window < batch_rows:
        raise ValueError(f"window ({window}) must be >= batch_rows ({batch_rows})")
    if window > n_rows:
        window = n_rows

    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    max_start = n_rows - window
    for _ in range(n_batches):
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        if window == batch_rows:
            idx = np.arange(start, start + batch_rows, dtype=np.int64)
            rng.shuffle(idx)  # order is shuffled, physical locality is not
        else:
            idx = _sample_without_replacement(window, batch_rows, rng) + start
        out.append(idx)
    return out


def window_sweep(n_rows: int, batch_rows: int, n_points: int = 6) -> list[int]:
    """Log-spaced windows from ``batch_rows`` (sequential) to ``n_rows`` (random)."""
    if n_rows <= batch_rows:
        return [batch_rows]
    lo, hi = np.log10(batch_rows), np.log10(n_rows)
    vals = np.unique(np.round(np.logspace(lo, hi, n_points)).astype(np.int64))
    vals = np.clip(vals, batch_rows, n_rows)
    return sorted(set(int(v) for v in vals))


@dataclass(frozen=True)
class Access:
    """One access pattern, fully determined so a run is reproducible."""

    kind: str  # scan | shuffle | point | filter
    columns: tuple[str, ...] | None = None
    batch_rows: int = 65_536
    window: int | None = None       # shuffle only
    n_batches: int = 64             # shuffle / point only
    predicate: Predicate | None = None  # filter only
    seed: int = 0

    @property
    def label(self) -> str:
        ncol = "all" if self.columns is None else str(len(self.columns))
        if self.kind == "shuffle":
            return f"A3.shuffle(W={self.window},B={self.batch_rows},cols={ncol})"
        if self.kind == "point":
            return f"A4.point(n={self.n_batches},cols={ncol})"
        if self.kind == "filter":
            p = self.predicate
            return f"A5.filter({p.column}{p.op}{p.value},cols={ncol})"
        return f"A{'1' if self.columns is None else '2'}.scan(cols={ncol})"


def full_scan(batch_rows: int = 65_536) -> Access:
    return Access(kind="scan", columns=None, batch_rows=batch_rows)


def projection_scan(columns: Sequence[str], batch_rows: int = 65_536) -> Access:
    return Access(kind="scan", columns=tuple(columns), batch_rows=batch_rows)


def shuffle_read(
    window: int,
    *,
    columns: Sequence[str] | None = None,
    batch_rows: int = 4_096,
    n_batches: int = 64,
    seed: int = 0,
) -> Access:
    return Access(
        kind="shuffle",
        columns=tuple(columns) if columns else None,
        batch_rows=batch_rows,
        window=window,
        n_batches=n_batches,
        seed=seed,
    )


def point_lookup(
    n: int = 512, *, columns: Sequence[str] | None = None, seed: int = 0
) -> Access:
    return Access(
        kind="point",
        columns=tuple(columns) if columns else None,
        batch_rows=1,
        n_batches=n,
        seed=seed,
    )


def filtered_scan(
    predicate: Predicate,
    *,
    columns: Sequence[str] | None = None,
    batch_rows: int = 65_536,
) -> Access:
    return Access(
        kind="filter",
        columns=tuple(columns) if columns else None,
        batch_rows=batch_rows,
        predicate=predicate,
    )


def plan(access: Access, n_rows: int) -> list[np.ndarray] | None:
    """Precompute index arrays so index generation never lands inside a timer."""
    if access.kind == "shuffle":
        return shuffle_batches(
            n_rows,
            batch_rows=access.batch_rows,
            window=access.window or access.batch_rows,
            n_batches=access.n_batches,
            seed=access.seed,
        )
    if access.kind == "point":
        rng = np.random.default_rng(access.seed)
        idx = rng.choice(n_rows, size=access.n_batches, replace=False).astype(np.int64)
        return [np.array([i], dtype=np.int64) for i in idx]
    return None


def execute(
    adapter: FormatAdapter,
    path,
    access: Access,
    *,
    indices: list[np.ndarray] | None = None,
) -> int:
    """Run ``access`` and return the number of rows produced.

    Pass ``indices`` from :func:`plan` so that identical row ids are used for
    every format and every repetition.
    """
    cols = list(access.columns) if access.columns else None

    if access.kind == "scan":
        return sum(b.num_rows for b in adapter.scan(path, columns=cols,
                                                    batch_rows=access.batch_rows))

    if access.kind == "filter":
        return sum(b.num_rows for b in adapter.filter_scan(
            path, access.predicate, columns=cols, batch_rows=access.batch_rows))

    if access.kind in ("shuffle", "point"):
        if indices is None:
            raise ValueError(f"{access.kind} access requires precomputed indices")
        adapter.begin_take_access(path)
        try:
            return sum(adapter.take(path, idx, columns=cols).num_rows
                       for idx in indices)
        finally:
            adapter.end_take_access(path)

    raise ValueError(f"unknown access kind {access.kind!r}")
