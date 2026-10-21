#!/usr/bin/env python
'Can we manipulate kernel readahead without root or writable sysfs?'
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colbench import cache  # noqa: E402
from colbench.measure import _io_snapshot  # noqa: E402


ADVICE = {}
for name in ("POSIX_FADV_NORMAL", "POSIX_FADV_RANDOM", "POSIX_FADV_SEQUENTIAL"):
    if hasattr(os, name):
        ADVICE[name.replace("POSIX_FADV_", "").lower()] = getattr(os, name)


def check_route(path: Path) -> int | None:
    """Return an fd Parquet will read through, or None if the route is closed."""
    f = pa.OSFile(str(path), "rb")
    try:
        fd = f.fileno()
    except Exception as exc:
        print(f"  pa.OSFile.fileno() unavailable: {type(exc).__name__}: {exc}")
        f.close()
        return None
    try:
        pf = pq.ParquetFile(f)
        rows = pf.metadata.num_rows
    except Exception as exc:
        print(f"  ParquetFile(OSFile) rejected: {type(exc).__name__}: {exc}")
        f.close()
        return None
    print(f"  route OK: fd={fd}, ParquetFile sees {rows:,} rows")
    f.close()
    return fd


def one_pass(path: Path, advice_name: str, indices: list[np.ndarray],
             columns: list[str]) -> tuple[float, int]:
    """Cold-cache shuffle read under one readahead advice."""
    cache.evict([path])
    before = _io_snapshot()[0]
    f = pa.OSFile(str(path), "rb")
    fd = f.fileno()
    os.posix_fadvise(fd, 0, 0, ADVICE[advice_name])
    t0 = time.perf_counter()
    pf = pq.ParquetFile(f)
    md = pf.metadata
    starts, off = [], 0
    for i in range(md.num_row_groups):
        starts.append(off)
        off += md.row_group(i).num_rows
    starts = np.asarray(starts, dtype=np.int64)
    n = 0
    for idx in indices:
        rg = np.unique(np.searchsorted(starts, idx, side="right") - 1)
        for g in rg:
            n += pf.read_row_group(int(g), columns=columns).num_rows
    wall = time.perf_counter() - t0
    f.close()
    return wall, _io_snapshot()[0] - before


def main() -> int:
    if not hasattr(os, "posix_fadvise"):
        print("posix_fadvise unavailable: this probe is Linux-only")
        return 2
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    path = Path(sys.argv[1])
    print(f"target: {path}  ({path.stat().st_size / 1e9:.2f} GB)")
    print(f"advices available: {sorted(ADVICE)}")

    print("\n[1] descriptor route")
    if check_route(path) is None:
        print("\nRESULT: route closed -- fall back to the Aliyun sysfs sweep only")
        return 1

    print("\n[2] does the advice move kernel-accounted bytes?")
    pf = pq.ParquetFile(str(path))
    cols = [pf.schema_arrow.names[i] for i in range(min(2, pf.metadata.num_columns))]
    rng = np.random.default_rng(31)
    nrows = pf.metadata.num_rows
    idx = [np.sort(rng.choice(nrows, size=4096, replace=False)).astype(np.int64)
           for _ in range(8)]

    print(f"  8 batches x 4096 rows, columns={cols}")
    print(f"  {'advice':<12} {'wall_s':>9} {'MB read':>10}")
    base = None
    for name in ("normal", "random", "sequential"):
        if name not in ADVICE:
            continue
        wall, nbytes = one_pass(path, name, idx, cols)
        mb = nbytes / 1e6
        if base is None:
            base = mb
        print(f"  {name:<12} {wall:>9.3f} {mb:>10.1f}"
              + ("" if base in (None, 0) else f"   ({mb / base:.2f}x normal)"))

    print("\nRESULT: if random/sequential differ from normal by more than a few "
          "percent, the fadvise arm is a usable stand-in for the sysfs sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
