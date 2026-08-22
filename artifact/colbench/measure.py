"""Timing, I/O accounting, thread control and repetition statistics.

Three conditions this module enforces on every measured cell:

* Thread pools are pinned rather than left at the library default, which scales
  with the host's core count and so is not comparable across machines.
* Kernel-accounted read bytes and process CPU time are recorded alongside wall
  time, so a difference can be attributed to retrieval or to decoding.
* Statistics are the median and interquartile range over repetitions; a single
  run is reported as such and never as a measurement.
"""

from __future__ import annotations

import contextlib
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import psutil

_PROC = psutil.Process()

# Windows reports syscall bytes rather than device I/O.
DISK_IO_IS_REAL = sys.platform.startswith("linux")


@contextlib.contextmanager
def threads(n: int):
    """Pin pyarrow's CPU and I/O thread pools to ``n`` for the duration."""
    import pyarrow as pa

    prev_cpu = pa.cpu_count()
    prev_io = pa.io_thread_count()
    pa.set_cpu_count(n)
    pa.set_io_thread_count(n)
    try:
        yield
    finally:
        pa.set_cpu_count(prev_cpu)
        pa.set_io_thread_count(prev_io)


@dataclass
class Sample:
    wall_s: float
    cpu_s: float
    read_bytes: int
    read_chars: int
    rows: int
    disk_io_is_real: bool = DISK_IO_IS_REAL

    @property
    def rows_per_s(self) -> float:
        return self.rows / self.wall_s if self.wall_s > 0 else 0.0


def _io_snapshot() -> tuple[int, int]:
    try:
        c = _PROC.io_counters()
    except Exception:
        return 0, 0
    return getattr(c, "read_bytes", 0), getattr(c, "read_chars", getattr(c, "read_bytes", 0))


def run_once(fn: Callable[[], int]) -> Sample:
    """Time ``fn``; it must return the number of rows it produced."""
    rb0, rc0 = _io_snapshot()
    cpu0 = time.process_time()
    t0 = time.perf_counter()

    rows = fn()

    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    rb1, rc1 = _io_snapshot()
    return Sample(wall, cpu, max(0, rb1 - rb0), max(0, rc1 - rc0), rows)


@dataclass
class Result:
    samples: list[Sample] = field(default_factory=list)

    def _q(self, key: str, q: float) -> float:
        vals = sorted(getattr(s, key) for s in self.samples)
        if not vals:
            return 0.0
        if len(vals) == 1:
            return float(vals[0])
        pos = q * (len(vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        return float(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))

    def median(self, key: str = "wall_s") -> float:
        return self._q(key, 0.5)

    def iqr(self, key: str = "wall_s") -> float:
        return self._q(key, 0.75) - self._q(key, 0.25)

    def rel_iqr(self, key: str = "wall_s") -> float:
        """IQR as a fraction of the median -- the noise flag for a run."""
        m = self.median(key)
        return self.iqr(key) / m if m > 0 else 0.0

    def summary(self) -> dict:
        return {
            "n": len(self.samples),
            "wall_s_median": self.median("wall_s"),
            "wall_s_iqr": self.iqr("wall_s"),
            "wall_s_rel_iqr": self.rel_iqr("wall_s"),
            "cpu_s_median": self.median("cpu_s"),
            "read_bytes_median": self.median("read_bytes"),
            "read_chars_median": self.median("read_chars"),
            "rows": self.samples[0].rows if self.samples else 0,
            "rows_per_s_median": self.median("rows_per_s"),
            "disk_io_is_real": DISK_IO_IS_REAL,
            "samples": [
                {**asdict(sample), "rows_per_s": sample.rows_per_s}
                for sample in self.samples
            ],
        }


def repeat(
    fn: Callable[[], int],
    *,
    n: int = 5,
    setup: Callable[[], None] | None = None,
    warmup: int = 0,
) -> Result:
    """Run ``fn`` ``n`` times, calling ``setup`` before each run.

    ``setup`` is where cache preparation goes, so every repetition starts from
    the same cache state instead of the first run paying for all the others.
    """
    for _ in range(warmup):
        if setup:
            setup()
        fn()

    res = Result()
    for _ in range(n):
        if setup:
            setup()
        res.samples.append(run_once(fn))
    return res


def environment(path=None) -> dict:
    """Provenance for one host.

    ``path`` is the directory the measured layouts live on.  Passing it records
    the kernel I/O parameters of that device; readahead in particular changes
    kernel-accounted byte counts for identical files, so a byte model is only
    comparable across hosts that agree on them.
    """
    import pyarrow as pa

    from . import paths as _paths

    vm = psutil.virtual_memory()
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(vm.total / 1e9, 1),
        "pyarrow": pa.__version__,
        "disk_io_is_real": DISK_IO_IS_REAL,
        "native_thread_env": {
            name: os.environ.get(name)
            for name in (
                "LANCE_CPU_THREADS", "LANCE_IO_THREADS",
                "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS",
            )
        },
    }
    for mod in ("lance", "vortex"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    if path is not None:
        info["io_stack"] = _paths.io_stack(path)
    return info
