'Page-cache control.'

from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


class CacheMode(str, Enum):
    COLD = "cold"
    WARM = "warm"


IS_LINUX = sys.platform.startswith("linux")
IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"

_READ_BLOCK = 1 << 22

QUALIFICATION_TRIALS = 3


def evict(paths: Iterable[Path]) -> None:
    """Drop ``paths`` from the OS page cache, as far as the platform allows."""
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        if IS_LINUX:
            _evict_posix(path)
        elif IS_WINDOWS:
            _evict_windows(path)
        elif IS_MACOS:
            _evict_posix(path)  # advisory only on Darwin; verify before trusting


def _evict_posix(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(fd)
        except OSError:
            pass
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _evict_windows(path: Path) -> None:
    'Open the file unbuffered, which flushes its cached pages.'
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    OPEN_EXISTING = 3
    FILE_FLAG_NO_BUFFERING = 0x20000000
    INVALID_HANDLE = ctypes.c_void_p(-1).value

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    handle = kernel32.CreateFileW(
        str(path), GENERIC_READ, FILE_SHARE_READ, None,
        OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, None,
    )
    if handle and handle != INVALID_HANDLE:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def warm(paths: Iterable[Path]) -> None:
    """Pull ``paths`` fully into the page cache."""
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        with open(path, "rb", buffering=0) as fh:
            while fh.read(_READ_BLOCK):
                pass


def prepare(paths: Sequence[Path], mode: CacheMode) -> None:
    if mode is CacheMode.COLD:
        evict(paths)
    else:
        warm(paths)


@dataclass
class EvictionReport:
    supported: bool
    cold_s: float
    warm_s: float
    ratio: float
    detail: str

    def __str__(self) -> str:
        verdict = "WORKS" if self.supported else "DOES NOT WORK"
        return (
            f"page-cache eviction {verdict} here: "
            f"cold={self.cold_s * 1000:.1f}ms warm={self.warm_s * 1000:.1f}ms "
            f"worst-ratio={self.ratio:.2f}x -- {self.detail}"
        )


def verify_eviction(
    tmp_dir: Path,
    *,
    size_mb: int = 256,
    min_ratio: float = 2.0,
    trials: int = 5,
) -> EvictionReport:
    'Write a scratch file, then check that an evicted read is really slower.'
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    probe = tmp_dir / "_cache_probe.bin"

    payload = os.urandom(1 << 20)
    with open(probe, "wb") as fh:
        for _ in range(size_mb):
            fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())

    try:
        cold_times, warm_times, ratios = [], [], []
        for _ in range(trials):
            evict([probe])
            cold_t = _timed_read(probe)
            warm_t = _timed_read(probe)  # the previous read left it hot
            cold_times.append(cold_t)
            warm_times.append(warm_t)
            ratios.append(cold_t / warm_t if warm_t > 0 else float("inf"))

        cold = sorted(cold_times)[len(cold_times) // 2]
        warm_s = sorted(warm_times)[len(warm_times) // 2]
        worst = min(ratios)
        median_ratio = sorted(ratios)[len(ratios) // 2]

        if worst >= min_ratio:
            ok = True
            detail = f"every one of {trials} trials showed a cold penalty"
        elif median_ratio >= min_ratio:
            ok = False
            detail = (
                f"eviction is INTERMITTENT (worst trial {worst:.2f}x of {trials}) -- "
                "treated as unsupported, because a cold row that is sometimes warm "
                "is worse than one that is never cold"
            )
        else:
            ok = False
            detail = (
                "cold and warm reads are indistinguishable -- eviction is a no-op "
                "on this platform, so cold-cache numbers here are meaningless"
            )
        return EvictionReport(ok, cold, warm_s, worst, detail)
    finally:
        probe.unlink(missing_ok=True)


def _timed_read(path: Path) -> float:
    start = time.perf_counter()
    with open(path, "rb", buffering=0) as fh:
        while fh.read(_READ_BLOCK):
            pass
    return time.perf_counter() - start
