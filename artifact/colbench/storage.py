'Measured storage qualification for performance experiments.'

from __future__ import annotations

import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from colbench import cache


MAX_RANDOM_MEDIAN_MS = 5.0
MIN_RANDOM_IOPS = 250.0
GATE_SCHEMA_VERSION = 3


def md_raid_status(
    device: str | None,
    *,
    sys_block: Path = Path("/sys/block"),
    proc_mdstat: Path = Path("/proc/mdstat"),
) -> dict:
    """Capture Linux MD state for provenance; never mutate host maintenance."""
    name = Path((device or "").split("[", 1)[0]).name
    md = sys_block / name / "md"
    if not name.startswith("md") or not md.is_dir():
        return {"detected": False, "device": device}

    def read(field: str) -> str | None:
        try:
            return (md / field).read_text().strip()
        except OSError:
            return None

    try:
        mdstat = proc_mdstat.read_text().strip()
    except OSError:
        mdstat = None
    return {
        "detected": True,
        "device": device,
        "sync_action": read("sync_action"),
        "sync_completed": read("sync_completed"),
        "sync_speed_kib_s": read("sync_speed"),
        "degraded": read("degraded"),
        "mismatch_cnt": read("mismatch_cnt"),
        "mdstat": mdstat,
    }


@dataclass(frozen=True)
class RandomReadTrial:
    median_ms: float
    p95_ms: float
    iops: float


@dataclass(frozen=True)
class RandomReadReport:
    qualified: bool
    median_ms: float
    p95_ms: float
    iops: float
    worst_median_ms: float
    worst_iops: float
    trials: tuple[RandomReadTrial, ...]
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        verdict = "RULES OUT HDD-LIKE SEEKS" if self.qualified else "HDD-LIKE OR UNSTABLE"
        return (
            f"cold random read {verdict}: median={self.median_ms:.2f}ms "
            f"p95={self.p95_ms:.2f}ms iops={self.iops:.0f}; {self.detail}"
        )


@dataclass(frozen=True)
class StorageQualification:
    qualified: bool
    storage_class: str
    blockers: tuple[str, ...]
    caveats: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def boot_id() -> str | None:
    """Return the Linux boot identity so copied/stale gate reports fail closed."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def validate_gate_report(report: dict, work: Path, mount_info: dict) -> list[str]:
    """Explain why a saved gate report cannot authorize this experiment."""
    problems = []
    if report.get("gate_schema") != GATE_SCHEMA_VERSION:
        problems.append("gate report schema is missing or obsolete")
    if report.get("storage_qualified") is not True:
        problems.append("storage_qualified is not true")
    stability = report.get("gate_stability") or {}
    if stability.get("qualified") is not True:
        problems.append("repeated gate stability has not been qualified")

    recorded = report.get("device") or {}
    expected = {
        "path": str(Path(work).resolve()),
        "device": mount_info.get("device"),
        "fstype": mount_info.get("fstype"),
    }
    for key, value in expected.items():
        if recorded.get(key) != value:
            problems.append(
                f"gate device {key} changed: recorded={recorded.get(key)!r}, "
                f"current={value!r}"
            )

    current_boot = boot_id()
    if current_boot is not None and report.get("boot_id") != current_boot:
        problems.append("gate report is from another host boot")
    return problems


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def classify_random_trials(
    trial_latencies_ms: Sequence[Sequence[float]],
    *,
    max_median_ms: float = MAX_RANDOM_MEDIAN_MS,
    min_iops: float = MIN_RANDOM_IOPS,
) -> RandomReadReport:
    'Classify queue-depth-one cold random reads.'
    if not trial_latencies_ms or any(not trial for trial in trial_latencies_ms):
        raise ValueError("at least one non-empty random-read trial is required")

    reports = []
    all_values = []
    for values in trial_latencies_ms:
        values = list(values)
        all_values.extend(values)
        elapsed_s = sum(values) / 1000.0
        reports.append(RandomReadTrial(
            median_ms=statistics.median(values),
            p95_ms=_percentile(values, 0.95),
            iops=len(values) / elapsed_s if elapsed_s else float("inf"),
        ))

    worst_median = max(t.median_ms for t in reports)
    worst_iops = min(t.iops for t in reports)
    qualified = all(
        t.median_ms <= max_median_ms and t.iops >= min_iops
        for t in reports
    )
    detail = (
        f"all {len(reports)} trials passed HDD-exclusion thresholds "
        f"(median <= {max_median_ms:.1f}ms, IOPS >= {min_iops:.0f})"
        if qualified else
        f"at least one of {len(reports)} trials failed HDD-exclusion thresholds "
        f"(median <= {max_median_ms:.1f}ms, IOPS >= {min_iops:.0f})"
    )
    elapsed_s = sum(all_values) / 1000.0
    return RandomReadReport(
        qualified=qualified,
        median_ms=statistics.median(all_values),
        p95_ms=_percentile(all_values, 0.95),
        iops=len(all_values) / elapsed_s if elapsed_s else float("inf"),
        worst_median_ms=worst_median,
        worst_iops=worst_iops,
        trials=tuple(reports),
        detail=detail,
    )


def benchmark_cold_random_reads(
    work: Path,
    *,
    size_mb: int = 512,
    reads_per_trial: int = 128,
    trials: int = 3,
    block_size: int = 4096,
) -> RandomReadReport:
    'Measure cold 4-KiB random reads at queue depth one.'
    if not hasattr(os, "pread"):
        raise RuntimeError("cold random-read qualification requires os.pread")

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    probe = work / "_gate_random.bin"
    one_mb = 1 << 20
    payload = bytearray(os.urandom(one_mb))
    with open(probe, "wb", buffering=0) as fh:
        for index in range(size_mb):
            payload[:8] = index.to_bytes(8, "little")
            fh.write(payload)
        os.fsync(fh.fileno())

    try:
        total_blocks = (size_mb * one_mb) // block_size
        total_reads = reads_per_trial * trials
        if total_reads > total_blocks:
            raise ValueError("probe file is too small for unique random offsets")
        offsets = random.Random(0xC01B_EC7).sample(range(total_blocks), total_reads)
        measured: list[list[float]] = []

        for trial in range(trials):
            cache.evict([probe])
            fd = os.open(probe, os.O_RDONLY)
            try:
                if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_RANDOM"):
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
                latencies = []
                begin = trial * reads_per_trial
                for block in offsets[begin:begin + reads_per_trial]:
                    started = time.perf_counter_ns()
                    data = os.pread(fd, block_size, block * block_size)
                    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                    if len(data) != block_size:
                        raise OSError("short read in random storage probe")
                    latencies.append(elapsed_ms)
                measured.append(latencies)
            finally:
                os.close(fd)

        return classify_random_trials(measured)
    finally:
        probe.unlink(missing_ok=True)


def qualify_storage(
    *,
    suspect_filesystem: bool,
    rotational: bool | None,
    virtualized: bool,
    eviction_supported: bool,
    disk_io_is_real: bool,
    seq_read_mbps: float,
    min_seq_read_mbps: float,
    random_read: RandomReadReport | None,
) -> StorageQualification:
    """Combine semantics and measurements into a conservative gate verdict."""
    blockers: list[str] = []
    caveats: list[str] = []

    if suspect_filesystem:
        blockers.append("work/ is not on a directly mounted block filesystem")
    if not eviction_supported:
        blockers.append("page-cache eviction is not reliable; cold rows are invalid")
    if not disk_io_is_real:
        blockers.append("kernel-level storage I/O accounting is unavailable")
    if math.isnan(seq_read_mbps):
        blockers.append("sequential storage throughput could not be certified")
    elif seq_read_mbps < min_seq_read_mbps:
        blockers.append(
            f"certified sequential throughput is only {seq_read_mbps:.0f} MB/s "
            f"(< {min_seq_read_mbps:.0f} MB/s)"
        )
    if random_read is None:
        blockers.append("cold random-read storage class was not measured")
    elif not random_read.qualified:
        blockers.append("cold random access is HDD-like or unstable")

    if rotational is True and not virtualized:
        blockers.append("kernel identifies the physical block device as rotational")
    elif rotational is True and virtualized:
        caveats.append(
            "kernel ROTA=1 is ignored only as a virtual-device hint; measured cold "
            "random access rules out HDD-like seeks"
        )
    elif rotational is None:
        caveats.append("physical media type is not exposed by the host")

    if virtualized:
        caveats.append(
            "provider backend and transport are unknown; report virtual block "
            "storage and do not describe it as NVMe or local SSD"
        )

    qualified = not blockers
    if qualified and virtualized:
        storage_class = "verified provider-managed virtual block storage"
    elif qualified and rotational is False:
        storage_class = "verified non-rotational block storage"
    elif qualified:
        storage_class = "verified block storage (physical media undisclosed)"
    else:
        storage_class = "unqualified storage"
    return StorageQualification(qualified, storage_class, tuple(blockers), tuple(caveats))
