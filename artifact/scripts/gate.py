"""D1 go/no-go gate.

Answers, on *this* machine and with measurements rather than assumptions:

1. Are all three format adapters importable?
2. Does cold random access rule out HDD-like seek behaviour?
3. Does page-cache eviction actually work here, so "cold" means cold?
4. Is ``read_bytes`` real device I/O, or only syscall bytes?
5. Is there enough free space for the matrix?

Exit code 0 = green, 1 = qualified storage with reduced experiment scope,
2 = blocked.

    python scripts/gate.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colbench import cache, measure, paths, storage  # noqa: E402
from colbench.formats import ADAPTERS  # noqa: E402

MIN_FREE_GB = 60.0
MIN_SEQ_READ_MBPS = 300.0


def check_device(work: Path) -> dict:
    """Identify what work/ actually sits on.

    Every timed read comes from work/, so this decides what the whole study
    measured.  On a host with several volumes it is entirely possible to
    benchmark a network mount by accident and never notice, because the numbers
    look plausible -- just wrong.
    """
    _hdr("0. device under test")
    info = paths.describe_mount(work)
    spin = paths.rotational(info.get("device"))

    print(f"  work dir : {info['path']}")
    print(f"  mount    : {info['mount']}")
    print(f"  fstype   : {info['fstype']}")
    print(f"  device   : {info['device']}")
    is_virtual = paths.virtualized(info.get("device"))
    transport = paths.transport_hint(info.get("device"))
    if spin is True and is_virtual:
        print("  rotation : ROTA=1 (virtual-device hint; physical media unknown)")
    elif spin is True:
        print("  rotation : SPINNING DISK")
    elif spin is False:
        print("  rotation : solid state")

    if info["suspect"]:
        print(f"  SUSPECT  : {info['note']}")
        print("  -> set COLBENCH_WORK to a directly mounted block filesystem and rerun.")
        print("     Sources may stay on the slow volume (COLBENCH_DATA); they are")
        print("     read only during materialisation, never inside a timer.")
    elif info["device"] is None:
        # An unidentified device cannot be certified.
        print(f"  UNKNOWN  : {info['note']}")
        print("  -> the backing device could not be identified; verify by hand")
        print("     its storage semantics before trusting any timing.")
    else:
        print(f"  OK       : {info['note']}")

    info["rotational"] = spin
    info["virtualized"] = is_virtual
    info["transport_hint"] = transport
    return info


def _hdr(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def check_formats() -> tuple[list[str], list[str]]:
    _hdr("1. format adapters")
    live, dead = [], []
    for name, cls in ADAPTERS.items():
        ok, detail = cls.probe()
        print(f"  {'OK  ' if ok else 'MISS'}  {name:9s} {detail}")
        (live if ok else dead).append(name)
    return live, dead


def check_space(work: Path) -> float:
    _hdr("2. disk space")
    work.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(work).free / 1e9
    verdict = "OK" if free >= MIN_FREE_GB else "TIGHT"
    print(f"  {verdict}  {free:.1f} GB free at {work}  (want >= {MIN_FREE_GB:.0f} GB)")
    return free


def check_sequential_read(work: Path, size_mb: int, evict_works: bool) -> float:
    """Measure raw sequential read to sanity-check the storage class.

    Only meaningful when eviction works.  Without it the read is served from the
    page cache and reports RAM bandwidth, which would wrongly certify a slow
    device as fast -- so the number is reported as unverified instead.
    """
    _hdr("4. sequential read throughput")
    probe = work / "_gate_seq.bin"
    block = os.urandom(1 << 20)
    with open(probe, "wb") as fh:
        for _ in range(size_mb):
            fh.write(block)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        cache.evict([probe])
        t0 = time.perf_counter()
        with open(probe, "rb", buffering=0) as fh:
            while fh.read(1 << 22):
                pass
        mbps = size_mb / (time.perf_counter() - t0)
    finally:
        probe.unlink(missing_ok=True)

    if not evict_works:
        print(f"  ????  {mbps:.0f} MB/s  -- UNVERIFIED: eviction failed, so this")
        print("        is very likely page-cache bandwidth, not the device.")
        print("        Storage class cannot be certified on this host.")
        return float("nan")

    verdict = "OK" if mbps >= MIN_SEQ_READ_MBPS else "SLOW"
    print(f"  {verdict}  {mbps:.0f} MB/s  (want >= {MIN_SEQ_READ_MBPS:.0f} MB/s)")
    if mbps < MIN_SEQ_READ_MBPS:
        print("  -> looks like a spinning disk or network storage. Results would")
        print("     make the format comparison storage-bound and non-comparable.")
    return mbps


def check_eviction(work: Path) -> cache.EvictionReport:
    _hdr("3. page-cache eviction")
    rep = cache.verify_eviction(
        work, size_mb=256, trials=cache.QUALIFICATION_TRIALS)
    print(f"  {'OK  ' if rep.supported else 'FAIL'}  {rep}")
    if not rep.supported:
        print("  -> cold-cache numbers here are NOT evidence. Warm-cache")
        print("     comparisons and correctness testing remain valid.")
    return rep


def check_random_read(work: Path, evict_works: bool) -> storage.RandomReadReport | None:
    _hdr("5. cold random-read qualification")
    if not evict_works:
        print("  SKIP  eviction failed, so random reads cannot be certified cold")
        return None
    try:
        rep = storage.benchmark_cold_random_reads(work)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"  FAIL  random-read probe could not run: {exc}")
        return None
    print(f"  {'OK  ' if rep.qualified else 'FAIL'}  {rep}")
    return rep


def check_io_accounting() -> bool:
    _hdr("6. I/O accounting")
    real = measure.DISK_IO_IS_REAL
    print(f"  {'OK  ' if real else 'WARN'}  read_bytes is "
          f"{'kernel-accounted storage I/O' if real else 'syscall-level (counts cache hits)'}")
    if not real:
        print("  -> cannot attribute time to bytes-read vs decode on this OS.")
    return real


def main() -> int:
    work = paths.work_dir()
    print("colbench D1 gate")
    print(f"host: {measure.environment()['platform']}")
    print(f"work: {work}    data: {paths.data_dir()}")

    work.mkdir(parents=True, exist_ok=True)
    device = check_device(work)
    raid = storage.md_raid_status(device.get("device"))
    if raid.get("detected"):
        _hdr("0b. RAID background state")
        print(f"  action       : {raid.get('sync_action')}")
        print(f"  completed    : {raid.get('sync_completed')}")
        print(f"  speed KiB/s  : {raid.get('sync_speed_kib_s')}")
        print(f"  degraded     : {raid.get('degraded')}")
        print(f"  mismatch_cnt : {raid.get('mismatch_cnt')}")
    live, dead = check_formats()
    free = check_space(work)
    # Verify eviction before measuring throughput.
    evict = check_eviction(work)
    mbps = check_sequential_read(work, 512, evict.supported)
    random_read = check_random_read(work, evict.supported)
    io_real = check_io_accounting()

    _hdr("verdict")
    blockers, degraded = [], []

    if "parquet" not in live:
        blockers.append("pyarrow missing - nothing can run")
    qualification = storage.qualify_storage(
        suspect_filesystem=bool(device.get("suspect") or device.get("device") is None),
        rotational=device.get("rotational"),
        virtualized=bool(device.get("virtualized")),
        eviction_supported=evict.supported,
        disk_io_is_real=io_real,
        seq_read_mbps=mbps,
        min_seq_read_mbps=MIN_SEQ_READ_MBPS,
        random_read=random_read,
    )
    blockers.extend(qualification.blockers)
    if free < MIN_FREE_GB:
        degraded.append(f"only {free:.1f} GB free - add/free storage; do not shrink the matrix")
    if dead:
        degraded.append(f"formats unavailable: {', '.join(dead)}")
    for caveat in qualification.caveats:
        print(f"  CAVEAT: {caveat}")

    report = {
        "gate_schema": storage.GATE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "boot_id": storage.boot_id(),
        "platform": measure.environment(),
        "device": device,
        "raid": raid,
        "formats_live": live,
        "formats_missing": dead,
        "free_gb": round(free, 1),
        "seq_read_mbps": None if mbps != mbps else round(mbps),
        "seq_read_certified": mbps == mbps,
        "random_read": random_read.to_dict() if random_read else None,
        "eviction_supported": evict.supported,
        "eviction_ratio": round(evict.ratio, 2),
        "disk_io_is_real": io_real,
        "storage_qualification": qualification.to_dict(),
        "storage_qualified": qualification.qualified,
        "blockers": blockers,
        "degraded": degraded,
    }
    out = paths.results_dir() / "gate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if blockers:
        for b in blockers:
            print(f"  BLOCKED: {b}")
        print("\n  -> do not run performance measurements until the blockers are fixed")
        code = 2
    elif degraded:
        for d in degraded:
            print(f"  DEGRADED: {d}")
        print("\n  -> qualified-scope performance work can proceed here")
        print("     resolve or explicitly report the reduced-scope items")
        code = 1
    else:
        print(f"  GREEN - full matrix may run on {qualification.storage_class}")
        code = 0

    print(f"\nwrote {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
