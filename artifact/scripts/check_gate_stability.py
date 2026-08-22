"""Qualify repeated storage-gate snapshots on a quiescent storage device.

    python scripts/check_gate_stability.py results/gate_report_autodl_*.json

Formal data require an idle RAID (or no MD RAID). ``check`` is deliberately
blocked even when stable: provider schedulers may pause and resume a scrub, and
the resulting maintenance traffic is avoidable on the replacement host.
Recovery, repair, resync, degradation, mismatches, or unstable probe
performance also fail closed.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colbench import storage  # noqa: E402

MIN_GATES = 3
MIN_SEQ_MBPS = 300.0
MAX_SEQ_RATIO = 1.25
MAX_SCRUB_SPEED_RATIO = 1.50
ALLOWED_RAID_ACTIONS = {"idle"}


def _positive_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def assess(reports: list[dict]) -> tuple[list[str], str]:
    problems: list[str] = []
    if len(reports) < MIN_GATES:
        problems.append(f"need at least {MIN_GATES} independent gates; got {len(reports)}")
        return problems, "unknown"

    identities = set()
    seq = []
    actions = []
    scrub_speeds = []
    raid_detected = False

    for i, report in enumerate(reports, 1):
        prefix = f"gate {i}"
        if report.get("gate_schema") != storage.GATE_SCHEMA_VERSION:
            problems.append(f"{prefix}: obsolete gate schema")
        if report.get("storage_qualified") is not True:
            problems.append(f"{prefix}: storage_qualified is not true")
        if report.get("blockers"):
            problems.append(f"{prefix}: blockers are present: {report['blockers']}")
        if report.get("eviction_supported") is not True:
            problems.append(f"{prefix}: page-cache eviction is not verified")
        if report.get("disk_io_is_real") is not True:
            problems.append(f"{prefix}: kernel-level storage I/O accounting is unavailable")
        if report.get("seq_read_certified") is not True:
            problems.append(f"{prefix}: sequential read is not certified")

        speed = _positive_float(report.get("seq_read_mbps"))
        if speed is None or speed < MIN_SEQ_MBPS:
            problems.append(f"{prefix}: sequential read is below {MIN_SEQ_MBPS:.0f} MB/s")
        else:
            seq.append(speed)

        random_read = report.get("random_read") or {}
        if random_read.get("qualified") is not True:
            problems.append(f"{prefix}: cold random read did not qualify")

        device = report.get("device") or {}
        identities.add((
            report.get("boot_id"), device.get("path"),
            device.get("device"), device.get("fstype"),
        ))

        raid = report.get("raid") or {}
        if raid.get("detected"):
            raid_detected = True
            action = raid.get("sync_action")
            actions.append(action)
            if action not in ALLOWED_RAID_ACTIONS:
                problems.append(f"{prefix}: RAID action {action!r} is not admissible")
            if str(raid.get("degraded")) != "0":
                problems.append(f"{prefix}: RAID is degraded")
            if str(raid.get("mismatch_cnt")) != "0":
                problems.append(f"{prefix}: RAID mismatch_cnt is nonzero")
            scrub_speed = _positive_float(raid.get("sync_speed_kib_s"))
            if action == "check" and scrub_speed is not None:
                scrub_speeds.append(scrub_speed)

    if len(identities) != 1:
        problems.append("gate snapshots do not describe one boot/path/device/filesystem")
    if len(seq) == len(reports) and max(seq) / min(seq) > MAX_SEQ_RATIO:
        problems.append(
            f"sequential throughput varies {max(seq) / min(seq):.2f}x "
            f"(limit {MAX_SEQ_RATIO:.2f}x)"
        )
    if raid_detected:
        if len(actions) != len(reports) or len(set(actions)) != 1:
            problems.append("RAID action changed during qualification")
        if len(scrub_speeds) >= 2 and max(scrub_speeds) / min(scrub_speeds) > MAX_SCRUB_SPEED_RATIO:
            problems.append(
                f"RAID check speed varies {max(scrub_speeds) / min(scrub_speeds):.2f}x "
                f"(limit {MAX_SCRUB_SPEED_RATIO:.2f}x)"
            )

    mode = actions[0] if actions and len(set(actions)) == 1 else "idle-or-non-md"
    return problems, mode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    report_paths = sorted({Path(p) for pattern in args.reports for p in glob.glob(pattern)})
    if not report_paths:
        raise SystemExit(f"no gate reports matched {args.reports}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    problems, mode = assess(reports)
    print(f"gate stability: {len(reports)} snapshots; RAID mode={mode}")
    if problems:
        for problem in problems:
            print(f"  BLOCKED: {problem}")
        return 2

    # Bind repeated qualification to the report consumed by the runner.
    authorized = dict(reports[-1])
    authorized["gate_stability"] = {
        "qualified": True,
        "snapshot_count": len(reports),
        "raid_mode": mode,
        "source_reports": [str(path.resolve()) for path in report_paths],
    }
    out = args.out or ROOT / "results" / "gate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(authorized, indent=2), encoding="utf-8")
    print("  QUALIFIED")
    print(f"  wrote authorized report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
