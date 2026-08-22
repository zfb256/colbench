"""Repeated gates must fail closed when any RAID scrub is active."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbench import storage
from scripts import check_gate_stability
from scripts.check_gate_stability import assess


def report(*, action="check", seq=1000, scrub_speed=20_000):
    return {
        "gate_schema": storage.GATE_SCHEMA_VERSION,
        "boot_id": "boot-a",
        "storage_qualified": True,
        "blockers": [],
        "eviction_supported": True,
        "disk_io_is_real": True,
        "seq_read_certified": True,
        "seq_read_mbps": seq,
        "random_read": {"qualified": True},
        "device": {"path": "/work", "device": "/dev/md0", "fstype": "xfs"},
        "raid": {
            "detected": True,
            "sync_action": action,
            "sync_speed_kib_s": str(scrub_speed),
            "degraded": "0",
            "mismatch_cnt": "0",
        },
    }


def test_three_stable_check_gates_are_blocked():
    problems, mode = assess([report(seq=n) for n in (950, 1000, 1050)])
    assert all("not admissible" in problem for problem in problems)
    assert mode == "check"


def test_recovery_or_changed_action_is_blocked():
    reports = [report(), report(), report(action="recovery")]
    problems, _ = assess(reports)
    assert any("not admissible" in p for p in problems)
    assert any("changed" in p for p in problems)


def test_unstable_throughput_is_blocked():
    problems, _ = assess([report(seq=n) for n in (400, 800, 1000)])
    assert any("throughput varies" in p for p in problems)


def test_degraded_or_mismatched_raid_is_blocked():
    bad = copy.deepcopy(report())
    bad["raid"]["degraded"] = "1"
    bad["raid"]["mismatch_cnt"] = "2"
    problems, _ = assess([report(), report(), bad])
    assert any("degraded" in p for p in problems)
    assert any("mismatch_cnt" in p for p in problems)


def test_successful_checker_writes_runner_authorization(monkeypatch, tmp_path):
    result_dir = tmp_path / "results"
    paths = []
    for index, seq in enumerate((950, 1000, 1050), 1):
        path = tmp_path / f"gate-{index}.json"
        path.write_text(json.dumps(report(action="idle", seq=seq)), encoding="utf-8")
        paths.append(path)
    monkeypatch.setattr(check_gate_stability, "ROOT", tmp_path)
    assert check_gate_stability.main([
        *(str(path) for path in paths), "--out", str(result_dir / "gate_report.json")
    ]) == 0
    authorized = json.loads((result_dir / "gate_report.json").read_text())
    assert authorized["gate_stability"] == {
        "qualified": True,
        "snapshot_count": 3,
        "raid_mode": "idle",
        "source_reports": [str(path.resolve()) for path in paths],
    }
