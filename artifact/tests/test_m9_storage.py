"""Measured storage qualification must not turn an unknown disk into NVMe."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbench import storage


def fast_random_report() -> storage.RandomReadReport:
    return storage.classify_random_trials([
        [0.4] * 32,
        [0.7] * 32,
        [1.0] * 32,
    ])


def qualify(**overrides) -> storage.StorageQualification:
    args = {
        "suspect_filesystem": False,
        "rotational": False,
        "virtualized": False,
        "eviction_supported": True,
        "disk_io_is_real": True,
        "seq_read_mbps": 400.0,
        "min_seq_read_mbps": 300.0,
        "random_read": fast_random_report(),
    }
    args.update(overrides)
    return storage.qualify_storage(**args)


def test_random_report_requires_every_trial_to_pass():
    report = storage.classify_random_trials([
        [0.5] * 32,
        [6.0] * 32,
        [0.5] * 32,
    ])
    assert report.qualified is False
    assert report.worst_median_ms == pytest.approx(6.0)
    assert report.worst_iops < storage.MIN_RANDOM_IOPS


def test_random_report_rejects_empty_trials():
    with pytest.raises(ValueError):
        storage.classify_random_trials([])
    with pytest.raises(ValueError):
        storage.classify_random_trials([[0.5], []])


def test_virtual_rota_flag_can_be_overridden_only_by_measurement():
    result = qualify(rotational=True, virtualized=True)
    assert result.qualified is True
    assert result.storage_class == "verified provider-managed virtual block storage"
    assert any("ROTA=1" in caveat for caveat in result.caveats)
    assert any("do not describe it as NVMe" in caveat for caveat in result.caveats)


def test_physical_rotational_device_remains_blocked_even_if_random_probe_is_fast():
    result = qualify(rotational=True, virtualized=False)
    assert result.qualified is False
    assert any("rotational" in blocker for blocker in result.blockers)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"suspect_filesystem": True}, "directly mounted"),
        ({"eviction_supported": False}, "eviction"),
        ({"disk_io_is_real": False}, "I/O accounting"),
        ({"seq_read_mbps": 100.0}, "throughput"),
        ({"seq_read_mbps": math.nan}, "could not be certified"),
        ({"random_read": None}, "was not measured"),
    ],
)
def test_required_evidence_is_fail_closed(override, fragment):
    result = qualify(**override)
    assert result.qualified is False
    assert any(fragment in blocker for blocker in result.blockers)


def test_hdd_like_random_access_is_blocked():
    slow = storage.classify_random_trials([[8.0] * 32] * 3)
    result = qualify(random_read=slow)
    assert result.qualified is False
    assert any("HDD-like" in blocker for blocker in result.blockers)


def test_unknown_media_can_qualify_but_is_not_relabelled():
    result = qualify(rotational=None, virtualized=False)
    assert result.qualified is True
    assert "undisclosed" in result.storage_class
    assert any("not exposed" in caveat for caveat in result.caveats)


def test_gate_report_must_match_path_device_filesystem_and_boot(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "boot_id", lambda: "boot-a")
    mount = {"device": "/dev/vda3", "fstype": "ext4"}
    report = {
        "gate_schema": storage.GATE_SCHEMA_VERSION,
        "boot_id": "boot-a",
        "storage_qualified": True,
        "gate_stability": {"qualified": True},
        "device": {
            "path": str(tmp_path.resolve()),
            "device": "/dev/vda3",
            "fstype": "ext4",
        },
    }
    assert storage.validate_gate_report(report, tmp_path, mount) == []

    report["boot_id"] = "boot-b"
    report["device"]["device"] = "/dev/vdb"
    problems = storage.validate_gate_report(report, tmp_path, mount)
    assert any("another host boot" in problem for problem in problems)
    assert any("device changed" in problem for problem in problems)


def test_unqualified_or_obsolete_gate_report_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "boot_id", lambda: None)
    problems = storage.validate_gate_report(
        {"storage_qualified": False, "device": {}},
        tmp_path,
        {"device": "/dev/vda3", "fstype": "ext4"},
    )
    assert any("schema" in problem for problem in problems)
    assert any("storage_qualified" in problem for problem in problems)


def test_md_raid_status_captures_scrub_without_changing_it(tmp_path):
    sys_block = tmp_path / "sys" / "block"
    md = sys_block / "md0" / "md"
    md.mkdir(parents=True)
    values = {
        "sync_action": "check\n",
        "sync_completed": "10 / 100\n",
        "sync_speed": "20480\n",
        "degraded": "0\n",
        "mismatch_cnt": "0\n",
    }
    for name, value in values.items():
        (md / name).write_text(value, encoding="utf-8")
    mdstat = tmp_path / "mdstat"
    mdstat.write_text("md0 : active raid1 nvme0n1[0] nvme2n1[1]\n", encoding="utf-8")

    status = storage.md_raid_status(
        "/dev/md0[/container/subdir]", sys_block=sys_block, proc_mdstat=mdstat
    )
    assert status["detected"] is True
    assert status["sync_action"] == "check"
    assert status["sync_speed_kib_s"] == "20480"
    assert status["degraded"] == status["mismatch_cnt"] == "0"


def test_md_raid_status_ignores_non_md_device(tmp_path):
    assert storage.md_raid_status("/dev/nvme0n1", sys_block=tmp_path) == {
        "detected": False,
        "device": "/dev/nvme0n1",
    }
