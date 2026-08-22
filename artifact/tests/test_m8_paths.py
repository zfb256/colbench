"""Milestone 8 self-test: path resolution and device identification.

The point of this layer is to stop the study silently measuring the wrong
volume. On a host with several disks -- a small native NVMe and a larger rented
mount, say -- putting ``work/`` on the mount produces numbers that look entirely
plausible and describe the network rather than the format.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbench import paths


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("COLBENCH_WORK", "COLBENCH_DATA", "COLBENCH_RESULTS"):
        monkeypatch.delenv(var, raising=False)


def test_defaults_are_repo_relative():
    assert paths.work_dir() == (paths.ROOT / "work").resolve()
    assert paths.data_dir() == (paths.ROOT / "data").resolve()
    assert paths.results_dir() == (paths.ROOT / "results").resolve()


def test_env_overrides_are_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv("COLBENCH_WORK", str(tmp_path / "nvme"))
    monkeypatch.setenv("COLBENCH_DATA", str(tmp_path / "mount"))
    assert paths.work_dir() == (tmp_path / "nvme").resolve()
    assert paths.data_dir() == (tmp_path / "mount").resolve()


def test_work_and_data_can_be_on_different_volumes(monkeypatch, tmp_path):
    """The whole reason this module exists."""
    monkeypatch.setenv("COLBENCH_WORK", str(tmp_path / "a"))
    monkeypatch.setenv("COLBENCH_DATA", str(tmp_path / "b"))
    assert paths.work_dir() != paths.data_dir()


def test_describe_mount_returns_the_expected_shape(tmp_path):
    info = paths.describe_mount(tmp_path)
    for key in ("path", "mount", "fstype", "device", "suspect", "note"):
        assert key in info
    assert isinstance(info["suspect"], bool)
    assert info["note"], "must always explain its verdict"


def test_describe_mount_does_not_raise_off_linux(tmp_path):
    info = paths.describe_mount(tmp_path / "does" / "not" / "exist")
    assert isinstance(info, dict)


@pytest.mark.parametrize("fstype", ["nfs4", "overlay", "tmpfs", "virtiofs", "cifs"])
def test_network_and_ram_filesystems_are_flagged(fstype):
    """A rented mount is usually one of these; none can carry the study."""
    assert fstype in paths.SUSPECT_FILESYSTEMS


@pytest.mark.parametrize("fstype", ["ext4", "xfs", "btrfs", "zfs", "ntfs"])
def test_real_local_filesystems_are_not_flagged(fstype):
    assert fstype not in paths.SUSPECT_FILESYSTEMS


def test_mount_detection_prefers_the_longest_matching_prefix(tmp_path, monkeypatch):
    """/mnt/data must win over / for a path under /mnt/data."""
    fake = tmp_path / "mounts"
    fake.write_text(
        "/dev/sda1 / ext4 rw 0 0\n"
        f"/dev/vdb {tmp_path.as_posix()} nfs4 rw 0 0\n"
    )
    real_exists, real_read = Path.exists, Path.read_text
    payload = fake.read_text()

    # Path("/proc/mounts") renders with backslashes on Windows.
    def is_proc(p) -> bool:
        return p.as_posix() == "/proc/mounts"

    def exists(self):
        return True if is_proc(self) else real_exists(self)

    def read_text(self, *a, **k):
        return payload if is_proc(self) else real_read(self, *a, **k)

    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(Path, "read_text", read_text)

    info = paths.describe_mount(tmp_path / "sub" / "dir")
    assert info["fstype"] == "nfs4"
    assert info["suspect"] is True
    assert "not local block storage" in info["note"]


def test_rotational_unknown_for_non_block_devices():
    assert paths.rotational(None) is None
    assert paths.rotational("tmpfs") is None
    assert paths.rotational("/dev/definitely-not-a-real-device") is None


def test_md_rotational_hint_defers_to_nonrotational_members(tmp_path):
    md = tmp_path / "md0"
    (md / "queue").mkdir(parents=True)
    (md / "queue" / "rotational").write_text("1\n")
    for name in ("nvme0n1", "nvme1n1", "nvme2n1"):
        member = md / "slaves" / name
        (member / "queue").mkdir(parents=True)
        (member / "queue" / "rotational").write_text("0\n")

    assert paths.rotational("/dev/md0", sys_block=tmp_path) is False


def test_md_with_any_rotational_member_remains_rotational(tmp_path):
    md = tmp_path / "md0"
    (md / "queue").mkdir(parents=True)
    (md / "queue" / "rotational").write_text("1\n")
    for name, rota in (("nvme0n1", "0"), ("sda", "1")):
        member = md / "slaves" / name
        (member / "queue").mkdir(parents=True)
        (member / "queue" / "rotational").write_text(f"{rota}\n")

    assert paths.rotational("/dev/md0", sys_block=tmp_path) is True


@pytest.mark.parametrize("device", ["/dev/vda", "/dev/vda3", "/dev/xvdb1"])
def test_virtual_block_devices_are_identity_hints_not_physical_media(device):
    assert paths.virtualized(device) is True


@pytest.mark.parametrize("device", ["/dev/sda", "/dev/nvme0n1p3", None, "tmpfs"])
def test_physical_or_non_device_names_are_not_called_virtual(device):
    assert paths.virtualized(device) is False


def test_transport_hints_are_conservative():
    assert paths.transport_hint("/dev/vda3") == "virtio"
    assert paths.transport_hint("/dev/xvdb1") == "xen-virtual-block"
    assert paths.transport_hint("/dev/nvme0n1p3") == "nvme"
    assert paths.transport_hint("/dev/sda1") is None
