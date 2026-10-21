'Where the two kinds of bytes live, and why they must be able to differ.'

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def work_dir() -> Path:
    """Directory holding materialized datasets -- the device under test."""
    return Path(os.environ.get("COLBENCH_WORK") or (ROOT / "work")).resolve()


def data_dir() -> Path:
    """Directory holding downloaded / generated sources."""
    return Path(os.environ.get("COLBENCH_DATA") or (ROOT / "data")).resolve()


def results_dir() -> Path:
    return Path(os.environ.get("COLBENCH_RESULTS") or (ROOT / "results")).resolve()


SUSPECT_FILESYSTEMS = {
    "nfs", "nfs4", "cifs", "smb", "smb2", "smb3", "fuse", "fuse.sshfs",
    "fuseblk", "virtiofs", "9p", "overlay", "overlayfs", "tmpfs", "ramfs",
    "glusterfs", "ceph", "afs", "lustre",
}


def describe_mount(path: Path) -> dict:
    'Identify the filesystem and device backing ``path``.'
    path = Path(path).resolve()
    info = {"path": str(path), "mount": None, "fstype": None, "device": None,
            "suspect": False, "note": ""}

    proc = Path("/proc/mounts")
    if not proc.exists():
        info["note"] = "no /proc/mounts; cannot identify the backing device"
        return info

    best = None
    try:
        for line in proc.read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            device, mount, fstype = parts[0], parts[1].replace("\\040", " "), parts[2]
            try:
                mp = Path(mount).resolve()
            except OSError:
                continue
            if path == mp or mp in path.parents:
                if best is None or len(str(mp)) > len(str(best[1])):
                    best = (device, mp, fstype)
    except OSError as exc:
        info["note"] = f"could not read /proc/mounts: {exc}"
        return info

    if best is None:
        info["note"] = "no matching mount point found"
        return info

    device, mount, fstype = best
    info.update(mount=str(mount), fstype=fstype, device=device)

    if fstype in SUSPECT_FILESYSTEMS:
        info["suspect"] = True
        info["note"] = (
            f"{fstype!r} is not local block storage (or is RAM-backed); timings "
            "taken here do not describe a disk"
        )
    elif not device.startswith("/dev/"):
        info["suspect"] = True
        info["note"] = f"backing device {device!r} is not a block device"
    else:
        info["note"] = f"local filesystem {fstype} on {device}"
    return info


def _base_device_name(device: str) -> str:
    name = Path(device).name
    if name.startswith("nvme") and "p" in name:
        return name.rsplit("p", 1)[0]
    return name.rstrip("0123456789") or name


def virtualized(device: str | None) -> bool:
    'Whether ``device`` is a virtual block-device identity, not physical media.'
    if not device or not device.startswith("/dev/"):
        return False
    base = _base_device_name(device)
    return base.startswith(("vd", "xvd"))


def transport_hint(device: str | None) -> str | None:
    if not device or not device.startswith("/dev/"):
        return None
    base = _base_device_name(device)
    if base.startswith("vd"):
        return "virtio"
    if base.startswith("xvd"):
        return "xen-virtual-block"
    if base.startswith("nvme"):
        return "nvme"
    return None


def rotational(
    device: str | None,
    *,
    sys_block: Path = Path("/sys/block"),
) -> bool | None:
    'Return the physical media rotation flag when Linux exposes it.'
    if not device or not device.startswith("/dev/"):
        return None
    name = Path(device).name
    for candidate in (name, name.rstrip("0123456789"),
                      name.split("p")[0] if "nvme" in name else name):
        block = sys_block / candidate
        flag = block / "queue" / "rotational"
        if flag.exists():
            try:
                aggregate = flag.read_text().strip() == "1"
                slaves = block / "slaves"
                member_flags = []
                if slaves.is_dir():
                    for member in slaves.iterdir():
                        member_flag = member / "queue" / "rotational"
                        if member_flag.exists():
                            value = member_flag.read_text().strip()
                            if value in {"0", "1"}:
                                member_flags.append(value == "1")
                if member_flags:
                    return any(member_flags)
                return aggregate
            except OSError:
                return None
    return None


def _sysfs_block_dir(device: str | None) -> Path | None:
    """The ``/sys/block/<name>`` directory backing ``device``, if one exists."""
    if not device or not device.startswith("/dev/"):
        return None
    name = Path(device).name
    for candidate in (name, name.split("p")[0] if name.startswith("nvme") else name,
                      name.rstrip("0123456789")):
        d = Path(f"/sys/block/{candidate}")
        if d.is_dir():
            return d
    return None


def _read_sysfs(base: Path, *names: str) -> dict:
    out = {}
    for n in names:
        f = base / n
        try:
            out[n] = f.read_text().strip() if f.exists() else None
        except OSError:
            out[n] = None
    return out


def _mount_options(path: Path) -> str | None:
    proc = Path("/proc/mounts")
    if not proc.exists():
        return None
    path = Path(path).resolve()
    best = None
    try:
        for line in proc.read_text().splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            mp = Path(parts[1].replace("\040", " "))
            try:
                mp = mp.resolve()
            except OSError:
                continue
            if path == mp or mp in path.parents:
                if best is None or len(str(mp)) > len(str(best[0])):
                    best = (mp, parts[3])
    except OSError:
        return None
    return best[1] if best else None


QUEUE_KEYS = (
    "read_ahead_kb", "nr_requests", "scheduler", "max_sectors_kb",
    "logical_block_size", "physical_block_size", "rotational", "nomerges",
)


def io_stack(path) -> dict:
    'Kernel I/O parameters that decide how many bytes a read actually fetches.'
    path = Path(path)
    info = {"available": False, "device": None, "sysfs": None, "queue": {},
            "mount_options": None, "md": None, "note": ""}

    mount = describe_mount(path)
    info["device"] = mount.get("device")
    info["mount_options"] = _mount_options(path)

    base = _sysfs_block_dir(info["device"])
    if base is None:
        info["note"] = "no /sys/block entry; kernel I/O parameters unavailable"
        return info

    info["available"] = True
    info["sysfs"] = base.name
    info["queue"] = _read_sysfs(base / "queue", *QUEUE_KEYS)

    md_dir = base / "md"
    if md_dir.is_dir():
        md = _read_sysfs(md_dir, "level", "chunk_size", "stripe_cache_size")
        members = []
        for slave in sorted((base / "slaves").glob("*")) if (base / "slaves").is_dir() else []:
            sd = Path(f"/sys/block/{slave.name}")
            members.append({
                "name": slave.name,
                "queue": _read_sysfs(sd / "queue", *QUEUE_KEYS) if sd.is_dir() else {},
            })
        md["members"] = members
        info["md"] = md
    return info
