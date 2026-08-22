"""Regenerate MANIFEST.txt and SHA256SUMS.txt for the whole bundle.

Run this after any change to the bundle, and after transferring it, so the
recorded digests describe what is actually present.

    python scripts/refresh_checksums.py --verify      # check only, change nothing
    python scripts/refresh_checksums.py               # check, then rewrite

Paths are recorded **relative to the bundle root**, so

    cd <bundle-root> && sha256sum -c SHA256SUMS.txt

works on any machine. An earlier revision of this bundle recorded absolute
`/root/colbench/...` paths from the measurement host, which made the file
unverifiable anywhere else.

Exit codes: 0 = everything matched (or was rewritten), 1 = mismatches or
missing files were found. A mismatch on a large data file usually means the
file was still being copied when the digest was taken -- rerun once the copy
has finished rather than trusting the new value.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "MANIFEST.txt"
SUMS = ROOT / "SHA256SUMS.txt"

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", ".idea", "colbench-anon"}
SELF = {MANIFEST.name, SUMS.name}
SKIP_PATHS = {"paper/main.aux", "paper/main.blg", "paper/main.log"}


def iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_dir() or any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.parent == root and p.name in SELF:
            continue
        if p.relative_to(root).as_posix() in SKIP_PATHS:
            continue
        yield p


def sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def read_existing() -> dict[str, str]:
    """Parse the current SHA256SUMS, tolerating the legacy absolute-path form."""
    out: dict[str, str] = {}
    if not SUMS.exists():
        return out
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        rel = name.strip().lstrip("*")
        for legacy in ("/root/colbench/", "/root/colbench_transfer/"):
            if rel.startswith(legacy):
                rel = rel[len(legacy):]
        out[rel] = digest
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="report differences without rewriting the files")
    args = ap.parse_args(argv)

    print(f"bundle root: {ROOT}")
    digests: dict[str, str] = {}
    unreadable: list[tuple[str, str]] = []
    for path in iter_files(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        try:
            digests[rel] = sha256(path)
        except OSError as exc:
            # Never record a digest for a file that may still be in flight.
            unreadable.append((rel, f"{type(exc).__name__}: {exc}"))
    print(f"hashed {len(digests)} files")

    if unreadable:
        print(f"\n  UNREADABLE ({len(unreadable)}) -- refusing to record a digest:")
        for rel, why in unreadable:
            print(f"     {rel}\n       {why}")
        print("\n  These files are probably still being written. Wait for the")
        print("  transfer to finish and rerun; do not rewrite the checksum file")
        print("  while any file is in flight.")
        return 1

    old = read_existing()
    added = sorted(set(digests) - set(old))
    removed = sorted(set(old) - set(digests))
    changed = sorted(r for r in set(old) & set(digests) if old[r] != digests[r])

    print(f"\n  recorded previously : {len(old)}")
    print(f"  present now         : {len(digests)}")
    print(f"  added               : {len(added)}")
    print(f"  changed             : {len(changed)}")
    print(f"  missing             : {len(removed)}")

    for label, items in (("added", added), ("changed", changed),
                         ("missing", removed)):
        if items:
            print(f"\n  {label}:")
            for r in items:
                print(f"     {r}")

    if changed:
        print("\n  NOTE: a changed digest on a large file often means the file was")
        print("  still being written when it was hashed. Confirm the copy has")
        print("  finished before accepting the new value.")

    if args.verify:
        return 1 if (added or changed or removed) else 0

    # Newlines are forced to LF: on Windows the default translation writes CRLF,
    # and `sha256sum -c` on Linux then reads the trailing CR as part of every
    # filename, failing every entry on a bundle whose contents are intact.
    SUMS.write_text("".join(f"{digests[r]}  {r}\n" for r in sorted(digests)),
                    encoding="utf-8", newline="\n")
    MANIFEST.write_text(f"{MANIFEST.name}\n"
                        + "".join(f"{r}\n" for r in sorted(digests)),
                        encoding="utf-8", newline="\n")
    print(f"\nrewrote {SUMS.name} and {MANIFEST.name} ({len(digests)} entries,"
          " bundle-relative paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
