"""Fail closed unless one segmented T1 output satisfies the formal contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colbench import paths as path_mod  # noqa: E402
from colbench import storage  # noqa: E402


class ValidationError(ValueError):
    """A checkpoint or its provenance does not satisfy the formal contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _q(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot summarize an empty raw-sample sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _same_number(actual: Any, expected: float, field: str) -> None:
    _require(_finite_number(actual), f"{field} is not finite")
    _require(
        math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12),
        f"{field} disagrees with raw samples: recorded={actual!r}, expected={expected!r}",
    )


def _expected_access(row: dict) -> tuple[str, int]:
    access = row.get("access") or {}
    key = (access.get("kind"), access.get("n_columns"))
    contracts = {
        ("scan", None): {
            "label": "A1.scan(cols=all)", "batch_rows": 65_536,
            "window": None, "n_batches": 64, "seed": 0, "rows": 32_000_000,
        },
        ("scan", 10): {
            "label": "A2.scan(cols=10)", "batch_rows": 65_536,
            "window": None, "n_batches": 64, "seed": 0, "rows": 32_000_000,
        },
        ("shuffle", 10): {
            "label": "A3.shuffle(W=362039,B=4096,cols=10)", "batch_rows": 4_096,
            "window": 362_039, "n_batches": 64, "seed": 31, "rows": 262_144,
        },
    }
    _require(key in contracts, f"unexpected access contract: {access!r}")
    expected = contracts[key]
    for field in ("label", "batch_rows", "window", "n_batches", "seed"):
        _require(
            access.get(field) == expected[field],
            f"access {field} mismatch for {key}: {access.get(field)!r} != {expected[field]!r}",
        )
    columns = access.get("columns")
    if access.get("n_columns") is None:
        _require(columns is None, "full scan must not carry a column subset")
    else:
        _require(isinstance(columns, list) and len(columns) == 10, "projection must name 10 columns")
        _require(len(set(columns)) == 10, "projection columns must be unique")
    return expected["label"], expected["rows"]


def _validate_raw_samples(row: dict, expected_rows: int) -> None:
    samples = row.get("samples")
    _require(isinstance(samples, list) and len(samples) == 7, "every cell must retain 7 raw samples")
    required = {
        "wall_s", "cpu_s", "read_bytes", "read_chars", "rows",
        "disk_io_is_real", "rows_per_s",
    }
    for index, sample in enumerate(samples):
        _require(isinstance(sample, dict), f"raw sample {index} is not an object")
        _require(required <= sample.keys(), f"raw sample {index} is incomplete")
        for field in ("wall_s", "cpu_s", "read_bytes", "read_chars", "rows", "rows_per_s"):
            _require(_finite_number(sample[field]), f"raw sample {index} {field} is not finite")
        _require(sample["wall_s"] > 0, f"raw sample {index} wall_s must be positive")
        _require(sample["cpu_s"] >= 0, f"raw sample {index} cpu_s must be nonnegative")
        _require(sample["read_bytes"] >= 0 and sample["read_chars"] >= 0,
                 f"raw sample {index} I/O counters must be nonnegative")
        _require(int(sample["rows"]) == expected_rows, f"raw sample {index} row count is wrong")
        _require(sample["disk_io_is_real"] is True, f"raw sample {index} lacks real I/O accounting")
        _same_number(sample["rows_per_s"], expected_rows / sample["wall_s"],
                     f"samples[{index}].rows_per_s")

    def values(field: str) -> list[float]:
        return [float(sample[field]) for sample in samples]

    wall_median = _q(values("wall_s"), 0.5)
    wall_iqr = _q(values("wall_s"), 0.75) - _q(values("wall_s"), 0.25)
    _same_number(row.get("wall_s_median"), wall_median, "wall_s_median")
    _same_number(row.get("wall_s_iqr"), wall_iqr, "wall_s_iqr")
    _same_number(row.get("wall_s_rel_iqr"), wall_iqr / wall_median, "wall_s_rel_iqr")
    _same_number(row.get("cpu_s_median"), _q(values("cpu_s"), 0.5), "cpu_s_median")
    _same_number(row.get("read_bytes_median"), _q(values("read_bytes"), 0.5),
                 "read_bytes_median")
    _same_number(row.get("read_chars_median"), _q(values("read_chars"), 0.5),
                 "read_chars_median")
    _same_number(row.get("rows_per_s_median"), _q(values("rows_per_s"), 0.5),
                 "rows_per_s_median")
    _require(int(row.get("rows", -1)) == expected_rows, "summary row count is wrong")


def validate(path: Path, chunk: int, *, require_raw_samples: bool = False) -> None:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read checkpoint {path}: {exc}") from exc

    _require(len(rows) == 12, f"expected 12 records, got {len(rows)}")
    _require({int(row.get("chunk_rows", -1)) for row in rows} == {chunk}, "chunk_rows mismatch")
    _require({row.get("dataset", {}).get("name") for row in rows} == {"clickbench_32p"},
             "dataset name mismatch")
    _require({int(row.get("dataset", {}).get("n_rows", -1)) for row in rows} == {32_000_000},
             "dataset row count mismatch")
    _require({int(row.get("dataset", {}).get("n_cols", -1)) for row in rows} == {105},
             "dataset column count mismatch")
    _require({int(row.get("threads", -1)) for row in rows} == {1}, "thread count mismatch")
    _require({row.get("cache_mode") for row in rows} == {"cold", "warm"}, "cache modes mismatch")
    _require({row.get("format") for row in rows} == {"parquet", "lance"}, "formats mismatch")
    _require({row.get("compression") for row in rows} == {"zstd"}, "compression mismatch")
    _require(len({row.get("run_id") for row in rows}) == 1 and bool(rows[0].get("run_id")),
             "checkpoint must have one nonempty run_id")
    _require(all(int(row.get("n", -1)) == 7 for row in rows), "summary repeat count must be 7")
    _require(all(row.get("disk_io_is_real") is True for row in rows),
             "kernel-accounted storage I/O is required")
    _require(all(bool(row.get("eviction_verified")) for row in rows if row.get("cache_mode") == "cold"),
             "cold rows require verified eviction")

    cells = set()
    access_labels = set()
    for row in rows:
        label, expected_rows = _expected_access(row)
        access_labels.add(label)
        cells.add((row["format"], label, row["cache_mode"]))
        native = (row.get("env") or {}).get("native_thread_env") or {}
        _require(
            all(native.get(name) == "1" for name in (
                "LANCE_CPU_THREADS", "LANCE_IO_THREADS", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS"
            )),
            "native thread environment is not pinned to 1",
        )
        if require_raw_samples:
            _validate_raw_samples(row, expected_rows)

    _require(len(access_labels) == 3, f"expected 3 exact access patterns, got {access_labels}")
    _require(len(cells) == 12, "checkpoint contains duplicate or missing matrix cells")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _boundary_fields(path: Path) -> dict[str, str]:
    wanted = {
        "timestamp_utc", "phase", "exit_code", "git_commit", "boot_id",
        "raid_action", "raid_mismatch_cnt", "raid_degraded", "raid_members",
        "gate_report_sha256", "COLBENCH_WORK", "COLBENCH_DATA", "COLBENCH_RESULTS",
    }
    fields = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read boundary {path}: {exc}") from exc
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key in wanted:
            fields[key] = value
    return fields


def validate_boundaries(
    start_path: Path,
    end_path: Path,
    gate_path: Path,
    *,
    work: Path,
    data: Path,
    results: Path,
    check_current_host: bool = True,
) -> None:
    start = _boundary_fields(start_path)
    end = _boundary_fields(end_path)
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read gate report {gate_path}: {exc}") from exc

    _require(start.get("phase") == "start" and start.get("exit_code") == "not-run",
             "invalid start boundary phase/exit code")
    _require(end.get("phase") == "end" and end.get("exit_code") == "0",
             "invalid end boundary phase/exit code")
    gate_digest = _sha256(gate_path)
    _require(start.get("gate_report_sha256") == gate_digest, "start boundary gate fingerprint mismatch")
    _require(end.get("gate_report_sha256") == gate_digest, "end boundary gate fingerprint mismatch")
    _require(gate.get("storage_qualified") is True, "boundary gate is not storage-qualified")
    stability = gate.get("gate_stability") or {}
    _require(stability.get("qualified") is True and int(stability.get("snapshot_count", 0)) >= 3,
             "boundary gate lacks three-snapshot stability qualification")
    gate_boot = gate.get("boot_id")
    _require(bool(gate_boot) and start.get("boot_id") == end.get("boot_id") == gate_boot,
             "boundary boot ID does not match authorized gate")

    expected_paths = {
        "COLBENCH_WORK": str(work),
        "COLBENCH_DATA": str(data),
        "COLBENCH_RESULTS": str(results),
    }
    for key, expected in expected_paths.items():
        _require(start.get(key) == end.get(key) == expected, f"boundary {key} mismatch")

    raid_detected = bool((gate.get("raid") or {}).get("detected"))
    for name, boundary in (("start", start), ("end", end)):
        if raid_detected:
            _require(boundary.get("raid_action") == "idle", f"{name} RAID action is not idle")
            _require(boundary.get("raid_mismatch_cnt") == "0", f"{name} RAID mismatch is nonzero")
            _require(boundary.get("raid_degraded") == "0", f"{name} RAID is degraded")
            members = boundary.get("raid_members", "")
            _require(members.startswith("[") and members.endswith("]") and "_" not in members
                     and set(members[1:-1]) == {"U"}, f"{name} RAID members are not all online")
        else:
            _require(boundary.get("raid_action") == "none", f"{name} non-MD action must be none")

    if check_current_host:
        mount = path_mod.describe_mount(work)
        problems = storage.validate_gate_report(gate, work, mount)
        _require(not problems, "authorized gate no longer matches current host: " + "; ".join(problems))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("chunk", type=int)
    parser.add_argument("--require-raw-samples", action="store_true")
    parser.add_argument("--start-boundary", type=Path)
    parser.add_argument("--end-boundary", type=Path)
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args(argv)
    try:
        validate(args.path, args.chunk, require_raw_samples=args.require_raw_samples)
        boundary_args = (
            args.start_boundary, args.end_boundary, args.gate_report,
            args.work, args.data, args.results,
        )
        if any(value is not None for value in boundary_args):
            _require(all(value is not None for value in boundary_args),
                     "all boundary/gate/path arguments are required together")
            validate_boundaries(
                args.start_boundary, args.end_boundary, args.gate_report,
                work=args.work, data=args.data, results=args.results,
            )
    except ValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
