import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from colbench import access, measure, runner
from scripts import refresh_checksums, run_experiment
from scripts.validate_t1_checkpoint import ValidationError, validate, validate_boundaries


def test_t1_fig4_shuffle_seed_is_fixed():
    ds = runner.Dataset("tiny", spec=run_experiment.datagen.PRESETS["tiny"])
    accesses, _chunks = run_experiment.preset_fig4(ds, _Args())
    shuffle = [a for a in accesses if a.kind == "shuffle"]
    assert len(shuffle) == 1
    assert shuffle[0].seed == 31


def test_numeric_column_ignores_parquet_nulls(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "nullable.parquet"
    pq.write_table(pa.table({
        "nullable_i": pa.array([None, 10, 30], type=pa.int64()),
        "label": ["a", "b", "c"],
    }), source)
    ds = runner.Dataset("nullable", source=source)

    assert run_experiment._numeric_column(ds) == ("nullable_i", 10, 30)


def test_shuffle_plan_is_reproducible_for_same_seed():
    a = access.shuffle_read(window=4096, batch_rows=256, n_batches=4, seed=31)
    first = access.plan(a, 20_000)
    second = access.plan(a, 20_000)
    assert [x.tolist() for x in first] == [x.tolist() for x in second]


def _checkpoint_rows():
    rows = []
    contracts = (
        ({
            "kind": "scan", "label": "A1.scan(cols=all)", "n_columns": None,
            "columns": None, "batch_rows": 65_536, "window": None,
            "n_batches": 64, "seed": 0,
        }, 32_000_000),
        ({
            "kind": "scan", "label": "A2.scan(cols=10)", "n_columns": 10,
            "columns": [f"c{i}" for i in range(10)], "batch_rows": 65_536,
            "window": None, "n_batches": 64, "seed": 0,
        }, 32_000_000),
        ({
            "kind": "shuffle", "label": "A3.shuffle(W=362039,B=4096,cols=10)",
            "n_columns": 10, "columns": [f"c{i}" for i in range(10)],
            "batch_rows": 4_096, "window": 362_039, "n_batches": 64, "seed": 31,
        }, 262_144),
    )
    for fmt in ("parquet", "lance"):
        for access_spec, expected_rows in contracts:
            for cache_mode in ("cold", "warm"):
                samples = [
                    measure.Sample(
                        wall_s=1.0 + index / 100,
                        cpu_s=0.5 + index / 100,
                        read_bytes=100 + index,
                        read_chars=200 + index,
                        rows=expected_rows,
                        # Model a qualified Linux host.
                        disk_io_is_real=True,
                    )
                    for index in range(7)
                ]
                rows.append({
                    "run_id": "one-checkpoint",
                    "chunk_rows": 1024,
                    "dataset": {"name": "clickbench_32p", "n_rows": 32_000_000, "n_cols": 105},
                    "threads": 1,
                    "cache_mode": cache_mode,
                    "format": fmt,
                    "compression": "zstd",
                    "access": deepcopy(access_spec),
                    "eviction_verified": True,
                    "env": {"native_thread_env": {
                        "LANCE_CPU_THREADS": "1", "LANCE_IO_THREADS": "1",
                        "RAYON_NUM_THREADS": "1", "TOKIO_WORKER_THREADS": "1",
                    }},
                    **measure.Result(samples).summary(),
                    # Override the current host's accounting capability.
                    "disk_io_is_real": True,
                })
    return rows


def _write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_checkpoint_requires_full_matrix_and_raw_samples(tmp_path):
    rows = _checkpoint_rows()
    path = tmp_path / "checkpoint.jsonl"
    _write_rows(path, rows)
    validate(path, 1024, require_raw_samples=True)

    rows[-1] = dict(rows[0])
    _write_rows(path, rows)
    with pytest.raises(ValidationError, match="duplicate or missing"):
        validate(path, 1024, require_raw_samples=True)


def test_checkpoint_recomputes_summary_from_raw_samples(tmp_path):
    rows = _checkpoint_rows()
    rows[0]["wall_s_median"] += 1
    path = tmp_path / "checkpoint.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValidationError, match="wall_s_median disagrees"):
        validate(path, 1024, require_raw_samples=True)


def test_checkpoint_rejects_rows_without_kernel_io_accounting(tmp_path):
    rows = _checkpoint_rows()
    rows[0]["disk_io_is_real"] = False
    path = tmp_path / "checkpoint.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValidationError, match="kernel-accounted storage I/O is required"):
        validate(path, 1024, require_raw_samples=True)


def test_checkpoint_rejects_raw_samples_without_kernel_io_accounting(tmp_path):
    rows = _checkpoint_rows()
    rows[0]["samples"][0]["disk_io_is_real"] = False
    path = tmp_path / "checkpoint.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValidationError, match="lacks real I/O accounting"):
        validate(path, 1024, require_raw_samples=True)


def test_checkpoint_rejects_cold_rows_without_verified_eviction(tmp_path):
    rows = _checkpoint_rows()
    cold = next(row for row in rows if row["cache_mode"] == "cold")
    cold["eviction_verified"] = False
    path = tmp_path / "checkpoint.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValidationError, match="cold rows require verified eviction"):
        validate(path, 1024, require_raw_samples=True)


def test_checkpoint_validator_cannot_be_disabled_with_python_optimized(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "validate_t1_checkpoint.py"
    completed = subprocess.run(
        [sys.executable, "-O", str(script), str(path), "1024", "--require-raw-samples"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "expected 12 records" in completed.stderr


# Render fixture paths consistently across platforms.
BOUNDARY_WORK = Path("/work")
BOUNDARY_DATA = Path("/data")
BOUNDARY_RESULTS = Path("/results")


def _write_boundary(path, *, phase, exit_code, gate_digest, mismatch="0"):
    path.write_text(
        "\n".join((
            "timestamp_utc=2026-08-16T00:00:00+00:00",
            f"phase={phase}",
            f"exit_code={exit_code}",
            "git_commit=unavailable",
            "boot_id=boot-a",
            "raid_action=idle",
            f"raid_mismatch_cnt={mismatch}",
            "raid_degraded=0",
            "raid_members=[UU]",
            f"gate_report_sha256={gate_digest}",
            f"COLBENCH_WORK={BOUNDARY_WORK}",
            f"COLBENCH_DATA={BOUNDARY_DATA}",
            f"COLBENCH_RESULTS={BOUNDARY_RESULTS}",
        )) + "\n",
        encoding="utf-8",
    )


def test_boundaries_bind_gate_boot_paths_and_clean_raid(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "gate_schema": 3,
        "boot_id": "boot-a",
        "storage_qualified": True,
        "gate_stability": {"qualified": True, "snapshot_count": 3},
        "raid": {"detected": True},
    }), encoding="utf-8")
    digest = hashlib.sha256(gate.read_bytes()).hexdigest()
    start = tmp_path / "start.txt"
    end = tmp_path / "end.txt"
    _write_boundary(start, phase="start", exit_code="not-run", gate_digest=digest)
    _write_boundary(end, phase="end", exit_code="0", gate_digest=digest)
    validate_boundaries(
        start, end, gate,
        work=BOUNDARY_WORK, data=BOUNDARY_DATA, results=BOUNDARY_RESULTS,
        check_current_host=False,
    )

    _write_boundary(end, phase="end", exit_code="0", gate_digest=digest, mismatch="1")
    with pytest.raises(ValidationError, match="mismatch is nonzero"):
        validate_boundaries(
            start, end, gate,
            work=BOUNDARY_WORK, data=BOUNDARY_DATA, results=BOUNDARY_RESULTS,
            check_current_host=False,
        )


def test_segmented_runner_enforces_raid_health_and_gate_fingerprint():
    script = (Path(__file__).parents[1] / "scripts" / "run_t1_segmented.sh").read_text()
    for required in (
        "require_raid_healthy", "raid_mismatch_cnt", "raid_degraded",
        "raid_members", "gate_report_sha256", "--start-boundary", "--gate-report",
    ):
        assert required in script


def test_checksum_verify_rejects_unrecorded_file(tmp_path, monkeypatch):
    sums = tmp_path / "SHA256SUMS.txt"
    manifest = tmp_path / "MANIFEST.txt"
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    digest = hashlib.sha256(tracked.read_bytes()).hexdigest()
    sums.write_text(f"{digest}  tracked.txt\n", encoding="utf-8")

    monkeypatch.setattr(refresh_checksums, "ROOT", tmp_path)
    monkeypatch.setattr(refresh_checksums, "SUMS", sums)
    monkeypatch.setattr(refresh_checksums, "MANIFEST", manifest)
    (tmp_path / "unrecorded.txt").write_text("new\n", encoding="utf-8")

    assert refresh_checksums.main(["--verify"]) == 1
    assert refresh_checksums.main([]) == 0
    assert "unrecorded.txt" in sums.read_text(encoding="utf-8")


class _Args:
    batch_rows = 4096
    n_batches = 64
    formats = "parquet,lance"
