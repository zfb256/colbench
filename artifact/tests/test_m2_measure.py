"""Milestone 2 self-test: cache control and measurement layer."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbench import cache, measure


def test_run_once_reports_wall_and_rows():
    s = measure.run_once(lambda: (time.sleep(0.05), 42)[1])
    assert s.rows == 42
    assert s.wall_s >= 0.05
    assert s.rows_per_s > 0


def test_repeat_collects_n_samples_and_runs_setup_each_time():
    calls = {"setup": 0, "fn": 0}

    def setup():
        calls["setup"] += 1

    def fn():
        calls["fn"] += 1
        return 10

    res = measure.repeat(fn, n=5, setup=setup)
    assert len(res.samples) == 5
    assert calls["setup"] == 5 and calls["fn"] == 5


def test_warmup_runs_are_excluded_from_samples():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return 1

    res = measure.repeat(fn, n=3, warmup=2)
    assert calls["n"] == 5, "warmup must actually execute"
    assert len(res.samples) == 3, "warmup must not pollute the samples"


def test_quantiles_are_correct_on_known_values():
    res = measure.Result()
    for w in [1.0, 2.0, 3.0, 4.0, 5.0]:
        res.samples.append(measure.Sample(w, 0.0, 0, 0, 1))
    assert res.median("wall_s") == pytest.approx(3.0)
    assert res.iqr("wall_s") == pytest.approx(2.0)
    assert res.rel_iqr("wall_s") == pytest.approx(2.0 / 3.0)


def test_single_sample_does_not_crash_statistics():
    res = measure.Result([measure.Sample(1.0, 0.5, 0, 0, 7)])
    assert res.median() == 1.0
    assert res.iqr() == 0.0
    assert res.summary()["rows"] == 7


def test_summary_flags_whether_disk_io_is_trustworthy():
    res = measure.Result([measure.Sample(1.0, 0.5, 100, 200, 7)])
    s = res.summary()
    assert "disk_io_is_real" in s
    assert s["disk_io_is_real"] is measure.DISK_IO_IS_REAL


def test_summary_preserves_every_raw_sample():
    samples = [
        measure.Sample(1.0, 0.5, 100, 200, 7),
        measure.Sample(2.0, 0.75, 300, 400, 7),
    ]
    summary = measure.Result(samples).summary()
    assert [sample["wall_s"] for sample in summary["samples"]] == [1.0, 2.0]
    assert summary["samples"][0]["rows_per_s"] == 7.0


def test_threads_context_restores_previous_setting():
    import pyarrow as pa

    before = pa.cpu_count()
    with measure.threads(2):
        assert pa.cpu_count() == 2
        assert pa.io_thread_count() == 2
    assert pa.cpu_count() == before


def test_environment_captures_versions():
    env = measure.environment()
    for key in ("platform", "python", "logical_cores", "ram_gb", "pyarrow"):
        assert env.get(key) is not None
    assert "lance" in env and "vortex" in env
    assert set(env["native_thread_env"]) == {
        "LANCE_CPU_THREADS", "LANCE_IO_THREADS",
        "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS",
    }


def test_warm_then_evict_do_not_raise(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"x" * (1 << 20))
    cache.warm([f])
    cache.evict([f])
    cache.prepare([f], cache.CacheMode.WARM)
    cache.prepare([f], cache.CacheMode.COLD)


def test_cache_helpers_ignore_missing_and_directory_paths(tmp_path):
    cache.evict([tmp_path / "nope.bin", tmp_path])
    cache.warm([tmp_path / "nope.bin", tmp_path])


@pytest.mark.slow
def test_eviction_verification_runs_and_reports(tmp_path):
    'Records whether cold-cache measurement is trustworthy on this platform.'
    report = cache.verify_eviction(tmp_path, size_mb=64, trials=2)
    print(f"\n{report}")
    assert report.cold_s > 0 and report.warm_s > 0
    assert isinstance(report.supported, bool)
