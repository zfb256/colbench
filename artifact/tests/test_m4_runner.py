"""Milestone 4 self-test: the runner produces honest, complete records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbench import access, cache, datagen, runner
from colbench.formats import ADAPTERS, Predicate

LIVE = [n for n, c in ADAPTERS.items() if c.probe()[0]]
SPEC = datagen.TableSpec(n_rows=20_000, batch_rows=4_096, seed=5)


@pytest.fixture()
def tiny_matrix():
    return runner.Matrix(
        datasets=[runner.Dataset("unit", spec=SPEC)],
        formats=LIVE,
        chunk_rows=[5_000],
        accesses=[
            access.full_scan(batch_rows=4_096),
            access.projection_scan(["row_id", "f0"], batch_rows=4_096),
            access.shuffle_read(window=5_000, batch_rows=256, n_batches=4),
        ],
        cache_modes=[cache.CacheMode.WARM],
        threads=[1],
        repeats=2,
        warmup=0,
    )


@pytest.fixture()
def run(tmp_path, tiny_matrix):
    r = runner.Runner(tmp_path / "work", tmp_path / "out.jsonl",
                      verify_cache=False, quiet=True)
    return r, r.run(tiny_matrix)


def test_dataset_rejects_ambiguous_construction():
    with pytest.raises(ValueError):
        runner.Dataset("bad")
    with pytest.raises(ValueError):
        runner.Dataset("bad", spec=SPEC, source=Path("x.parquet"))


def test_dataset_describe_reports_shape():
    d = runner.Dataset("unit", spec=SPEC).describe()
    assert d["n_rows"] == SPEC.n_rows and d["synthetic"] is True


def test_dataset_reader_canonicalizes_unsigned_columns_for_every_format(tmp_path):
    source = tmp_path / "unsigned.parquet"
    pq.write_table(pa.table({
        "EventDate": pa.array([1, 2], type=pa.uint16()),
        "value": pa.array([3, 4], type=pa.int64()),
    }), source)
    ds = runner.Dataset("unsigned", source=source)

    assert ds.reader().schema.field("EventDate").type == pa.int32()
    assert ds.describe()["canonical_schema_changes"] == [{
        "column": "EventDate",
        "from": "uint16",
        "to": "int32",
    }]


def test_matrix_cell_count_is_the_full_product(tiny_matrix):
    assert len(list(tiny_matrix.cells())) == len(LIVE) * 1 * 3 * 1 * 1


def test_matrix_counterbalances_format_order():
    m = runner.Matrix(
        datasets=[runner.Dataset("unit", spec=SPEC)],
        formats=["parquet", "lance"],
        chunk_rows=[1, 2],
        accesses=[access.full_scan(batch_rows=4096)],
        cache_modes=[cache.CacheMode.WARM],
        threads=[1],
    )
    cells = list(m.cells())
    assert [cell[1] for cell in cells[:2]] == ["parquet", "lance"]
    assert [cell[1] for cell in cells[2:4]] == ["lance", "parquet"]


def test_run_produces_one_record_per_cell(run, tiny_matrix):
    _, records = run
    assert len(records) == len(list(tiny_matrix.cells()))


def test_records_are_written_as_valid_jsonl(run):
    r, records = run
    lines = [json.loads(l) for l in r.results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == len(records)


def test_every_record_carries_provenance(run):
    _, records = run
    required = {
        "run_id", "format", "chunk_rows", "file_size_bytes", "access",
        "cache_mode", "threads", "eviction_verified", "disk_io_is_real",
        "wall_s_median", "wall_s_iqr", "wall_s_rel_iqr",
        "cpu_parallelism_median", "rows", "env",
    }
    for rec in records:
        assert required <= set(rec), f"missing: {required - set(rec)}"


def test_row_counts_agree_with_the_access_pattern(run):
    _, records = run
    for rec in records:
        if rec["access"]["kind"] == "scan":
            assert rec["rows"] == SPEC.n_rows
        elif rec["access"]["kind"] == "shuffle":
            assert rec["rows"] == 4 * 256


def test_all_formats_agree_on_row_counts(run):
    """A format that silently returns fewer rows would look fast and be wrong."""
    _, records = run
    by_access: dict[str, set[int]] = {}
    for rec in records:
        by_access.setdefault(rec["access"]["label"], set()).add(rec["rows"])
    for label, counts in by_access.items():
        assert len(counts) == 1, f"{label}: formats disagree on row count {counts}"


def test_file_sizes_are_positive(run):
    _, records = run
    assert all(rec["file_size_bytes"] > 0 for rec in records)


def test_every_live_format_records_a_layout_signature(run):
    _, records = run
    assert all(rec["layout_emitted"] is not None for rec in records)
    assert all(rec["layout_signature"] is not None for rec in records)
    assert all(rec["layout_emitted"].get("n_groups", 0) > 0 for rec in records)
    for rec in records:
        if rec["format"] == "orc":
            assert rec["layout_signature"]["stripe_rows"]


def test_dataset_is_written_once_and_reused(tmp_path, tiny_matrix):
    r = runner.Runner(tmp_path / "w", tmp_path / "o.jsonl",
                      verify_cache=False, quiet=True)
    records = r.run(tiny_matrix)
    assert len(r._materialized) == len(LIVE)
    written = [rec for rec in records if rec["write_wall_s"] == rec["write_wall_s"]]
    assert len(written) == len(LIVE), "write time recorded exactly once per format"


def test_unavailable_formats_are_skipped_not_fatal(tmp_path):
    m = runner.Matrix(
        datasets=[runner.Dataset("unit", spec=SPEC)],
        formats=list(ADAPTERS),
        chunk_rows=[5_000],
        accesses=[access.full_scan(batch_rows=4_096)],
        cache_modes=[cache.CacheMode.WARM],
        repeats=1,
        warmup=0,
    )
    r = runner.Runner(tmp_path / "w", tmp_path / "o.jsonl",
                      verify_cache=False, quiet=True)
    records = r.run(m)
    assert {rec["format"] for rec in records} == set(LIVE)


def test_eviction_flag_is_false_when_unverified(tmp_path, tiny_matrix):
    r = runner.Runner(tmp_path / "w", tmp_path / "o.jsonl",
                      verify_cache=False, quiet=True)
    for rec in r.run(tiny_matrix):
        assert rec["eviction_verified"] is False


def test_each_cell_drops_its_final_warm_cache(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cache, "evict", lambda files: calls.append(tuple(files)))
    m = runner.Matrix(
        datasets=[runner.Dataset("unit", spec=SPEC)],
        formats=["parquet"],
        chunk_rows=[5_000],
        accesses=[access.full_scan(batch_rows=4_096)],
        cache_modes=[cache.CacheMode.WARM],
        threads=[1],
        repeats=1,
        warmup=0,
    )
    r = runner.Runner(
        tmp_path / "work", tmp_path / "out.jsonl",
        verify_cache=False, quiet=True,
    )
    r.run(m)
    assert len(calls) == 1
    assert calls[0] == tuple(runner.get_adapter("parquet").files(
        tmp_path / "work" / "unit__parquet__c5000.parquet"
    ))


def test_load_results_roundtrip(run):
    r, records = run
    assert len(runner.load_results(r.results_path)) == len(records)
