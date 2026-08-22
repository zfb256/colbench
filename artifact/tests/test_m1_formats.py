"""Milestone 1 self-test: format adapters round-trip correctly.

Run:  .venv\\Scripts\\python.exe -m pytest tests/test_m1_formats.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbench import datagen
from colbench.formats import ADAPTERS, Predicate, get_adapter

SPEC = datagen.PRESETS["tiny"]
LIVE = [name for name, (ok, _) in
        {n: c.probe() for n, c in ADAPTERS.items()}.items() if ok]


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    """Write the tiny spec once per format; every test reads those files."""
    root = tmp_path_factory.mktemp("m1")
    out = {}
    for name in LIVE:
        ad = get_adapter(name)
        path = root / f"tiny_{name}"
        ad.write(datagen.reader(SPEC), path, chunk_rows=10_000, compression="zstd")
        out[name] = (ad, path)
    return out


def test_at_least_parquet_available():
    assert "parquet" in LIVE, "pyarrow must be importable for anything to work"


@pytest.mark.parametrize("name", LIVE)
def test_row_count(written, name):
    ad, path = written[name]
    assert ad.n_rows(path) == SPEC.n_rows


@pytest.mark.parametrize("name", LIVE)
def test_size_is_positive(written, name):
    ad, path = written[name]
    assert ad.size_bytes(path) > 0
    assert ad.files(path), "files() must enumerate backing files for cache eviction"


@pytest.mark.parametrize("name", LIVE)
def test_full_scan_roundtrip(written, name):
    """Scan must return every row exactly once, in order."""
    ad, path = written[name]
    seen = [b.column("row_id").to_numpy(zero_copy_only=False)
            for b in ad.scan(path, batch_rows=4_096)]
    got = np.concatenate(seen)
    np.testing.assert_array_equal(got, np.arange(SPEC.n_rows, dtype=np.int64))


@pytest.mark.parametrize("name", LIVE)
def test_projection_returns_only_requested_columns(written, name):
    ad, path = written[name]
    cols = ["row_id", "f0", "s0"]
    batch = next(iter(ad.scan(path, columns=cols, batch_rows=1_024)))
    assert batch.schema.names == cols


@pytest.mark.parametrize("name", LIVE)
def test_take_returns_requested_rows_in_order(written, name):
    """The critical one: random access must be positionally faithful.

    A `take` that silently reorders would make A3/A4 timings meaningless.
    """
    ad, path = written[name]
    rng = np.random.default_rng(7)
    idx = rng.choice(SPEC.n_rows, size=500, replace=False).astype(np.int64)

    tbl = ad.take(path, idx, columns=["row_id", "i0"])
    assert tbl.num_rows == len(idx)
    np.testing.assert_array_equal(
        tbl.column("row_id").to_numpy(zero_copy_only=False), idx
    )


@pytest.mark.parametrize("name", LIVE)
def test_take_handles_unsorted_and_duplicate_free_edges(written, name):
    ad, path = written[name]
    idx = np.array([SPEC.n_rows - 1, 0, 12345, 7], dtype=np.int64)
    tbl = ad.take(path, idx, columns=["row_id"])
    np.testing.assert_array_equal(
        tbl.column("row_id").to_numpy(zero_copy_only=False), idx
    )


@pytest.mark.parametrize("name", LIVE)
def test_filter_scan_matches_reference(written, name):
    """Filtered scan must agree with a brute-force scan over the same data."""
    ad, path = written[name]
    pred = Predicate("i0", "<", 100)

    filtered = [b.column("row_id").to_numpy(zero_copy_only=False)
                for b in ad.filter_scan(path, pred, columns=["row_id", "i0"])]
    got = np.sort(np.concatenate(filtered)) if filtered else np.array([], dtype=np.int64)

    ref_ids, ref_vals = [], []
    for b in ad.scan(path, columns=["row_id", "i0"], batch_rows=8_192):
        ref_ids.append(b.column("row_id").to_numpy(zero_copy_only=False))
        ref_vals.append(b.column("i0").to_numpy(zero_copy_only=False))
    ref_ids = np.concatenate(ref_ids)
    ref_vals = np.concatenate(ref_vals)
    expected = np.sort(ref_ids[ref_vals < 100])

    np.testing.assert_array_equal(got, expected)
    assert len(expected) > 0, "predicate selected nothing; test would be vacuous"


def test_parquet_chunk_rows_controls_row_group_size(tmp_path):
    """C4 depends on chunk_rows really setting the physical grouping.

    If this silently no-ops, the whole default-parameter contribution collapses,
    so assert it against the file metadata rather than trusting the writer.
    """
    import pyarrow.parquet as pq

    ad = get_adapter("parquet")
    path = tmp_path / "rg.parquet"
    ad.write(datagen.reader(SPEC), path, chunk_rows=10_000)

    md = pq.ParquetFile(str(path)).metadata
    assert md.num_row_groups == 5
    assert [md.row_group(i).num_rows for i in range(5)] == [10_000] * 5


def test_parquet_metadata_is_hoisted_and_invalidated(tmp_path):
    ad = get_adapter("parquet")
    ad._ROW_GROUP_CACHE.clear()
    path = tmp_path / "cached.parquet"
    ad.write(datagen.reader(SPEC), path, chunk_rows=10_000)

    first_file, first_starts = ad._file_and_starts(path)
    second_file, second_starts = ad._file_and_starts(path)
    assert second_file is first_file
    np.testing.assert_array_equal(second_starts, first_starts)
    assert first_starts.tolist() == [0, 10_000, 20_000, 30_000, 40_000]

    ad.write(datagen.reader(SPEC), path, chunk_rows=5_000)
    third_file, third_starts = ad._file_and_starts(path)
    assert third_file is not first_file
    assert third_starts.tolist() == list(range(0, 50_000, 5_000))


def test_orc_stripe_cache_is_scoped_to_one_access(tmp_path):
    ad = get_adapter("orc")
    if not ad.probe()[0]:
        pytest.skip("ORC unavailable")
    path = tmp_path / "scoped.orc"
    ad.write(datagen.reader(SPEC), path, chunk_rows=10_000,
             bytes_per_row=32.0)

    ad.begin_take_access(path)
    first = ad._stripe_rows(path)
    second = ad._stripe_rows(path)
    assert second is first
    ad.end_take_access(path)

    ad.begin_take_access(path)
    third = ad._stripe_rows(path)
    assert third == first
    assert third is not first
    ad.end_take_access(path)


def test_rechunk_preserves_rows_and_respects_size():
    from colbench.formats import _rechunk

    spec = datagen.TableSpec(n_rows=25_000, batch_rows=4_096, seed=11)
    chunks = list(_rechunk(datagen.reader(spec), 10_000))
    assert [c.num_rows for c in chunks] == [10_000, 10_000, 5_000]

    ids = np.concatenate([c.column("row_id").to_numpy() for c in chunks])
    np.testing.assert_array_equal(ids, np.arange(25_000))


def test_rechunk_when_chunk_larger_than_input():
    from colbench.formats import _rechunk

    spec = datagen.TableSpec(n_rows=1_000, batch_rows=256, seed=12)
    chunks = list(_rechunk(datagen.reader(spec), 10_000))
    assert [c.num_rows for c in chunks] == [1_000]
