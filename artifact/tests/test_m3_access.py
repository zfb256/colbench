"""Milestone 3 self-test: access patterns, above all the shuffle-window knob.

These tests pin the shuffle-window semantics down numerically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbench import access, datagen
from colbench.formats import ADAPTERS, Predicate, get_adapter

SPEC = datagen.PRESETS["tiny"]
LIVE = [n for n, c in ADAPTERS.items() if c.probe()[0]]


def test_window_equal_batch_gives_contiguous_rows():
    """W == B is the sequential endpoint: each batch is one contiguous run."""
    batches = access.shuffle_batches(100_000, batch_rows=1_000, window=1_000,
                                     n_batches=8, seed=0)
    for idx in batches:
        assert len(idx) == 1_000
        span = idx.max() - idx.min()
        assert span == 999, "contiguous block expected at the sequential endpoint"


def test_window_equal_n_rows_spreads_across_whole_dataset():
    """W == N is the random endpoint: a batch should span nearly everything."""
    n = 100_000
    batches = access.shuffle_batches(n, batch_rows=1_000, window=n,
                                     n_batches=8, seed=0)
    for idx in batches:
        span = idx.max() - idx.min()
        assert span > 0.8 * n, f"expected near-full spread, got span={span}"


def test_span_grows_monotonically_with_window():
    """The knob must actually be a knob: bigger window => wider spread."""
    n, b = 200_000, 512
    spans = []
    for w in access.window_sweep(n, b, n_points=6):
        batches = access.shuffle_batches(n, batch_rows=b, window=w,
                                         n_batches=16, seed=3)
        spans.append(np.median([idx.max() - idx.min() for idx in batches]))
    assert all(x < y for x, y in zip(spans, spans[1:])), f"not monotonic: {spans}"


def test_indices_are_distinct_and_in_range():
    n = 50_000
    for w in access.window_sweep(n, 256, n_points=5):
        for idx in access.shuffle_batches(n, batch_rows=256, window=w,
                                          n_batches=4, seed=1):
            assert len(np.unique(idx)) == len(idx), "duplicate rows in a batch"
            assert idx.min() >= 0 and idx.max() < n


def test_shuffle_is_reproducible_across_calls():
    a = access.shuffle_batches(10_000, batch_rows=128, window=2_048, n_batches=4, seed=42)
    b = access.shuffle_batches(10_000, batch_rows=128, window=2_048, n_batches=4, seed=42)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_window_smaller_than_batch_is_rejected():
    with pytest.raises(ValueError, match="window"):
        access.shuffle_batches(10_000, batch_rows=512, window=256, n_batches=1)


def test_window_larger_than_dataset_is_clamped():
    idx = access.shuffle_batches(1_000, batch_rows=100, window=10**9,
                                 n_batches=2, seed=0)
    for i in idx:
        assert i.max() < 1_000


def test_window_sweep_endpoints():
    sweep = access.window_sweep(1_000_000, 4_096, n_points=6)
    assert sweep[0] == 4_096
    assert sweep[-1] == 1_000_000
    assert all(x < y for x, y in zip(sweep, sweep[1:]))


def test_floyd_path_matches_choice_path_statistically():
    """Large windows take the O(k) Floyd branch; it must stay uniform."""
    rng = np.random.default_rng(0)
    k, hi = 200, 10_000_000
    s = access._sample_without_replacement(hi, k, rng)
    assert len(np.unique(s)) == k
    assert s.min() >= 0 and s.max() < hi


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    root = tmp_path_factory.mktemp("m3")
    out = {}
    for name in LIVE:
        ad = get_adapter(name)
        p = root / f"t_{name}"
        ad.write(datagen.reader(SPEC), p, chunk_rows=10_000)
        out[name] = (ad, p)
    return out


@pytest.mark.parametrize("name", LIVE)
def test_full_scan_returns_all_rows(written, name):
    ad, p = written[name]
    assert access.execute(ad, p, access.full_scan(batch_rows=8_192)) == SPEC.n_rows


@pytest.mark.parametrize("name", LIVE)
def test_projection_scan_returns_all_rows(written, name):
    ad, p = written[name]
    acc = access.projection_scan(["row_id", "f0"], batch_rows=8_192)
    assert access.execute(ad, p, acc) == SPEC.n_rows


@pytest.mark.parametrize("name", LIVE)
def test_shuffle_returns_expected_row_count(written, name):
    ad, p = written[name]
    acc = access.shuffle_read(window=10_000, batch_rows=512, n_batches=8,
                              columns=["row_id", "f0"])
    idx = access.plan(acc, SPEC.n_rows)
    assert access.execute(ad, p, acc, indices=idx) == 8 * 512


@pytest.mark.parametrize("name", LIVE)
def test_point_lookup_returns_one_row_each(written, name):
    ad, p = written[name]
    acc = access.point_lookup(n=32, columns=["row_id"])
    idx = access.plan(acc, SPEC.n_rows)
    assert all(len(i) == 1 for i in idx)
    assert access.execute(ad, p, acc, indices=idx) == 32


@pytest.mark.parametrize("name", LIVE)
def test_filtered_scan_selectivity_is_plausible(written, name):
    ad, p = written[name]
    acc = access.filtered_scan(Predicate("i0", "<", 100), columns=["row_id", "i0"])
    got = access.execute(ad, p, acc)
    expected = SPEC.n_rows * 100 / SPEC.int_range
    assert 0.7 * expected < got < 1.3 * expected


@pytest.mark.parametrize("name", LIVE)
def test_all_formats_see_identical_indices(written, name):
    """Fairness check: the plan is format-independent by construction."""
    acc = access.shuffle_read(window=20_000, batch_rows=256, n_batches=4, seed=9)
    a = access.plan(acc, SPEC.n_rows)
    b = access.plan(acc, SPEC.n_rows)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_access_labels_are_distinct():
    labels = {
        access.full_scan().label,
        access.projection_scan(["a", "b"]).label,
        access.shuffle_read(1024).label,
        access.point_lookup().label,
        access.filtered_scan(Predicate("i0", "<", 1)).label,
    }
    assert len(labels) == 5
