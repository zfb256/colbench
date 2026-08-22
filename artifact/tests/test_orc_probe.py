from scripts.probe_orc_layout import probe


def test_orc_stripe_size_changes_emitted_layout():
    report = probe(20_000, 20260817, (64 << 10, 1 << 20))

    assert report["stripe_layout_responds"]
    assert report["stripe_counts_nonincreasing"]
    assert report["row_index_stride_preserved"]
