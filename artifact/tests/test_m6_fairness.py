"""Cross-format fairness checks.

These guard the comparison rather than any single adapter. A writer that
accepts a compression or grouping argument and drops it produces a file built
to different rules from its peers, and the resulting timing difference is not
a format difference.

Assertions here test responsiveness, not a particular compression ratio: how
well a format compresses is a result, whether it applied what was requested is
a precondition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbench import datagen
from colbench.formats import ADAPTERS, get_adapter

LIVE = [n for n, c in ADAPTERS.items() if c.probe()[0]]

    # Avoid incompressible random floats in codec checks.
SPEC = datagen.TableSpec(n_rows=200_000, n_float=0, n_int=4, n_str=4,
                         str_cardinality=16, int_range=64,
                         batch_rows=16_384, seed=21)


@pytest.mark.parametrize("name", LIVE)
def test_compression_request_changes_the_output(tmp_path, name):
    """Two different codecs must produce two different files.

    Deliberately not "zstd must beat none by Nx": Lance's built-in encodings
    already do most of the work on this data, so its zstd gain is only ~1.1x --
    small, but real.  Requiring a fixed ratio would fail a correct adapter.
    """
    ad = get_adapter(name)
    sizes = {}
    for comp in ("none", "zstd"):
        p = tmp_path / f"{name}_{comp}"
        ad.write(datagen.reader(SPEC), p, chunk_rows=32_768, compression=comp)
        sizes[comp] = ad.size_bytes(p)

    assert sizes["none"] != sizes["zstd"], (
        f"{name}: 'none' and 'zstd' produced byte-identical files, so the "
        "compression setting is being ignored and the comparison is unfair"
    )
    assert sizes["zstd"] < sizes["none"], f"{name}: zstd made the file larger: {sizes}"


def test_parquet_records_the_requested_codec(tmp_path):
    """Parquet exposes the codec in metadata, so assert it directly."""
    import pyarrow.parquet as pq

    ad = get_adapter("parquet")
    for requested, expected in (("none", "UNCOMPRESSED"), ("zstd", "ZSTD"),
                                ("snappy", "SNAPPY")):
        p = tmp_path / f"codec_{requested}.parquet"
        ad.write(datagen.reader(SPEC), p, chunk_rows=32_768, compression=requested)
        got = pq.ParquetFile(str(p)).metadata.row_group(0).column(0).compression
        assert got == expected, f"asked for {requested}, file says {got}"


@pytest.mark.parametrize("name", LIVE)
def test_unknown_codec_fails_loudly(tmp_path, name):
    """A silently-ignored codec is how the original unfairness crept in."""
    ad = get_adapter(name)
    with pytest.raises(Exception):
        ad.write(datagen.reader(SPEC), tmp_path / f"{name}_bogus",
                 chunk_rows=32_768, compression="definitely-not-a-codec")


@pytest.mark.skipif(len(LIVE) < 2, reason="needs at least two formats")
def test_formats_are_within_a_plausible_size_band(tmp_path):
    """Same data, same codec, same chunking: sizes should be comparable.

    Not a claim that formats must compress equally -- they must not, that is
    part of what we measure.  But an order-of-magnitude gap on identical
    settings means the harness is asking them for different things.
    """
    sizes = {}
    for name in LIVE:
        ad = get_adapter(name)
        p = tmp_path / f"band_{name}"
        ad.write(datagen.reader(SPEC), p, chunk_rows=32_768, compression="zstd")
        sizes[name] = ad.size_bytes(p)

    spread = max(sizes.values()) / min(sizes.values())
    assert spread < 5.0, f"implausible size spread across formats: {sizes}"


@pytest.mark.skipif(len(LIVE) < 2, reason="needs at least two formats")
def test_formats_return_identical_data_for_identical_indices(tmp_path):
    """Formats returning different rows are not answering the same query."""
    written = {}
    for name in LIVE:
        ad = get_adapter(name)
        p = tmp_path / f"same_{name}"
        ad.write(datagen.reader(SPEC), p, chunk_rows=32_768, compression="zstd")
        written[name] = (ad, p)

    rng = np.random.default_rng(3)
    idx = rng.choice(SPEC.n_rows, size=256, replace=False).astype(np.int64)
    cols = ["row_id", "i0", "s0"]

    reference = None
    for name, (ad, p) in written.items():
        tbl = ad.take(p, idx, columns=cols).to_pydict()
        if reference is None:
            reference = tbl
        else:
            for col in cols:
                assert tbl[col] == reference[col], f"{name} disagrees on column {col}"


@pytest.mark.parametrize("name", LIVE)
def test_chunk_rows_is_accepted_and_data_survives(tmp_path, name):
    """chunk_rows must be accepted as an argument and never corrupt the data.

    Whether it changes the emitted layout is format-specific, so responsiveness
    is checked from each format's physical metadata below.
    """
    ad = get_adapter(name)
    for chunk in (4_096, 262_144):
        p = tmp_path / f"{name}_c{chunk}"
        ad.write(datagen.reader(SPEC), p, chunk_rows=chunk, compression="zstd")
        assert ad.n_rows(p) == SPEC.n_rows
        first = next(iter(ad.scan(p, columns=["row_id"], batch_rows=1_000)))
        assert first.column("row_id").to_pylist()[0] == 0


def _parquet_layout_signature(path):
    import pyarrow.parquet as pq

    md = pq.ParquetFile(str(path)).metadata
    return tuple(md.row_group(i).num_rows for i in range(md.num_row_groups))


def _lance_layout_signature(path):
    import lance
    from lance.file import LanceFileReader

    signature = []
    for fragment in lance.dataset(str(path)).get_fragments():
        for data_file in fragment.data_files():
            reader = LanceFileReader(str(Path(path) / "data" / data_file.path))
            pages = tuple(col.num_pages for col in reader.file_statistics().columns)
            signature.append((reader.num_rows(), pages))
    return tuple(sorted(signature))


def chunk_responsiveness(tmp_path, spec) -> dict[str, bool]:
    """Which formats change physical metadata when chunk_rows changes."""
    out = {}
    for name in LIVE:
        ad = get_adapter(name)
        signatures = []
        for chunk in (1_024, 8_192, 32_768, 131_072, 524_288, 1_048_576):
            p = tmp_path / f"resp_{name}_{chunk}"
            ad.write(datagen.reader(spec), p, chunk_rows=chunk, compression="zstd")
            if name == "parquet":
                signatures.append(_parquet_layout_signature(p))
            elif name == "lance":
                signatures.append(_lance_layout_signature(p))
            else:
                signatures.append(ad.size_bytes(p))
        out[name] = len(set(signatures)) > 1
    return out


def test_chunk_knob_is_real_for_at_least_one_format(tmp_path):
    """C4 needs the chunk knob to move *something* somewhere.

    Measured behaviour, verified from physical metadata at 200k rows:

    * parquet: row-group metadata changes across the six requested settings.
    * lance: fragment row counts and page counts do not change. Every column
      retains one page per fragment across the six settings in the evaluated
      Lance 9.0 writer path.

    That asymmetry is itself a C4 result -- the knob practitioners are told to
    tune does not exist in every format -- so it belongs in the paper, not in a
    failing test.
    """
    responds = chunk_responsiveness(tmp_path, SPEC)
    print(f"\nchunk_rows responsiveness: {responds}")
    assert responds.get("parquet") is True, (
        "parquet's row_group_size stopped being a physical knob; the C4 sweep "
        f"would be measuring nothing: {responds}"
    )
    assert any(responds.values()), (
        f"no format responded to chunk_rows at all: {responds}"
    )
    if "lance" in responds:
        assert responds["lance"] is False
