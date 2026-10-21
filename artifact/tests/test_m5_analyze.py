'Milestone 5 self-test: the analysis layer, especially what it refuses.'

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbench import analyze
from scripts.plot_cross_dataset_chunk import shuffle_points


def rec(fmt, window, wall, *, cache_mode="warm", evicted=True, rel_iqr=0.05,
        chunk=131_072, kind="shuffle", n_cols=2, label=None):
    return {
        "run_id": "test", "ts": 0.0, "format": fmt, "chunk_rows": chunk,
        "compression": "zstd", "file_size_bytes": 1000, "n_files": 1,
        "write_wall_s": 1.0, "cache_mode": cache_mode,
        "threads": 1, "eviction_verified": evicted, "eviction_ratio": 3.0,
        "wall_s_median": wall, "wall_s_iqr": wall * rel_iqr,
        "wall_s_rel_iqr": rel_iqr, "cpu_s_median": wall * 0.5,
        "read_bytes_median": 1e6, "read_chars_median": 1e6,
        "rows": 4096, "rows_per_s_median": 4096 / wall, "n": 5,
        "disk_io_is_real": True,
        "dataset": {"name": "unit", "n_rows": 1_000_000},
        "env": {"platform": "test"},
        "access": {
            "kind": kind,
            "label": label or f"A3.shuffle(W={window})",
            "n_columns": n_cols, "columns": None, "batch_rows": 4096,
            "window": window, "n_batches": 32, "seed": 0,
        },
    }


def frame(records):
    return pd.DataFrame([
        {**{k: v for k, v in r.items() if k not in ("access", "dataset", "env")},
         **{f"access_{k}": v for k, v in r["access"].items() if k != "columns"},
         "dataset": r["dataset"]["name"], "dataset_rows": r["dataset"]["n_rows"]}
        for r in records
    ])


def test_load_flattens_jsonl(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(rec("parquet", 4096, 0.1)) for _ in range(3)),
                 encoding="utf-8")
    df = analyze.load(p)
    assert len(df) == 3
    assert {"format", "access_window", "wall_s_median", "dataset"} <= set(df.columns)


def test_load_of_empty_file_is_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert analyze.load(p).empty


def test_cross_dataset_plot_keeps_all_group_sizes_in_one_file(tmp_path):
    p = tmp_path / "sweep.jsonl"
    rows = [
        {"format": "parquet", "cache_mode": "cold", "chunk_rows": chunk,
         "wall_s_median": wall, "access": {"kind": "shuffle"}}
        for chunk, wall in ((8192, 2.0), (131072, 1.0))
    ]
    p.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    assert shuffle_points([p]) == [(8192, 2.0), (131072, 1.0)]


def test_unverified_cold_rows_are_rejected():
    df = frame([
        rec("parquet", 4096, 0.1, cache_mode="cold", evicted=False),
        rec("lance", 4096, 0.1, cache_mode="warm", evicted=False),
    ])
    g = analyze.usable(df)
    assert len(g.kept) == 1 and g.kept.iloc[0]["format"] == "lance"
    assert "cold-cache row from a host that cannot evict" in g.reasons


def test_verified_cold_rows_are_kept():
    df = frame([rec("parquet", 4096, 0.1, cache_mode="cold", evicted=True)])
    assert len(analyze.usable(df).kept) == 1


def test_noisy_cells_are_rejected():
    df = frame([rec("parquet", 4096, 0.1, rel_iqr=0.9),
                rec("lance", 4096, 0.1, rel_iqr=0.02)])
    g = analyze.usable(df)
    assert len(g.kept) == 1 and g.kept.iloc[0]["format"] == "lance"


def test_guard_report_is_human_readable():
    df = frame([rec("parquet", 4096, 0.1, cache_mode="cold", evicted=False)])
    text = analyze.usable(df).report()
    assert "rejected 1" in text and "cannot evict" in text


def test_guard_on_empty_frame_does_not_crash():
    g = analyze.usable(pd.DataFrame())
    assert g.kept.empty and g.rejected.empty


def test_require_real_io_filters_syscall_only_rows():
    df = frame([rec("parquet", 4096, 0.1), rec("lance", 4096, 0.1)])
    df.loc[0, "disk_io_is_real"] = False
    assert len(analyze.require_real_io(df)) == 1


def test_measurement_scope_refuses_to_mix_cache_modes():
    df = frame([
        rec("parquet", 4096, 0.1, cache_mode="cold"),
        rec("parquet", 4096, 0.05, cache_mode="warm"),
    ])
    with pytest.raises(ValueError, match="multiple cache_mode"):
        analyze.select_measurement_scope(df)
    cold = analyze.select_measurement_scope(df, cache_mode="cold")
    assert len(cold) == 1 and cold.iloc[0]["cache_mode"] == "cold"


def test_measurement_scope_refuses_to_mix_threads():
    df = frame([rec("parquet", 4096, 0.1), rec("lance", 4096, 0.1)])
    df.loc[df.index[1], "threads"] = 8
    with pytest.raises(ValueError, match="multiple threads"):
        analyze.select_measurement_scope(df)


def test_regular_report_scope_refuses_to_mix_datasets():
    df = frame([rec("parquet", 4096, 0.1), rec("lance", 4096, 0.1)])
    df.loc[df.index[1], "dataset"] = "other"
    with pytest.raises(ValueError, match="multiple datasets"):
        analyze.select_measurement_scope(df, allow_multiple_datasets=False)
    assert len(analyze.select_measurement_scope(
        df, dataset="unit", allow_multiple_datasets=False
    )) == 1


def test_crossover_detected_when_ranking_flips():
    """parquet wins when local, lance wins when spread -- one crossover."""
    df = frame([
        rec("parquet", 4_096, 0.010), rec("lance", 4_096, 0.020),
        rec("parquet", 65_536, 0.050), rec("lance", 65_536, 0.030),
        rec("parquet", 1_000_000, 0.400), rec("lance", 1_000_000, 0.060),
    ])
    cs = analyze.find_crossovers(df)
    assert len(cs) == 1
    assert cs[0].leader_lo == "parquet" and cs[0].leader_hi == "lance"
    assert cs[0].window_lo == 4_096 and cs[0].window_hi == 65_536


def test_no_crossover_when_one_format_always_wins():
    """The negative outcome for C2 must be reported as an empty list."""
    df = frame([
        rec("parquet", 4_096, 0.02), rec("lance", 4_096, 0.01),
        rec("parquet", 65_536, 0.05), rec("lance", 65_536, 0.02),
        rec("parquet", 1_000_000, 0.40), rec("lance", 1_000_000, 0.06),
    ])
    assert analyze.find_crossovers(df) == []


def test_multiple_crossovers_are_all_reported():
    df = frame([
        rec("parquet", 1_000, 0.01), rec("lance", 1_000, 0.02),
        rec("parquet", 10_000, 0.05), rec("lance", 10_000, 0.02),
        rec("parquet", 100_000, 0.06), rec("lance", 100_000, 0.09),
    ])
    assert len(analyze.find_crossovers(df)) == 2


def test_crossover_needs_two_formats():
    df = frame([rec("parquet", 4_096, 0.01), rec("parquet", 65_536, 0.05)])
    assert analyze.find_crossovers(df) == []


def test_window_table_shape():
    df = frame([rec("parquet", 4_096, 0.01), rec("lance", 4_096, 0.02),
                rec("parquet", 65_536, 0.05), rec("lance", 65_536, 0.02)])
    t = analyze.window_table(df)
    assert list(t.index) == [4_096, 65_536]
    assert set(t.columns) == {"parquet", "lance"}


def test_speedup_vs_baseline_is_one_for_baseline():
    df = frame([rec("parquet", 4_096, 0.10), rec("lance", 4_096, 0.05)])
    s = analyze.speedup_vs(df, "parquet")
    assert s.loc[4_096, "parquet"] == pytest.approx(1.0)
    assert s.loc[4_096, "lance"] == pytest.approx(2.0)


def test_chunk_sensitivity_reports_swing_and_best_chunk():
    df = frame([
        rec("lance", 4_096, 0.10, chunk=8_192, label="A"),
        rec("lance", 4_096, 0.02, chunk=131_072, label="A"),
    ])
    cs = analyze.chunk_sensitivity(df)
    row = cs.iloc[0]
    assert row["swing"] == pytest.approx(5.0)
    assert row["best_chunk_rows"] == 131_072


def test_gap_shrinks_when_defaults_are_neutralized():
    """The C4 claim: an advantage at defaults partly survives tuning, or not."""
    df = frame([
        rec("parquet", 4_096, 0.40, chunk=1_048_576, label="A"),
        rec("parquet", 4_096, 0.10, chunk=131_072, label="A"),
        rec("lance", 4_096, 0.10, chunk=1_024, label="A"),
        rec("lance", 4_096, 0.08, chunk=131_072, label="A"),
    ])
    g = analyze.gap_at_defaults_vs_tuned(
        df, {"parquet": 1_048_576, "lance": 1_024}, baseline="parquet")
    row = g.iloc[0]
    assert row["speedup_at_defaults"] == pytest.approx(4.0)
    assert row["speedup_when_tuned"] == pytest.approx(1.25)
    assert row["gap_shrinkage"] == pytest.approx(3.2)


def test_fixed_layout_reuses_its_default_estimate():
    df = frame([
        rec("parquet", 4_096, 0.40, chunk=1_048_576, label="A"),
        rec("parquet", 4_096, 0.10, chunk=131_072, label="A"),
        rec("lance", 4_096, 0.10, chunk=1_024, label="A"),
        rec("lance", 4_096, 0.08, chunk=131_072, label="A"),
    ])
    g = analyze.gap_at_defaults_vs_tuned(
        df, {"parquet": 1_048_576, "lance": 1_024}, baseline="parquet",
        fixed_layout_formats={"lance"})
    row = g.iloc[0]
    assert row["speedup_when_tuned"] == pytest.approx(1.0)
    assert row["gap_shrinkage"] == pytest.approx(4.0)


def test_bootstrap_identifies_a_clearly_faster_chunk():
    result = analyze.bootstrap_chunk_uncertainty(
        {8_192: [9, 10, 11, 10, 10], 131_072: [1, 2, 1, 2, 1]},
        n_boot=500,
    )
    assert result[1]["chunk_rows"] == 131_072
    assert result[1]["probability_best"] == 1.0


def test_bootstrap_splits_exact_ties_evenly():
    result = analyze.bootstrap_chunk_uncertainty(
        {8_192: [1, 1, 1], 131_072: [1, 1, 1]}, n_boot=20
    )
    assert [cell["probability_best"] for cell in result] == [0.5, 0.5]


def test_window_plot_is_written(tmp_path):
    df = frame([rec("parquet", 4_096, 0.01), rec("lance", 4_096, 0.02),
                rec("parquet", 65_536, 0.05), rec("lance", 65_536, 0.02)])
    out = analyze.plot_window_sweep(df, tmp_path / "fig2.png")
    assert out and out.exists() and out.stat().st_size > 0


def test_plots_return_none_instead_of_crashing_on_empty(tmp_path):
    empty = pd.DataFrame()
    assert analyze.plot_window_sweep(empty, tmp_path / "a.png") is None
    assert analyze.plot_projection(empty, tmp_path / "b.png") is None
    assert analyze.plot_chunk_effect(empty, tmp_path / "c.png") is None


def test_projection_plot_is_written(tmp_path):
    df = frame([
        rec("parquet", None, 0.10, kind="scan", n_cols=2, label="A2(2)"),
        rec("lance", None, 0.05, kind="scan", n_cols=2, label="A2(2)"),
        rec("parquet", None, 0.40, kind="scan", n_cols=20, label="A2(20)"),
        rec("lance", None, 0.20, kind="scan", n_cols=20, label="A2(20)"),
    ])
    out = analyze.plot_projection(df, tmp_path / "fig1.png")
    assert out and out.exists()


def test_chunk_plot_is_written(tmp_path):
    df = frame([
        rec("parquet", 4_096, 0.10, chunk=8_192, label="A"),
        rec("parquet", 4_096, 0.05, chunk=131_072, label="A"),
        rec("lance", 4_096, 0.08, chunk=8_192, label="A"),
        rec("lance", 4_096, 0.04, chunk=131_072, label="A"),
    ])
    out = analyze.plot_chunk_effect(df, tmp_path / "fig4.png")
    assert out and out.exists()
