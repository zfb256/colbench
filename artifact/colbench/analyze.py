'Analysis and figures.'

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

NOISE_LIMIT = 0.25


def load(*paths: Path) -> pd.DataFrame:
    rows = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            flat = {k: v for k, v in rec.items()
                    if k not in ("access", "dataset", "env")}
            flat.update({f"access_{k}": v for k, v in rec["access"].items()
                         if k != "columns"})
            flat["dataset"] = rec["dataset"]["name"]
            flat["dataset_rows"] = rec["dataset"]["n_rows"]
            flat["dataset_n_columns"] = rec["dataset"].get("n_cols")
            rows.append(flat)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


@dataclass
class Guard:
    kept: pd.DataFrame
    rejected: pd.DataFrame
    reasons: dict[str, int]

    def report(self) -> str:
        if self.rejected.empty:
            return f"all {len(self.kept)} records usable"
        lines = [f"kept {len(self.kept)}, rejected {len(self.rejected)}:"]
        lines += [f"  {n:5d}  {why}" for why, n in sorted(self.reasons.items())]
        return "\n".join(lines)


def usable(df: pd.DataFrame, *, noise_limit: float = NOISE_LIMIT) -> Guard:
    """Split ``df`` into records that can carry a claim and records that cannot."""
    if df.empty:
        return Guard(df, df, {})

    why = pd.Series("", index=df.index)

    unverified_cold = (df["cache_mode"] == "cold") & (~df["eviction_verified"])
    why[unverified_cold & (why == "")] = "cold-cache row from a host that cannot evict"

    noisy = df["wall_s_rel_iqr"] > noise_limit
    why[noisy & (why == "")] = f"relative IQR above {noise_limit:.0%}"

    bad = why != ""
    reasons = why[bad].value_counts().to_dict()
    return Guard(df[~bad].copy(), df[bad].copy(), reasons)


def require_real_io(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where ``read_bytes`` reflects the device, for I/O-vs-decode splits."""
    if df.empty or "disk_io_is_real" not in df:
        return df
    return df[df["disk_io_is_real"]].copy()


def select_measurement_scope(
    df: pd.DataFrame,
    *,
    cache_mode: str | None = None,
    threads: int | None = None,
    dataset: str | None = None,
    allow_multiple_datasets: bool = True,
) -> pd.DataFrame:
    'Select one comparable measurement scope, failing closed on ambiguity.'
    if df.empty:
        return df.copy()

    out = df
    requested = {"cache_mode": cache_mode, "threads": threads, "dataset": dataset}
    for column, value in requested.items():
        if value is not None:
            if column not in out.columns:
                raise ValueError(f"records have no {column!r} column")
            out = out[out[column] == value]
            if out.empty:
                raise ValueError(f"no records match {column}={value!r}")

    for column in ("cache_mode", "threads"):
        if requested[column] is None and column in out.columns:
            values = sorted(out[column].dropna().unique().tolist())
            if len(values) > 1:
                raise ValueError(
                    f"multiple {column} values {values}; select one explicitly"
                )

    if dataset is None and not allow_multiple_datasets and "dataset" in out.columns:
        values = sorted(out["dataset"].dropna().unique().tolist())
        if len(values) > 1:
            raise ValueError(
                f"multiple datasets {values}; select one explicitly"
            )
    return out.copy()


def window_table(df: pd.DataFrame, *, metric: str = "wall_s_median") -> pd.DataFrame:
    """Median time per (shuffle window x format), the raw material for Fig.2."""
    if df.empty or "access_kind" not in df.columns:
        return pd.DataFrame()
    sub = df[df["access_kind"] == "shuffle"]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="access_window", columns="format",
                           values=metric, aggfunc="median").sort_index()


@dataclass
class Crossover:
    window_lo: int
    window_hi: int
    leader_lo: str
    leader_hi: str

    def __str__(self) -> str:
        return (f"leader changes {self.leader_lo} -> {self.leader_hi} "
                f"between W={self.window_lo:,} and W={self.window_hi:,}")


def find_crossovers(df: pd.DataFrame, *, metric: str = "wall_s_median") -> list[Crossover]:
    'Windows at which the fastest format changes.'
    table = window_table(df, metric=metric)
    if table.empty or table.shape[1] < 2:
        return []

    leaders = table.idxmin(axis=1)
    out: list[Crossover] = []
    windows = list(table.index)
    for lo, hi in zip(windows, windows[1:]):
        if leaders[lo] != leaders[hi]:
            out.append(Crossover(int(lo), int(hi), leaders[lo], leaders[hi]))
    return out


def speedup_vs(df: pd.DataFrame, baseline: str = "parquet",
               *, metric: str = "wall_s_median") -> pd.DataFrame:
    """Each format's speedup over ``baseline`` at every shuffle window."""
    table = window_table(df, metric=metric)
    if table.empty or baseline not in table.columns:
        return pd.DataFrame()
    return table[baseline].values[:, None] / table


def chunk_sensitivity(df: pd.DataFrame, *, metric: str = "wall_s_median") -> pd.DataFrame:
    'Per (access, format) spread of ``metric`` across chunk sizes.'
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["access_label", "format"])[metric]
    out = g.agg(best="min", worst="max", n_chunks="count").reset_index()
    out["swing"] = out["worst"] / out["best"]
    best_chunk = (df.loc[df.groupby(["access_label", "format"])[metric].idxmin()]
                    [["access_label", "format", "chunk_rows"]]
                    .rename(columns={"chunk_rows": "best_chunk_rows"}))
    return out.merge(best_chunk, on=["access_label", "format"])


def bootstrap_chunk_uncertainty(
    samples_by_chunk: dict[int, Sequence[float]], *, n_boot: int = 10_000, seed: int = 31
) -> list[dict]:
    """Bootstrap median intervals and best-chunk probabilities."""
    if n_boot < 1 or not samples_by_chunk:
        raise ValueError("bootstrap needs samples and at least one draw")
    chunks = sorted(samples_by_chunk)
    rng = np.random.default_rng(seed)
    medians = np.empty((n_boot, len(chunks)))
    for column, chunk in enumerate(chunks):
        samples = np.asarray(samples_by_chunk[chunk], dtype=float)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError(f"chunk {chunk} has no finite samples")
        draws = samples[rng.integers(0, samples.size, size=(n_boot, samples.size))]
        medians[:, column] = np.median(draws, axis=1)
    best = medians == medians.min(axis=1, keepdims=True)
    best_credit = best / best.sum(axis=1, keepdims=True)
    return [
        {
            "chunk_rows": chunk,
            "median_s": float(np.median(samples_by_chunk[chunk])),
            "median_ci95_s": [
                float(np.quantile(medians[:, column], 0.025)),
                float(np.quantile(medians[:, column], 0.975)),
            ],
            "probability_best": float(best_credit[:, column].mean()),
        }
        for column, chunk in enumerate(chunks)
    ]


def gap_at_defaults_vs_tuned(
    df: pd.DataFrame,
    defaults: dict[str, int],
    *,
    metric: str = "wall_s_median",
    baseline: str = "parquet",
    fixed_layout_formats: set[str] | None = None,
) -> pd.DataFrame:
    "Compare defaults with each format's best verified layout."
    if df.empty:
        return pd.DataFrame()
    fixed_layout_formats = fixed_layout_formats or set()
    rows = []
    for label, grp in df.groupby("access_label"):
        at_default, at_best = {}, {}
        for fmt, sub in grp.groupby("format"):
            d = sub[sub["chunk_rows"] == defaults.get(fmt)]
            if not d.empty:
                at_default[fmt] = d[metric].median()
            at_best[fmt] = (at_default[fmt]
                            if fmt in fixed_layout_formats and fmt in at_default
                            else sub[metric].min())
        if baseline not in at_best or len(at_best) < 2:
            continue
        for fmt in at_best:
            if fmt == baseline:
                continue
            rows.append({
                "access_label": label,
                "format": fmt,
                "speedup_at_defaults": (at_default.get(baseline, float("nan"))
                                        / at_default[fmt]) if fmt in at_default else None,
                "speedup_when_tuned": at_best[baseline] / at_best[fmt],
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["gap_shrinkage"] = out["speedup_at_defaults"] / out["speedup_when_tuned"]
    return out


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "pdf.fonttype": 42,
        "figure.dpi": 150, "font.size": 9, "axes.grid": True,
        "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def plot_window_sweep(df: pd.DataFrame, out: Path, *, title: str = "") -> Path | None:
    """Fig.2 -- throughput against shuffle locality, with crossovers marked."""
    table = window_table(df)
    if table.empty:
        return None
    plt = _style()

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for fmt in table.columns:
        ax.plot(table.index, table[fmt] * 1000, marker="o", ms=4, label=fmt)
    for c in find_crossovers(df):
        ax.axvspan(c.window_lo, c.window_hi, color="0.85", zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("shuffle window W (rows)   sequential $\\rightarrow$ random")
    ax.set_ylabel("batch read time (ms, median)")
    ax.set_title(title or "Shuffle locality sweep")
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_projection(df: pd.DataFrame, out: Path, *, title: str = "") -> Path | None:
    """Fig.1 -- scan time against projection width."""
    if df.empty or not {"access_kind", "access_n_columns"} <= set(df.columns):
        return None
    sub = df[(df["access_kind"] == "scan") & df["access_n_columns"].notna()]
    if sub.empty:
        return None
    plt = _style()
    table = sub.pivot_table(index="access_n_columns", columns="format",
                            values="wall_s_median", aggfunc="median").sort_index()

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for fmt in table.columns:
        ax.plot(table.index, table[fmt] * 1000, marker="s", ms=4, label=fmt)
    ax.set_xlabel("columns projected")
    ax.set_ylabel("scan time (ms, median)")
    ax.set_title(title or "Projection width")
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_chunk_effect(df: pd.DataFrame, out: Path, *, title: str = "") -> Path | None:
    """The same access across chunk sizes, per format."""
    if df.empty or "chunk_rows" not in df.columns or df["chunk_rows"].nunique() < 2:
        return None
    plt = _style()
    labels = sorted(df["access_label"].unique())
    fig = plt.figure(figsize=(9.6, 5.8))
    grid = fig.add_gridspec(2, 6)
    slots = [(0, 0, 2), (0, 2, 4), (0, 4, 6), (1, 1, 3), (1, 3, 5)]
    axes = [fig.add_subplot(grid[row, start:stop])
            for row, start, stop in slots[:len(labels)]]
    for ax, label in zip(axes, labels):
        sub = df[df["access_label"] == label]
        table = sub.pivot_table(index="chunk_rows", columns="format",
                                values="wall_s_median", aggfunc="median").sort_index()
        for fmt in table.columns:
            label_name = "ORC" if fmt == "orc" else fmt.title()
            ax.plot(table.index, table[fmt] * 1000, marker="o", ms=4, label=label_name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Requested group size (rows)")
        kind = sub["access_kind"].iloc[0]
        n_columns = sub["access_n_columns"].iloc[0]
        if kind == "shuffle":
            panel_title = f"Local shuffle ({int(n_columns)} columns)"
        elif kind == "point":
            panel_title = f"Point lookup ({int(n_columns)} columns)"
        elif kind == "filter":
            panel_title = f"Selective filter ({int(n_columns)} columns)"
        elif pd.isna(n_columns):
            panel_title = "Full scan"
        else:
            panel_title = f"Projection ({int(n_columns)} columns)"
        ax.set_title(panel_title, fontsize=8)
    axes[0].set_ylabel("Median time (ms)")
    if len(axes) > 3:
        axes[3].set_ylabel("Median time (ms)")
    axes[-1].legend(frameon=False)

    if not title and "cache_mode" in df.columns:
        modes = sorted(df["cache_mode"].unique())
        suffix = f" ({modes[0]} cache)" if len(modes) == 1 else ""
    else:
        suffix = ""
    fig.suptitle((title or "Requested group-size sensitivity") + suffix, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
