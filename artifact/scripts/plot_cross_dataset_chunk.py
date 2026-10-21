"""Plot normalized ClickBench, TPC-H, and TPC-DS Parquet shuffle curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def shuffle_points(paths: list[Path]) -> list[tuple[int, float]]:
    points = []
    for path in paths:
        for row in read_jsonl(path):
            if (
                row["format"] == "parquet"
                and row["cache_mode"] == "cold"
                and row["access"]["kind"] == "shuffle"
            ):
                points.append((int(row["chunk_rows"]), float(row["wall_s_median"])))
    return sorted(points)


def normalized(points: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
    best = min(wall for _, wall in points)
    return [chunk for chunk, _ in points], [wall / best for _, wall in points]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path,
                    help="explicit directory containing the qualified P2 checkpoints")
    ap.add_argument("--out", type=Path, default=FIGURES / "fig6_cross_dataset_chunk.png")
    args = ap.parse_args(argv)

    series = {
        "ClickBench-32p": shuffle_points(sorted(args.results_dir.glob("p2_clickbench_32p_c*.jsonl"))),
        "TPC-H SF1 lineitem": shuffle_points(sorted(args.results_dir.glob("p2_tpch_sf1_lineitem_c*.jsonl"))),
        "TPC-DS SF1 store_sales": shuffle_points(sorted(args.results_dir.glob("p2_tpcds_sf1_store_sales_c*.jsonl"))),
    }
    expected = [1024, 8192, 32768, 131072, 524288, 1048576]
    for label, points in series.items():
        chunks = [chunk for chunk, _ in points]
        if chunks != expected:
            raise SystemExit(f"{label}: expected one qualified shuffle row per chunk; got {chunks}")

    clickbench = series["ClickBench-32p"]
    tpch = series["TPC-H SF1 lineitem"]
    tpcds = series["TPC-DS SF1 store_sales"]

    cb_x, cb_y = normalized(clickbench)
    tpch_x, tpch_y = normalized(tpch)
    tpcds_x, tpcds_y = normalized(tpcds)

    fig, ax = plt.subplots(figsize=(4.9, 2.75))
    ax.plot(cb_x, cb_y, marker="o", linewidth=1.8, label="ClickBench-32p")
    ax.plot(tpch_x, tpch_y, marker="s", linewidth=1.8, label="TPC-H SF1 lineitem")
    ax.plot(tpcds_x, tpcds_y, marker="^", linewidth=1.8, label="TPC-DS SF1 store_sales")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2**e for e in (10, 13, 15, 17, 19, 20)], [f"$2^{{{e}}}$" for e in (10, 13, 15, 17, 19, 20)])
    ax.minorticks_off()
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Parquet row-group size (rows)")
    ax.set_ylabel("Cold shuffle time / best")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
