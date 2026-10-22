"""Plot measured and modeled Parquet shuffle read bytes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def shuffle_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["format"] == "parquet" and r["access_kind"] == "shuffle"]


def mb(values: list[float]) -> list[float]:
    return [v / 1_000_000 for v in values]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_csv", type=Path,
                    help="explicit read-amplification CSV from the qualified run")
    ap.add_argument("--out", type=Path, default=FIGURES / "fig5_read_amplification.pdf")
    args = ap.parse_args(argv)

    rows = sorted(shuffle_rows(args.input_csv), key=lambda r: int(r["chunk_rows"]))
    chunks = [int(r["chunk_rows"]) for r in rows]
    expected = [1024, 8192, 32768, 131072, 524288, 1048576]
    if chunks != expected:
        raise SystemExit(f"expected one qualified shuffle row per chunk; got {chunks}")
    x = [int(r["chunk_rows"]) for r in rows]
    measured = mb([float(r["measured_read_bytes_median"]) for r in rows])
    lower = mb([float(r["predicted_bytes_access_union"]) for r in rows])
    per_call = mb([float(r["predicted_bytes_per_call"]) for r in rows])

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(x, measured, marker="o", linewidth=1.8, label="measured cold read bytes")
    ax.plot(x, lower, marker="s", linewidth=1.8, label="column-chunk lower bound")
    ax.plot(x, per_call, marker="^", linewidth=1.8, label="per-call chunk sum")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2**e for e in (10, 13, 15, 17, 19, 20)], [f"$2^{{{e}}}$" for e in (10, 13, 15, 17, 19, 20)])
    ax.minorticks_off()
    ax.set_xlabel("Parquet row-group size (rows)")
    ax.set_ylabel("Bytes read for shuffle (MB)")
    ax.set_title("Measured vs layout-implied shuffle reads")
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
