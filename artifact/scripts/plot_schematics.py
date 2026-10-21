'Draw the two schematic figures of the paper: the attribution problem and the audit protocol.'

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

BLUE, GREEN, ORANGE = "#4C72B0", "#55A868", "#DD8452"
INK, MUTED, LINE = "#22252A", "#6B7280", "#9CA3AF"
PANEL, BORDER = "#FFFFFF", "#E3E6EA"
FS = 0.74


class Canvas:
    def __init__(self, width, height, figsize):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.fig.patch.set_facecolor("white")

    def card(self, x, y, w, h, r=12, ec=BORDER, fc=PANEL, lw=1.0, z=3, ls="-"):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                         linewidth=lw, edgecolor=ec, facecolor=fc,
                                         linestyle=ls, zorder=z))

    def bar(self, x, y, w, h, colour, r=2.5, z=4):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                         linewidth=0, facecolor=colour, zorder=z))

    def txt(self, x, y, s, size, color=INK, ha="left", weight="normal", z=6):
        self.ax.text(x, y, s, fontsize=size * FS, color=color, ha=ha, va="center",
                     zorder=z, weight=weight)

    def elbow(self, pts, color=LINE, lw=1.4, head=True, z=5, ms=10, ls="-"):
        for i in range(len(pts) - 1):
            last = i == len(pts) - 2
            self.ax.add_patch(FancyArrowPatch(
                pts[i], pts[i + 1], arrowstyle="-|>" if (last and head) else "-",
                mutation_scale=ms, linewidth=lw, color=color, linestyle=ls,
                shrinkA=0, shrinkB=0, joinstyle="round", zorder=z))

    def badge(self, x, y, num, accent):
        self.ax.add_patch(Circle((x, y), 12.5, facecolor=accent, linewidth=0, zorder=4))
        self.txt(x, y + 0.5, num, 12, "white", "center", "bold")

    def save(self, out_dir, stem):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        self.fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=220)
        plt.close(self.fig)


def make_attribution(out_dir):
    c = Canvas(900, 330, (10.6, 3.9))

    c.txt(8, 12, "DEFAULT-POINT BENCHMARK", 10, MUTED, weight="bold")
    c.txt(892, 12, "COLBENCH", 10, GREEN, ha="right", weight="bold")

    lanes = (
        (30, GREEN, "FORMAT DESIGN", "Encodings and statistics",
         "what a format comparison should credit",
         "Optimized comparison", "advantage that survives tuning, Eq. (1)"),
        (130, BLUE, "WRITER POLICY", "Default layout",
         "pyarrow and DuckDB defaults differ 8.5-fold",
         "Capability audit", "verified layout space, penalty $P_X$"),
        (230, ORANGE, "HARNESS POLICY", "Reader lifetime",
         "per batch, per access, or per session",
         "Controlled reader lifetime", "per-access grid, per-batch probe"),
    )
    for y, accent, label, title, note, ctitle, cnote in lanes:
        c.card(8, y, 290, 80, r=13)
        c.bar(8, y, 5, 80, accent, z=4)
        c.txt(28, y + 18, label, 10.5, accent, weight="bold")
        c.txt(28, y + 40, title, 12.5, weight="bold")
        c.txt(28, y + 61, note, 10.5, MUTED)
        c.elbow([(298, y + 40), (316, y + 40), (316, 170), (334, 170)], accent, 1.6)

        c.card(602, y, 290, 80, r=13)
        c.bar(602, y, 5, 80, accent, z=4)
        c.txt(622, y + 30, ctitle, 12.5, weight="bold")
        c.txt(622, y + 54, cnote, 10.5, MUTED)
        c.elbow([(568, 170), (584, 170), (584, y + 40), (598, y + 40)], GREEN, 1.6)

    c.card(338, 110, 230, 120, r=18, ec=LINE, fc="#F6F7F9", lw=1.2)
    c.txt(453, 140, "Measured time", 13.5, ha="center", weight="bold")
    c.txt(453, 166, "one number per access", 10.5, MUTED, ha="center")
    c.txt(453, 186, "credited to the format", 10.5, MUTED, ha="center")
    c.txt(453, 210, "attribution problem", 10.5, ORANGE, ha="center", weight="bold")

    c.save(out_dir, "fig1_attribution")


def make_protocol(out_dir):
    c = Canvas(900, 430, (10.6, 5.1))

    for x, num, title in (
        (8, "1", "Qualify and write"),
        (312, "2", "Audit"),
        (616, "3", "Measure and attribute"),
    ):
        c.card(x, 8, 276, 414, r=14, z=1)
        c.bar(x, 8, 276, 5, MUTED, z=2)
        c.badge(x + 26, 36, num, MUTED)
        c.txt(x + 48, 36.5, title, 12.8, weight="bold")

    def step(x, y, w, h, title, notes, ec=BORDER, fc=PANEL, lw=1.0):
        c.card(x, y, w, h, r=12, ec=ec, fc=fc, lw=lw)
        c.txt(x + 16, y + 22, title, 12.5, weight="bold")
        for i, s in enumerate(notes):
            c.txt(x + 16, y + 44 + i * 17, s, 10.5, MUTED)

    step(28, 66, 236, 86, "Storage gate",
         ("verified page-cache eviction", "byte accounting · ≥ 300 MB/s"))
    step(28, 196, 236, 120, "Layout sweep",
         ("six group-size requests", "sent to every writer"),
         ec=BLUE, fc=BLUE + "14", lw=1.3)
    for i, h in enumerate((6, 10, 14, 18, 23, 28)):
        c.bar(44 + i * 30, 296 - h, 22, h, BLUE, z=5)
    c.elbow([(146, 152), (146, 192)], LINE, 1.35)

    step(332, 66, 236, 86, "Capability audit",
         ("emitted layout read back", "from every written file"),
         ec=BLUE, fc=BLUE + "14", lw=1.3)
    step(332, 196, 236, 70, "Verified layout space $\\mathcal{L}_X$",
         ("distinct emitted layouts",), ec=BLUE, lw=1.0)
    step(332, 300, 236, 70, "Default point $d_X$",
         ("written with the control omitted",), ec=BLUE, lw=1.0)
    c.elbow([(450, 152), (450, 192)], LINE, 1.35)
    c.elbow([(450, 266), (450, 296)], LINE, 1.35)

    step(636, 66, 236, 104, "Timed measurements",
         ("one reader per access · one thread", "seven repetitions", "cold and warm cache"),
         ec=ORANGE, fc=ORANGE + "14", lw=1.3)
    c.card(636, 196, 236, 56, r=12, ec=ORANGE, fc=PANEL, lw=1.1, ls=(0, (3, 2)))
    c.txt(652, 216, "Reader-lifetime probe", 11.5, weight="bold")
    c.txt(652, 236, "per-batch vs per-access, same host", 10, MUTED)
    c.card(636, 290, 236, 114, r=17, ec=GREEN, fc="#EAF3EC", lw=1.3)
    c.txt(754, 314, "Default vs optimized", 12.8, ha="center", weight="bold")
    c.txt(754, 340, "default penalty $P_X$", 11, ha="center")
    c.txt(754, 360, "inflation $\\Phi_{\\mathrm{default}}$", 11, ha="center")
    c.txt(754, 384, "per dataset, access, and cache mode", 10, MUTED, ha="center")
    c.elbow([(754, 196), (754, 174)], ORANGE, 1.2, ls=(0, (3, 2)))
    c.elbow([(872, 118), (886, 118), (886, 347), (876, 347)], ORANGE, 1.7)

    c.elbow([(264, 256), (298, 256), (298, 109), (328, 109)], BLUE, 1.7)
    c.elbow([(568, 109), (632, 109)], BLUE, 1.7)
    c.elbow([(568, 231), (600, 231)], BLUE, 1.7, head=False)
    c.elbow([(568, 335), (600, 335)], BLUE, 1.7, head=False)
    c.elbow([(600, 231), (600, 347), (632, 347)], BLUE, 1.7)

    c.save(out_dir, "fig2_protocol")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=FIGURES)
    args = ap.parse_args(argv)
    make_attribution(args.out)
    make_protocol(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
