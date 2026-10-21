'Regenerate the full-grid inflation table and the read-versus-cost table from the formal results.'

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ACCESS = {"A1": "scan", "A2": "projection", "A3": "shuffle", "A4": "point", "A5": "filter"}
DATASET = {"clickbench_32p": "ClickBench", "tpch_sf1_lineitem": "TPC-H",
           "tpcds_sf1_store_sales": "TPC-DS"}
PARQUET_DEFAULT = 1_048_576
ORC_NEIGHBOURS = (1_048_576, 524_288)


def load(results_dir: Path):
    cells = defaultdict(dict)
    files = {}
    for path in sorted(results_dir.glob("p2_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ds = DATASET[r["dataset"]["name"]]
            key = (ds, ACCESS[r["access"]["label"][:2]], r["cache_mode"])
            cells[key][(r["format"], r["chunk_rows"])] = r["wall_s_median"]
            w = r.get("write_wall_s")
            if w is not None and not math.isnan(w):
                files[(ds, r["format"], r["chunk_rows"])] = (r["file_size_bytes"], w)
    return cells, files


def grid_rows(cells):
    rows = []
    for (ds, acc, cache), d in sorted(cells.items()):
        pq = {c: t for (f, c), t in d.items() if f == "parquet"}
        orc = {c: t for (f, c), t in d.items() if f == "orc"}
        lance = sorted(t for (f, _), t in d.items() if f == "lance")
        p_def, p_best = pq[PARQUET_DEFAULT], min(pq.values())
        l_med = (lance[len(lance) // 2 - 1] + lance[len(lance) // 2]) / 2
        penalty = p_def / p_best
        lead = "L" if l_med < p_def else "P"
        phi_pl = penalty if lead == "L" else 1 / penalty
        if lance[0] <= p_best <= lance[-1]:
            tuned = "T"
        else:
            tuned = "L" if p_best > lance[-1] else "P"
        ratio_lo, ratio_hi = sorted((p_best / lance[-1], p_best / lance[0]))
        o_best = min(orc.values())
        phi_po = []
        for o_def_rows in ORC_NEIGHBOURS:
            o_def = orc[o_def_rows]
            if o_def < p_def:
                phi_po.append((p_def / o_def) / (p_best / o_best))
            else:
                phi_po.append((o_def / p_def) / (o_best / p_best))
        rows.append(dict(dataset=ds, access=acc, cache=cache, P_pq=round(penalty, 2), lead=lead,
                         Phi_PL=round(phi_pl, 2), tuned_PL=f"{ratio_lo:.2f}--{ratio_hi:.2f}",
                         tuned=tuned, Phi_PO="/".join(f"{v:.2f}" for v in phi_po)))
    return rows


def cost_rows(cells, files):
    rows = []
    for ds in DATASET.values():
        for fmt in ("parquet", "orc", "lance"):
            sizes = [s for (d, f, _), (s, _) in files.items() if d == ds and f == fmt]
            writes = [w for (d, f, _), (_, w) in files.items() if d == ds and f == fmt]
            best = {}
            for acc in ("scan", "shuffle", "point"):
                d = cells[(ds, acc, "cold")]
                best[acc] = min(t for (f, _), t in d.items() if f == fmt)
            rows.append(dict(dataset=ds, format=fmt,
                             size_mb=f"{min(sizes) / 1e6:,.0f}--{max(sizes) / 1e6:,.0f}",
                             write_s=f"{min(writes):.1f}--{max(writes):.1f}",
                             scan_s=round(best["scan"], 2), shuffle_s=round(best["shuffle"], 2),
                             point_ms=round(best["point"] * 1000)))
    return rows


def write_csv(path: Path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)
    cells, files = load(args.results_dir)
    grid, cost = grid_rows(cells), cost_rows(cells, files)
    out = args.out_dir or args.results_dir
    write_csv(out / "full_grid_inflation.csv", grid)
    write_csv(out / "read_vs_cost.csv", cost)
    for r in grid:
        print(" | ".join(str(v) for v in r.values()))
    print()
    for r in cost:
        print(" | ".join(str(v) for v in r.values()))


if __name__ == "__main__":
    main()
