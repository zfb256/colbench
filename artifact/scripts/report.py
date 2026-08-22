"""Turn results JSONL into figures and tables.

    python scripts/report.py results/fig2.jsonl --cache-mode cold --threads 1

One report must describe one dataset, cache mode, and thread count. If an input
contains more than one value on those axes, select the scope explicitly; the
tool refuses to average unlike experiments.

Prints the guard report first, on purpose: a reader should see what was thrown
away before seeing what survived.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from colbench import analyze  # noqa: E402
from scripts.run_experiment import DEFAULT_CHUNKS  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "figures"))
    ap.add_argument("--baseline", default="parquet")
    ap.add_argument("--keep-noisy", action="store_true")
    ap.add_argument("--cache-mode", choices=("cold", "warm"))
    ap.add_argument("--threads", type=int)
    ap.add_argument("--dataset")
    ap.add_argument("--only-chunk", action="store_true",
                    help="generate only the chunk-sensitivity report and figure")
    args = ap.parse_args(argv)

    paths = [Path(p) for pat in args.results for p in glob.glob(pat)]
    if not paths:
        raise SystemExit(f"no result files matched {args.results}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = analyze.load(*paths)
    if df.empty:
        raise SystemExit("no records found")

    print(f"loaded {len(df)} records from {len(paths)} file(s)")
    try:
        df = analyze.select_measurement_scope(
            df, cache_mode=args.cache_mode, threads=args.threads,
            dataset=args.dataset, allow_multiple_datasets=False,
        )
    except ValueError as exc:
        raise SystemExit(f"ambiguous measurement scope: {exc}") from exc
    guard = analyze.usable(df, noise_limit=10.0 if args.keep_noisy else analyze.NOISE_LIMIT)
    print(guard.report())
    data = guard.kept
    if data.empty:
        print("\nNothing usable survived the guard. Nothing will be plotted.")
        return 1

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 40)

    if not args.only_chunk:
        table = analyze.window_table(data)
        if not table.empty:
            print("\n== shuffle-locality sweep (median seconds) ==")
            print(table.to_string(float_format=lambda v: f"{v * 1000:9.2f}ms"))

            speed = analyze.speedup_vs(data, args.baseline)
            print(f"\n== speedup over {args.baseline} (>1 means faster) ==")
            print(speed.to_string(float_format=lambda v: f"{v:6.2f}x"))

            crossovers = analyze.find_crossovers(data)
            print("\n== crossovers ==")
            if crossovers:
                for c in crossovers:
                    print(f"  {c}")
            else:
                leader = table.idxmin(axis=1).unique()
                print(f"  NONE - {leader[0]!r} leads at every window.")
                print("  This is the negative outcome for C2: shuffle locality does")
                print("  not change the recommendation. Lean on C4 instead.")

            p = analyze.plot_window_sweep(data, out_dir / "fig2_window_sweep.png",
                                          title="Throughput vs shuffle locality")
            print(f"  wrote {p}")

        p = analyze.plot_projection(data, out_dir / "fig1_projection.png")
        if p:
            print(f"\nwrote {p}")

    if data["chunk_rows"].nunique() > 1:
        print("\n== chunk-size sensitivity (swing = worst/best across chunk sizes) ==")
        print(analyze.chunk_sensitivity(data).to_string(index=False))

        gap = analyze.gap_at_defaults_vs_tuned(data, DEFAULT_CHUNKS,
                                               baseline=args.baseline,
                                               fixed_layout_formats={"lance"})
        if not gap.empty:
            print("\n== C4: library defaults vs verified-layout optimization ==")
            print(gap.to_string(index=False))
            print("\n  gap_shrinkage > 1 means the advantage reported at defaults")
            print("  partly disappears after optimization over verified layouts.")

        p = analyze.plot_chunk_effect(data, out_dir / "fig4_chunk_effect.png")
        if p:
            print(f"  wrote {p}")
    else:
        print("\n(only one chunk size present; run the fig4 preset for C4)")

    sizes = data.drop_duplicates(["format", "chunk_rows"])[
        ["format", "chunk_rows", "file_size_bytes"]
    ].sort_values(["format", "chunk_rows"])
    if not sizes.empty:
        sizes = sizes.assign(size_mb=(sizes["file_size_bytes"] / 1e6).round(1))
        print("\n== on-disk size ==")
        print(sizes[["format", "chunk_rows", "size_mb"]].to_string(index=False))

    if not analyze.require_real_io(data).equals(data):
        print("\nNOTE: some rows lack kernel-accounted storage I/O; the")
        print("      bytes-read vs decode-time split is not available for them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
