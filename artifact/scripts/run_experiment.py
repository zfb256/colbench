'Run an experiment preset.'

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colbench import access, cache, datagen, paths, runner, storage  # noqa: E402
from colbench.formats import ADAPTERS, Predicate  # noqa: E402

DATA = paths.data_dir()

CHUNK_SWEEP = [8_192, 32_768, 131_072, 524_288]
DEFAULT_CHUNKS = {"parquet": 1_048_576, "lance": 1_024, "vortex": 65_536}


EXPANSION = {"parquet": 1.0, "lance": 2.6, "vortex": 2.6}


def estimate_disk_bytes(ds: runner.Dataset, formats, chunks) -> int:
    'Space the run will consume under work/, plus one copy of transient slack.'
    source = ds.source.stat().st_size if ds.source else ds.n_rows() * 100
    per_format = {f: source * EXPANSION.get(f, 2.6) for f in formats}
    total = sum(v * len(chunks) for v in per_format.values())
    return int(total + max(per_format.values(), default=0))


def check_disk(ds, formats, chunks, work: Path) -> None:
    import shutil as _shutil

    need = estimate_disk_bytes(ds, formats, chunks)
    work.mkdir(parents=True, exist_ok=True)
    free = _shutil.disk_usage(work).free
    print(f"disk: need ~{need / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
    if free < need:
        raise SystemExit(
            f"\nNot enough disk: need ~{need / 1e9:.1f} GB, have {free / 1e9:.1f} GB.\n"
            "Options, cheapest first:\n"
            "  * run formats whose chunk knob is a no-op at one chunk value:\n"
            "      --formats lance --chunks 131072\n"
            "    (Lance writes byte-identical files at every chunk_rows, so the\n"
            "     other copies are pure waste -- see REPRODUCIBILITY.md)\n"
            "  * use fewer ClickBench partitions\n"
            "  * clear work/ between runs\n"
            "Override with --skip-disk-check if the estimate is wrong for your data."
        )


def resolve_dataset(name: str) -> runner.Dataset:
    if name in datagen.PRESETS:
        return runner.Dataset(name, spec=datagen.PRESETS[name])
    path = DATA / f"{name}.parquet"
    if path.exists():
        return runner.Dataset(name, source=path)
    raise SystemExit(
        f"unknown dataset {name!r}\n"
        f"  synthetic presets: {sorted(datagen.PRESETS)}\n"
        f"  prepared files:    {sorted(p.stem for p in DATA.glob('*.parquet'))}"
    )


def pick_columns(ds: runner.Dataset, fraction: float) -> list[str]:
    """A deterministic column subset of roughly ``fraction`` of the schema."""
    names = ds.column_names()
    k = max(1, round(len(names) * fraction))
    step = max(1, len(names) // k)
    return names[::step][:k]


def preset_smoke(ds, args):
    cols = pick_columns(ds, 0.1)
    return [
        access.full_scan(batch_rows=65_536),
        access.projection_scan(cols, batch_rows=65_536),
        access.shuffle_read(window=4_096, batch_rows=4_096, n_batches=16, columns=cols),
        access.shuffle_read(window=ds.n_rows(), batch_rows=4_096, n_batches=16,
                            columns=cols),
    ], [131_072]


def preset_fig1(ds, args):
    """Scan and projection width -- reproduces the axis vendors report on."""
    accesses = [access.full_scan(batch_rows=65_536)]
    for frac in (0.01, 0.05, 0.10, 0.50):
        accesses.append(access.projection_scan(pick_columns(ds, frac),
                                               batch_rows=65_536))
    return accesses, [131_072]


def preset_fig2(ds, args):
    """Shuffle-locality sweep."""
    n = ds.n_rows()
    b = args.batch_rows
    cols = pick_columns(ds, 0.1)
    return [
        access.shuffle_read(window=w, batch_rows=b, n_batches=args.n_batches,
                            columns=cols, seed=17)
        for w in access.window_sweep(n, b, n_points=args.window_points)
    ], [131_072]


def preset_fig3(ds, args):
    cols = pick_columns(ds, 0.1)
    accesses = [access.point_lookup(n=args.n_batches, columns=cols, seed=23)]
    num = _numeric_column(ds)
    if num is not None:
        col, lo, hi = num
        for sel in (0.0001, 0.001, 0.01, 0.1):
            threshold = lo + max(1, int((hi - lo) * sel))
            accesses.append(access.filtered_scan(
                Predicate(col, "<", threshold), columns=cols))
    return accesses, [131_072]


def preset_fig4(ds, args):
    'C1: same access, sweeping the physical chunk size across formats.'
    cols = pick_columns(ds, 0.1)
    b = args.batch_rows
    n = ds.n_rows()
    mid = access.window_sweep(n, b, n_points=5)[2]
    live = {f.strip() for f in args.formats.split(",")} & set(ADAPTERS)
    chunks = sorted(set(CHUNK_SWEEP) | {DEFAULT_CHUNKS[f] for f in live
                                        if f in DEFAULT_CHUNKS})
    return [
        access.full_scan(batch_rows=65_536),
        access.projection_scan(cols, batch_rows=65_536),
        access.shuffle_read(window=mid, batch_rows=b, n_batches=args.n_batches,
                            columns=cols, seed=31),
    ], chunks


def preset_p2(ds, args):
    """Paper-2 matrix checkpoint: A1--A5 at one physical grouping target."""
    names = ds.column_names()
    width = min(10, len(names))
    step = max(1, len(names) // width)
    cols = names[::step][:width]
    batch_rows = args.batch_rows
    middle_window = access.window_sweep(ds.n_rows(), batch_rows, n_points=5)[2]
    accesses = [
        access.full_scan(batch_rows=65_536),
        access.projection_scan(cols, batch_rows=65_536),
        access.shuffle_read(window=middle_window, batch_rows=batch_rows,
                            n_batches=args.n_batches, columns=cols, seed=31),
        access.point_lookup(n=args.n_batches, columns=cols, seed=23),
    ]
    numeric = _numeric_column(ds)
    if numeric is not None:
        column, lo, hi = numeric
        threshold = lo + max(1, int((hi - lo) * 0.01))
        accesses.append(access.filtered_scan(
            Predicate(column, "<", threshold), columns=cols))
    return accesses, [1_024, 8_192, 32_768, 131_072, 524_288, 1_048_576]


def _numeric_column(ds: runner.Dataset):
    """First integer column plus its observed range, for selectivity sweeps."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if ds.spec is not None:
        return ("i0", 0, ds.spec.int_range)
    pf = pq.ParquetFile(str(ds.source))
    for field in pf.schema_arrow:
        if pa.types.is_integer(field.type) and field.name.replace("_", "").isalnum():
            batch = next(pf.iter_batches(batch_size=200_000, columns=[field.name]))
            bounds = pc.min_max(batch.column(0)).as_py()
            if bounds["min"] is None or bounds["max"] is None:
                continue
            lo, hi = int(bounds["min"]), int(bounds["max"])
            if hi > lo:
                return (field.name, lo, hi)
    return None


PRESETS = {
    "smoke": preset_smoke,
    "fig1": preset_fig1,
    "fig2": preset_fig2,
    "fig3": preset_fig3,
    "fig4": preset_fig4,
    "p2": preset_p2,
}

NATIVE_THREAD_ENVS = (
    "LANCE_CPU_THREADS",
    "LANCE_IO_THREADS",
    "RAYON_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
)


def configure_native_threads(spec: str) -> list[int]:
    'Configure native format runtimes before Lance is imported.'
    values = [int(value) for value in spec.split(",") if value.strip()]
    if not values or any(value < 1 for value in values):
        raise SystemExit("--threads must contain positive integers")
    if len(values) != 1:
        raise SystemExit(
            "multiple thread counts in one process cannot control Lance fairly; "
            "run one --threads value per command"
        )
    for name in NATIVE_THREAD_ENVS:
        os.environ[name] = str(values[0])
    return values


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", nargs="?", choices=sorted(PRESETS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dataset", default="smoke")
    ap.add_argument("--formats", default=",".join(ADAPTERS))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--threads", default="1")

    ap.add_argument("--batch-rows", type=int, default=4_096)
    ap.add_argument("--n-batches", type=int, default=64)
    ap.add_argument("--window-points", type=int, default=6)
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--orc-rate", type=float, default=None)
    ap.add_argument("--chunks", default=None,
                    help="comma-separated chunk_rows, overriding the preset. Use "
                         "this to run a format whose chunk knob is a no-op at a "
                         "single value instead of writing N identical copies "
                         "(Lance: ~2.2 GB each on ClickBench-8p).")
    ap.add_argument("--skip-disk-check", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.list or not args.preset:
        print("presets:", ", ".join(sorted(PRESETS)))
        print("datasets:", ", ".join(sorted(datagen.PRESETS)
                                     + [p.stem for p in DATA.glob("*.parquet")]))
        return 0

    thread_values = configure_native_threads(args.threads)
    ds = resolve_dataset(args.dataset)
    accesses, chunks = PRESETS[args.preset](ds, args)
    if args.chunks:
        chunks = sorted({int(c) for c in args.chunks.split(",") if c.strip()})

    if args.preset == "p2":
        if len(chunks) != 1:
            raise SystemExit("p2 is checkpointed one chunk per process; pass --chunks <one value>")
        if chunks[0] == 1_024:
            accesses = [a for a in accesses if a.kind not in {"point", "filter"}]
        elif not any(a.kind == "filter" for a in accesses):
            raise SystemExit(f"dataset {ds.name!r} has no numeric column for P2 A5")

    work = paths.work_dir()
    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
    live_fmts = [f for f in fmts if f in ADAPTERS and ADAPTERS[f].probe()[0]]

    dev = paths.describe_mount(work)
    if dev.get("suspect") and not args.skip_disk_check:
        raise SystemExit(
            f"\nwork/ is on {dev.get('fstype')} ({dev.get('device')}): {dev['note']}\n"
            f"  work dir: {work}\n"
            "Every timed read comes from work/, so this would measure that volume,\n"
            "not the formats. Point COLBENCH_WORK at a directly mounted block device:\n"
            "  export COLBENCH_WORK=/path/on/block-device/colbench-work\n"
            "Sources can stay where they are (COLBENCH_DATA) -- they are read only\n"
            "during materialization, never inside a timer.\n"
            "Run scripts/gate.py and use --skip-disk-check only for a documented override."
        )
    if not args.skip_disk_check:
        gate_path = paths.results_dir() / "gate_report.json"
        try:
            gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"\nNo valid storage gate report at {gate_path}: {exc}\n"
                "Run .venv/bin/python scripts/gate.py before the experiment."
            ) from exc
        gate_problems = storage.validate_gate_report(gate_report, work, dev)
        if gate_problems:
            if os.environ.get("COLBENCH_ALLOW_UNQUALIFIED_STORAGE") == "1":
                detail = "\n".join(f"  - {problem}" for problem in gate_problems)
                print(
                    "\nWARNING: running with an explicit unqualified-storage override.\n"
                    f"{detail}\n"
                    "Record this run as provider virtual block storage, not NVMe/local SSD."
                )
                gate_report["unqualified_storage_override"] = {
                    "enabled": True,
                    "reason": os.environ.get(
                        "COLBENCH_UNQUALIFIED_STORAGE_REASON",
                        "manual documented override",
                    ),
                    "diagnostic": os.environ.get(
                        "COLBENCH_UNQUALIFIED_STORAGE_DIAGNOSTIC",
                        "",
                    ),
                }
            else:
                detail = "\n".join(f"  - {problem}" for problem in gate_problems)
                raise SystemExit(
                    "\nThe saved storage gate does not authorize this run:\n"
                    f"{detail}\nRun .venv/bin/python scripts/gate.py again."
                )
        check_disk(ds, live_fmts, chunks, work)

    out = Path(args.out) if args.out else paths.results_dir() / f"{args.preset}.jsonl"
    seeded_rates = {ds.name: args.orc_rate} if args.orc_rate is not None else None
    r = runner.Runner(work, out, orc_rates=seeded_rates)

    modes = [cache.CacheMode.WARM]
    if r.eviction and r.eviction.supported:
        modes.insert(0, cache.CacheMode.COLD)
    else:
        print("NOTE: running WARM-cache only; this host cannot evict.")

    matrix = runner.Matrix(
        datasets=[ds],
        formats=fmts,
        chunk_rows=chunks,
        accesses=accesses,
        cache_modes=modes,
        threads=thread_values,
        compression=args.compression,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    n_cells = sum(1 for _ in matrix.cells())
    print(f"preset={args.preset} dataset={ds.name} rows={ds.n_rows():,} "
          f"cells={n_cells} repeats={args.repeats}")
    records = r.run(matrix)
    print(f"\nwrote {len(records)} records -> {out}")

    noisy = [x for x in records if x["wall_s_rel_iqr"] > 0.25]
    if noisy:
        print(f"WARNING: {len(noisy)}/{len(records)} cells have relative IQR > 25%; "
              "raise --repeats or quiesce the machine before trusting them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
