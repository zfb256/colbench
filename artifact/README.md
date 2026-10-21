# colbench

Independent, protocol-controlled benchmark of columnar storage formats
under sparse and shuffled access. Parquet, ORC and Lance are timed; Vortex is
carried as a capability result only, because the evaluated writer exposes
neither a grouping control nor a codec selector and so cannot satisfy the
compression contract.

The reproducibility contract is summarized in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Why this exists

Columnar-format comparisons often mix format differences with physical layout
choices. This harness measures several access patterns under the same protocol
for every format.

One measured knob is **shuffle locality**: a batch of `B` rows drawn from a
window of `W` consecutive rows.
`W = B` is sequential, `W = N` is fully random, and production sits in between.

---

## Setup

Requires Python **3.11+** (Vortex ships no wheels below that, and none for
Windows at all).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

On Windows, `vortex-data` will not install and page-cache eviction does not
work. Windows is usable for correctness work only.

## 1. Check the host

```bash
python scripts/gate.py
```

Exit `0` green, `1` qualified storage with scope or free-space warnings, `2` blocked.
It verifies by measurement that cache eviction works, `read_bytes` reflects
kernel-accounted storage I/O, and sequential and random reads meet the gate. A
virtual block device may qualify, but is never relabelled as NVMe. The full
provenance is written to `results/gate_report.json`. Formal runs require three
saved gate reports checked with `scripts/check_gate_stability.py`; use its
`--out` option when results are isolated in a separate directory. Rerun the
gate after every host reboot; the experiment runner rejects stale reports.

## 2. Get data

The commands below are a small quick-start only. Formal runs should follow
the fixed-seed and storage-gate contract in `REPRODUCIBILITY.md`.

```bash
python scripts/get_data.py clickbench --partitions 8
python scripts/get_data.py tpch --sf 1
python scripts/get_data.py tpcds --sf 1 --table store_sales
python scripts/get_data.py status
```

Each ClickBench partition is ~117 MB / 1 M rows (100 available). TPC-H and
TPC-DS are generated in-process by duckdb, no external generator binary needed.
Budget ~3x the source size on disk: the benchmark rewrites the data once per
format.

## 3. Run

```bash
python scripts/run_experiment.py --list
python scripts/run_experiment.py fig2 --dataset clickbench_8p --repeats 5
```

| preset | measures | current role |
|---|---|---|
| `smoke` | pipeline shakeout | correctness check |
| `fig1`  | scan + projection width (A1/A2) | artifact diagnostic |
| `fig2`  | shuffle-locality sweep (A3 / C2) | artifact diagnostic |
| `fig3`  | point lookup + selectivity (A4/A5) | artifact diagnostic |
| `fig4`  | chunk-size sweep (C1) | paper group-size figure source |
The experiment entry point normally rejects an invalid storage gate. It does
not automatically convert cold cells to warm; explicitly overridden runs may
retain unverified cold rows, which the analysis guard excludes.

## 4. Report

```bash
python scripts/report.py results/fig2.jsonl --cache-mode cold --threads 1
```

`report.py` prints the guard report before result tables.

---

## Layout

```
colbench/
  formats.py   format adapters; all per-format special casing lives here
  datagen.py   synthetic tables for the controlled sweeps
  access.py    access patterns A1-A5, incl. the shuffle-window knob
  cache.py     page-cache eviction + verification that it worked
  measure.py   timing, I/O accounting, thread pinning, repetitions
  runner.py    experiment matrix -> JSONL with provenance
  analyze.py   guards, crossover detection, figures
scripts/
  gate.py              D1 go/no-go
  get_data.py          dataset download / generation
  run_experiment.py    preset driver
  run_column_count_scan.py  TPC-H projection-width mechanism scans
  tpch_model_screen.py      TPC-H Parquet byte-model screening
  plot_cross_dataset_chunk.py normalized ClickBench/TPC-H/TPC-DS shuffle plot
  plot_read_amplification.py explicit-input Parquet byte-model plot
  plot_schematics.py   attribution and protocol schematics
  report_grid.py       full-grid inflation and read-versus-cost tables
  read_amplification_report.py explicit-input Parquet footer analysis
  run_p2_segmented.sh  formal 18-checkpoint P2 driver
  report_rerun.py      formal P2 tables and validation summary
  summarize_extra_results.py TPC-H, TPC-DS, and column-count summaries
  seed_sensitivity.py     analytical 64-seed shuffle-plan check
  bootstrap_chunk_uncertainty.py raw-sample median intervals and best probabilities
  probe_orc_layout.py  emitted-stripe capability check for the ORC writer
  report.py            figures and tables
```

---

## Design rules

These are load-bearing; violating one invalidates results rather than merely
degrading them.

1. **Bound materialization where possible.** Scans stream batches, but sparse
   reads decode whole row groups or stripes and collect selected rows. Memory
   needs depend on layout, projection, and reader; no fixed 16 GB bound is guaranteed.
2. **Never leave thread count to the default.** pyarrow will use every core and
   make numbers irreproducible. Pinned explicitly, reported per run.
3. **Never trust a cold number without verified eviction.** Every record carries
   `eviction_verified`; the analysis layer refuses to plot cold rows that lack
   it.
4. **Never report a single run or a mean.** Median and IQR over repeats; cells
   above 25% relative IQR are flagged and dropped by default.
5. **Ask every format for the same logical access, then audit what it emits.**
   Compression and row indices are fixed across formats, but physical layout
   controls are not assumed equivalent: Parquet uses row counts, ORC uses byte
   targets, and the tested Lance grouping argument is non-responsive under the
   fixed file-cap policy. Every result records the emitted layout.

## Tests

```bash
python -m pytest tests/ -q
```

The previously recorded packaged test result was `197 passed`; rerun the
command above to validate a modified checkout. The tests need no network access. The
formal timing runs were measured on Linux with Python 3.12.3, PyArrow 25.0.0
and Lance 9.0.0; correctness tests do not reproduce the reported timings.
`tests/test_m6_fairness.py` is the one to
extend when adding a format: it checks that a new adapter honours compression,
responds to chunking, and returns byte-identical data to the others.
