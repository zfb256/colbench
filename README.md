# ColBench Reviewer Artifact

This repository contains the benchmark implementation, tests, formal raw
measurements, validation records, and scripts used to regenerate the reported
tables and figures. The manuscript is intentionally distributed separately.

## Repository layout

- `artifact/colbench/`: benchmark implementation
- `artifact/scripts/`: data preparation, execution, validation, and plotting
- `artifact/tests/`: correctness and protocol tests
- `artifact/results/p2_qualified_20260821/`: formal 504-cell result set
- `artifact/results/p2_qualified_20260821/probe_metadata_*.log`: same-host
  reader-lifetime probes
- `artifact/results/autodl_nvme_clean_20260816/`: earlier per-batch-reader run,
  kept for audit history
- `artifact/figures/`: figures regenerated from the formal results
- `artifact/REPRODUCIBILITY.md`: full protocol and reproduction commands

## Data policy

The input Parquet files are not stored in Git because they are generated from
public inputs and range from 107 MB to 2.95 GB. The repository instead provides
the data-generation commands and the SHA-256 digest of the main ClickBench
source. See `artifact/data/README.md`.

## Quick verification

```bash
cd artifact
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
cd ..
sha256sum -c SHA256SUMS.txt
```

The tests exercise logical correctness and protocol guards; they do not rerun
storage timings. Timing replication requires Linux, a storage-qualified block
device, and the source data described in `artifact/REPRODUCIBILITY.md`.

## Regenerate the packaged figures and tables

From `artifact/`, with the installed virtual environment activated. The
outputs map to the paper as follows: `fig1_attribution` and `fig2_protocol` are
Figures 1 and 2, `fig4_chunk_effect`, `fig5_read_amplification`, and
`fig6_cross_dataset_chunk` are Figures 3 to 5, and `report_grid.py` writes
Tables 7 and 8.

```bash
source .venv/bin/activate
python scripts/plot_schematics.py
python scripts/report.py \
  'results/p2_qualified_20260821/p2_clickbench_32p_c*.jsonl' \
  --cache-mode cold --threads 1 --dataset clickbench_32p \
  --only-chunk --out figures
python scripts/plot_read_amplification.py \
  results/p2_qualified_20260821/read_amplification_p2.csv
python scripts/plot_cross_dataset_chunk.py \
  results/p2_qualified_20260821
python scripts/report_grid.py results/p2_qualified_20260821
```

These commands select the formal result set explicitly. The byte and
cross-dataset plots validate the six-point sweep; input provenance must also
be checked against `artifact/results/README.md`.
