# ColBench Reviewer Artifact

This repository contains the benchmark implementation, tests, formal raw
measurements, validation records, and scripts used to regenerate the reported
tables and figures. The manuscript is intentionally distributed separately.

## Repository layout

- `artifact/colbench/`: benchmark implementation
- `artifact/scripts/`: data preparation, execution, validation, and plotting
- `artifact/tests/`: correctness and protocol tests
- `artifact/results/p2_qualified_20260821/`: formal 504-cell result set
- `artifact/results/autodl_nvme_clean_20260816/`: discarded-reader-protocol
  control used for the reader-lifetime comparison
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

## Regenerate the packaged figures

From `artifact/`:

```bash
python scripts/report.py \
  'results/p2_qualified_20260821/p2_clickbench_32p_c*.jsonl' \
  --cache-mode cold --threads 1 --dataset clickbench_32p \
  --only-chunk --out figures
python scripts/plot_read_amplification.py \
  results/p2_qualified_20260821/read_amplification_p2.csv
python scripts/plot_cross_dataset_chunk.py \
  results/p2_qualified_20260821
```

Each plotting entry point requires an explicit qualified input and validates
the expected six-point sweep.
