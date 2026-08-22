# Storage-qualified run under the discarded reader protocol

This directory contains the six-checkpoint ClickBench-32p run completed on
2026-08-16 on XFS over a two-device Intel NVMe RAID1. Three storage gates
passed, and all 2,275 runtime RAID samples recorded `idle`, `mismatch_cnt=0`,
and `degraded=0`. The 72 JSONL records retain seven raw samples per cell.
This is not the paper's formal P2 run because it reopened the reader for every
batch; it is retained only as the reader-lifetime control.

The saved gate reports, checkpoint boundaries, runtime RAID monitor, and raw
samples record the run provenance.

Reanalysis against the Alibaba Cloud run gives a 10.33x Parquet cold-shuffle
swing over the four shared chunk sizes and 77.93x over all six. Absolute
kernel-accounted storage reads differ by up to 43.82% across the two Linux
hosts; the earlier 0.23% figure applies only to the superseded historical
AutoDL run in `../autodl_nvme/` and must not be attributed to this clean run.
