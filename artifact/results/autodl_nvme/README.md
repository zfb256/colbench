# AutoDL NVMe replication

This directory preserves the independent ClickBench-32p replication run from
2026-08-15. The measured path was XFS on `/dev/md0`, a RAID1 over two Samsung
NVMe devices. All checkpoint boundaries record an idle, nondegraded `[UU]`
array, and the three gate probes measured 2,062--2,221 MB/s sequential reads.

The MD array also reported a pre-existing `mismatch_cnt=384`. The project's
conservative repeated-gate checker rejects any nonzero mismatch counter, so
these records are secondary cross-host evidence rather than a replacement for
the principal run. The saved gate reports and checkpoint boundaries record the
provenance and scope.
