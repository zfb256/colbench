# Result sets

Result directories are separate evidence strata and must not be pooled.

| path | status | purpose |
|---|---|---|
| `p2_qualified_20260821/` | formal | Qualified RAID5/XFS run: 18 checkpoints, 504 cells, seven samples per cell |
| `autodl_nvme_clean_20260816/` | control | Storage-qualified run under the discarded per-batch reader protocol |
| `autodl_nvme/` | historical | Earlier AutoDL byte-accounting control; not a formal replication |
| `aliyun_readahead_20260820/` | auxiliary | Readahead dose-response and metadata mechanism checks |
| `legacy_aliyun_20260813/` | auxiliary | Earlier unqualified-host layout and byte-model measurements |
| `windows_reference/` | auxiliary | Warm-cache correctness and early access-family probes |

The formal figure entry points require explicit paths under
`p2_qualified_20260821/` and reject incomplete six-point sweeps. The formal
directory excludes abandoned intermediate runs; those measurements are not
needed to reproduce any reported table or figure.

Top-level derived CSV and JSON files belong to auxiliary analyses unless their
provenance fields explicitly name the formal P2 directory.
