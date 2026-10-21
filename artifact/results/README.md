# Result sets

Result directories are separate evidence strata and must not be pooled.

| path | status | purpose |
|---|---|---|
| `p2_qualified_20260821/` | formal | Qualified RAID5/XFS run: 18 checkpoints, 504 cells, seven samples per cell |
| `autodl_nvme_clean_20260816/` | historical | Earlier run under the per-batch reader protocol; the paper's reader-lifetime comparison uses the same-host probes in `p2_qualified_20260821/` |
| `autodl_nvme/` | historical | Earlier AutoDL byte-accounting control; not a formal replication |
| `aliyun_readahead_20260820/` | auxiliary | Readahead dose-response and metadata mechanism checks |
| `legacy_aliyun_20260813/` | auxiliary | Earlier unqualified-host layout and byte-model measurements |
| `windows_reference/` | auxiliary | Warm-cache correctness and early access-family probes |

Use explicit inputs from `p2_qualified_20260821/` for formal figures. The byte
and cross-dataset plotting scripts check the six-point sweep but do not
authenticate a directory's provenance from its name. The formal
directory excludes abandoned intermediate runs; those measurements are not
needed to reproduce any reported table or figure.

Top-level derived CSV and JSON files belong to auxiliary analyses unless their
provenance fields explicitly name the formal P2 directory.

Historical experiment directories are evidence from this study, not separate
projects. Keep their host and protocol labels when distributing or analyzing
them. `autodl_nvme/` is retained for audit history only and is not a source of
the paper's headline results.
