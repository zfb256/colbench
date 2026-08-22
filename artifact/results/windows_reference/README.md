# Windows reference results — NOT formal results

These records come from a **different host, a different storage class, a
different dataset scale, and warm cache only**. They are retained as an
auxiliary historical check showing that a chunk-size effect also appears on a
local NVMe device; they are not part of the formal qualified run.

**Never pool these records with the Linux T1 records.** They are kept separate
deliberately. `analyze.select_measurement_scope()` will refuse to mix hosts,
cache modes, and datasets; do not work around it.

## Provenance

| property | value |
|---|---|
| host | Windows 10, Intel 16 logical / 12 physical cores, 16.9 GB RAM |
| storage | KIOXIA EXCERIA PLUS G2, local consumer NVMe SSD |
| dataset | ClickBench-8p, 8,000,000 rows x 105 columns |
| chunk sizes | 8,192 / 32,768 / 131,072 / 524,288 |
| threads | 1 |
| compression | zstd |
| cache mode | **warm only** |
| storage gate | **DEGRADED** — see `gate_report_windows.json` |

The gate failed for two reasons, both platform limitations rather than hardware
ones:

- `eviction_supported: false` (ratio 1.44) — Windows offers no reliable
  per-file page-cache eviction, so no cold-cache measurement is possible.
- `disk_io_is_real: false` — Windows reports syscall bytes, not block-level
  `read_bytes`, so device I/O cannot be attributed.

Consequently these records carry **no cold-cache rows and no read-byte
evidence**. They measure reader and decode cost with the working set already
resident in memory.

## What they can support

One narrow observation: **a chunk-size effect is also visible with warm cache
on this local device.** Because these records use a different scale, cache
state, and protocol generation, they do not establish cross-host equality and
do not qualify any headline result.

For the four chunk sizes shared with the historical Alibaba Cloud sweep,
`scripts/crosshost_check.py` recomputes the two swing factors separately.

## What they cannot support

- **Not** the formal six-point span or its endpoints. This sweep omits the
  1,024-row and 1,048,576-row settings.
- **Not** any read-amplification claim. There are no device bytes here.
- **Not** any absolute timing comparison against the main run: different CPU,
  different dataset size, different cache state.

## Files

- `fig4.jsonl` — chunk sweep, both formats, three access patterns (24 records)
- `fig2.jsonl` — shuffle-window sweep at a single chunk size
- `fullgrid.jsonl` — Parquet-only factorial on a TPC-H lineitem table
- `gate_report_windows.json` — the failing gate that classifies this host

## Regenerating the source data

These records were produced from datasets that are **not shipped** with the
artifact, because both are reproducible from public sources:

```bash
python scripts/get_data.py clickbench --partitions 8   # -> clickbench_8p
python scripts/get_data.py tpch --sf 1                 # -> tpch_sf1_lineitem
```

`clickbench_8p` is not a slice of `clickbench_32p`: it is written independently
and therefore has its own row-group layout. Do not attempt to derive one from
the other.

## Reproducing the comparison

```bash
python scripts/crosshost_check.py --csv results/crosshost_shuffle_swing.csv
```

The script summarises each host separately and compares only the resulting
swing factors. It restricts itself to the four chunk sizes both sweeps share,
and reports disagreement rather than agreement if the hosts diverge.
