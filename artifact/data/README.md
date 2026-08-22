# Input data

Input Parquet files are intentionally excluded from Git:

| file | size in the reference run | reconstruction |
|---|---:|---|
| `clickbench_32p.parquet` | 2.95 GB | `python scripts/get_data.py clickbench --partitions 32` |
| `tpch_sf1_lineitem.parquet` | 149 MB | `python scripts/get_data.py tpch --sf 1` |
| `tpcds_sf1_store_sales.parquet` | 107 MB | `python scripts/get_data.py tpcds --sf 1 --table store_sales` |

After reconstructing ClickBench-32p, verify it from `artifact/`:

```bash
sha256sum -c data/clickbench_32p.parquet.sha256
```

DuckDB-generated TPC-H and TPC-DS output is version dependent. The pinned
package version in `requirements.txt` is part of the reproduction contract.
