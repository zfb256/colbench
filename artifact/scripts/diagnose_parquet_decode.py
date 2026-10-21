'Diagnostic Parquet single-thread hot full-scan throughput.'

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colbench import measure, paths  # noqa: E402


ROWS = 8_000_000
CHUNK_ROWS = 131_072
BATCH_ROWS = 65_536
REPEATS = 5


def materialize_slice(source: Path, target: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(str(source))
    writer = pq.ParquetWriter(str(target), pf.schema_arrow, compression="zstd")
    remaining = ROWS
    try:
        for batch in pf.iter_batches(batch_size=CHUNK_ROWS):
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
            writer.write_table(pa.Table.from_batches([batch]))
            remaining -= batch.num_rows
    finally:
        writer.close()
    if remaining:
        raise RuntimeError(f"source ended {remaining} rows short")


def scan_all(path: Path) -> int:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    return sum(batch.num_rows for batch in pf.iter_batches(batch_size=BATCH_ROWS))


def main() -> int:
    import pyarrow as pa

    data = paths.data_dir() / "clickbench_32p.parquet"
    if not data.exists():
        raise SystemExit(f"missing source: {data}")

    work = paths.work_dir()
    target = work / f"diagnostic_clickbench_8m_parquet_c{CHUNK_ROWS}.parquet"
    materialize_slice(data, target)
    size = target.stat().st_size

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    warm_rows = scan_all(target)
    samples = []
    for _ in range(REPEATS):
        with measure.threads(1):
            sample = measure.run_once(lambda: scan_all(target))
        samples.append(sample)
        time.sleep(1)

    wall = sorted(sample.wall_s for sample in samples)[len(samples) // 2]
    cpu = sorted(sample.cpu_s for sample in samples)[len(samples) // 2]
    mbps = size / wall / 1e6

    out = {
        "diagnostic": "parquet_single_thread_hot_full_scan",
        "formal_result": False,
        "source": str(data),
        "target": str(target),
        "rows": warm_rows,
        "file_size_bytes": size,
        "file_size_mb": size / 1e6,
        "chunk_rows": CHUNK_ROWS,
        "batch_rows": BATCH_ROWS,
        "repeats": REPEATS,
        "wall_s_median": wall,
        "cpu_s_median": cpu,
        "file_mb_per_wall_s": mbps,
        "gate_seq_read_mbps": 163,
        "pid": os.getpid(),
    }
    dest = paths.results_dir() / "diagnostic_parquet_decode_hot_8m.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
