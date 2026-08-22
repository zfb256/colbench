"""Validate one P2 dataset/chunk checkpoint and its storage boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colbench import access as access_mod  # noqa: E402
from scripts.validate_t1_checkpoint import (  # noqa: E402
    ValidationError,
    _require,
    _validate_raw_samples,
    validate_boundaries,
)

FORMATS = {"parquet", "lance", "orc"}
CACHE_MODES = {"cold", "warm"}
THREAD_ENVS = (
    "LANCE_CPU_THREADS",
    "LANCE_IO_THREADS",
    "RAYON_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
)


def validate(path: Path, dataset: str, chunk: int, source: Path) -> None:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read checkpoint {path}: {exc}") from exc

    metadata = pq.ParquetFile(str(source)).metadata
    n_rows, n_cols = metadata.num_rows, metadata.num_columns
    access_count = 3 if chunk == 1_024 else 5
    expected_records = len(FORMATS) * len(CACHE_MODES) * access_count

    _require(len(rows) == expected_records,
             f"expected {expected_records} records, got {len(rows)}")
    # The reader lifetime decides what a timed cell measures, so a checkpoint
    # that does not state it cannot be pooled with one that does. Historical
    # rows predate the field and are reported as such rather than assumed.
    protocols = {json.dumps(row.get("protocol"), sort_keys=True) for row in rows}
    _require(len(protocols) == 1, "checkpoint mixes measurement protocols")
    protocol = rows[0].get("protocol")
    if protocol is None:
        print(f"  note: {path.name} predates protocol recording; "
              "reader scope is documented outside the JSONL")
    else:
        _require(protocol.get("reader_scope") == "per_timed_access",
                 f"reader_scope is {protocol.get('reader_scope')!r}, "
                 "not per_timed_access")
        _require(isinstance(protocol.get("source_sha256"), str)
                 and len(protocol["source_sha256"]) == 64,
                 "protocol lacks a usable source digest")

    _require({row.get("format") for row in rows} == FORMATS, "format set mismatch")
    _require({row.get("cache_mode") for row in rows} == CACHE_MODES, "cache set mismatch")
    _require({int(row.get("chunk_rows", -1)) for row in rows} == {chunk},
             "chunk_rows mismatch")
    _require({row.get("dataset", {}).get("name") for row in rows} == {dataset},
             "dataset name mismatch")
    _require({int(row.get("dataset", {}).get("n_rows", -1)) for row in rows} == {n_rows},
             "dataset row count mismatch")
    _require({int(row.get("dataset", {}).get("n_cols", -1)) for row in rows} == {n_cols},
             "dataset column count mismatch")
    _require({row.get("compression") for row in rows} == {"zstd"},
             "compression mismatch")
    _require({int(row.get("threads", -1)) for row in rows} == {1},
             "thread count mismatch")
    _require(len({row.get("run_id") for row in rows}) == 1
             and bool(rows[0].get("run_id")), "checkpoint run_id mismatch")
    _require(all(int(row.get("n", -1)) == 7 for row in rows),
             "every summary must contain 7 repeats")
    _require(all(row.get("disk_io_is_real") is True for row in rows),
             "kernel-accounted I/O is required")
    _require(all(row.get("eviction_verified") is True
                 for row in rows if row.get("cache_mode") == "cold"),
             "cold rows require verified eviction")

    width = min(10, n_cols)
    expected_kinds = Counter({"scan": 2, "shuffle": 1})
    if chunk != 1_024:
        expected_kinds.update({"point": 1, "filter": 1})

    representative = {}
    cells = set()
    for row in rows:
        access = row.get("access") or {}
        kind = access.get("kind")
        label = access.get("label")
        representative.setdefault(label, row)
        cells.add((row.get("format"), label, row.get("cache_mode")))

        _require(kind in expected_kinds, f"unexpected access kind: {kind!r}")
        if kind == "scan" and access.get("n_columns") is None:
            _require(label == "A1.scan(cols=all)", "A1 contract mismatch")
        elif kind == "scan":
            _require(access.get("n_columns") == width, "A2 width mismatch")
        elif kind == "shuffle":
            expected_window = access_mod.window_sweep(n_rows, 4_096, n_points=5)[2]
            _require(access.get("n_columns") == width, "A3 width mismatch")
            _require(access.get("batch_rows") == 4_096, "A3 batch size mismatch")
            _require(access.get("n_batches") == 64, "A3 batch count mismatch")
            _require(access.get("window") == expected_window, "A3 window mismatch")
            _require(access.get("seed") == 31, "A3 seed mismatch")
        elif kind == "point":
            _require(access.get("n_columns") == width, "A4 width mismatch")
            _require(access.get("n_batches") == 64, "A4 point count mismatch")
            _require(access.get("seed") == 23, "A4 seed mismatch")
        elif kind == "filter":
            predicate = access.get("predicate")
            _require(isinstance(predicate, dict), "A5 predicate provenance missing")
            _require(predicate.get("op") == "<", "A5 predicate operator mismatch")
            _require(access.get("n_columns") == width, "A5 width mismatch")

        native = (row.get("env") or {}).get("native_thread_env") or {}
        _require(all(native.get(name) == "1" for name in THREAD_ENVS),
                 "native thread environment is not pinned to 1")
        io_stack = (row.get("env") or {}).get("io_stack") or {}
        _require(io_stack.get("available") is True, "io_stack provenance missing")
        _require((io_stack.get("queue") or {}).get("read_ahead_kb") == "2048",
                 "P2 must use native 2048 KB readahead")

        emitted = row.get("layout_emitted")
        signature = row.get("layout_signature")
        _require(isinstance(emitted, dict) and emitted.get("n_groups", 0) > 0,
                 "emitted layout missing")
        _require(isinstance(signature, dict) and "error" not in signature,
                 "layout signature missing")
        if row.get("format") == "parquet":
            _require(bool(signature.get("row_group_rows")),
                     "Parquet row-group signature missing")
        elif row.get("format") == "orc":
            _require(bool(signature.get("stripe_rows")),
                     "ORC stripe signature missing")
        elif row.get("format") == "lance":
            _require(signature.get("pages_per_column_distinct") == [1],
                     "Lance page layout changed")
            _require(bool(signature.get("fragment_rows")),
                     "Lance fragment signature missing")

        _validate_raw_samples(row, int(row.get("rows", -1)))

    kind_counts = Counter((row.get("access") or {}).get("kind")
                          for row in representative.values())
    _require(kind_counts == expected_kinds,
             f"access matrix mismatch: {kind_counts} != {expected_kinds}")
    _require(len(cells) == expected_records, "duplicate or missing matrix cells")

    for label in representative:
        counts = {int(row["rows"]) for row in rows
                  if (row.get("access") or {}).get("label") == label}
        _require(len(counts) == 1, f"formats disagree on rows for {label}: {counts}")

    changes = {
        json.dumps(row.get("dataset", {}).get("canonical_schema_changes"),
                   sort_keys=True)
        for row in rows
    }
    _require(len(changes) == 1, "canonical schema provenance differs across cells")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("dataset")
    parser.add_argument("chunk", type=int)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start-boundary", type=Path, required=True)
    parser.add_argument("--end-boundary", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        validate(args.path, args.dataset, args.chunk, args.source)
        validate_boundaries(
            args.start_boundary,
            args.end_boundary,
            args.gate_report,
            work=args.work,
            data=args.data,
            results=args.results,
        )
    except ValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

