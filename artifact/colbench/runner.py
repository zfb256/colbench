"""Experiment driver.

Turns a declarative matrix (datasets x formats x chunk sizes x access patterns x
cache modes x thread counts) into JSONL records carrying enough provenance that
a reviewer can tell what was actually measured.

Two fields exist purely so a result can never be over-read:

* ``eviction_verified`` -- was cold-cache eviction *demonstrated* to work on this
  machine, or merely requested?
* ``disk_io_is_real``   -- does ``read_bytes`` reflect kernel-accounted storage
  I/O, or syscall bytes that include page-cache hits?

Any cold-cache row with ``eviction_verified == false`` is not evidence, and the
analysis layer must refuse to plot it as such.
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow as pa

from . import access as acc_mod
from . import cache, datagen, measure
from .datagen import TableSpec
from .formats import ADAPTERS, FormatAdapter, canonical_reader, canonical_schema, get_adapter


#: Bumped whenever a change alters what a timed cell measures, so rows produced
#: under different conventions can never be pooled by accident.
PROTOCOL_SCHEMA = 1


def _source_digest() -> str:
    """SHA-256 over the measurement package, binding rows to the code that made them.

    Git metadata does not survive the transfers this bundle goes through, so the
    digest is taken over the sources themselves. It is the fallback identity a
    third party can recompute from the artifact alone.
    """
    import hashlib

    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(here.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


@dataclass
class Dataset:
    """Either a synthetic spec or a Parquet file on disk (the real datasets)."""

    name: str
    spec: TableSpec | None = None
    source: Path | None = None

    def __post_init__(self):
        if (self.spec is None) == (self.source is None):
            raise ValueError("exactly one of spec / source must be set")
        if self.source is not None:
            self.source = Path(self.source)

    @property
    def synthetic(self) -> bool:
        return self.spec is not None

    def reader(self, batch_rows: int = 65_536) -> pa.RecordBatchReader:
        if self.spec is not None:
            base = datagen.reader(self.spec)
        else:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(str(self.source))
            base = pa.RecordBatchReader.from_batches(
                pf.schema_arrow, pf.iter_batches(batch_size=batch_rows)
            )
        return canonical_reader(base)[0]

    def schema(self) -> pa.Schema:
        if self.spec is not None:
            return datagen.reader(self.spec).schema
        import pyarrow.parquet as pq

        return pq.ParquetFile(str(self.source)).schema_arrow

    def n_rows(self) -> int:
        if self.spec is not None:
            return self.spec.n_rows
        import pyarrow.parquet as pq

        return pq.ParquetFile(str(self.source)).metadata.num_rows

    def column_names(self) -> list[str]:
        if self.spec is not None:
            return self.spec.column_names()
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(str(self.source)).schema_arrow.names)

    def describe(self) -> dict:
        d = {
            "name": self.name,
            "synthetic": self.synthetic,
            "n_rows": self.n_rows(),
            "n_cols": len(self.column_names()),
            "canonical_schema_changes": [
                {"column": name, "from": old, "to": new}
                for name, old, new in canonical_schema(self.schema())[1]
            ],
        }
        if self.spec is not None:
            d["spec"] = asdict(self.spec)
        else:
            d["source"] = str(self.source)
        return d


@dataclass
class Matrix:
    datasets: Sequence[Dataset]
    formats: Sequence[str]
    chunk_rows: Sequence[int]
    accesses: Sequence[acc_mod.Access]
    cache_modes: Sequence[cache.CacheMode] = (cache.CacheMode.COLD, cache.CacheMode.WARM)
    threads: Sequence[int] = (1,)
    compression: str = "zstd"
    repeats: int = 5
    warmup: int = 1

    def cells(self) -> Iterator[tuple]:
        """Yield a deterministic, counterbalanced full-factorial order.

        Formats are adjacent within each workload/cache/thread cell and their
        order reverses on alternating access/chunk blocks.  This prevents a
        slow drift in shared-host or RAID-scrub load from being perfectly
        confounded with format (the old format-major order ran all of one
        format before the next).
        """
        formats = tuple(self.formats)
        for ds in self.datasets:
            for access_i, a in enumerate(self.accesses):
                for chunk_i, chunk in enumerate(self.chunk_rows):
                    for cm in self.cache_modes:
                        for th in self.threads:
                            ordered = formats if (access_i + chunk_i) % 2 == 0 else formats[::-1]
                            for fmt in ordered:
                                yield ds, fmt, chunk, a, cm, th


class Runner:
    def __init__(
        self,
        work_dir: Path,
        results_path: Path,
        *,
        verify_cache: bool = True,
        quiet: bool = False,
        orc_rates: dict[str, float] | None = None,
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = Path(results_path)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.quiet = quiet
        self.run_id = uuid.uuid4().hex[:12]
        self.env = measure.environment(self.work_dir)

        # Match the eviction check used by scripts/gate.py.
        self.eviction = (
            cache.verify_eviction(
                self.work_dir, size_mb=256, trials=cache.QUALIFICATION_TRIALS)
            if verify_cache
            else None
        )
        if self.eviction and not self.eviction.supported:
            self._log(f"WARNING: {self.eviction}")
            self._log("WARNING: cold-cache rows from this host are NOT evidence.")

        # A reader lifetime is an experimental condition, not an implementation
        # detail: an earlier protocol reopened the file per batch and charged
        # the resulting metadata cost to the layout. Recording the protocol in
        # every row lets a reader tell which convention produced it without
        # relying on the directory it happens to sit in.
        self.protocol = {
            "schema": PROTOCOL_SCHEMA,
            "reader_scope": "per_timed_access",
            "source_sha256": _source_digest(),
        }

        self._materialized: dict[tuple[str, str, int], Path] = {}
        self._orc_rates: dict[str, float] = dict(orc_rates or {})
        self._orc_rate_logged: set[str] = set()

    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg, flush=True)

    def _dataset_path(self, ds: Dataset, fmt: str, chunk: int) -> Path:
        suffix = ".parquet" if fmt == "parquet" else f".{fmt}"
        return self.work_dir / f"{ds.name}__{fmt}__c{chunk}{suffix}"

    def materialize(
        self, ds: Dataset, fmt: str, chunk: int, compression: str
    ) -> tuple[Path, float, int]:
        """Write ``ds`` in ``fmt``; reuse if already written this session."""
        key = (ds.name, fmt, chunk)
        adapter = get_adapter(fmt)
        path = self._dataset_path(ds, fmt, chunk)

        if key in self._materialized and path.exists():
            return path, float("nan"), adapter.size_bytes(path)

        self._log(f"  writing {ds.name} -> {fmt} (chunk_rows={chunk})")
        extra = {}
        if fmt == "orc":
            # ORC groups by bytes, so the shared rows-per-group target has to be
            # converted with a rate measured once per dataset. Measuring it per
            # setting would let the conversion absorb the very effect we sweep.
            extra["bytes_per_row"] = self._orc_rate(ds, compression)
        t0 = time.perf_counter()
        adapter.write(ds.reader(), path, chunk_rows=chunk, compression=compression,
                      **extra)
        write_s = time.perf_counter() - t0
        self._materialized[key] = path
        return path, write_s, adapter.size_bytes(path)

    def _emitted_layout(self, fmt: str, path: Path) -> dict | None:
        """What the writer actually produced, not what it was asked for.

        The requested value is already in ``chunk_rows``; recording the emitted
        grouping next to it is what lets a reader tell a tuned writer from one
        that accepted the argument and ignored it.
        """
        adapter = get_adapter(fmt)
        fn = getattr(adapter, "emitted_layout", None)
        if fn is None:
            return None
        try:
            sig = fn(path)
        except Exception as exc:  # never let provenance break a measurement
            return {"error": f"{type(exc).__name__}: {exc}"}
        if not sig:
            return {"n_groups": 0}
        ordered = sorted(sig)
        mid = len(ordered) // 2
        median = (ordered[mid] if len(ordered) % 2
                  else (ordered[mid - 1] + ordered[mid]) / 2)
        return {"n_groups": len(sig), "median_rows": median,
                "min_rows": ordered[0], "max_rows": ordered[-1]}

    def _layout_signature(self, fmt: str, path: Path):
        """Format-specific structural fingerprint, where the adapter offers one."""
        fn = getattr(get_adapter(fmt), "layout_signature", None)
        if fn is None:
            return None
        try:
            return fn(path)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _orc_rate(self, ds: Dataset, compression: str) -> float:
        cached = self._orc_rates.get(ds.name)
        if cached is None:
            from .formats import measure_bytes_per_row

            cached = measure_bytes_per_row(ds.reader(), compression=compression)
            self._orc_rates[ds.name] = cached
        if ds.name not in self._orc_rate_logged:
            self._log(f"  orc rate for {ds.name}: {cached:.3f} bytes/row")
            self._orc_rate_logged.add(ds.name)
        return cached

    def run(self, matrix: Matrix) -> list[dict]:
        skipped = {
            f: ADAPTERS[f].probe()[1]
            for f in matrix.formats
            if f in ADAPTERS and not ADAPTERS[f].probe()[0]
        }
        for f, why in skipped.items():
            self._log(f"SKIP format {f!r}: unavailable ({why})")
        live = [f for f in matrix.formats if f not in skipped]

        records: list[dict] = []
        with open(self.results_path, "a", encoding="utf-8") as sink:
            for ds, fmt, chunk, a, cm, th in matrix.cells():
                if fmt not in live:
                    continue
                rec = self._run_cell(ds, fmt, chunk, a, cm, th, matrix)
                records.append(rec)
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
        return records

    def _run_cell(self, ds, fmt, chunk, a, cache_mode, n_threads, matrix) -> dict:
        adapter = get_adapter(fmt)
        path, write_s, size_bytes = self.materialize(ds, fmt, chunk, matrix.compression)
        files = adapter.files(path)
        n_rows = ds.n_rows()

        indices = acc_mod.plan(a, n_rows)

        def once() -> int:
            return acc_mod.execute(adapter, path, a, indices=indices)

        def setup() -> None:
            cache.prepare(files, cache_mode)

        try:
            with measure.threads(n_threads):
                result = measure.repeat(
                    once, n=matrix.repeats, setup=setup, warmup=matrix.warmup
                )
        finally:
            # Cleanup is outside the timer; setup restores the requested cache state.
            cache.evict(files)

        wall = result.median("wall_s")
        cpu_parallelism = result.median("cpu_s") / wall if wall > 0 else 0.0
        # Ignore sub-second cells where startup work distorts CPU parallelism.
        if wall >= 1.0 and cpu_parallelism > n_threads * 1.25:
            raise RuntimeError(
                f"{fmt} used effective CPU parallelism {cpu_parallelism:.2f}x "
                f"with threads={n_threads}; refusing an unfair timing"
            )

        rec = {
            "run_id": self.run_id,
            "ts": time.time(),
            "dataset": ds.describe(),
            "format": fmt,
            "chunk_rows": chunk,
            "layout_emitted": self._emitted_layout(fmt, path),
            "layout_signature": self._layout_signature(fmt, path),
            "compression": matrix.compression,
            "file_size_bytes": size_bytes,
            "n_files": len(files),
            "write_wall_s": write_s,
            "access": {
                "kind": a.kind,
                "label": a.label,
                "n_columns": None if a.columns is None else len(a.columns),
                "columns": list(a.columns) if a.columns else None,
                "batch_rows": a.batch_rows,
                "window": a.window,
                "n_batches": a.n_batches,
                "seed": a.seed,
                "predicate": asdict(a.predicate) if a.predicate is not None else None,
            },
            "cache_mode": cache_mode.value,
            "threads": n_threads,
            "protocol": self.protocol,
            "cpu_parallelism_median": cpu_parallelism,
            "eviction_verified": bool(self.eviction and self.eviction.supported),
            "eviction_ratio": self.eviction.ratio if self.eviction else None,
            "env": self.env,
            **result.summary(),
        }

        noise = rec["wall_s_rel_iqr"]
        flag = "  <-- NOISY" if noise > 0.25 else ""
        self._log(
            f"  {fmt:8s} c={chunk:<8d} {a.label:38s} {cache_mode.value:4s} "
            f"t={n_threads} {rec['wall_s_median'] * 1000:9.2f}ms "
            f"(relIQR {noise:.0%}){flag}"
        )
        return rec


def load_results(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
