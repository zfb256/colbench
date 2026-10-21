'Format adapters.'

from __future__ import annotations

import abc
import importlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pyarrow as pa


@dataclass(frozen=True)
class Predicate:
    'A single-column comparison, e.g. ``Predicate("v0", "<", 0.01)``.'

    column: str
    op: str  # one of < <= > >= == !=
    value: object

    def to_sql(self) -> str:
        if not self.column.replace("_", "").isalnum():
            raise ValueError(
                f"column {self.column!r} needs quoting, which is not portable "
                "across the format SQL dialects; rename it instead"
            )
        val = self.value
        lit = f"'{val}'" if isinstance(val, str) else repr(val)
        return f"{self.column} {self.op} {lit}"

    def to_arrow(self):
        import pyarrow.dataset as ds

        f = ds.field(self.column)
        return {
            "<": f < self.value,
            "<=": f <= self.value,
            ">": f > self.value,
            ">=": f >= self.value,
            "==": f == self.value,
            "!=": f != self.value,
        }[self.op]


class FormatAdapter(abc.ABC):
    name: str = "base"
    writes_single_file: bool = True

    @classmethod
    def probe(cls) -> tuple[bool, str]:
        """Return ``(available, detail)`` without raising."""
        try:
            mod = importlib.import_module(cls._module)
        except Exception as exc:  # pragma: no cover - environment dependent
            return False, f"{type(exc).__name__}: {exc}"
        version = getattr(mod, "__version__", None)
        if version is None:  # submodules such as pyarrow.parquet carry no version
            root = importlib.import_module(cls._module.split(".")[0])
            version = getattr(root, "__version__", "unknown")
        return True, version

    @property
    def available(self) -> bool:
        return self.probe()[0]

    def remove(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    def files(self, path: Path) -> list[Path]:
        'Every on-disk file backing ``path``.'
        path = Path(path)
        if path.is_dir():
            return [p for p in path.rglob("*") if p.is_file()]
        return [path] if path.exists() else []

    def size_bytes(self, path: Path) -> int:
        return sum(p.stat().st_size for p in self.files(path))

    @abc.abstractmethod
    def write(
        self,
        reader: pa.RecordBatchReader,
        path: Path,
        *,
        chunk_rows: int,
        compression: str = "zstd",
    ) -> None:
        'Write the requested layout; emitted metadata determines whether the control took effect.'

    @abc.abstractmethod
    def n_rows(self, path: Path) -> int: ...

    @abc.abstractmethod
    def scan(
        self,
        path: Path,
        *,
        columns: Sequence[str] | None = None,
        batch_rows: int = 65_536,
    ) -> Iterator[pa.RecordBatch]: ...

    @abc.abstractmethod
    def take(
        self,
        path: Path,
        indices: np.ndarray,
        *,
        columns: Sequence[str] | None = None,
    ) -> pa.Table:
        """Random access: fetch exactly the rows in ``indices``."""

    def begin_take_access(self, path: Path) -> None:
        """Start one logical random-access call (one timed repetition)."""

    def end_take_access(self, path: Path) -> None:
        """Release reader state scoped to one logical random-access call."""

    @abc.abstractmethod
    def filter_scan(
        self,
        path: Path,
        predicate: Predicate,
        *,
        columns: Sequence[str] | None = None,
        batch_rows: int = 65_536,
    ) -> Iterator[pa.RecordBatch]: ...


class ParquetAdapter(FormatAdapter):
    name = "parquet"
    _module = "pyarrow.parquet"

    def write(self, reader, path, *, chunk_rows: int, compression: str = "zstd") -> None:
        import pyarrow.parquet as pq

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._clear_take_cache(path)
        self.remove(path)

        writer = pq.ParquetWriter(str(path), reader.schema, compression=compression)
        try:
            for chunk in _rechunk(reader, chunk_rows):
                writer.write_table(chunk)
        finally:
            writer.close()

    _ROW_GROUP_CACHE: dict[tuple, tuple[object, np.ndarray]] = {}

    def _file_and_starts(self, path):
        import pyarrow.parquet as pq

        path = Path(path)
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        hit = self._ROW_GROUP_CACHE.get(key)
        if hit is not None:
            return hit

        pf = pq.ParquetFile(str(path))
        md = pf.metadata
        starts = np.zeros(md.num_row_groups, dtype=np.int64)
        if md.num_row_groups > 1:
            rows = np.fromiter(
                (md.row_group(i).num_rows for i in range(md.num_row_groups - 1)),
                dtype=np.int64, count=md.num_row_groups - 1)
            starts[1:] = np.cumsum(rows)
        self._ROW_GROUP_CACHE[key] = (pf, starts)
        return pf, starts

    def _clear_take_cache(self, path) -> None:
        target = str(Path(path))
        for key in [k for k in self._ROW_GROUP_CACHE if k[0] == target]:
            pf, _ = self._ROW_GROUP_CACHE.pop(key)
            pf.close()

    def begin_take_access(self, path) -> None:
        self._clear_take_cache(path)

    def end_take_access(self, path) -> None:
        self._clear_take_cache(path)

    def n_rows(self, path) -> int:
        pf, _ = self._file_and_starts(path)
        return pf.metadata.num_rows

    def emitted_layout(self, path) -> list[int]:
        """Row count of every emitted row group: the layout signature."""
        pf, _ = self._file_and_starts(path)
        md = pf.metadata
        return [md.row_group(i).num_rows for i in range(md.num_row_groups)]

    def layout_signature(self, path) -> dict:
        """Exact row-group sequence used by the Parquet read path."""
        return {"row_group_rows": self.emitted_layout(path)}

    def scan(self, path, *, columns=None, batch_rows: int = 65_536):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(path))
        yield from pf.iter_batches(batch_size=batch_rows, columns=list(columns) if columns else None)

    def take(self, path, indices, *, columns=None):
        'Parquet has no row-level index, so we do what a real engine does: map each requested row to its row group, read only the row groups that are actually needed, and gather within them.'
        pf, starts = self._file_and_starts(path)
        cols = list(columns) if columns else None

        indices = np.asarray(indices, dtype=np.int64)
        rg_of = np.searchsorted(starts, indices, side="right") - 1

        order = np.argsort(rg_of, kind="stable")
        tables: list[pa.Table] = []
        for rg in np.unique(rg_of):
            local = indices[order][rg_of[order] == rg] - starts[rg]
            tbl = pf.read_row_group(int(rg), columns=cols)
            tables.append(tbl.take(pa.array(local)))

        if not tables:
            return pf.schema_arrow.empty_table() if cols is None else pa.table({})
        out = pa.concat_tables(tables)
        inverse = np.argsort(order, kind="stable")
        return out.take(pa.array(inverse))

    def filter_scan(self, path, predicate, *, columns=None, batch_rows: int = 65_536):
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(path), format="parquet")
        scanner = dataset.scanner(
            columns=list(columns) if columns else None,
            filter=predicate.to_arrow(),
            batch_size=batch_rows,
        )
        yield from scanner.to_batches()


class LanceAdapter(FormatAdapter):
    name = "lance"
    _module = "lance"
    writes_single_file = False

    COMPRESSION_KEY = b"lance-encoding:compression"

    @classmethod
    def _apply_compression(cls, schema: pa.Schema, compression: str) -> pa.Schema:
        if not compression or compression.lower() == "none":
            return schema
        fields = [
            f.with_metadata({**(f.metadata or {}), cls.COMPRESSION_KEY:
                             compression.encode()})
            for f in schema
        ]
        return pa.schema(fields, metadata=schema.metadata)

    def write(self, reader, path, *, chunk_rows: int, compression: str = "zstd") -> None:
        import lance

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._TAKE_DATASET_CACHE.pop(str(path), None)
        self.remove(path)

        schema = self._apply_compression(reader.schema, compression)

        def batches():
            for batch in reader:
                yield pa.RecordBatch.from_arrays(batch.columns, schema=schema)

        lance.write_dataset(
            pa.RecordBatchReader.from_batches(schema, batches()),
            str(path),
            mode="create",
            max_rows_per_group=chunk_rows,
            max_rows_per_file=1 << 40,
        )

    _TAKE_DATASET_CACHE: dict[str, object] = {}

    def begin_take_access(self, path) -> None:
        self._TAKE_DATASET_CACHE.pop(str(Path(path)), None)

    def end_take_access(self, path) -> None:
        self._TAKE_DATASET_CACHE.pop(str(Path(path)), None)

    def _take_dataset(self, path):
        import lance

        key = str(Path(path))
        dataset = self._TAKE_DATASET_CACHE.get(key)
        if dataset is None:
            dataset = lance.dataset(key)
            self._TAKE_DATASET_CACHE[key] = dataset
        return dataset

    def n_rows(self, path) -> int:
        import lance

        return lance.dataset(str(path)).count_rows()

    def emitted_layout(self, path) -> list[int]:
        'Rows per emitted fragment.'
        import lance

        return [f.count_rows() for f in lance.dataset(str(path)).get_fragments()]

    def layout_signature(self, path) -> dict:
        'Fragment rows *and* per-column page counts.'
        import lance
        from lance.file import LanceFileReader

        rows, pages = [], set()
        root = Path(path)
        for fragment in lance.dataset(str(root)).get_fragments():
            for data_file in fragment.data_files():
                reader = LanceFileReader(str(root / "data" / data_file.path))
                rows.append(reader.num_rows())
                pages.update(col.num_pages
                             for col in reader.file_statistics().columns)
        return {"fragment_rows": sorted(rows),
                "pages_per_column_distinct": sorted(pages)}

    def scan(self, path, *, columns=None, batch_rows: int = 65_536):
        import lance

        dataset = lance.dataset(str(path))
        scanner = dataset.scanner(
            columns=list(columns) if columns else None,
            batch_size=batch_rows,
        )
        yield from scanner.to_batches()

    def take(self, path, indices, *, columns=None):
        dataset = self._take_dataset(path)
        return dataset.take(
            np.asarray(indices, dtype=np.int64),
            columns=list(columns) if columns else None,
        )

    def filter_scan(self, path, predicate, *, columns=None, batch_rows: int = 65_536):
        import lance

        dataset = lance.dataset(str(path))
        scanner = dataset.scanner(
            columns=list(columns) if columns else None,
            filter=predicate.to_sql(),
            batch_size=batch_rows,
        )
        yield from scanner.to_batches()


class OrcAdapter(FormatAdapter):
    'ORC, whose horizontal unit is denominated in *bytes*, not rows.'

    name = "orc"
    _module = "pyarrow.orc"

    _CODECS = {"none": "UNCOMPRESSED", "uncompressed": "UNCOMPRESSED"}

    def _codec(self, compression: str) -> str:
        return self._CODECS.get((compression or "").lower(), compression)

    def _reject_unsigned(self, schema: pa.Schema) -> None:
        bad = [f.name for f in schema if pa.types.is_unsigned_integer(f.type)]
        if bad:
            raise ValueError(
                f"ORC cannot represent unsigned columns {bad}; run the source "
                "through canonical_schema() so all formats receive one schema"
            )

    def write(self, reader, path, *, chunk_rows: int, compression: str = "zstd",
              bytes_per_row: float | None = None) -> None:
        import pyarrow.orc as orc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._clear_take_state(path)
        self.remove(path)
        self._reject_unsigned(reader.schema)

        if bytes_per_row is None:
            bytes_per_row, reader = self._rate_from_stream(reader, compression)

        stripe_bytes = max(1, int(round(chunk_rows * bytes_per_row)))
        writer = orc.ORCWriter(str(path), compression=self._codec(compression),
                               stripe_size=stripe_bytes)
        try:
            for chunk in _rechunk(reader, chunk_rows):
                writer.write(chunk)
        finally:
            writer.close()

    def _rate_from_stream(self, reader, compression: str,
                          sample_rows: int = 200_000):
        'Measure bytes per row from the head of ``reader`` and give it back.'
        import pyarrow.orc as orc

        buffered, taken = [], 0
        for batch in reader:
            buffered.append(batch)
            taken += batch.num_rows
            if taken >= sample_rows:
                break
        if not buffered:
            raise ValueError("empty source: cannot measure a bytes-per-row rate")

        sample = pa.Table.from_batches(buffered, schema=reader.schema)
        rate = _orc_bytes_per_row(sample, self._codec(compression))

        rest = reader

        def rebuilt():
            yield from buffered
            yield from rest

        return rate, pa.RecordBatchReader.from_batches(reader.schema, rebuilt())

    def n_rows(self, path) -> int:
        import pyarrow.orc as orc

        return orc.ORCFile(str(path)).nrows

    _STRIPE_CACHE: dict[tuple, list[int]] = {}
    _TAKE_FILE_CACHE: dict[str, object] = {}

    def _clear_take_state(self, path) -> None:
        target = str(Path(path))
        self._TAKE_FILE_CACHE.pop(target, None)
        for key in [k for k in self._STRIPE_CACHE if k[0] == target]:
            self._STRIPE_CACHE.pop(key, None)

    def begin_take_access(self, path) -> None:
        self._clear_take_state(path)

    def end_take_access(self, path) -> None:
        self._clear_take_state(path)

    def _take_file(self, path):
        import pyarrow.orc as orc

        key = str(Path(path))
        f = self._TAKE_FILE_CACHE.get(key)
        if f is None:
            f = orc.ORCFile(key)
            self._TAKE_FILE_CACHE[key] = f
        return f

    @staticmethod
    def _narrowest_column(schema: pa.Schema) -> str:
        """Cheapest column to scan when only row counts are wanted."""
        def width(field):
            t = field.type
            if pa.types.is_boolean(t):
                return 1
            try:
                return t.bit_width // 8
            except (ValueError, AttributeError):
                return 1 << 20        # varlen: never preferred
        return min(schema, key=width).name

    def _stripe_rows(self, path) -> list[int]:
        import pyarrow.orc as orc

        path = Path(path)
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        hit = self._STRIPE_CACHE.get(key)
        if hit is not None:
            return hit

        f = orc.ORCFile(str(path))
        col = self._narrowest_column(f.schema)
        rows = [f.read_stripe(i, columns=[col]).num_rows for i in range(f.nstripes)]
        self._STRIPE_CACHE[key] = rows
        return rows

    def emitted_layout(self, path) -> list[int]:
        """Row count of every emitted stripe: the layout signature for ORC."""
        return list(self._stripe_rows(path))

    def layout_signature(self, path) -> dict:
        """Exact stripe sequence used by the ORC read path."""
        return {"stripe_rows": self.emitted_layout(path)}

    def _stripe_starts(self, path) -> np.ndarray:
        rows = self._stripe_rows(path)
        starts = np.zeros(len(rows), dtype=np.int64)
        if rows:
            starts[1:] = np.cumsum(np.asarray(rows[:-1], dtype=np.int64))
        return starts

    def scan(self, path, *, columns=None, batch_rows: int = 65_536):
        import pyarrow.orc as orc

        f = orc.ORCFile(str(path))
        cols = list(columns) if columns else None
        for i in range(f.nstripes):
            stripe = f.read_stripe(i, columns=cols)
            tbl = pa.Table.from_batches([stripe]) if isinstance(
                stripe, pa.RecordBatch) else stripe
            for b in tbl.to_batches(max_chunksize=batch_rows):
                yield b

    def take(self, path, indices, *, columns=None):
        'Stripe-granular gather, the ORC analogue of Parquet row groups.'
        starts = self._stripe_starts(path)
        f = self._take_file(path)
        cols = list(columns) if columns else None

        indices = np.asarray(indices, dtype=np.int64)
        stripe_of = np.searchsorted(starts, indices, side="right") - 1

        order = np.argsort(stripe_of, kind="stable")
        tables: list[pa.Table] = []
        for s in np.unique(stripe_of):
            local = indices[order][stripe_of[order] == s] - starts[s]
            stripe = f.read_stripe(int(s), columns=cols)
            tbl = pa.Table.from_batches([stripe]) if isinstance(
                stripe, pa.RecordBatch) else stripe
            tables.append(tbl.take(pa.array(local)))

        if not tables:
            return pa.table({})
        out = pa.concat_tables(tables)
        inverse = np.argsort(order, kind="stable")
        return out.take(pa.array(inverse))

    def filter_scan(self, path, predicate, *, columns=None, batch_rows: int = 65_536):
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(path), format="orc")
        scanner = dataset.scanner(
            columns=list(columns) if columns else None,
            filter=predicate.to_arrow(),
            batch_size=batch_rows,
        )
        yield from scanner.to_batches()


class VortexAdapter(FormatAdapter):
    name = "vortex"
    _module = "vortex"

    @classmethod
    def probe(cls) -> tuple[bool, str]:
        'Reject Vortex versions that cannot satisfy the benchmark contract.'
        ok, version = super().probe()
        if not ok:
            return ok, version
        return False, (
            f"{version} imports but is benchmark-incompatible: Python writer "
            "has no none/zstd codec selector or physical chunk_rows control"
        )

    def write(self, reader, path, *, chunk_rows: int, compression: str = "zstd") -> None:
        raise RuntimeError(self.probe()[1])

    def n_rows(self, path) -> int:
        import vortex as vx

        return len(vx.io.open(str(path)))

    def scan(self, path, *, columns=None, batch_rows: int = 65_536):
        import vortex as vx

        f = vx.io.open(str(path))
        scan = f.scan(projection=list(columns) if columns else None)
        for chunk in scan.to_arrow_table().to_batches(max_chunksize=batch_rows):
            yield chunk

    def take(self, path, indices, *, columns=None):
        import vortex as vx

        f = vx.io.open(str(path))
        idx = np.asarray(indices, dtype=np.int64)
        return f.take(idx, projection=list(columns) if columns else None).to_arrow_table()

    def filter_scan(self, path, predicate, *, columns=None, batch_rows: int = 65_536):
        import vortex as vx

        f = vx.io.open(str(path))
        scan = f.scan(
            projection=list(columns) if columns else None,
            filter=predicate.to_sql(),
        )
        for chunk in scan.to_arrow_table().to_batches(max_chunksize=batch_rows):
            yield chunk


def _rechunk(reader: Iterable[pa.RecordBatch], chunk_rows: int) -> Iterator[pa.Table]:
    'Regroup an arbitrary batch stream into tables of exactly ``chunk_rows``.'
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    schema = reader.schema
    pending: list[pa.RecordBatch] = []
    held = 0
    for batch in reader:
        pending.append(batch)
        held += batch.num_rows
        while held >= chunk_rows:
            table = pa.Table.from_batches(pending, schema=schema)
            yield table.slice(0, chunk_rows).combine_chunks()
            rest = table.slice(chunk_rows)
            held = rest.num_rows
            pending = rest.to_batches() if held else []
    if held:
        yield pa.Table.from_batches(pending, schema=schema).combine_chunks()



def _orc_bytes_per_row(sample: pa.Table, compression: str) -> float:
    'Compressed ORC bytes per row for ``sample``, measured in memory.'
    import pyarrow.orc as orc

    sink = pa.BufferOutputStream()
    orc.write_table(sample, sink, compression=compression, stripe_size=1 << 30)
    return len(sink.getvalue()) / sample.num_rows


# Widen every writer's schema consistently; uint64 cannot fit losslessly in a signed type.
UNSIGNED_TO_SIGNED = {
    "uint8": pa.int16(),
    "uint16": pa.int32(),
    "uint32": pa.int64(),
}


def canonical_schema(schema: pa.Schema) -> tuple[pa.Schema, list[tuple[str, str, str]]]:
    'One schema every candidate writer can represent, plus what changed.'
    unsupported = [f.name for f in schema if str(f.type) == "uint64"]
    if unsupported:
        raise ValueError(
            f"uint64 columns {unsupported} cannot be canonicalized without "
            "possible value loss; cast or drop them before benchmarking"
        )
    fields, changes = [], []
    for f in schema:
        key = str(f.type)
        if key in UNSIGNED_TO_SIGNED:
            new = UNSIGNED_TO_SIGNED[key]
            changes.append((f.name, key, str(new)))
            fields.append(pa.field(f.name, new, nullable=f.nullable))
        else:
            fields.append(f)
    return pa.schema(fields, metadata=schema.metadata), changes


def canonical_reader(reader: pa.RecordBatchReader) -> tuple[pa.RecordBatchReader,
                                                            list[tuple[str, str, str]]]:
    """Wrap ``reader`` so every batch arrives in the canonical schema."""
    schema, changes = canonical_schema(reader.schema)
    if not changes:
        return reader, changes

    def batches():
        for batch in reader:
            yield pa.RecordBatch.from_arrays(
                [batch.column(i).cast(schema.field(i).type)
                 for i in range(batch.num_columns)],
                schema=schema,
            )

    return pa.RecordBatchReader.from_batches(schema, batches()), changes


def measure_bytes_per_row(source_reader: pa.RecordBatchReader, *,
                          compression: str = "zstd",
                          sample_rows: int = 200_000) -> float:
    'Compressed ORC bytes per row, measured once per dataset.'
    rows, taken = [], 0
    for batch in source_reader:
        rows.append(batch)
        taken += batch.num_rows
        if taken >= sample_rows:
            break
    if not rows:
        raise ValueError("empty source: cannot measure a bytes-per-row rate")
    sample = pa.Table.from_batches(rows).slice(0, sample_rows)

    return _orc_bytes_per_row(sample, compression)


ADAPTERS: dict[str, type[FormatAdapter]] = {
    "parquet": ParquetAdapter,
    "lance": LanceAdapter,
    "orc": OrcAdapter,
    "vortex": VortexAdapter,
}


def get_adapter(name: str) -> FormatAdapter:
    try:
        return ADAPTERS[name]()
    except KeyError:
        raise KeyError(f"unknown format {name!r}; known: {sorted(ADAPTERS)}") from None


def available_adapters() -> dict[str, tuple[bool, str]]:
    return {name: cls.probe() for name, cls in ADAPTERS.items()}
