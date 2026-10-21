#!/usr/bin/env python
"Does omitting Lance's row-group argument give a different layout?"
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colbench.formats import LanceAdapter, canonical_reader  # noqa: E402

REQUESTS = (1_024, 8_192, 32_768, 131_072, 524_288, 1_048_576)


def signature(path: Path) -> dict:
    ad = LanceAdapter()
    sig = ad.layout_signature(path)
    return {
        "n_files": len(ad.files(path)),
        "fragment_rows": sig["fragment_rows"],
        "pages_per_column_distinct": sig["pages_per_column_distinct"],
        "size_bytes": ad.size_bytes(path),
    }


def write_default(reader, path: Path, compression: str,
                  keep_file_cap: bool) -> None:
    'Lance write with ``max_rows_per_group`` omitted.'
    import lance

    schema = LanceAdapter._apply_compression(reader.schema, compression)

    def batches():
        for b in reader:
            yield pa.RecordBatch.from_arrays(b.columns, schema=schema)

    kwargs = {"max_rows_per_file": 1 << 40} if keep_file_cap else {}
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(schema, batches()),
        str(path), mode="create", **kwargs)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    rows = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000
    comp = "zstd"

    def reader():
        pf = pq.ParquetFile(str(src))

        def gen():
            taken = 0
            for b in pf.iter_batches(batch_size=65_536):
                if taken >= rows:
                    break
                if taken + b.num_rows > rows:
                    b = b.slice(0, rows - taken)
                taken += b.num_rows
                yield b

        base = pa.RecordBatchReader.from_batches(pf.schema_arrow, gen())
        return canonical_reader(base)[0]

    ad = LanceAdapter()
    work = Path(tempfile.mkdtemp(prefix="lance_default_"))
    result = {"source": str(src), "rows": rows, "compression": comp,
              "lance": __import__("lance").__version__, "settings": {}}
    try:
        p = work / "omitted_group_only.lance"
        write_default(reader(), p, comp, keep_file_cap=True)
        result["omitted_group_only"] = signature(p)
        print("omit max_rows_per_group only ->",
              json.dumps(result["omitted_group_only"]))

        q0 = work / "omitted_both.lance"
        write_default(reader(), q0, comp, keep_file_cap=False)
        result["omitted_both"] = signature(q0)
        print("omit group and file cap    ->",
              json.dumps(result["omitted_both"]))

        for req in REQUESTS:
            q = work / f"req_{req}.lance"
            ad.write(reader(), q, chunk_rows=req, compression=comp)
            result["settings"][str(req)] = signature(q)
            print("request %9d ->" % req, json.dumps(result["settings"][str(req)]))

        base = result["omitted_group_only"]
        same = [r for r, s in result["settings"].items()
                if s["fragment_rows"] == base["fragment_rows"]
                and s["pages_per_column_distinct"] == base["pages_per_column_distinct"]]
        result["requests_matching_omitted_group_only"] = sorted(same, key=int)
        result["omitted_group_only_matches_all_requests"] = len(same) == len(REQUESTS)
        print("\nomitted-argument layout matches %d/%d requested settings"
              % (len(same), len(REQUESTS)))
        print("=> the swept cells are valid stand-ins for the library default"
              if result["omitted_group_only_matches_all_requests"]
              else "=> the default differs; name the true default cell")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if out:
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
