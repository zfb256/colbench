'Cross-host robustness check for the chunk-size effect.'

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SHARED_CHUNKS = (8_192, 32_768, 131_072, 524_288)

MAIN_GLOB = "results/autodl_nvme_clean_20260816/fig4_clickbench_32p_c*.jsonl"
ALIYUN_GLOB = "results/legacy_aliyun_20260813/fig4_clickbench_32p_c*.jsonl"
REF_GLOB = "results/windows_reference/fig4.jsonl"


def load(pattern: str) -> list[dict]:
    out: list[dict] = []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
    return out


def summarize(records: list[dict], host: str) -> list[dict]:
    'Swing factor per (cache mode, format, access) over the shared chunks.'
    grouped: dict[tuple, dict[int, float]] = defaultdict(dict)
    for rec in records:
        chunk = rec["chunk_rows"]
        if chunk not in SHARED_CHUNKS:
            continue
        key = (rec["cache_mode"], rec["format"], rec["access"]["label"])
        grouped[key][chunk] = rec["wall_s_median"]

    rows = []
    for (cache_mode, fmt, label), by_chunk in sorted(grouped.items()):
        if len(by_chunk) < 2:
            continue
        best_chunk = min(by_chunk, key=by_chunk.get)
        worst_chunk = max(by_chunk, key=by_chunk.get)
        rows.append({
            "host": host,
            "cache_mode": cache_mode,
            "format": fmt,
            "access_label": label,
            "n_chunks": len(by_chunk),
            "best_s": by_chunk[best_chunk],
            "worst_s": by_chunk[worst_chunk],
            "swing": by_chunk[worst_chunk] / by_chunk[best_chunk],
            "best_chunk_rows": best_chunk,
            "worst_chunk_rows": worst_chunk,
        })
    return rows


def describe(records: list[dict]) -> str:
    if not records:
        return "no records"
    env = records[0].get("env", {})
    ds = records[0]["dataset"]
    return (f"{env.get('platform', '?')} | "
            f"{env.get('physical_cores', '?')} physical cores | "
            f"{ds['name']} {ds['n_rows']:,} rows | "
            f"cache={sorted({r['cache_mode'] for r in records})} | "
            f"device_io={sorted({r['disk_io_is_real'] for r in records})}")


def cold_shuffle_bytes(records: list[dict]) -> dict[int, float]:
    return {int(r["chunk_rows"]): float(r["read_bytes_median"])
            for r in records if r["format"] == "parquet"
            and r["cache_mode"] == "cold"
            and r["access"]["kind"] == "shuffle"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="write the summary table here")
    args = ap.parse_args(argv)

    main_recs = load(MAIN_GLOB)
    aliyun_recs = load(ALIYUN_GLOB)
    ref_recs = load(REF_GLOB)
    if not main_recs:
        raise SystemExit(f"no main-run records matched {MAIN_GLOB}")
    if not aliyun_recs:
        raise SystemExit(f"no Alibaba records matched {ALIYUN_GLOB}")
    if not ref_recs:
        raise SystemExit(f"no reference records matched {REF_GLOB}")

    print("qualified :", describe(main_recs))
    print("Alibaba   :", describe(aliyun_recs))
    print("Windows   :", describe(ref_recs))
    print(f"\nshared chunk sizes: {list(SHARED_CHUNKS)}")
    print("records are summarized separately and never pooled.\n")

    rows = (summarize(ref_recs, "windows-local-nvme")
            + summarize(aliyun_recs, "aliyun-cloud-block")
            + summarize(main_recs, "autodl-nvme-raid1"))

    hdr = (f"{'host':20s} {'cache':5s} {'format':8s} {'access':38s} "
           f"{'swing':>8s} {'best@':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['host']:20s} {r['cache_mode']:5s} {r['format']:8s} "
              f"{r['access_label']:38s} {r['swing']:7.2f}x {r['best_chunk_rows']:9,d}")

    print("\n== verdict ==")
    shuffle = {(r["host"], r["cache_mode"]): r for r in rows
               if r["format"] == "parquet" and r["access_label"].startswith("A3.")}
    for host, cache in (("windows-local-nvme", "warm"),
                        ("aliyun-cloud-block", "cold"),
                        ("autodl-nvme-raid1", "cold")):
        row = shuffle.get((host, cache))
        if row:
            print(f"  {host:20s}: {row['swing']:.2f}x, best chunk "
                  f"{row['best_chunk_rows']:,}")

    main_bytes = cold_shuffle_bytes(main_recs)
    aliyun_bytes = cold_shuffle_bytes(aliyun_recs)
    common = main_bytes.keys() & aliyun_bytes.keys()
    if not common:
        raise SystemExit("no shared cold Parquet shuffle records")
    max_delta = max(abs(main_bytes[c] - aliyun_bytes[c]) / aliyun_bytes[c]
                    for c in common)
    print(f"  max cold shuffle byte difference: {max_delta:.2%}")

    print("\n  Scope: compare curve shape and sensitivity, not absolute latency.")
    print("  The Windows run cannot support device-byte claims. The clean AutoDL")
    print("  run passed three storage gates and continuous RAID-health monitoring.")

    if args.csv:
        out = Path(args.csv)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
