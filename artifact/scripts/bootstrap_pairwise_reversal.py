#!/usr/bin/env python
'How often does the tuned format actually beat the default-point leader?'
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np

DRAWS = 10_000
SEED = 31
CLAIMS = [
    ("clickbench_32p", "shuffle", "cold", "lance", 1_024, "parquet"),
    ("clickbench_32p", "shuffle", "warm", "lance", 1_024, "parquet"),
]


def load(results: Path) -> list[dict]:
    recs = []
    for f in sorted(results.glob("*.jsonl")):
        for line in f.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def samples(rec: dict) -> list[float]:
    raw = rec.get("samples") or rec.get("wall_s_samples") or []
    if raw and isinstance(raw[0], dict):
        return [x["wall_s"] for x in raw]
    return list(raw)


def median_distribution(values: list[float]) -> list[tuple[float, float]]:
    'Exact distribution of the bootstrap median of ``len(values)`` draws.'
    n = len(values)
    need = (n + 1) // 2                      # rank of the median among n draws
    ordered = sorted(values)
    distinct = sorted(set(ordered))
    out, prev = [], 0.0
    for v in distinct:
        p = sum(1 for x in ordered if x <= v) / n          # empirical CDF at v
        at_most_v = sum(comb(n, j) * p ** j * (1 - p) ** (n - j)
                        for j in range(need, n + 1))
        out.append((v, at_most_v - prev))
        prev = at_most_v
    return out


def exact_p_smaller(a: list[float], b: list[float]) -> float:
    """P(median of a bootstrap resample of ``a`` < that of ``b``), exactly."""
    da, db = median_distribution(a), median_distribution(b)
    return sum(pa * pb for va, pa in da for vb, pb in db if va < vb)


def monte_carlo_p_smaller(a: list[float], b: list[float],
                          draws: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    aa, bb = np.asarray(a, float), np.asarray(b, float)
    wins = 0
    for _ in range(draws):
        if (np.median(rng.choice(aa, aa.size, replace=True))
                < np.median(rng.choice(bb, bb.size, replace=True))):
            wins += 1
    return wins / draws


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    results = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    recs = load(results)
    if not recs:
        print("no JSONL records under", results)
        return 1

    report = {
        "resampling_unit": "one measured repetition (wall_s)",
        "paired": False,
        "statistic": "median",
        "direction": "P(tuned challenger median < default-point leader median)",
        "primary_estimate": "exact_empirical_bootstrap",
        "monte_carlo": {"generator": "numpy.random.default_rng",
                        "seed": SEED, "draws": DRAWS,
                        "role": "cross-check of the exact value only"},
        "claims": [],
    }

    for ds, kind, cache, lead_fmt, lead_chunk, other_fmt in CLAIMS:
        def cell(fmt, chunk=None):
            for r in recs:
                if (r["dataset"]["name"] == ds and r["format"] == fmt
                        and r["cache_mode"] == cache
                        and r["access"]["kind"] == kind
                        and (chunk is None or r["chunk_rows"] == chunk)):
                    yield r

        lead = next(iter(cell(lead_fmt, lead_chunk)), None)
        tuned = min(cell(other_fmt), key=lambda r: r["wall_s_median"], default=None)
        if lead is None or tuned is None:
            print("skip %s/%s: cells missing" % (ds, cache))
            continue
        a, b = samples(tuned), samples(lead)
        if not a or not b:
            print("skip %s/%s: raw samples absent" % (ds, cache))
            continue

        exact = exact_p_smaller(a, b)
        mc = monte_carlo_p_smaller(a, b, DRAWS, SEED)
        entry = {
            "dataset": ds, "access": kind, "cache": cache,
            "selector": {
                "default_point_leader": "format %s at its default request %d"
                                        % (lead_fmt, lead_chunk),
                "tuned_challenger": "format %s at its fastest swept setting"
                                    % other_fmt,
            },
            "default_leader": {"format": lead_fmt, "chunk_rows": lead_chunk,
                               "median_s": lead["wall_s_median"],
                               "rel_iqr": lead["wall_s_rel_iqr"],
                               "n": len(b), "samples_s": b},
            "tuned_challenger": {"format": other_fmt,
                                 "chunk_rows": tuned["chunk_rows"],
                                 "median_s": tuned["wall_s_median"],
                                 "rel_iqr": tuned["wall_s_rel_iqr"],
                                 "n": len(a), "samples_s": a},
            "margin_pct": (lead["wall_s_median"] / tuned["wall_s_median"] - 1) * 100,
            "p_tuned_faster_exact": exact,
            "p_tuned_faster_monte_carlo": mc,
        }
        report["claims"].append(entry)
        print("%s %s: tuned %s@%d %.4f s vs default %s@%d %.4f s | margin %+.1f%%"
              % (ds, cache, other_fmt, tuned["chunk_rows"], tuned["wall_s_median"],
                 lead_fmt, lead_chunk, lead["wall_s_median"], entry["margin_pct"]))
        print("    exact %.5f | monte carlo (seed %d, %d draws) %.5f"
              % (exact, SEED, DRAWS, mc))

    if out:
        out.write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
