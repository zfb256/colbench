#!/usr/bin/env python
"""Turn a results directory into every number the paper needs.

Written so the post-run turnaround is one command rather than an afternoon of
ad-hoc analysis, and so a partial run still yields whatever it can: every
section degrades to "insufficient data" instead of failing.

Emits, for each stratum it can support:

  1. sweep swing per format per access          -> Table IV
  2. default-pairing inflation per format pair  -> Table V
  3. capability audit from emitted layouts      -> Table II
  4. per-projected-column-chunk residual k      -> Section V-E/F
  5. LaTeX-ready rows for 1-3

Run:  python scripts/report_rerun.py <results-dir> [more dirs] [--tex]
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DEFAULTS = {"parquet": 1_048_576, "lance": 1_024, "orc": 1_048_576}
#: Formats whose sweep maps to one physical layout: reusing the default cell
#: keeps repeat-measurement noise from being counted as a tuning opportunity.
FIXED_LAYOUT = {"lance"}


def load(dirs: list[str]) -> list[dict]:
    recs = []
    for d in dirs:
        p = Path(d)
        # Results directories intentionally contain pre-fix archives and
        # quarantined partial checkpoints. Recursing here silently mixed those
        # records into the formal rerun. A directory argument means its direct
        # result files; callers can still pass an archive or file explicitly.
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        files = [f for f in files if ".partial-" not in f.name]
        for f in files:
            for line in f.read_text(errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def key(r: dict) -> tuple:
    return (r["dataset"]["name"], r["access"]["label"], r["cache_mode"],
            r["format"], r["chunk_rows"])


def short(access: str) -> str:
    head = access.split("(")[0]
    return {"A1.scan": "full scan", "A2.scan": "projection",
            "A3.shuffle": "shuffle", "A4.point": "point",
            "A5.filter": "filter"}.get(head, head)


def swings(recs: list[dict]) -> None:
    print("\n=== 1. sweep swing (worst/best median wall) ===")
    by = defaultdict(dict)
    for r in recs:
        ds, acc, cm, fmt, chunk = key(r)
        by[(ds, acc, cm, fmt)][chunk] = r["wall_s_median"]
    print("  %-22s %-14s %-5s %-8s %8s %12s %10s"
          % ("dataset", "access", "cache", "format", "swing", "best chunk", "n"))
    for k in sorted(by):
        t = by[k]
        if len(t) < 2:
            continue
        ds, acc, cm, fmt = k
        print("  %-22s %-14s %-5s %-8s %7.2fx %12d %10d"
              % (ds[:22], short(acc), cm, fmt,
                 max(t.values()) / min(t.values()), min(t, key=t.get), len(t)))


def inflation(recs: list[dict]) -> list[tuple]:
    print("\n=== 2. default-pairing inflation, per format pair ===")
    by = defaultdict(dict)
    for r in recs:
        ds, acc, cm, fmt, chunk = key(r)
        by[(ds, acc, cm)].setdefault(fmt, {})[chunk] = r["wall_s_median"]

    rows = []
    print("  %-20s %-12s %-5s %-16s %9s %9s %8s"
          % ("dataset", "access", "cache", "pair", "def adv", "opt adv", "Phi"))
    for (ds, acc, cm), fmts in sorted(by.items()):
        names = sorted(fmts)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                ta, tb = fmts[a], fmts[b]
                da, db = ta.get(DEFAULTS.get(a)), tb.get(DEFAULTS.get(b))
                if da is None or db is None:
                    continue
                ma = da if a in FIXED_LAYOUT else min(ta.values())
                mb = db if b in FIXED_LAYOUT else min(tb.values())
                # orient on whichever format leads at the default point
                if da < db:
                    lead, adv_d, adv_o = a, db / da, mb / ma
                else:
                    lead, adv_d, adv_o = b, da / db, ma / mb
                if adv_o <= 0:
                    continue
                phi = adv_d / adv_o
                rows.append((ds, short(acc), cm, f"{a}-{b}", lead, adv_d, adv_o, phi))
                print("  %-20s %-12s %-5s %-16s %9.2f %9.2f %8.3f"
                      % (ds[:20], short(acc), cm, f"{a} vs {b}", adv_d, adv_o, phi))
    if not rows:
        print("  insufficient data: need two formats with their default cells")
    return rows


def capability(recs: list[dict]) -> None:
    print("\n=== 3. capability audit from emitted layouts ===")
    by = defaultdict(dict)
    for r in recs:
        le = r.get("layout_emitted")
        if not isinstance(le, dict) or "median_rows" not in le:
            continue
        by[(r["dataset"]["name"], r["format"])][r["chunk_rows"]] = (
            le["median_rows"], le.get("n_groups"))
    if not by:
        print("  no layout_emitted fields: run produced by pre-audit code")
        return
    print("  %-22s %-8s %10s %14s %12s %s"
          % ("dataset", "format", "requests", "distinct emitted", "responds", "emitted median rows"))
    for (ds, fmt), t in sorted(by.items()):
        med = [t[c][0] for c in sorted(t)]
        distinct = len(set(med))
        print("  %-22s %-8s %10d %14d %12s %s"
              % (ds[:22], fmt, len(t), distinct,
                 "yes" if distinct > 1 else "NO",
                 ", ".join(str(int(m)) for m in med)))
    print("  (a writer with one distinct emitted layout across many requests")
    print("   accepted the argument without changing the file)")


def residual(recs: list[dict], layout_json: Path | None) -> None:
    print("\n=== 4. per-projected-column-chunk residual k ===")
    if layout_json is None or not layout_json.exists():
        print("  skipped: needs group_overhead_fit.json for the layout term")
        return
    fit = json.load(open(layout_json))
    layout = {r["chunk_rows"]: r for r in fit["rows"]}
    got = defaultdict(dict)
    for r in recs:
        if (r["format"] != "parquet" or r["cache_mode"] != "cold"
                or r["access"]["kind"] != "shuffle"
                or not r["dataset"]["name"].startswith("clickbench")):
            continue
        ncol = r["access"].get("n_columns")
        got[r["dataset"]["name"]][r["chunk_rows"]] = (r["read_bytes_median"],
                                                     ncol, r["access"]["label"])
    if not got:
        print("  skipped: no cold ClickBench Parquet shuffle cells")
        return
    for ds, meas in sorted(got.items()):
        print("  %s" % ds)
        ks = []
        labels = {v[2] for v in meas.values()}
        if len(labels) > 1:
            print("    WARNING: mixed access plans %s; k is not comparable" % labels)
        for c in sorted(meas):
            if c not in layout:
                continue
            nbytes, ncol, _ = meas[c]
            if not ncol:
                print("    chunk %9d  skipped: projected column count unknown" % c)
                continue
            row = layout[c]
            # divide by the columns this access actually projected, not a constant
            k = ((nbytes - row["predicted_MB"] * 1e6)
                 / (row["row_groups_touched"] * ncol) / 1000)
            if c <= 131_072:
                ks.append(k)
            print("    chunk %9d  measured %8.1f MB  cols %2d  k = %7.2f kB"
                  % (c, nbytes / 1e6, ncol, k))
        if len(ks) >= 2:
            spread = (max(ks) - min(ks)) / statistics.mean(ks) * 100
            print("    small/mid k: mean %.3f kB, spread %.2f%%" %
                  (statistics.mean(ks), spread))
            print("    (Alibaba baseline: 6.21 kB, 4.4%; qualified host under"
                  " the old reader convention: 26 kB, 124%)")


def tex(rows: list[tuple]) -> None:
    print("\n=== 5. LaTeX rows for tab:inflation ===")
    for ds, acc, cm, pair, lead, ad, ao, phi in rows:
        print("%s & %s & %s & %s & %.2f / %.2f & %.2f \\\\"
              % (acc, cm, pair.replace("_", "\\_"), lead, ad, ao, phi))


def _find_layout_json(start: Path) -> Path | None:
    """Locate group_overhead_fit.json by walking up from the results directory."""
    here = start if start.is_dir() else start.parent
    for d in [here, *here.parents][:5]:
        cand = d / "group_overhead_fit.json"
        if cand.exists():
            return cand
    return None


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_tex = "--tex" in sys.argv
    if not args:
        print(__doc__)
        return 2
    recs = load(args)
    if not recs:
        print("no JSONL records found")
        return 1

    fmts = sorted({r["format"] for r in recs})
    dss = sorted({r["dataset"]["name"] for r in recs})
    accs = sorted({short(r["access"]["label"]) for r in recs})
    print("%d records | formats %s | datasets %s | access %s"
          % (len(recs), fmts, dss, accs))
    cold = [r for r in recs if r["cache_mode"] == "cold"]
    bad = [r for r in cold if not r.get("eviction_verified")]
    if bad:
        print("WARNING: %d cold cells without verified eviction -- not evidence"
              % len(bad))
    noisy = [r for r in recs if r.get("wall_s_rel_iqr", 0) > 0.25]
    if noisy:
        print("WARNING: %d cells with rel_iqr > 0.25" % len(noisy))
        for r in sorted(noisy, key=key):
            print("  NOISY dataset=%s format=%s chunk=%s access=%s cache=%s rel_iqr=%.3f"
                  % (r["dataset"]["name"], r["format"], r["chunk_rows"],
                     r["access"]["label"], r["cache_mode"],
                     r["wall_s_rel_iqr"]))

    swings(recs)
    rows = inflation(recs)
    capability(recs)
    residual(recs, _find_layout_json(Path(args[0])))
    if want_tex:
        tex(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
