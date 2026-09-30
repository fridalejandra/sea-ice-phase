#!/usr/bin/env python3
"""
patch_period_filter.py -- add the period == "FULL" guard to the scripts that
still read annual_params.csv without it.

Background: annual_params.csv carries two fits per sector-year for 1979-2018
(period == "FULL", the 1979-2025 fit, and period == "HR2018", the 1979-2018
replication used for Table S1). Most Ch3 scripts got a filter for this on
2026-09-18; three did not:

    ch3_stats.py                      (done already by patch_ch3_stats_dedupe.py)
    compute_asl_ross_sweep.py         -> Sect. 3.3, Fig. 6, Table S3: the
                                         "+0.71 / -0.12, p < 1e-5", the split-
                                         year sweep, the LOO shift p
    compute_atmospheric_correlations.py  (not used in the manuscript, fixed for hygiene)

Same guard as the ch3_stats patch: keep period == FULL if the column exists,
then stop with the offending rows printed if any (sector, Year) is still
duplicated. Dry run by default.

    python patch_period_filter.py            # show
    python patch_period_filter.py --apply    # write (a .bak of each file is kept)
then
    python compute_asl_ross_sweep.py
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APPLY = "--apply" in sys.argv

# file -> (variable name the script reads into, exact anchor line)
TARGETS = {
    "compute_asl_ross_sweep.py":          ("ann",    "ann = pd.read_csv(ANNUAL_CSV)\n"),
    "compute_atmospheric_correlations.py": ("annual", "annual = pd.read_csv(ANNUAL_CSV)\n"),
}

GUARD = '''{anchor}# ── one row per (sector, Year), or stop ──────────────────────────────────────
# annual_params.csv carries two fits per sector-year for 1979-2018 (period FULL
# and HR2018). Every statistic below assumes one observation per year.
import sys  # harmless if already imported
if "period" in {v}.columns:
    _periods = sorted({v}["period"].astype(str).unique())
    if "FULL" not in _periods:
        sys.exit(f"annual_params has a period column but no FULL rows: {{_periods}}")
    {v} = {v}[{v}["period"].astype(str) == "FULL"].copy()
    print(f"annual_params: kept period == FULL (file also had {{_periods}})")
_dup = {v}.duplicated(["sector", "Year"], keep=False)
if _dup.any():
    print({v}.loc[_dup].sort_values(["sector", "Year"]).head(12).to_string())
    sys.exit(f"annual_params has {{int(_dup.sum())}} rows sharing a (sector, Year); "
             "fix the file before any statistic is trusted")
'''

for name, (var, anchor) in TARGETS.items():
    path = os.path.join(HERE, name)
    print(f"\n== {name}")
    if not os.path.exists(path):
        print("   not present, skipped")
        continue
    src = open(path, encoding="utf-8").read()
    if "one row per (sector, Year), or stop" in src:
        print("   already has the guard")
        continue
    k = src.count(anchor)
    if k != 1:
        print(f"   expected exactly one `{anchor.strip()}`, found {k} -- not touched")
        continue
    guard = GUARD.format(anchor=anchor, v=var)
    new = src.replace(anchor, guard)
    print("   will insert after `%s`:" % anchor.strip())
    for line in guard.splitlines()[1:]:
        print("      " + line)
    if APPLY:
        shutil.copy2(path, path + ".bak")
        open(path, "w", encoding="utf-8").write(new)
        print(f"   written; backup {name}.bak")

print("\n" + ("done." if APPLY else "dry run -- add --apply to write."))
