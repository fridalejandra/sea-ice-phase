#!/usr/bin/env python3
"""
patch_ch3_stats_dedupe.py -- make ch3_stats.py refuse duplicated years.

Why: make_table_s2.py reported n = 85-87 per correlation for a 47-year
record. ch3_stats.py reads annual_params.csv and never checks that there is
one row per (sector, Year); if the file now carries more than one fit per
year (e.g. a `period` column with FULL and the separate-period fits), every
correlation in the script -- the 420-scan, S1, 3.3, the seven pairs -- is
computed on stacked rows, with roughly double the n and p-values that are
far too small. That is where "63 with p < 0.05, 16 with q < 0.05" came from;
the manuscript's "28, none" was from a run with one row per year.

This inserts, right after `ann = pd.read_csv(ANNUAL_CSV)`:
  - keep only period == "FULL" if a `period` column exists
  - assert exactly one row per (sector, Year), printing the offenders if not

Run from the figures folder:
    python patch_ch3_stats_dedupe.py          # shows the change, writes nothing
    python patch_ch3_stats_dedupe.py --apply  # edits ch3_stats.py in place (backup kept)
then
    python ch3_stats.py
    python make_table_s2.py
and check the console: "annual_params: N rows, 47 years, 6 sectors" must have
N = 282, and make_table_s2.py must report n per correlation of 47 (46/45 for
the indices that end in 2024).
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "ch3_stats.py")
APPLY = "--apply" in sys.argv

ANCHOR = "ann = pd.read_csv(ANNUAL_CSV)\n"
GUARD = '''ann = pd.read_csv(ANNUAL_CSV)
# ── one row per (sector, Year), or stop ──────────────────────────────────────
# annual_params.csv may carry more than one fit per year (a `period` column:
# FULL plus the separate-period fits). Every correlation below assumes one
# observation per year; stacked rows double n and make every p far too small.
import sys  # harmless if already imported
if "period" in ann.columns:
    _periods = sorted(ann["period"].astype(str).unique())
    if "FULL" not in _periods:
        sys.exit(f"annual_params has a period column but no FULL rows: {_periods}")
    ann = ann[ann["period"].astype(str) == "FULL"].copy()
    print(f"annual_params: kept period == FULL (file also had {_periods})")
_dup = ann.duplicated(["sector", "Year"], keep=False)
if _dup.any():
    print(ann.loc[_dup].sort_values(["sector", "Year"]).head(12).to_string())
    sys.exit(f"annual_params has {int(_dup.sum())} rows sharing a (sector, Year); "
             "fix the file (or the filter above) before any statistic is trusted")
'''

if not os.path.exists(TARGET):
    sys.exit(f"{TARGET} not found -- run this from the figures folder")
src = open(TARGET, encoding="utf-8").read()
if "one row per (sector, Year), or stop" in src:
    sys.exit("ch3_stats.py already has the guard; nothing to do")
if src.count(ANCHOR) != 1:
    sys.exit(f"expected exactly one line `{ANCHOR.strip()}` in ch3_stats.py, found {src.count(ANCHOR)}")

new = src.replace(ANCHOR, GUARD)
print("will insert after `%s`:\n" % ANCHOR.strip())
print("\n".join("    " + l for l in GUARD.splitlines()[1:]))
if APPLY:
    shutil.copy2(TARGET, TARGET + ".bak")
    open(TARGET, "w", encoding="utf-8").write(new)
    print(f"\nwritten; backup at {os.path.basename(TARGET)}.bak")
else:
    print("\n(dry run -- add --apply to write)")
