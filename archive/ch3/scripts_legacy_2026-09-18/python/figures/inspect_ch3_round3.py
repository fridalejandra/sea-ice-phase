#!/usr/bin/env python3
"""
inspect_ch3_round3.py -- last round before writing the volatility figure
and the component-comparison heatmap.

  1. Full 3.4c ledger rows (value/n/p/extra), the actual day-to-day
     volatility numbers.
  2. Whatever season-window definition ch3_config exposes (so the new
     residual-volatility-per-year computation uses the exact same season
     boundaries as the rest of the pipeline, not a guessed one).
  3. A peek at ch3_config's full attribute list, in case there's already
     a season-filter helper (e.g. a function or dict) worth reusing
     directly instead of reimplementing.

Read-only. Paste full output back.
"""
import os
import sys
import inspect
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.max_rows", 60)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR
import ch3_config as cfg

def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

# ---------------------------------------------------------------- 3.4c full rows
section("ch3_numbers.csv -- full 3.4c rows (value, n, p, extra)")
path = os.path.join(TABLES_DIR, "ch3_numbers.csv")
n = pd.read_csv(path)
sub = n[n["section"] == "3.4c"]
print(sub.to_string(index=False))

# ---------------------------------------------------------------- ch3_config season info
section("ch3_config -- all module-level attributes (name + type + short repr)")
for name in dir(cfg):
    if name.startswith("_"):
        continue
    val = getattr(cfg, name)
    if inspect.ismodule(val):
        continue
    r = repr(val)
    if len(r) > 300:
        r = r[:300] + " ...(truncated)"
    print(f"  {name:20s} {type(val).__name__:10s} {r}")

# ---------------------------------------------------------------- season-specific attrs
section("ch3_config -- anything with ADV/RET/DOY/SEASON/MONTH in the name")
for name in dir(cfg):
    if name.startswith("_"):
        continue
    if any(k in name.upper() for k in ["ADV", "RET", "DOY", "SEASON", "MONTH", "WINDOW"]):
        val = getattr(cfg, name)
        print(f"  {name} = {val!r}")

print("\ndone.")
