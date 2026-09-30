#!/usr/bin/env python3
"""
check_level_shift.py -- does El Nino lift the whole Weddell cycle (max AND min up)?

If Nino3.4 correlates positively with both the winter maximum and the summer
minimum, ENSO raises the level of the cycle, which explains why the extent
anomaly follows ENSO while the observed amplitude (max - min) does not.

Computed straight from daily extent, so it does not depend on annual_params
column names:
  max           largest daily extent, Jun-Nov of year Y
  min before    smallest daily extent, Jan-Mar of year Y   (the minimum the cycle starts from)
  min after     smallest daily extent, Jan-Mar of year Y+1 (the minimum it ends at)
  amplitude     max - min after
  mean          mean extent over the year
All detrended, 1979-2023, against Nino3.4 annual, Mar-Aug and Oct-Jan.

Usage: python check_level_shift.py [SECTOR] [INDEX]   (default SIE_Weddell Nino34)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import DAILY_CSV, INDEX_CSV

SECTOR = sys.argv[1] if len(sys.argv) > 1 else "SIE_Weddell"
INDEX = sys.argv[2] if len(sys.argv) > 2 else "Nino34"

d = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
if "period" in d.columns:
    d = d[d["period"] == "FULL"]
d = d[d["sector"] == SECTOR].copy()
if d.empty:
    sys.exit(f"no rows for {SECTOR}")
col = next((c for c in ("Extent", "extent", "SIE", "sie", "extent_obs", "obs") if c in d.columns), None)
if col is None:
    sys.exit(f"no extent column found; columns are {list(d.columns)}")
d["Y"], d["m"] = d["Date"].dt.year, d["Date"].dt.month

mx = d[d.m.between(6, 11)].groupby("Y")[col].max()
mn = d[d.m.between(1, 3)].groupby("Y")[col].min()
mean = d.groupby("Y")[col].mean()
t = pd.DataFrame({"max": mx, "min before": mn, "min after": mn.shift(-1).reindex(mx.index), "mean": mean})
t["amplitude"] = t["max"] - t["min after"]
t = t.loc[1979:2023]

idx = pd.read_csv(INDEX_CSV).set_index("Year")


def detrend(x):
    tt = np.arange(len(x), dtype=float); m, b = np.polyfit(tt, x, 1); return x - (m * tt + b)


print(f"{SECTOR} ({col}) vs {INDEX}, detrended, 1979-2023\n")
cols = ["max", "min before", "min after", "amplitude", "mean"]
print(f"{'index':16s}" + "".join(f"{c:>13s}" for c in cols))
for s in ("annual", "ADV", "RET"):
    name = f"{INDEX}_{s}"
    line = f"{name:16s}"
    for c in cols:
        j = t[[c]].join(idx[name], how="inner").dropna()
        r, p = stats.pearsonr(detrend(j[name].values), detrend(j[c].values))
        line += f"{r:+12.2f}{'*' if p < 0.05 else ' '}"
    print(line)
print("\n* p < 0.05.  Level shift = max and a minimum both positive, amplitude near zero.")