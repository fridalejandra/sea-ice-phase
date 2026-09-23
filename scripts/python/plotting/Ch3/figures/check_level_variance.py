#!/usr/bin/env python3
"""
check_level_variance.py -- is the rise in extent variability carried by the LEVEL of the cycle?

Hobbs et al. (2024) and Abram et al. (2025) report that Antarctic extent became more variable
(summer SD doubled, 1979-2006 vs 2007-2022). The chapter finds the AMPLITUDE (max - min) became
LESS variable after 2016. If both hold, the extra variability must sit in the level of the cycle.

For each sector and year, from the daily record:
  min      smallest daily extent, Jan-Mar           (summer minimum)
  max      largest daily extent, Jun-Nov            (winter maximum)
  mean     mean extent over the calendar year       (the LEVEL of the cycle)
  amp      max - min                                (the SIZE of the cycle)
Each series is linearly detrended over 1979-2025 (as in the chapter), then the variance after a
split year is compared with the variance before it (ratio > 1 = more variable after), with a
two-sided F-test p. Splits: 2007 (Hobbs et al. 2024) and 2016.

How to read it:
  mean (and min) ratio > 1 while amp ratio < 1  -> the 4.3 sentence holds as written.
  mean ratio ~ 1                                -> soften: the rise is not in the level either.
Usage: python check_level_variance.py [LAST_YEAR]   (default 2025; try 2023 to drop the noisy years)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import DAILY_CSV, SECTORS

Y1 = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
d = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
if "period" in d.columns:
    d = d[d["period"] == "FULL"]
col = next(c for c in ("Extent", "extent", "SIE", "sie") if c in d.columns)
d["Y"], d["m"] = d["Date"].dt.year, d["Date"].dt.month
d = d[d["Y"].between(1979, Y1)]


def detrend(s):
    t = s.index.values.astype(float)
    m, b = np.polyfit(t, s.values, 1)
    return s - (m * t + b)


def vratio(s, split):
    a, b = s[s.index < split].dropna(), s[s.index >= split].dropna()
    r = b.var(ddof=1) / a.var(ddof=1)
    f = stats.f.cdf(r, len(b) - 1, len(a) - 1)
    return r, 2 * min(f, 1 - f)


rows = []
for sec in SECTORS:
    g = d[d["sector"] == sec]
    t = pd.DataFrame({
        "min": g[g.m.between(1, 3)].groupby("Y")[col].min(),
        "max": g[g.m.between(6, 11)].groupby("Y")[col].max(),
        "mean": g.groupby("Y")[col].mean(),
    })
    t["amp"] = t["max"] - t["min"]
    for q in ("mean", "min", "max", "amp"):
        s = detrend(t[q].dropna())
        for split in (2007, 2016):
            r, p = vratio(s, split)
            rows.append(dict(sector=sec.replace("SIE_", ""), quantity=q, split=split, ratio=r, p=p))

out = pd.DataFrame(rows)
tab = out.pivot_table(index=["sector", "quantity"], columns="split", values=["ratio", "p"])
tab = tab.reindex(columns=[("ratio", 2007), ("p", 2007), ("ratio", 2016), ("p", 2016)])
order = {"mean": 0, "min": 1, "max": 2, "amp": 3}
tab = tab.sort_index(key=lambda ix: ix.map(order) if ix.name == "quantity" else ix)
print(f"Variance after split / before split, detrended 1979-{Y1} (ratio > 1 = more variable after)\n")
print(tab.round(2).to_string())
print("\nmean = level of the cycle, amp = size of the cycle. Check: mean/min ratios > 1 while amp < 1?")
