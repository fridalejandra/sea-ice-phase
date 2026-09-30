#!/usr/bin/env python3
"""
check_abs_growth_season.py -- every number in Sect. 3.4.2, in one place.

Amundsen-Bellingshausen growth-season length (day of max - day of min) against
amplitude, linearly detrended within each period, as fig_08 and ch3_stats.py
3.3c do. Prints:
  1. r over 1979-2015 and 2016-YEAR_MAX, its p, and the Fisher-z p of the change
  2. leave-one-year-out over the post-2016 years: r and p with each year removed
  3. the change p for every boundary 2012-2017 ("holds for any boundary ...")
  4. the post-2016 years ordered by season length (the "ten years order" sentence)
  5. the index scan: growth-season length against all 35 index-season columns
     over the full record, with Benjamini-Hochberg q ("none does")

Reads annual_params.csv (period == FULL) and master_index_detrended.csv via
ch3_config. Run from the figures directory.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import ANNUAL_CSV, INDEX_CSV, BREAK_YEAR

SECTOR = "SIE_Amundsen_Bellingshausen"


def detrend(x):
    x = np.asarray(x, float); t = np.arange(len(x), dtype=float)
    b, a = np.polyfit(t, x, 1)
    return x - (a + b * t)


def pear_dt(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return stats.pearsonr(detrend(x[ok]), detrend(y[ok])) + (int(ok.sum()),)


def zshift(r1, n1, r2, n2):
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(2 * stats.norm.sf(abs(z)))


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"].astype(str) == "FULL"]
a = ann[ann["sector"] == SECTOR].sort_values("Year").copy()
if a.duplicated("Year").any():
    sys.exit("duplicate years after the FULL filter")
a["len"] = a["max_doy_raw"] - a["min_doy_raw"]
AMP = next(c for c in ("amplitude_raw_yr", "amplitude_raw", "amplitude") if c in a.columns)
Y0, Y1 = int(a.Year.min()), int(a.Year.max())
print(f"{SECTOR}: {Y0}-{Y1}, amplitude column {AMP!r}\n")

# 1
pre, post = a[a.Year < BREAK_YEAR], a[a.Year >= BREAK_YEAR]
r1, p1, n1 = pear_dt(pre["len"], pre[AMP])
r2, p2, n2 = pear_dt(post["len"], post[AMP])
print(f"1. {Y0}-{BREAK_YEAR-1}: r = {r1:+.2f} (n={n1}, p={p1:.2f})")
print(f"   {BREAK_YEAR}-{Y1}: r = {r2:+.2f} (n={n2}, p={p2:.3f})")
print(f"   change: p = {zshift(r1, n1, r2, n2):.3f}")

# 2
print(f"\n2. leave-one-year-out over {BREAK_YEAR}-{Y1}")
worst = (0, None)
for y in post.Year:
    q = post[post.Year != y]
    r, p, n = pear_dt(q["len"], q[AMP])
    ps = zshift(r1, n1, r, n)
    print(f"   without {int(y)}: r = {r:+.2f} (p={p:.3f})   change p = {ps:.3f}")
    worst = max(worst, (ps, int(y)))
print(f"   worst change p: {worst[0]:.3f} (dropping {worst[1]})")

# 3
print("\n3. boundary sensitivity (change p)")
for B in range(2012, 2018):
    q1, q2 = a[a.Year < B], a[a.Year >= B]
    ra, _, na = pear_dt(q1["len"], q1[AMP]); rb, _, nb = pear_dt(q2["len"], q2[AMP])
    print(f"   boundary {B}: {ra:+.2f} -> {rb:+.2f} (n_post={nb})   change p = {zshift(ra, na, rb, nb):.3f}")

# 4
print(f"\n4. {BREAK_YEAR}-{Y1} years ordered by season length")
print(post.sort_values("len")[["Year", "min_doy_raw", "max_doy_raw", "len", AMP]].to_string(index=False))
rho, prho = stats.spearmanr(post["len"], post[AMP])
print(f"   Spearman rank agreement of length and amplitude: {rho:+.2f} (p={prho:.3f})")

# 5
idx = pd.read_csv(INDEX_CSV)
ai = a.merge(idx, on="Year", how="left")
icols = [c for c in idx.columns if c != "Year" and pd.api.types.is_numeric_dtype(idx[c])]
rows = []
for c in icols:
    r, p, n = pear_dt(ai[c], ai["len"])
    rows.append(dict(index=c, r=r, p=p, n=n))
g = pd.DataFrame(rows).sort_values("p"); g["q"] = bh(g.p.values)
print(f"\n5. growth-season length vs {len(icols)} index columns, full record (detrended; BH q)")
print(g.head(8).round(3).to_string(index=False))
print(f"   {int((g.p < 0.05).sum())} of {len(g)} at p < 0.05, {int((g.p < 0.1).sum())} at p < 0.1; "
      f"index record ends {int(idx.Year.max())}")
for fam in ("SAM", "Nino", "ASL"):
    sub = g[g["index"].str.contains(fam, case=False)]
    if len(sub):
        print(f"   best {fam}: {sub.iloc[0]['index']} r = {sub.iloc[0].r:+.2f}, p = {sub.iloc[0].p:.3f}")
