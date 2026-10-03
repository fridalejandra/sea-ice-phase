#!/usr/bin/env python3
"""
check_level_synchrony.py -- where does the ~4x rise in circumpolar LEVEL variance after 2007 come from?

check_level_variance.py: circumpolar level (annual-mean extent) variance is 3.9x larger after 2007
(4.2x ending 2023), but no sector's level variance is anywhere near that (only Ross is significant,
~2.3-2.8x). A sum can become ~4x more variable while its parts barely change only if the parts start
moving together. Three checks, each with the record ending in END (default 2025; pass 2023 to drop the
2024-2025 years whose day-to-day variance jumps in every sector at once):

  A. SYNCHRONY. var(sum of five sectors) = sum of the five variances + 2 x sum of covariances. Before vs
     after the split: how much of the circumpolar rise is the sectors' own variance and how much is
     covariance; the ten pairwise correlations of sector levels before and after (Fisher-z p for each
     change), their mean; Weddell--ABS (the dipole pair) shown first.
  B. MEAN SHIFT OR VARIABILITY? The post-2007 window holds both the record highs of 2012-2014 and the lows
     after 2016. If the circumpolar level variance after 2007 is mostly that high-then-low swing (a change
     in mean state), the variance AROUND each sub-period's own mean (2007-2015, 2016-END) is much smaller
     than the variance of the whole window. If it's year-to-year variability, the two are similar.
  C. The post-2016 timing--amplitude coupling (pooled over the five sectors, Pearson and Spearman) with
     leave-one-year-out, for this END -- same method as check_coupling_loo.py.

Level = mean of the daily extent over the calendar year, linearly detrended over 1979-END (as in
check_level_variance.py). Splits: 2007 (Hobbs et al. 2024) and 2016.

Reads   DAILY_CSV, ANNUAL_CSV (via ch3_config)
Writes  results/ch3/tables/t_level_synchrony_<END>.csv
Run     python checks/check_level_synchrony.py [END]      (from the figures folder)
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from ch3_config import DAILY_CSV, ANNUAL_CSV, TABLES_DIR, SECTORS_ONLY, BREAK_YEAR, YEAR_MIN

END = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
SPLITS = (2007, 2016)
SHORT = {"SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "ABS", "SIE_Ross": "Ross",
         "SIE_East_Antarctica": "EA", "SIE_King_Haakon": "KH"}
PH, AMP = "max_doy_raw_anom", "amplitude_raw_anom"


def lin_detrend(s):
    t = s.index.values.astype(float); m, b = np.polyfit(t, s.values, 1)
    return s - (m * t + b)


def fz(r): return np.arctanh(np.clip(r, -0.999999, 0.999999))


def zdiff_p(r1, n1, r2, n2):
    z = (fz(r2) - fz(r1)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3)); return 2 * stats.norm.sf(abs(z))


# ---- levels ----------------------------------------------------------------
d = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
if "period" in d.columns:
    d = d[d["period"] == "FULL"]
col = next(c for c in ("Extent", "extent", "SIE", "sie") if c in d.columns)
d["Y"] = d["Date"].dt.year
d = d[d["Y"].between(YEAR_MIN, END)]
L = pd.DataFrame({SHORT[s]: d[d.sector == s].groupby("Y")[col].mean() for s in SECTORS_ONLY}).dropna()
L = L.apply(lin_detrend)
L["sum5"] = L[list(SHORT.values())].sum(axis=1)
names = list(SHORT.values())
pairs = [("Weddell", "ABS")] + [p for p in itertools.combinations(names, 2) if set(p) != {"Weddell", "ABS"}]

rows = []
print(f"Sector LEVELS (annual-mean extent), detrended {YEAR_MIN}-{END}\n")
for split in SPLITS:
    pre, post = L[L.index < split], L[L.index >= split]
    v_pre, v_post = pre["sum5"].var(ddof=1), post["sum5"].var(ddof=1)
    own_pre, own_post = pre[names].var(ddof=1).sum(), post[names].var(ddof=1).sum()
    cov_pre, cov_post = v_pre - own_pre, v_post - own_post
    print(f"A. split {split} (n = {len(pre)} / {len(post)} years)")
    print(f"   var(sum of five): {v_pre:.4f} -> {v_post:.4f}  (ratio {v_post / v_pre:.2f})")
    print(f"     sectors' own variances: {own_pre:.4f} -> {own_post:.4f}  (ratio {own_post / own_pre:.2f})")
    print(f"     2 x covariances:        {cov_pre:+.4f} -> {cov_post:+.4f}")
    print(f"     share of the rise from covariance: {(cov_post - cov_pre) / (v_post - v_pre):.0%}"
          if v_post != v_pre else "     (no change)")
    rs = []
    for a, b in pairs:
        r1 = pre[a].corr(pre[b]); r2 = post[a].corr(post[b]); p = zdiff_p(r1, len(pre), r2, len(post))
        rs.append((r1, r2))
        rows.append(dict(end=END, split=split, pair=f"{a}-{b}", r_pre=r1, r_post=r2, p_change=p))
        flag = "*" if p < 0.05 else " "
        print(f"     {a + '-' + b:16s} r {r1:+.2f} -> {r2:+.2f}   p(change) = {p:.3f}{flag}")
    m1, m2 = np.mean([x for x, _ in rs]), np.mean([y for _, y in rs])
    print(f"     mean of the ten pairwise r: {m1:+.2f} -> {m2:+.2f}\n")
    rows.append(dict(end=END, split=split, pair="mean_pairwise", r_pre=m1, r_post=m2, p_change=np.nan))
    rows.append(dict(end=END, split=split, pair="var_sum5", r_pre=v_pre, r_post=v_post, p_change=np.nan))

# ---- B. mean shift vs variability in the post-2007 window --------------------
s = L["sum5"]
pre07 = s[s.index < 2007]
a, b = s[(s.index >= 2007) & (s.index < BREAK_YEAR)], s[s.index >= BREAK_YEAR]
whole = s[s.index >= 2007]
within = (((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()) / (len(a) + len(b) - 2)
print(f"B. Circumpolar (sum of five) level, post-2007 window: mean shift or more variability?")
print(f"   variance 1979-2006: {pre07.var(ddof=1):.4f}")
print(f"   variance 2007-{END}, whole window: {whole.var(ddof=1):.4f}  ({whole.var(ddof=1) / pre07.var(ddof=1):.2f}x)")
print(f"   variance around each sub-period's own mean (2007-{BREAK_YEAR - 1} | {BREAK_YEAR}-{END}): "
      f"{within:.4f}  ({within / pre07.var(ddof=1):.2f}x)")
print(f"   sub-period means (detrended): 2007-{BREAK_YEAR - 1} {a.mean():+.3f}, {BREAK_YEAR}-{END} {b.mean():+.3f}  "
      f"(Welch p = {stats.ttest_ind(a, b, equal_var=False).pvalue:.3f})")
share = 1 - within / whole.var(ddof=1)
print(f"   -> about {share:.0%} of the post-2007 level variance is the shift between the two sub-periods\n")
rows.append(dict(end=END, split=2007, pair="B_within_over_whole", r_pre=within, r_post=whole.var(ddof=1), p_change=np.nan))

# ---- C. post-2016 coupling with leave-one-year-out ----------------------------
def detrend(x):
    x = np.asarray(x, float); ok = np.isfinite(x)
    if ok.sum() < 3: return x
    t = np.arange(len(x), dtype=float); bb, aa = np.polyfit(t[ok], x[ok], 1)
    out = x.copy(); out[ok] = x[ok] - (aa + bb * t[ok]); return out

def pear(x, y):
    x, y = detrend(x), detrend(y); ok = np.isfinite(x) & np.isfinite(y); n = int(ok.sum())
    if n < 6: return np.nan, np.nan, n
    r, p = stats.pearsonr(x[ok], y[ok]); return float(r), float(p), n

def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); n = int(ok.sum())
    if n < 6: return np.nan, np.nan, n
    r, p = stats.spearmanr(x[ok], y[ok]); return float(r), float(p), n

def meta(rs, ns):
    rs, ns = np.asarray(rs, float), np.asarray(ns, float)
    ok = np.isfinite(rs) & (ns > 3); rs, ns = rs[ok], ns[ok]; k = len(rs)
    z = fz(rs); v = 1 / (ns - 3); w = 1 / v
    z_fe = np.sum(w * z) / np.sum(w); Q = np.sum(w * (z - z_fe) ** 2); df = k - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w); tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_re = 1 / (v + tau2); z_re = np.sum(w_re * z) / np.sum(w_re); se = np.sqrt(1 / np.sum(w_re))
    return np.tanh(z_re), 2 * stats.norm.sf(abs(z_re / se))

ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"] == "FULL"]
post = ann[ann.Year.between(BREAK_YEAR, END)]
yrs = sorted(post.Year.unique())

def pooled(df, fn):
    rs, ns = [], []
    for sec in SECTORS_ONLY:
        g = df[df.sector == sec].sort_values("Year"); r, _, n = fn(g[PH].values, g[AMP].values)
        rs.append(r); ns.append(n)
    return meta(rs, ns)

print(f"C. Post-{BREAK_YEAR} timing-amplitude coupling, five sectors, {yrs[0]}-{yrs[-1]} (n = {len(yrs)} years)")
for meth, fn in (("Pearson, detrended", pear), ("Spearman", spear)):
    r, p = pooled(post, fn)
    loo = [(y, *pooled(post[post.Year != y], fn)) for y in yrs]
    worst = max(loo, key=lambda t: t[2]); lost = sum(1 for t in loo if t[2] >= 0.05)
    print(f"   {meth:19s} r = {r:+.2f}  p = {p:.3f} | leave-one-out r {min(t[1] for t in loo):+.2f} to "
          f"{max(t[1] for t in loo):+.2f}, worst p = {worst[2]:.3f} (dropping {worst[0]}), {lost}/{len(loo)} lose p<0.05")
    rows.append(dict(end=END, split=BREAK_YEAR, pair=f"C_coupling_{meth.split(',')[0]}", r_pre=r, r_post=worst[2], p_change=p))

out = os.path.join(TABLES_DIR, f"t_level_synchrony_{END}.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nwrote {out}")
