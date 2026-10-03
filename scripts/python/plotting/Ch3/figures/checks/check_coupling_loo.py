#!/usr/bin/env python3
"""
check_coupling_loo.py -- is the post-2016 coupling between the cycle's timing
(day of maximum) and its size (amplitude) robust to leaving out any one year?

ch3_stats.py tests this for the Pearson version only: pooled r = +0.34 after
2016 (p = 0.024), but leaving out 2022 gives p = 0.16, and 4 of the 10 drops
lose p < 0.05. The Spearman version is stronger (+0.40, p = 0.006) but its
leave-one-year-out was never run. This runs both, the same way, and also
with and without the circumpolar total in the pool -- ch3_stats.py pools all
six series, but the circumpolar total is the sum of the five sectors, so
pooling it with them counts the same years twice (check_coupling_shift.py
already leaves it out for that reason).

Pooling is DerSimonian-Laird random effects on Fisher-z, copied verbatim from
ch3_stats.py (meta, pear, spear), so the "six series, Pearson" row reproduces
the ledger's +0.34 / worst p 0.156 exactly.

Reads   ANNUAL_CSV (via ch3_config)
Writes  results/ch3/tables/t_coupling_loo.csv
Run     python checks/check_coupling_loo.py      (from the figures folder)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from ch3_config import ANNUAL_CSV, TABLES_DIR, SECTORS, SECTORS_ONLY, BREAK_YEAR, YEAR_MIN, YEAR_MAX

PH, AMP = "max_doy_raw_anom", "amplitude_raw_anom"


# ---- helpers, verbatim from ch3_stats.py -----------------------------------
def detrend(x):
    x = np.asarray(x, float); ok = np.isfinite(x)
    if ok.sum() < 3: return x
    t = np.arange(len(x), dtype=float); b, a = np.polyfit(t[ok], x[ok], 1)
    out = x.copy(); out[ok] = x[ok] - (a + b * t[ok]); return out

def fz(r): return 0.5 * np.log((1 + r) / (1 - r))
def fz_inv(z): return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def pear(x, y, dt=True):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if dt: x, y = detrend(x), detrend(y)
    ok = np.isfinite(x) & np.isfinite(y); n = int(ok.sum())
    if n < 6: return np.nan, np.nan, n
    r, p = stats.pearsonr(x[ok], y[ok]); return float(r), float(p), n

def spear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); n = int(ok.sum())
    if n < 6: return np.nan, np.nan, n
    r, p = stats.spearmanr(x[ok], y[ok]); return float(r), float(p), n

def meta(rs, ns):
    rs, ns = np.asarray(rs, float), np.asarray(ns, float)
    ok = np.isfinite(rs) & (ns > 3); rs, ns = rs[ok], ns[ok]; k = len(rs)
    if k < 2: return None
    z = fz(rs); v = 1 / (ns - 3); w = 1 / v
    z_fe = np.sum(w * z) / np.sum(w); Q = np.sum(w * (z - z_fe) ** 2); df = k - 1
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w); tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_re = 1 / (v + tau2); z_re = np.sum(w_re * z) / np.sum(w_re); se = np.sqrt(1 / np.sum(w_re))
    return dict(k=k, r_re=fz_inv(z_re), lo=fz_inv(z_re - 1.96 * se), hi=fz_inv(z_re + 1.96 * se),
                se_z=se, p=2 * stats.norm.sf(abs(z_re / se)), I2=I2)


ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"] == "FULL"]
ann = ann[ann.Year.between(YEAR_MIN, YEAR_MAX)]
post = ann[ann.Year >= BREAK_YEAR]
YEARS = sorted(post.Year.unique())


def pooled(df, sectors, fn):
    rs, ns = [], []
    for sec in sectors:
        a = df[df.sector == sec].sort_values("Year")
        r, _, n = fn(a[PH].values, a[AMP].values); rs.append(r); ns.append(n)
    return meta(rs, ns), rs


rows = []
print(f"Post-{BREAK_YEAR} pooled r(day of maximum, amplitude), {YEARS[0]}-{YEARS[-1]} (n = {len(YEARS)} years per series)\n")
for pool_name, sectors in (("six series (as ch3_stats.py)", SECTORS), ("five sectors (no circumpolar)", SECTORS_ONLY)):
    for meth, fn in (("Pearson, detrended", pear), ("Spearman", spear)):
        m, rs = pooled(post, sectors, fn)
        loo = []
        for y in YEARS:
            mm, _ = pooled(post[post.Year != y], sectors, fn)
            loo.append((y, mm["r_re"], mm["p"]))
            rows.append(dict(pool=pool_name, method=meth, dropped=y, r=mm["r_re"], p=mm["p"]))
        rows.append(dict(pool=pool_name, method=meth, dropped="none", r=m["r_re"], p=m["p"]))
        worst = max(loo, key=lambda t: t[2])
        n_lost = sum(1 for _, _, p in loo if p >= 0.05)
        verdict = "ROBUST" if worst[2] < 0.05 else ("borderline" if worst[2] < 0.10 else "not robust")
        print(f"  {pool_name:30s} {meth:19s} r = {m['r_re']:+.2f}  p = {m['p']:.3f}  I2 = {m['I2']:.0f}% | "
              f"leave-one-out: r {min(t[1] for t in loo):+.2f} to {max(t[1] for t in loo):+.2f}, "
              f"worst p = {worst[2]:.3f} (dropping {worst[0]}), {n_lost}/{len(loo)} drops lose p<0.05  -> {verdict}")
    print()

print("Per-year detail (p when that year is left out), Spearman, five sectors:")
d = pd.DataFrame(rows)
sub = d[(d.pool.str.startswith("five")) & (d.method == "Spearman") & (d.dropped != "none")]
print("  " + "  ".join(f"{int(r.dropped)}:{r.p:.3f}" for r in sub.itertuples()))
out = os.path.join(TABLES_DIR, "t_coupling_loo.csv")
d.to_csv(out, index=False)
print(f"\nwrote {out}")
