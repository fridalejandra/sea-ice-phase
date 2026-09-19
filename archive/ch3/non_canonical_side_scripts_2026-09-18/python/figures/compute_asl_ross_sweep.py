#!/usr/bin/env python3
"""
compute_asl_ross_sweep.py -- standalone extraction of the "Ross ~ ASL detail"
block from ch3_stats.py (the part that builds t36_ross_asl_detail.csv and
feeds Fig 9). Pulled out on its own so you can run and inspect just this
piece without the full ~40-file pipeline run, and so it's easy to tweak
(e.g. add/remove candidate split years) without touching the canonical
script.

Run this AFTER 01_fit_apac.R and compute_atmospheric_correlations.py have
produced annual_params.csv and master_index_detrended.csv -- it reads those
via ch3_config, same as the rest of the pipeline, so the numbers will match
ch3_stats.py's own t36_ross_asl_detail.csv exactly (same detrend, same
correlation function, same SPLIT_YEAR).

-------------------------------------------------------------------------
"How did we know it was 2001?" -- short answer: we didn't derive it, we
CHOSE it, on purpose, before running any test. ch3_config.SPLIT_YEAR = 2001
is documented there as "half-record split" -- the record runs 1979-2023,
45 years, and 2001 is just the midpoint (1979 + 44/2). It is not a fitted
breakpoint, not the year that maximizes significance, and not chosen after
looking at the ASL-Ross result. Per S2.3 of the draft: "The record is split
at 2016, following the abrupt decline documented by Fogt et al. (2022)...
and separately at 2001, which halves it." Both boundaries are pre-specified
for reasons that have nothing to do with this particular result -- 2016 for
an external, literature-based reason, 2001 for a purely structural one (cut
the record in half).

That's exactly why the split-year sweep in this script matters: 2001 is an
arbitrary-but-principled choice, not a cherry-picked one, and the sweep
(testing 1996, 2001, 2006, 2011, 2016 as alternative boundaries) is what
demonstrates the break isn't an artifact of that specific choice. It's
significant everywhere from 1996 through 2011, and stops being significant
only if you push the boundary all the way out to 2016 -- so 2001 sitting in
the middle of that range is what makes it a reasonable single number to
quote, not what makes the underlying break real. The break's evidence is
the whole 1996-2011 stretch, not the number 2001 specifically.
-------------------------------------------------------------------------

Outputs
    results/ch3/tables/t36_ross_asl_detail.csv   (same schema as ch3_stats.py's)
    console: the headline pre/post-2001 numbers, the five-boundary sweep,
             and the leave-one-year-out worst case

Paste the console output back if anything looks different from the draft's
quoted numbers (r_pre=+0.71, r_post=-0.21, shift p<0.0001; sweep p at
1996/2001/2006/2011/2016 = 0.008, <0.0001, <0.0001, 0.026, 0.57).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import (
    ANNUAL_CSV, INDEX_CSV, INDEX_DIR, INDEX_FILES, TABLES_DIR,
    YEAR_MIN, YEAR_MAX, SPLIT_YEAR, SEASONS,
)

AMP = "amplitude_raw_anom"
SWEEP_YEARS = (1996, 2001, 2006, 2011, 2016)  # same five candidates as ch3_stats.py


def detrend(x):
    """Linear detrend against sequence position (not calendar year) --
    matches ch3_stats.py's own detrend() exactly, including using arange()
    rather than Year, so results agree to the last decimal even if there
    were ever a gap year."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return x
    t = np.arange(len(x), dtype=float)
    b, a = np.polyfit(t[ok], x[ok], 1)
    out = x.copy()
    out[ok] = x[ok] - (a + b * t[ok])
    return out


def pear(x, y, dt=True):
    """Pearson r,p,n with optional pre-detrending of both series -- the
    pipeline's convention throughout, since the atmospheric indices are
    already detrended and the sea-ice scalars need to match that footing."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if dt:
        x, y = detrend(x), detrend(y)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 6:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x[ok], y[ok])
    return float(r), float(p), n


def zshift(r1, n1, r2, n2):
    """Fisher z-test for whether two independent correlations differ."""
    if not (np.isfinite(r1) and np.isfinite(r2)) or n1 < 4 or n2 < 4:
        return np.nan, np.nan
    fz = lambda r: 0.5 * np.log((1 + r) / (1 - r))
    z = (fz(r1) - fz(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(z), float(2 * stats.norm.sf(abs(z)))


print("compute_asl_ross_sweep — Ross amplitude ~ ASL, season test + split-year sweep + LOO")

ann = pd.read_csv(ANNUAL_CSV)
ann = ann[ann.Year.between(YEAR_MIN, YEAR_MAX)].copy()
idx = pd.read_csv(INDEX_CSV)
ai = ann.merge(idx, "left", "Year")

ross = ai[ai.sector == "SIE_Ross"].sort_values("Year")
print(f"  Ross rows: {len(ross)}  years {int(ross.Year.min())}-{int(ross.Year.max())}")

detail = []

# ---- (a) by season, fixed split at SPLIT_YEAR ------------------------------
print(f"\n{'season':>8s}  {'r_pre':>7s}  {'r_post':>7s}  {'shift p':>9s}")
for s in SEASONS:
    ic = f"ASL_{s}"
    if ic not in ross.columns:
        print(f"  (skip {ic}: not in index table)")
        continue
    a1, a2 = ross[ross.Year < SPLIT_YEAR], ross[ross.Year >= SPLIT_YEAR]
    r1, _, n1 = pear(a1[ic], a1[AMP])
    r2, _, n2 = pear(a2[ic], a2[AMP])
    _, p = zshift(r1, n1, r2, n2)
    detail.append(dict(test="season", key=s, r_pre=r1, r_post=r2, p_shift=p))
    print(f"{s:>8s}  {r1:+7.2f}  {r2:+7.2f}  {p:9.2g}")

# ---- (b) split-year sweep, ASL_annual only ---------------------------------
print(f"\nsplit-year sweep ({', '.join(str(y) for y in SWEEP_YEARS)}), ASL_annual vs Ross amplitude:")
print(f"{'boundary':>8s}  {'r_pre':>7s}  {'r_post':>7s}  {'shift p':>9s}")
for yr in SWEEP_YEARS:
    a1, a2 = ross[ross.Year < yr], ross[ross.Year >= yr]
    r1, _, n1 = pear(a1["ASL_annual"], a1[AMP])
    r2, _, n2 = pear(a2["ASL_annual"], a2[AMP])
    _, p = zshift(r1, n1, r2, n2)
    detail.append(dict(test="split_sweep", key=yr, r_pre=r1, r_post=r2, p_shift=p))
    print(f"{yr:>8d}  {r1:+7.2f}  {r2:+7.2f}  {p:9.2g}")

# ---- (c) leave-one-year-out on the fixed SPLIT_YEAR test -------------------
ps = []
for y in ross.Year:
    b = ross[ross.Year != y]
    a1, a2 = b[b.Year < SPLIT_YEAR], b[b.Year >= SPLIT_YEAR]
    r1, _, n1 = pear(a1["ASL_annual"], a1[AMP])
    r2, _, n2 = pear(a2["ASL_annual"], a2[AMP])
    ps.append(zshift(r1, n1, r2, n2)[1])
worst_p = max(ps)
detail.append(dict(test="loo_shift", key="worst p", r_pre=np.nan, r_post=np.nan, p_shift=worst_p))
n_lost = sum(p > 0.05 for p in ps)
print(f"\nleave-one-year-out on the {SPLIT_YEAR} split: worst shift p = {worst_p:.4g}"
      f"   ({n_lost}/{len(ps)} single-year drops lose significance)")

# ---- (d) ASL position (longitude/latitude/pressure), if the raw file is present
asl_path = os.path.join(INDEX_DIR, INDEX_FILES.get("ASL", ""))
if os.path.exists(asl_path):
    asl = pd.read_csv(asl_path, comment="#")
    asl["time"] = pd.to_datetime(asl["time"])
    asl["year"], asl["month"] = asl.time.dt.year, asl.time.dt.month

    def seas_mean(col, months=None, jan_prev=False):
        d = asl if months is None else asl[asl.month.isin(months)].copy()
        if jan_prev:
            d.loc[d.month == 1, "year"] -= 1
        return d.groupby("year")[col].mean()

    yrs = ross.Year.values
    print("\nASL position vs Ross amplitude, pre/post split:")
    for col in ("lon", "lat", "ActCenPres", "RelCenPres"):
        for s, months, jp in (("annual", None, False), ("SON", [9, 10, 11], False),
                               ("RET", [10, 11, 12, 1], True)):
            if col not in asl.columns:
                continue
            x = seas_mean(col, months, jp).reindex(yrs).values
            y = ross[AMP].values
            m1, m2 = yrs < SPLIT_YEAR, yrs >= SPLIT_YEAR
            r1, _, n1 = pear(x[m1], y[m1])
            r2, _, n2 = pear(x[m2], y[m2])
            _, p = zshift(r1, n1, r2, n2)
            detail.append(dict(test=f"asl_property:{col}", key=s, r_pre=r1, r_post=r2, p_shift=p))
            if col == "lon" and s == "annual":
                print(f"  longitude, annual: {r1:+.2f} -> {r2:+.2f}  (shift p={p:.2g}, "
                      f"mean {np.nanmean(x[m1]):.1f}E -> {np.nanmean(x[m2]):.1f}E)")
else:
    print(f"\n(skipping ASL position block -- {asl_path} not found; "
          f"that's fine, it isn't needed for the season/sweep/LOO numbers above)")

out = pd.DataFrame(detail)
os.makedirs(TABLES_DIR, exist_ok=True)
out_path = os.path.join(TABLES_DIR, "t36_ross_asl_detail.csv")
out.to_csv(out_path, index=False)
print(f"\nwrote {out_path}  ({len(out)} rows)")
