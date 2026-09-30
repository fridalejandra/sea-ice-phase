#!/usr/bin/env python3
"""
check_amplitude_attribution.py -- is the post-2016 rise in the amplitude
component (Fig. 3 / Sect. 3.1, paragraph 105) actually a level effect?

Paragraph 105 already argues this in words: "the amplitude contributes more
since 2016 even though it varies less from year to year ... because the cycle
has been smaller than its long-term average in most years since 2016: a
component that is consistently negative has a large average size even when
it varies little." That is a level-shift story told using the word
"amplitude." This script checks it against the real numbers instead of
trusting the prose.

It does NOT touch the "amplitude" used throughout Sect. 3.2-3.5 and the
Discussion (observed max - min) -- that one is clean by construction and is
not what this checks. This is specifically about `amplitude_component`, the
one place the word "amplitude" means something else: a term in the four-part
sum (trend + amplitude_component + phase_component + raw_anomaly = extent
anomaly).

What it prints, for the circumpolar total (or --sector):
  1. A reproduction of paragraph 105's own numbers: mean |trend_component|,
     mean |amplitude_component|, mean |phase_component|, mean |raw_anomaly|,
     one cycle-average per year, split at 1979-2015 vs 2016-2024 (as Fig. 3
     is captioned) and also vs 2016-2025 (as Sect. 3.4 is split), since the
     chapter uses both endpoints in different places -- flagged if they
     disagree.
  2. Whether amplitude_component's MEAN became more negative after 2016
     (paragraph 105's claim) and whether its VARIANCE fell (Sect. 3.4.1's
     claim, but checked here for the component, not the observed range).
  3. How much amplitude_component actually tracks level (annual-mean extent,
     or min_extent) versus how much it tracks observed amplitude (max-min),
     by simple correlation and by a two-predictor regression. This is the
     number that says whether "the amplitude" in paragraph 105 is mostly
     size, mostly level, or a real mix.

Usage:
    python check_amplitude_attribution.py [path/to/daily_fitted.csv] [--sector NAME]
"""
import os
import sys

import numpy as np
import pandas as pd

args = sys.argv[1:]
sector_arg = None
if "--sector" in args:
    i = args.index("--sector")
    sector_arg = args[i + 1]
    args = args[:i] + args[i + 2:]

path = next((a for a in args if a.lower().endswith(".csv")), None)
if path is None:
    for cand in [
        os.environ.get("SEAICE_DAILY_FITTED"),
        os.path.expanduser("~/Research/repos/sea-ice-phase/data/ch3/daily_fitted.csv"),
        "daily_fitted.csv",
    ]:
        if cand and os.path.exists(cand):
            path = cand
            break
if path is None or not os.path.exists(path):
    sys.exit(
        "Could not find daily_fitted.csv. Pass it explicitly:\n"
        "    python check_amplitude_attribution.py /path/to/daily_fitted.csv"
    )

pd.set_option("display.width", 200)
d = pd.read_csv(path)
if "period" in d.columns:
    d = d[d["period"] == "FULL"].copy()

need = {"sector", "Year", "trend_component", "amplitude_component",
        "phase_component", "raw_anomaly"}
missing = need - set(d.columns)
if missing:
    sys.exit(f"Missing expected columns: {sorted(missing)}\n"
              f"Columns present: {list(d.columns)}")

print(f"reading {path}  ({len(d):,} rows)")
sectors = sorted(d["sector"].unique())
print(f"sectors found: {sectors}\n")

if sector_arg:
    sector = sector_arg
else:
    guess = [s for s in sectors if "circump" in str(s).lower() or "total" in str(s).lower()]
    if guess:
        sector = guess[0]
        print(f"(no --sector given; auto-detected the circumpolar sector: '{sector}')\n")
    else:
        sector = sectors[0]
        print(f"(no --sector given, and no sector looked circumpolar; using '{sector}' -- "
              f"pass --sector NAME to pick the right one from {sectors})\n")

ds = d[d["sector"] == sector].copy()

# level and amplitude proxies, per cycle-year, alongside the four components
agg = {"trend_component": "mean", "amplitude_component": "mean",
       "phase_component": "mean", "raw_anomaly": "mean"}
if "Extent" in ds.columns:
    agg["Extent"] = ["mean", "max", "min"]
if "min_extent" in ds.columns:
    agg["min_extent"] = "first"
if "amplitude" in ds.columns:
    agg["amplitude"] = "first"

g = ds.groupby("Year").agg(agg)
g.columns = ["_".join(c) if isinstance(c, tuple) and c[1] else c[0] if isinstance(c, tuple) else c
             for c in g.columns]
g = g.rename(columns={c: c.replace("_mean", "") for c in g.columns
                       if c.endswith("_mean") and not c.startswith("Extent")})
g = g.sort_index()

level_col = None
for cand in ["min_extent_first", "min_extent"]:
    if cand in g.columns:
        level_col = cand
        break
if level_col is None and "Extent_mean" in g.columns:
    level_col = "Extent_mean"

amp_obs_col = None
for cand in ["amplitude_first", "amplitude"]:
    if cand in g.columns:
        amp_obs_col = cand
        break
if amp_obs_col is None and {"Extent_max", "Extent_min"} <= set(g.columns):
    g["amplitude_obs_derived"] = g["Extent_max"] - g["Extent_min"]
    amp_obs_col = "amplitude_obs_derived"

print(f"level proxy column: {level_col}")
print(f"observed-amplitude proxy column: {amp_obs_col}\n")

def period_table(g, post_start, post_end, label):
    pre = g[g.index <= 2015]
    post = g[(g.index >= post_start) & (g.index <= post_end)]
    rows = []
    for comp in ["trend_component", "amplitude_component", "phase_component", "raw_anomaly"]:
        if comp not in g.columns:
            continue
        rows.append(dict(component=comp,
                          pre_mean_abs=pre[comp].abs().mean(),
                          post_mean_abs=post[comp].abs().mean(),
                          pre_mean_signed=pre[comp].mean(),
                          post_mean_signed=post[comp].mean(),
                          pre_var=pre[comp].var(ddof=1),
                          post_var=post[comp].var(ddof=1)))
    t = pd.DataFrame(rows)
    print(f"--- {label}: pre = 1979-2015 (n={len(pre)}), post = {post_start}-{post_end} (n={len(post)}) ---")
    print(t.round(4).to_string(index=False))
    print()
    return pre, post

print("1. REPRODUCING PARAGRAPH 105'S TABLE\n")
pre24, post24 = period_table(g, 2016, 2024, "as Fig. 3 is captioned (2016-2024)")
pre25, post25 = period_table(g, 2016, 2025, "as Sect. 3.4 is split (2016-2025)")

print("2. IS THE AMPLITUDE COMPONENT 'CONSISTENTLY NEGATIVE' SINCE 2016, WITH LOWER VARIANCE?\n")
for label, (pre, post) in [("2016-2024", (pre24, post24)), ("2016-2025", (pre25, post25))]:
    ac_pre, ac_post = pre["amplitude_component"], post["amplitude_component"]
    print(f"  [{label}] mean amplitude_component: {ac_pre.mean():+.4f} (pre) -> {ac_post.mean():+.4f} (post)")
    print(f"  [{label}] variance ratio (post/pre): {ac_post.var(ddof=1)/ac_pre.var(ddof=1):.2f}")
    frac_neg = (ac_post < 0).mean()
    print(f"  [{label}] fraction of post years with amplitude_component < 0: {frac_neg:.2f}\n")

if level_col and amp_obs_col:
    print("3. DOES THE AMPLITUDE COMPONENT TRACK LEVEL, OR OBSERVED AMPLITUDE, OR BOTH?\n")
    sub = g[["amplitude_component", level_col, amp_obs_col]].dropna()
    r_level = np.corrcoef(sub["amplitude_component"], sub[level_col])[0, 1]
    r_amp = np.corrcoef(sub["amplitude_component"], sub[amp_obs_col])[0, 1]
    print(f"  correlation(amplitude_component, {level_col})   = {r_level:+.3f}   (level)")
    print(f"  correlation(amplitude_component, {amp_obs_col}) = {r_amp:+.3f}   (observed amplitude / size)")
    print()

    # standardized two-predictor regression: which one actually explains amplitude_component?
    X = sub[[level_col, amp_obs_col]].copy()
    X = (X - X.mean()) / X.std(ddof=1)
    y = (sub["amplitude_component"] - sub["amplitude_component"].mean()) / sub["amplitude_component"].std(ddof=1)
    X["const"] = 1.0
    beta, *_ = np.linalg.lstsq(X[[level_col, amp_obs_col, "const"]].values, y.values, rcond=None)
    pred = X[[level_col, amp_obs_col, "const"]].values @ beta
    r2 = 1 - np.sum((y.values - pred) ** 2) / np.sum((y.values - y.values.mean()) ** 2)
    print(f"  standardized regression: amplitude_component ~ {level_col} + {amp_obs_col}")
    print(f"    level coefficient   = {beta[0]:+.3f}")
    print(f"    amplitude coefficient = {beta[1]:+.3f}")
    print(f"    R^2 = {r2:.3f}")
    print("    (bigger |coefficient| = that predictor explains more of amplitude_component's")
    print("     year-to-year movement, holding the other one fixed. If the level coefficient")
    print("     dominates, paragraph 105's 'amplitude' is substantially a level story.)\n")

    print("4. HOW MUCH OF THE POST-2016 MEAN SHIFT IN AMPLITUDE_COMPONENT LINES UP WITH LEVEL VS SIZE?\n")
    for label, (pre, post) in [("2016-2024", (pre24, post24)), ("2016-2025", (pre25, post25))]:
        d_ac = post["amplitude_component"].mean() - pre["amplitude_component"].mean()
        d_level = post[level_col].mean() - pre[level_col].mean()
        d_amp = post[amp_obs_col].mean() - pre[amp_obs_col].mean()
        # simple slopes from the pooled regression above, applied to each predictor's own shift,
        # as a rough share-of-shift attribution (not a causal claim, just bookkeeping)
        b_level_raw = beta[0] * sub["amplitude_component"].std(ddof=1) / sub[level_col].std(ddof=1)
        b_amp_raw = beta[1] * sub["amplitude_component"].std(ddof=1) / sub[amp_obs_col].std(ddof=1)
        implied_from_level = b_level_raw * d_level
        implied_from_amp = b_amp_raw * d_amp
        print(f"  [{label}] amplitude_component shift: {d_ac:+.4f}")
        print(f"  [{label}]   level shifted by {d_level:+.4f} -> implies {implied_from_level:+.4f} of that")
        print(f"  [{label}]   observed amplitude shifted by {d_amp:+.4f} -> implies {implied_from_amp:+.4f} of that")
        print(f"  [{label}]   (these two need not sum to the total shift; R^2 above shows how much")
        print(f"  [{label}]    of amplitude_component's variation these two predictors jointly explain)\n")
else:
    print("Could not find both a level proxy and an observed-amplitude proxy column -- "
          "skipping sections 3 and 4. Check the column names printed above.")