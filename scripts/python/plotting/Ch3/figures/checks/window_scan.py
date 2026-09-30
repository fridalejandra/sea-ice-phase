#!/usr/bin/env python3
"""
window_scan.py

Question: is the Oct-Dec raw-anomaly -> next-winter amplitude link specific to
Oct-Dec, or does it show up in other windows too?

Method
  1. Slide a window (60 and 90 days) across the calendar year. For each window,
     take the mean daily raw anomaly per sector per year.
  2. Correlate that with the amplitude of the FOLLOWING year (lag 1). Lag 0
     (same-year amplitude) is also run, as a reference only.
     Same scan is repeated with the observed day of maximum as the target.
  3. Both series are linearly detrended over years, then Spearman-correlated,
     to match how the rest of the chapter does it.
  4. Sector-averaged r is the headline (mean of the 5 sectors, as in the draft).
     Circumpolar is shown separately, not in the average.
  5. Permutation test (years shuffled, same shuffle for all sectors):
       p_single = chance of |r| this big for that ONE window
       p_family = chance that the BEST of all windows is this big
     p_family is the honest number if the window was chosen after looking.
  6. Partial correlations for the named Oct-Dec window and the best window:
     controlling for the target's own value the year before, and for the
     Jan-Mar raw anomaly of the intervening summer.

Usage
  python3 window_scan.py                       # uses DATA_DIR below
  python3 window_scan.py /path/to/data/ch3/    # or pass a folder

Needs: pandas, numpy, scipy, matplotlib
Writes: window_scan_results.csv, window_scan.png (in the current folder)
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------ CONFIG ------------------------------------
DATA_DIR = os.path.expanduser("~/Research/repos/sea-ice-phase/data/ch3/")
DAILY_FILE = "daily_fitted.csv"
ANNUAL_FILE = "annual_params.csv"

FIRST_YEAR, LAST_YEAR = 1988, 2023   # 2024-25 excluded (known data bug)
AMP_COL = "amplitude_raw_yr"         # or "amplitude_fitted"
DOY_COL = "max_doy_raw"              # observed day of maximum (phase)
LENGTHS = [60, 90]                   # window lengths, days
STEP = 10                            # window start step, days
MIN_COVERAGE = 0.8                   # fraction of days needed in a window
N_PERM = 5000
SEED = 1
OCTDEC = ("OctDec", 274, 92)         # label, start day-of-year, length
WRAP = True                          # let windows run into Jan-Mar of the FOLLOWING year
# Optional extra target that is NOT tied to the minimum (amplitude = max - min is).
# Seasonal-mean winter extent anomaly; skipped automatically if the column is missing.
OPTIONAL_TARGETS = {"winter_extent": "sie_JJA"}
# --------------------------------------------------------------------------


def norm(s):
    s = str(s).lower().replace("sie_", "")
    return re.sub(r"[^a-z0-9]", "", s)


def find_col(df, name):
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    raise KeyError(f"column '{name}' not found. Columns are: {list(df.columns)}")


def detrend(y, yrs):
    out = np.full(len(y), np.nan)
    m = ~np.isnan(y)
    if m.sum() < 5:
        return out
    b = np.polyfit(yrs[m], y[m], 1)
    out[m] = y[m] - np.polyval(b, yrs[m])
    return out


def spearman(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 5:
        return np.nan
    a, b = rankdata(x[m]), rankdata(y[m])
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def partial(x, y, z):
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den > 0 else np.nan


def zrank(a):
    """rank then standardise along last axis (so mean of products = Spearman r)"""
    r = np.apply_along_axis(rankdata, -1, a)
    return (r - r.mean(-1, keepdims=True)) / r.std(-1, keepdims=True)


# ------------------------------ LOAD --------------------------------------
if len(sys.argv) > 1:
    DATA_DIR = os.path.expanduser(sys.argv[1])

daily = pd.read_csv(os.path.join(DATA_DIR, DAILY_FILE))
annual = pd.read_csv(os.path.join(DATA_DIR, ANNUAL_FILE))

# daily
if any(c.lower() == "period" for c in daily.columns):
    daily = daily[daily[find_col(daily, "period")].astype(str) == "FULL"]
dcol_date = find_col(daily, "Date")
dcol_sec = find_col(daily, "sector")
dcol_anom = find_col(daily, "raw_anomaly")
daily = daily[[dcol_date, dcol_sec, dcol_anom]].copy()
daily.columns = ["date", "sector", "anom"]
daily["date"] = pd.to_datetime(daily["date"])
daily = daily[~((daily.date.dt.month == 2) & (daily.date.dt.day == 29))]
daily["year"] = daily.date.dt.year
leap = daily.date.dt.is_leap_year
daily["doy"] = daily.date.dt.dayofyear - ((leap) & (daily.date.dt.month > 2)).astype(int)
daily = daily[(daily.year >= FIRST_YEAR) & (daily.year <= LAST_YEAR)]
daily["key"] = daily["sector"].map(norm)

# annual
if any(c.lower() == "period" for c in annual.columns):
    annual = annual[annual[find_col(annual, "period")].astype(str) == "FULL"]
acol_year = find_col(annual, "Year")
acol_sec = find_col(annual, "sector")
annual["key"] = annual[acol_sec].map(norm)

daily_keys = sorted(daily["key"].unique())
print("Sectors in daily file :", sorted(daily["sector"].unique()))
print("Sectors in annual file:", sorted(annual[acol_sec].unique()))


def match(k):
    if k in daily_keys:
        return k
    for d in daily_keys:
        if d.startswith(k) or k.startswith(d):
            return d
    return None


is_circ = lambda k: any(t in k for t in ("circum", "total", "all"))
ann_keys = sorted(annual["key"].unique())
pairs = [(a, match(a)) for a in ann_keys]
pairs = [(a, d) for a, d in pairs if d is not None]
sectors5 = [(a, d) for a, d in pairs if not is_circ(a)]
circ = [(a, d) for a, d in pairs if is_circ(a)]
print("Matched sectors used  :", [a for a, _ in sectors5],
      "| circumpolar:", [a for a, _ in circ])
if len(sectors5) < 3:
    sys.exit("Could not match sectors between the two files - check names printed above.")

years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
nY = len(years)


def daily_matrix(dkey):
    sub = daily[daily.key == dkey]
    M = np.full((nY, 365), np.nan)
    yi = (sub.year.values - FIRST_YEAR).astype(int)
    di = (sub.doy.values - 1).astype(int)
    ok = (di >= 0) & (di < 365)
    M[yi[ok], di[ok]] = sub.anom.values[ok]
    return M


def annual_series(akey, col):
    sub = annual[annual.key == akey].drop_duplicates(subset=[acol_year])
    s = pd.Series(sub[find_col(annual, col)].values, index=sub[acol_year].values)
    return s.reindex(years).values.astype(float)


allpairs = sectors5 + circ
D = {a: daily_matrix(d) for a, d in allpairs}
TARGETS = {"amplitude": AMP_COL, "day_of_max": DOY_COL}
for _k, _c in OPTIONAL_TARGETS.items():
    if any(col.lower() == _c.lower() for col in annual.columns):
        TARGETS[_k] = _c
    else:
        print(f"(optional target '{_k}': column '{_c}' not found - skipped.")
        print(f"  Annual-file columns: {list(annual.columns)})")
T = {(name, a): annual_series(a, col) for name, col in TARGETS.items() for a, _ in allpairs}

# window list
windows = []
end_limit = 365 + (90 if WRAP else 0)   # last day allowed = Mar 31 of next year
for L in LENGTHS:
    for s in range(1, end_limit - L + 2, STEP):
        windows.append((f"{s}-{s + L - 1}", s, L))
windows.append(OCTDEC)
lab_idx = {w[0]: i for i, w in enumerate(windows)}
nW = len(windows)


def window_means(M, s, L):
    seg = M[:, s - 1: s - 1 + L]
    cnt = np.sum(~np.isnan(seg), axis=1)
    with np.errstate(all="ignore"):
        mu = np.nanmean(seg, axis=1)
    mu[cnt < MIN_COVERAGE * L] = np.nan
    return mu


# row Y = days 1-365 of year Y followed by days 1-365 of year Y+1 (NaN for the last year)
D2 = {a: np.concatenate([D[a], np.vstack([D[a][1:], np.full((1, 365), np.nan)])], axis=1)
      for a, _ in allpairs}
W = {(a, i): window_means(D2[a], s, L) for a, _ in allpairs for i, (_, s, L) in enumerate(windows)}
JFM = {a: window_means(D[a], 1, 90) for a, _ in allpairs}   # Jan-Mar of a given year

rng = np.random.default_rng(SEED)
rows = []
figdata = {}
summary_lines = []

for lag in (0, 1):
    for tname in TARGETS:
        # residual years Y, target years Y+lag
        Y0 = np.arange(nY - lag)             # index of residual year
        yrs_res = years[Y0]
        X = np.full((len(sectors5), nW, len(Y0)), np.nan)
        Yt = np.full((len(sectors5), len(Y0)), np.nan)
        Yprev = np.full((len(sectors5), len(Y0)), np.nan)
        Zsum = np.full((len(sectors5), len(Y0)), np.nan)
        for si, (a, _) in enumerate(sectors5):
            tv = T[(tname, a)]
            Yt[si] = detrend(tv[Y0 + lag], yrs_res)
            Yprev[si] = detrend(tv[Y0], yrs_res)
            if lag == 1:
                Zsum[si] = detrend(JFM[a][Y0 + 1], yrs_res)
            for wi in range(nW):
                X[si, wi] = detrend(W[(a, wi)][Y0], yrs_res)

        # complete-case years across everything
        ok = ~np.isnan(Yt).any(0)
        ok &= ~np.isnan(X).any(axis=(0, 1))
        dropped = int((~ok).sum())
        if dropped > 3:
            print(f"  WARNING lag{lag} {tname}: {dropped} years dropped for missing data")
        Xc, Yc = X[..., ok], Yt[..., ok]
        n = Xc.shape[-1]

        # observed
        r_obs = np.full((len(sectors5), nW), np.nan)
        for si in range(len(sectors5)):
            for wi in range(nW):
                r_obs[si, wi] = spearman(Xc[si, wi], Yc[si])
        r_avg = np.nanmean(r_obs, axis=0)

        # circumpolar
        r_circ = np.full(nW, np.nan)
        if circ:
            a = circ[0][0]
            tv = T[(tname, a)]
            yt = detrend(tv[Y0 + lag], yrs_res)[ok]
            for wi in range(nW):
                r_circ[wi] = spearman(detrend(W[(a, wi)][Y0], yrs_res)[ok], yt)

        # permutation (same shuffle of years for every sector)
        Xz = zrank(Xc)                       # (S, W, n)
        Yz = zrank(Yc)                       # (S, n)
        max_null = np.empty(N_PERM)
        null_abs = np.empty((N_PERM, nW))
        for p in range(N_PERM):
            perm = rng.permutation(n)
            r_p = (Xz * Yz[:, None, perm]).mean(-1)   # (S, W)
            ra = np.abs(r_p.mean(0))
            null_abs[p] = ra
            max_null[p] = ra.max()
        p_single = (null_abs >= np.abs(r_avg)[None, :]).mean(0)
        p_family = (max_null[:, None] >= np.abs(r_avg)[None, :]).mean(0)
        fw_thresh = np.quantile(max_null, 0.95)

        for wi, (lab, s, L) in enumerate(windows):
            row = dict(lag=lag, target=tname, window=lab, start_doy=s, length=L,
                       r_sector_avg=r_avg[wi], r_circumpolar=r_circ[wi],
                       p_single=p_single[wi], p_family=p_family[wi], n_years=n)
            for si, (a, _) in enumerate(sectors5):
                row[f"r_{a}"] = r_obs[si, wi]
            rows.append(row)
        figdata[(lag, tname)] = dict(r_avg=r_avg, r_obs=r_obs, r_circ=r_circ, fw=fw_thresh)

        # report
        best = int(np.nanargmax(np.abs(r_avg)))
        oi = lab_idx[OCTDEC[0]]
        tag = f"lag {lag} | target = {tname} | n = {n} years"
        summary_lines.append("=" * 74)
        summary_lines.append(tag)
        summary_lines.append(f"  Oct-Dec (doy {OCTDEC[1]}-{OCTDEC[1] + OCTDEC[2] - 1}): "
                             f"sector-avg r = {r_avg[oi]:+.2f}   "
                             f"p_single = {p_single[oi]:.3f}   p_family = {p_family[oi]:.3f}")
        summary_lines.append(f"  Best window {windows[best][0]}: sector-avg r = {r_avg[best]:+.2f}   "
                             f"p_single = {p_single[best]:.3f}   p_family = {p_family[best]:.3f}")
        order = np.argsort(-np.abs(r_avg))[:6]
        summary_lines.append("  Top 6 windows by |r|: " +
                             ", ".join(f"{windows[i][0]} ({r_avg[i]:+.2f})" for i in order))

        # partial correlations (only meaningful for lag 1)
        if lag == 1:
            for name_w, wi in (("Oct-Dec", oi), (f"best {windows[best][0]}", best)):
                pr_prev, pr_sum = [], []
                wrapped = (windows[wi][1] + windows[wi][2] - 1) > 365
                for si in range(len(sectors5)):
                    x, y = Xc[si, wi], Yc[si]
                    pr_prev.append(partial(x, y, Yprev[si][ok]))
                    pr_sum.append(np.nan if wrapped else partial(x, y, Zsum[si][ok]))
                sum_txt = ("n/a (window overlaps next Jan-Mar)" if wrapped
                           else f"{np.nanmean(pr_sum):+.2f}")
                summary_lines.append(
                    f"  {name_w}: raw r = {r_avg[wi]:+.2f} | "
                    f"partial, controlling the target's own value the year before = {np.nanmean(pr_prev):+.2f} | "
                    f"partial, controlling the Jan-Mar summer anomaly = {sum_txt}")

pd.DataFrame(rows).to_csv("window_scan_results.csv", index=False)

# ------------------------------ FIGURE ------------------------------------
fig, axes = plt.subplots(2, len(TARGETS), figsize=(6 * len(TARGETS), 7), sharex=True)
for ri, lag in enumerate((0, 1)):
    for ci, tname in enumerate(TARGETS):
        ax = axes[ri, ci]
        fd = figdata[(lag, tname)]
        idx = [i for i, w in enumerate(windows) if w[2] == 90 and w[0] != OCTDEC[0]]
        idx = sorted(idx, key=lambda i: windows[i][1])
        cx = [windows[i][1] + 45 for i in idx]
        for si in range(len(sectors5)):
            ax.plot(cx, fd["r_obs"][si, idx], color="0.75", lw=0.9)
        ax.plot(cx, fd["r_circ"][idx], color="k", ls="--", lw=1.2, label="circumpolar")
        ax.plot(cx, fd["r_avg"][idx], color="C3", lw=2.4, label="mean of 5 sectors")
        ax.axhspan(-fd["fw"], fd["fw"], color="C0", alpha=0.12,
                   label="5% family-wise band (sector mean)")
        ax.axvspan(274, 365, color="C1", alpha=0.10)
        ax.axvline(321, color="0.4", ls=":", lw=1)   # right of this, windows include next-year Jan+
        ax.set_xlim(30, 415)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"lag {lag}  |  target: {tname}   (90-day windows)", fontsize=10)
        ax.set_ylim(-0.8, 0.8)
        if ri == 1:
            ax.set_xlabel("window centre (day of year)")
        if ci == 0:
            ax.set_ylabel("Spearman r (detrended)")
        if ri == 0 and ci == 0:
            ax.legend(fontsize=8, loc="upper left")
fig.text(0.5, 0.005, "orange shading = Oct-Dec; right of the dotted line the window runs into the following Jan-Mar",
         ha="center", fontsize=9)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("window_scan.png", dpi=150)

print("\n".join(summary_lines))
print("=" * 74)
print("Wrote window_scan_results.csv and window_scan.png")
print("How to read: lag 1 / amplitude is your claim. If Oct-Dec sits inside the blue")
print("band, or p_family is large, it is not distinguishable from picking the best")
print("of many windows. Lag 0 is a reference only (windows before Sept lie inside")
print("the same cycle). Check which lag reproduces your r ~ +0.27; that also tells")
print("you how 'Year' is labelled in annual_params.csv.")