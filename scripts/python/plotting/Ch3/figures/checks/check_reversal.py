#!/usr/bin/env python3
"""
coupling_test.py

Stress test for the claim: "amplitude and phase were independent before 2016
and became correlated after 2016."

Uses annual_params.csv only (one row per sector per year).

What it checks (each for day of maximum, day of minimum, and growth-season
length, always against the observed amplitude):

  1. BASELINE   correlation over FIRST_YEAR-2015 (should be ~0 if "independent")
  2. POST       correlation over 2016-last year, with an exact permutation p-value
                (years shuffled, same shuffle for every sector)
  3. BUG YEARS  same, but with 2024-25 removed (suspect data)
  4. DETRENDED  same, but with a linear trend removed INSIDE the window
                (a trend in both variables alone can create a correlation)
  5. LEAVE-ONE-OUT  does any single year carry it?
  6. TRAILING WINDOWS  the same 10-year correlation for every window in the
                record: how unusual is 2016-2025 compared with earlier decades?
  7. A z-score table for the circumpolar series in the post years, so you can
     see whether 2024-25 look like outliers.

Headline number everywhere = mean Spearman r over the 5 sectors (as in the
draft). Circumpolar is reported separately.

Usage
  python3 coupling_test.py                       # uses DATA_DIR below
  python3 coupling_test.py /path/to/data/ch3/

Needs: pandas, numpy, scipy, matplotlib
Writes: coupling_test_windows.csv, coupling_test.png (current folder)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------ CONFIG ------------------------------------
DATA_DIR = os.path.expanduser("~/Research/repos/sea-ice-phase/data/ch3/")
ANNUAL_FILE = "annual_params.csv"
AMP_COL = "amplitude_raw_yr"
PHASE_COLS = {"day_of_max": "max_doy_raw", "day_of_min": "min_doy_raw"}
FIRST_YEAR = 1988          # avoids the every-other-day SMMR era; set 1979 to use more
BREAK_YEAR = 2016          # first year of the "post" period
BAD_YEARS = (2024, 2025)   # suspect data (known bug)
WIN = 10                   # trailing-window length, years
MIN_YEARS = 7              # minimum valid years for a correlation
N_PERM = 10000
SEED = 1
# --------------------------------------------------------------------------


def find_col(df, name):
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    raise KeyError(f"column '{name}' not found. Columns are: {list(df.columns)}")


def detrend(v, yrs):
    out = np.full(len(v), np.nan)
    m = ~np.isnan(v)
    if m.sum() < 4:
        return out
    b = np.polyfit(yrs[m], v[m], 1)
    out[m] = v[m] - np.polyval(b, yrs[m])
    return out


def spearman(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < MIN_YEARS:
        return np.nan
    a, b = rankdata(x[m]), rankdata(y[m])
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def zrank(a):
    r = np.apply_along_axis(rankdata, 0, a)
    return (r - r.mean(0, keepdims=True)) / r.std(0, keepdims=True)


# ------------------------------ LOAD --------------------------------------
if len(sys.argv) > 1:
    DATA_DIR = os.path.expanduser(sys.argv[1])
annual = pd.read_csv(os.path.join(DATA_DIR, ANNUAL_FILE))
if any(c.lower() == "period" for c in annual.columns):
    annual = annual[annual[find_col(annual, "period")].astype(str) == "FULL"]
ycol, scol = find_col(annual, "Year"), find_col(annual, "sector")
annual = annual[annual[ycol] >= FIRST_YEAR]


def wide(col):
    c = find_col(annual, col)
    w = annual.pivot_table(index=ycol, columns=scol, values=c, aggfunc="first")
    return w.sort_index()


AMP = wide(AMP_COL)
PH = {name: wide(col) for name, col in PHASE_COLS.items()}
if len(PH) == 2:
    PH["season_length"] = PH["day_of_max"] - PH["day_of_min"]
years_all = AMP.index.values.astype(int)
LAST = int(years_all.max())
cols = list(AMP.columns)
circ = [c for c in cols if any(t in c.lower() for t in ("circum", "total"))]
sec5 = [c for c in cols if c not in circ]
print("Sectors used:", sec5, "| circumpolar:", circ)
print(f"Years {years_all.min()}-{LAST}; post period = {BREAK_YEAR}-{LAST}; bug years = {BAD_YEARS}")
rng = np.random.default_rng(SEED)


def sub(df, y0, y1, drop=()):
    keep = [y for y in df.index if y0 <= y <= y1 and y not in drop]
    return df.loc[keep]


def stats(X, Y, detr, nperm=0):
    """X, Y: DataFrames (years x sectors). Returns dict."""
    yrs = X.index.values.astype(float)
    xs, ys = X[sec5].values.astype(float), Y[sec5].values.astype(float)
    ok = ~(np.isnan(xs).any(1) | np.isnan(ys).any(1))
    if ok.sum() < MIN_YEARS:
        return dict(n=int(ok.sum()), r=np.nan, r_circ=np.nan, p=np.nan, r_sec=[np.nan] * len(sec5))
    yr_ok = yrs[ok]
    xs, ys = xs[ok], ys[ok]
    if detr:
        xs = np.column_stack([detrend(xs[:, j], yr_ok) for j in range(xs.shape[1])])
        ys = np.column_stack([detrend(ys[:, j], yr_ok) for j in range(ys.shape[1])])
    zx, zy = zrank(xs), zrank(ys)
    r_sec = (zx * zy).mean(0)
    r = float(r_sec.mean())
    r_c = np.nan
    if circ:
        xc = X[circ[0]].values.astype(float)[ok]
        yc = Y[circ[0]].values.astype(float)[ok]
        if detr:
            xc, yc = detrend(xc, yr_ok), detrend(yc, yr_ok)
        r_c = spearman(xc, yc)
    p = np.nan
    if nperm:
        n = len(zy)
        null = np.empty(nperm)
        for i in range(nperm):
            null[i] = (zx * zy[rng.permutation(n)]).mean(0).mean()
        p = float((np.abs(null) >= abs(r) - 1e-12).mean())
    return dict(n=int(ok.sum()), r=r, r_circ=r_c, p=p, r_sec=list(r_sec))


rows = []
lines = []
figdata = {}

for pname, PHX in PH.items():
    lines.append("=" * 78)
    lines.append(f"{pname}  vs  amplitude ({AMP_COL})")

    # 1. baseline
    b_raw = stats(sub(PHX, FIRST_YEAR, BREAK_YEAR - 1), sub(AMP, FIRST_YEAR, BREAK_YEAR - 1), False)
    b_det = stats(sub(PHX, FIRST_YEAR, BREAK_YEAR - 1), sub(AMP, FIRST_YEAR, BREAK_YEAR - 1), True)
    lines.append(f"  1. BASELINE {FIRST_YEAR}-{BREAK_YEAR - 1} (n={b_raw['n']}): "
                 f"sector-avg r = {b_raw['r']:+.2f} raw | {b_det['r']:+.2f} detrended")

    # 2-4. post windows
    specs = [("2. POST all years", BREAK_YEAR, LAST, ()),
             ("3. POST excluding bug years", BREAK_YEAR, LAST, BAD_YEARS)]
    for label, y0, y1, drop in specs:
        for detr in (False, True):
            s = stats(sub(PHX, y0, y1, drop), sub(AMP, y0, y1, drop), detr, N_PERM)
            tag = "detrended in window" if detr else "raw"
            per_sec = " ".join(f"{v:+.2f}" for v in s["r_sec"])
            yr_txt = f"{y0}-{y1}" + (f" without {', '.join(str(d) for d in drop)}" if drop else "")
            lines.append(f"  {label} ({yr_txt}, n={s['n']}, {tag}): sector-avg r = {s['r']:+.2f} "
                         f"| perm p = {s['p']:.3f} | circumpolar r = {s['r_circ']:+.2f}")
            lines.append(f"       per sector: {per_sec}")
            rows.append(dict(phase=pname, test=label, detrended=detr, y0=y0, y1=y1, n=s["n"],
                             r_sector_avg=s["r"], r_circumpolar=s["r_circ"], perm_p=s["p"]))

    # 5. leave one out (raw and detrended)
    post_years = [y for y in AMP.index if BREAK_YEAR <= y <= LAST]
    for detr in (False, True):
        vals = {}
        for y in post_years:
            s = stats(sub(PHX, BREAK_YEAR, LAST, (y,)), sub(AMP, BREAK_YEAR, LAST, (y,)), detr)
            vals[y] = s["r"]
        arr = np.array(list(vals.values()), dtype=float)
        lo_y = min(vals, key=lambda k: vals[k])
        hi_y = max(vals, key=lambda k: vals[k])
        lines.append(f"  5. LEAVE-ONE-OUT ({'detrended' if detr else 'raw'}): "
                     f"r ranges {np.nanmin(arr):+.2f} (drop {lo_y}) to {np.nanmax(arr):+.2f} (drop {hi_y})")

    # 6. trailing windows
    ends, r_raw, r_det, r_circ_w, r_secs = [], [], [], [], []
    for e in range(FIRST_YEAR + WIN - 1, LAST + 1):
        Xw, Yw = sub(PHX, e - WIN + 1, e), sub(AMP, e - WIN + 1, e)
        a, d_, = stats(Xw, Yw, False), stats(Xw, Yw, True)
        ends.append(e)
        r_raw.append(a["r"]); r_det.append(d_["r"]); r_circ_w.append(a["r_circ"]); r_secs.append(a["r_sec"])
        rows.append(dict(phase=pname, test="trailing_window", detrended=False, y0=e - WIN + 1, y1=e,
                         n=a["n"], r_sector_avg=a["r"], r_circumpolar=a["r_circ"], perm_p=np.nan))
    ends, r_raw, r_det = np.array(ends), np.array(r_raw), np.array(r_det)
    r_circ_w, r_secs = np.array(r_circ_w), np.array(r_secs)
    post_r = r_raw[ends == LAST][0]
    pre = r_raw[(ends < BREAK_YEAR) & ~np.isnan(r_raw)]
    if len(pre):
        frac = float((np.abs(pre) >= abs(post_r)).mean())
        lines.append(f"  6. TRAILING {WIN}-YR WINDOWS: post window r = {post_r:+.2f}; among the "
                     f"{len(pre)} windows ending before {BREAK_YEAR}, r ranged {pre.min():+.2f} to {pre.max():+.2f}; "
                     f"{100 * frac:.0f}% had |r| >= the post window")
    order = np.argsort(-np.abs(np.nan_to_num(r_raw)))[:5]
    lines.append("     top windows by |r| (end year): " +
                 ", ".join(f"{ends[i]} ({r_raw[i]:+.2f})" for i in order))
    lines.append("     (windows overlap heavily, so treat these as a picture, not independent trials)")
    figdata[pname] = dict(ends=ends, raw=r_raw, det=r_det, circ=r_circ_w, secs=r_secs, pre=pre)

# 7. z-scores of the circumpolar series
if circ:
    c = circ[0]
    lines.append("=" * 78)
    lines.append(f"7. Circumpolar z-scores relative to {FIRST_YEAR}-{BREAK_YEAR - 1} (are the bug years outliers?)")
    hdr = "  year  amplitude " + " ".join(f"{k:>14s}" for k in PH)
    lines.append(hdr)
    base = lambda s: (s.loc[FIRST_YEAR:BREAK_YEAR - 1].mean(), s.loc[FIRST_YEAR:BREAK_YEAR - 1].std())
    am, asd = base(AMP[c])
    for y in [y for y in AMP.index if y >= BREAK_YEAR]:
        parts = [f"{(AMP.loc[y, c] - am) / asd:+9.2f}"]
        for k, df in PH.items():
            m, sd = base(df[c])
            parts.append(f"{(df.loc[y, c] - m) / sd:+14.2f}")
        flag = "  <- bug year" if y in BAD_YEARS else ""
        lines.append(f"  {int(y)} " + " ".join(parts) + flag)

pd.DataFrame(rows).to_csv("coupling_test_windows.csv", index=False)

# ------------------------------ FIGURE ------------------------------------
fig, axes = plt.subplots(1, len(PH), figsize=(5.6 * len(PH), 4.6), sharey=True)
axes = np.atleast_1d(axes)
for ax, (pname, fd) in zip(axes, figdata.items()):
    if len(fd["pre"]):
        ax.axhspan(np.percentile(fd["pre"], 2.5), np.percentile(fd["pre"], 97.5), color="C0", alpha=0.13,
                   label="2.5-97.5% of windows ending before 2016")
    for j in range(fd["secs"].shape[1]):
        ax.plot(fd["ends"], fd["secs"][:, j], color="0.78", lw=0.9)
    ax.plot(fd["ends"], fd["circ"], color="k", ls="--", lw=1.2, label="circumpolar")
    ax.plot(fd["ends"], fd["raw"], color="C3", lw=2.4, label="mean of 5 sectors (raw)")
    ax.plot(fd["ends"], fd["det"], color="C3", lw=1.6, ls=":", label="mean of 5 sectors (detrended in window)")
    ax.axvline(BREAK_YEAR - 1, color="0.4", ls=":", lw=1)
    ax.axvspan(BAD_YEARS[0], LAST + 0.5, color="0.5", alpha=0.15)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{pname} vs amplitude", fontsize=10)
    ax.set_xlabel(f"end year of trailing {WIN}-yr window")
    ax.set_ylim(-1, 1)
axes[0].set_ylabel("Spearman r")
axes[0].legend(fontsize=7, loc="lower left")
fig.text(0.5, 0.005, "gray shading: windows that include the suspect 2024-25 data", ha="center", fontsize=9)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("coupling_test.png", dpi=150)

print("\n".join(lines))
print("=" * 78)
print("Wrote coupling_test_windows.csv and coupling_test.png")
print("How to read: the claim survives if (a) the post r is clearly outside the range of")
print("earlier windows, (b) it is still there with 2024-25 removed, (c) it is still there")
print("after detrending inside the window, and (d) no single year carries it.")