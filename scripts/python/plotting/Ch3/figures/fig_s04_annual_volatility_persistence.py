#!/usr/bin/env python3
"""
check_anomaly_persistence.py -- three diagnostics for Sect. 3.4, prompted by
Fig. 7 (2026-09-19): the interannual spread of the fitted cycle's slope did
NOT widen after 2016, so the rise in dSIE volatility (Fig. 6, Table 4) is
not coming from the per-year amplitude/phase adjustment. Since
    dSIE = (fitted slope) + delta(raw anomaly)
and neither the fitted slope's spread nor the raw anomaly's own variance
rose, the only term left is the variance of the day-to-day CHANGE in the
raw anomaly, which can rise with var(raw anomaly) fixed only if the anomaly
became less persistent:  var(delta ra) = 2 var(ra) (1 - rho1).

  1. PERSISTENCE of the raw APAC anomaly, pre-2016 vs 2016+, per sector:
     lag-1 autocorrelation rho1 (consecutive days only), e-folding time of
     the autocorrelation, var(ra), var(delta ra), and the identity check
     2 var(ra)(1 - rho1) vs var(delta ra). Also rho1 by YEAR, so the
     pre/post difference has a spread (n = 28 vs 10 years) and a p-value.
     Prediction if the "choppier, not bigger" reading is right: rho1 lower
     after 2016, var(ra) about the same, var(delta ra) up.

  2. CROSS-SECTOR SYNCHRONY of day-to-day change, pre vs post: correlation
     matrix of deseasonalised dSIE among the five sectors, the mean pairwise
     correlation, and var(sum of five) / sum(var of five) -- equal to 1 if
     the sectors' day-to-day changes are independent, > 1 if they co-vary,
     < 1 if they compensate. Motivated by the circumpolar dSIE ratio (3.0)
     far exceeding any sector's (<= 1.6): the sum's variance can outgrow the
     parts only if the parts became positively correlated.

  3. YEAR-BY-YEAR volatility (SD of deseasonalised dSIE and of the raw
     anomaly, per year, relative to the 1988-2015 mean) and year-by-year
     rho1, to see whether the change STEPS at 2016 or RAMPS. The
     passive-microwave record moved from F17 to F18 in 2016 (F17 37V
     channel degradation); the sensor term in the gamlss handles only
     SSM/I -> SSMIS in 2008. A step exactly at 2016 with no further rise is
     what a retrieval change would look like; a ramp through 2025 is not.

Deseasonalising: day-of-year mean over the full 1988-2025 record, per
sector, subtracted from dSIE and from the raw anomaly (the raw anomaly's
DOY mean should already be ~0; removing it costs nothing).

Inputs   daily_fitted.csv via ch3_data.load_daily(period="FULL")
Outputs  results/ch3/tables/t34e_anomaly_persistence.csv
         results/ch3/tables/t34e_anomaly_persistence_by_year.csv
         results/ch3/tables/t34f_dsie_cross_sector_corr.csv
         results/ch3/tables/t34g_annual_volatility.csv
         results/ch3/figures/fig_s04_annual_volatility_persistence.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ch3_data as D
from ch3_config import TABLES_DIR, OUTPUT_DIR, SECTOR_ORDER_BY_LONGITUDE, SECTOR_LABELS
import ch3_style

BREAK = 2016
START_YEAR, END_YEAR = 1988, 2025
COLOR_PRE, COLOR_POST = "#2a78d6", "#eb6834"
MAX_LAG = 30

try:
    from scipy import stats as _st
    def welch_p(a, b):
        return float(_st.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)
except Exception:
    def welch_p(a, b):
        return float("nan")

print("check_anomaly_persistence")
d = D.load_daily(period="FULL")
d = d[(d["Year"] >= START_YEAR) & (d["Year"] <= END_YEAR)].copy()
d["Date"] = pd.to_datetime(d["Date"])
d = d.sort_values(["sector", "Date"]).reset_index(drop=True)
d["ra"] = d["residual_apac"]

sector_codes = [s for s in SECTOR_ORDER_BY_LONGITUDE if s in SECTOR_LABELS and s in set(d["sector"])]
circ = [s for s in d["sector"].unique() if "circumpolar" in s.lower()]
all_codes = sector_codes + circ
lab = lambda s: SECTOR_LABELS.get(s, s)
period_of = lambda y: np.where(y >= BREAK, f"{BREAK}+", f"pre-{BREAK}")

# ── day-to-day change and deseasonalised series ─────────────────────────────
g = d.groupby("sector")
gap = g["Date"].diff().dt.days
d["dSIE"] = np.where(gap == 1, g["Extent"].diff(), np.nan)
d["ra_lag1"] = np.where(gap == 1, g["ra"].shift(1), np.nan)
for col in ("dSIE", "ra"):
    clim = d.groupby(["sector", "DOY"])[col].transform("mean")
    d[f"{col}_a"] = d[col] - clim
d["ra_a_lag1"] = np.where(gap == 1, d.groupby("sector")["ra_a"].shift(1), np.nan)
d["dra_a"] = d["ra_a"] - d["ra_a_lag1"]
d["period"] = period_of(d["Year"])


def acf_by_lag(x, dates, max_lag):
    """Autocorrelation at lags 1..max_lag using only pairs exactly `lag` days apart."""
    s = pd.Series(x.values, index=pd.DatetimeIndex(dates.values)).dropna()
    out = {}
    for L in range(1, max_lag + 1):
        s2 = s.copy(); s2.index = s2.index + pd.Timedelta(days=L)
        j = s.to_frame("x").join(s2.to_frame("xlag"), how="inner")
        out[L] = j["x"].corr(j["xlag"]) if len(j) > 30 else np.nan
    return out


def efold(acf):
    thr = 1 / np.e
    prev_L, prev_v = 0, 1.0
    for L in sorted(acf):
        v = acf[L]
        if np.isnan(v):
            return np.nan
        if v < thr:
            return prev_L + (prev_v - thr) / (prev_v - v) * (L - prev_L)
        prev_L, prev_v = L, v
    return np.nan


# ── 1. persistence, pre vs post and by year ────────────────────────────────
rows, rows_y = [], []
for sec in all_codes:
    a = d[d["sector"] == sec]
    for per in (f"pre-{BREAK}", f"{BREAK}+"):
        b = a[a["period"] == per]
        acf = acf_by_lag(b["ra_a"], b["Date"], MAX_LAG)
        var_ra = b["ra_a"].var(ddof=1)
        var_dra = b["dra_a"].var(ddof=1)
        rows.append(dict(sector=lab(sec), period=per, n_days=int(b["ra_a"].notna().sum()),
                         rho1=acf[1], rho5=acf.get(5, np.nan), rho10=acf.get(10, np.nan),
                         efold_days=efold(acf), var_ra=var_ra, var_dra=var_dra,
                         identity_2var_1_minus_rho1=2 * var_ra * (1 - acf[1])))
    for y, b in a.groupby("Year"):
        ok = b["ra_a"].notna() & b["ra_a_lag1"].notna()
        rho1 = b.loc[ok, "ra_a"].corr(b.loc[ok, "ra_a_lag1"]) if ok.sum() > 30 else np.nan
        rows_y.append(dict(sector=lab(sec), Year=int(y), period=str(period_of(np.array([y]))[0]),
                           rho1=rho1, sd_dSIE=b["dSIE_a"].std(ddof=1), sd_ra=b["ra_a"].std(ddof=1),
                           sd_dra=b["dra_a"].std(ddof=1), n_days=int(ok.sum())))
pers = pd.DataFrame(rows)
pers_y = pd.DataFrame(rows_y)

os.makedirs(TABLES_DIR, exist_ok=True)
pers.to_csv(os.path.join(TABLES_DIR, "t34e_anomaly_persistence.csv"), index=False)
pers_y.to_csv(os.path.join(TABLES_DIR, "t34e_anomaly_persistence_by_year.csv"), index=False)

print("\n== 1. raw-anomaly persistence, pre vs post (consecutive days only) ==")
print("   identity check: var(delta ra) should equal 2 var(ra) (1 - rho1) if the anomaly is ~AR(1)")
for sec in all_codes:
    p = pers[(pers.sector == lab(sec))].set_index("period")
    pre, post = p.loc[f"pre-{BREAK}"], p.loc[f"{BREAK}+"]
    yy = pers_y[pers_y.sector == lab(sec)]
    r_pre, r_post = yy[yy.period == f"pre-{BREAK}"]["rho1"], yy[yy.period == f"{BREAK}+"]["rho1"]
    print(f"  {lab(sec):24s} rho1 {pre.rho1:.3f} -> {post.rho1:.3f}  "
          f"(by-year means {r_pre.mean():.3f}±{r_pre.std(ddof=1)/np.sqrt(len(r_pre)):.3f} -> "
          f"{r_post.mean():.3f}±{r_post.std(ddof=1)/np.sqrt(len(r_post)):.3f}, Welch p = {welch_p(r_pre, r_post):.3f})")
    print(f"  {'':24s} e-fold {pre.efold_days:5.1f} -> {post.efold_days:5.1f} d   "
          f"var(ra) x{post.var_ra / pre.var_ra:.2f}   var(delta ra) x{post.var_dra / pre.var_dra:.2f}   "
          f"identity: {pre.var_dra:.4f} vs {pre.identity_2var_1_minus_rho1:.4f} (pre), "
          f"{post.var_dra:.4f} vs {post.identity_2var_1_minus_rho1:.4f} (post)")

# ── 2. cross-sector synchrony of day-to-day change ─────────────────────────
print("\n== 2. cross-sector correlation of deseasonalised dSIE (five sectors) ==")
xs_rows = []
for per in (f"pre-{BREAK}", f"{BREAK}+"):
    w = (d[(d["period"] == per) & (d["sector"].isin(sector_codes))]
         .pivot_table(index="Date", columns="sector", values="dSIE_a")).dropna()
    w.columns = [lab(c) for c in w.columns]
    C = w.corr()
    off = C.values[np.triu_indices_from(C.values, k=1)]
    ratio = w.sum(axis=1).var(ddof=1) / w.var(ddof=1).sum()
    print(f"\n  {per}  (n = {len(w)} days)   mean pairwise r = {off.mean():+.3f}   "
          f"var(sum)/sum(var) = {ratio:.2f}  (1 = independent; >1 co-vary; <1 compensate)")
    print(C.round(2).to_string())
    for i, a_ in enumerate(C.index):
        for j, b_ in enumerate(C.columns):
            if j > i:
                xs_rows.append(dict(period=per, sector_a=a_, sector_b=b_, r=C.iloc[i, j], n_days=len(w)))
    xs_rows.append(dict(period=per, sector_a="ALL", sector_b="var(sum)/sum(var)", r=ratio, n_days=len(w)))
    xs_rows.append(dict(period=per, sector_a="ALL", sector_b="mean pairwise r", r=off.mean(), n_days=len(w)))
pd.DataFrame(xs_rows).to_csv(os.path.join(TABLES_DIR, "t34f_dsie_cross_sector_corr.csv"), index=False)

# same for the raw anomaly level, for contrast
print("\n   (same, for the deseasonalised raw anomaly LEVEL)")
for per in (f"pre-{BREAK}", f"{BREAK}+"):
    w = (d[(d["period"] == per) & (d["sector"].isin(sector_codes))]
         .pivot_table(index="Date", columns="sector", values="ra_a")).dropna()
    C = w.corr(); off = C.values[np.triu_indices_from(C.values, k=1)]
    ratio = w.sum(axis=1).var(ddof=1) / w.var(ddof=1).sum()
    print(f"  {per}: mean pairwise r = {off.mean():+.3f}   var(sum)/sum(var) = {ratio:.2f}")

# ── 3. year-by-year volatility and persistence ─────────────────────────────
pers_y.to_csv(os.path.join(TABLES_DIR, "t34g_annual_volatility.csv"), index=False)
print(f"\n== 3. year-by-year SD of dSIE, relative to the {START_YEAR}-{BREAK - 1} mean ==")
for sec in all_codes:
    yy = pers_y[pers_y.sector == lab(sec)].set_index("Year")
    base = yy.loc[:BREAK - 1, "sd_dSIE"].mean()
    rel = (yy["sd_dSIE"] / base)
    print(f"  {lab(sec):24s} " + " ".join(f"{y}:{rel[y]:.2f}" for y in range(BREAK - 4, END_YEAR + 1) if y in rel.index))

bold = ch3_style.bold_font_properties(size=10.5)
ncol = 3
nrow = int(np.ceil(len(all_codes) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.9 * nrow), sharex=True)
axes = np.atleast_1d(axes).ravel()
for k, sec in enumerate(all_codes):
    ax = axes[k]
    yy = pers_y[pers_y.sector == lab(sec)].set_index("Year").sort_index()
    base_d = yy.loc[:BREAK - 1, "sd_dSIE"].mean()
    base_r = yy.loc[:BREAK - 1, "sd_ra"].mean()
    ax.plot(yy.index, yy["sd_dSIE"] / base_d, color=COLOR_PRE, lw=1.8, marker="o", ms=3, label="SD of day-to-day change")
    ax.plot(yy.index, yy["sd_ra"] / base_r, color=COLOR_POST, lw=1.8, marker="o", ms=3, label="SD of raw anomaly")
    ax2 = ax.twinx()
    ax2.plot(yy.index, yy["rho1"], color="0.35", lw=1.2, ls="--", label=r"$\rho_1$ of raw anomaly")
    ax2.set_ylim(0, 1); ax2.tick_params(labelsize=7.5, colors="0.35")
    ax2.spines[["top"]].set_visible(False)
    if k % ncol == ncol - 1 or k == len(all_codes) - 1:
        ax2.set_ylabel(r"lag-1 autocorrelation", fontsize=8, color="0.35")
    else:
        ax2.set_yticklabels([]); ax2.tick_params(axis="y", length=0)
    ax.axhline(1, color="k", lw=0.6)
    ax.axvline(BREAK - 0.5, color="0.35", lw=0.9, ls=(0, (2, 2)))
    ax.axvline(2008 - 0.5, color="0.7", lw=0.8, ls=(0, (1, 2)))
    ax.set_title(lab(sec), pad=3, fontproperties=bold)
    ax.text(0.01, 0.97, f"({chr(97 + k)})", transform=ax.transAxes, va="top",
            fontproperties=ch3_style.bold_font_properties(size=9))
    ax.tick_params(labelsize=8)
    ax.spines[["top"]].set_visible(False)
    ax.set_ylim(bottom=0)
    if k % ncol == 0:
        ax.set_ylabel(f"SD relative to\n{START_YEAR}–{BREAK - 1} mean", fontsize=8.5)
for k in range(len(all_codes), len(axes)):
    axes[k].set_visible(False)
handles = [plt.Line2D([0], [0], color=COLOR_PRE, lw=1.8, marker="o", ms=3, label="SD of day-to-day change (left)"),
           plt.Line2D([0], [0], color=COLOR_POST, lw=1.8, marker="o", ms=3, label="SD of raw anomaly (left)"),
           plt.Line2D([0], [0], color="0.35", lw=1.2, ls="--", label=r"lag-1 autocorrelation of raw anomaly (right)"),
           plt.Line2D([0], [0], color="0.35", lw=0.9, ls=(0, (2, 2)), label=str(BREAK)),
           plt.Line2D([0], [0], color="0.7", lw=0.8, ls=(0, (1, 2)), label="2008 SSM/I to SSMIS")]
fig.legend(handles=handles, ncol=5, loc="lower center", frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.join(OUTPUT_DIR, "fig_s04_annual_volatility_persistence.png")
fig.savefig(out, dpi=200)
plt.close(fig)
print(f"\nwrote {out}")
print("wrote t34e_anomaly_persistence.csv, t34e_anomaly_persistence_by_year.csv, "
      "t34f_dsie_cross_sector_corr.csv, t34g_annual_volatility.csv")