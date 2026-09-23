#!/usr/bin/env python3
"""
fig_09_raw_anomaly_persistence.py -- Fig. 9, Sect. 3.5 (restructured 2026-09-19
around the raw anomaly rather than volatility).

The raw APAC anomaly -- observed SIE minus the fitted trend-, amplitude- and
phase-adjusted cycle -- is what the decomposition leaves behind. This figure
characterises its PERSISTENCE: the autocorrelation function out to 60 days,
per sector, for 1988-2015 and 2016-2023, with the e-folding time marked.
Lag-1 autocorrelations of ~0.97-0.985 and e-folding times of two to three
weeks say the raw anomaly is not day-to-day noise but a departure that
persists for weeks: the ice edge sits above or below its adjusted cycle for
a fortnight or more at a time. Overlapping curves say that persistence did
not change after 2016.

Record ends 2023 by default (END_YEAR): the day-to-day variance of the
recorded extent rises 2-4x in 2024 and 3-8x in 2025 with no change in the
anomaly and synchronously across sectors (check_anomaly_persistence.py,
fig_s05) -- the signature of a change in retrieval or processing, not in
the ice -- so those two years are excluded pending a provenance check of
merged_bootstrap_SH_latest.nc.

Autocorrelation at lag L uses only pairs of days exactly L days apart, so
the every-other-day era and any gaps do not bias it. The anomaly is
deseasonalised by its 1988-2023 day-of-year mean first (which should be
near zero anyway). E-folding time is the lag at which the ACF first drops
below 1/e, linearly interpolated. Its 95 % CI is a bootstrap that resamples
whole years with replacement (N_BOOT replicates).

Inputs   daily_fitted.csv via ch3_data.load_daily(period="FULL")
Outputs  results/ch3/figures/fig06_raw_anomaly_persistence.png
         results/ch3/tables/t34h_raw_anomaly_persistence.csv
             sector, period, n_years, sd (10^6 km^2), rho1, rho7, rho14,
             efold_days, efold_lo, efold_hi
         (console) cross-sector correlation of the deseasonalised raw
             anomaly LEVEL, pre and post, five sectors -- the dipole check
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ch3_data as D
from ch3_config import TABLES_DIR, OUTPUT_DIR, SECTORS, SECTOR_LABELS
import ch3_style

# panel titles: chapter order (Figs 3-8) and full names, no abbreviations
TITLE_LABEL = {"SIE_Amundsen_Bellingshausen": "Amundsen-Bellingshausen",
               "SIE_circumpolar": "Circumpolar total"}
FIG_NAME = os.environ.get("FIG_NAME", "fig09_raw_anomaly_persistence.png")

START_YEAR = 1988
END_YEAR = int(os.environ.get("END_YEAR", "2023"))       # 2025 to include the suspect years
PRE_END = int(os.environ.get("PRE_END", "2015"))         # last year of the "before" period
POST_START = int(os.environ.get("POST_START", "2016"))   # first year of the "after" period;
#   the decline began ~Sept 2016, so calendar 2016 is mostly pre-event: run with
#   POST_START=2017 as well and check the numbers don't hinge on where 2016 goes.
MAX_LAG = 60
N_BOOT = int(os.environ.get("NBOOT", "1000"))
COLOR_PRE, COLOR_POST = "#2a78d6", "#eb6834"
PERIODS = [(f"{START_YEAR}–{PRE_END}", lambda y: (y >= START_YEAR) & (y <= PRE_END), COLOR_PRE),
           (f"{POST_START}–{END_YEAR}", lambda y: (y >= POST_START) & (y <= END_YEAR), COLOR_POST)]

try:
    from scipy import stats as _st
    def welch_p(a, b):
        return float(_st.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)
except Exception:
    def welch_p(a, b):
        return float("nan")

print(f"fig09 -- raw anomaly persistence, {START_YEAR}-{END_YEAR}")
d = D.load_daily(period="FULL")
d = d[(d["Year"] >= START_YEAR) & (d["Year"] <= END_YEAR)].copy()
d["Date"] = pd.to_datetime(d["Date"])
d = d.sort_values(["sector", "Date"]).reset_index(drop=True)
d["ra"] = d["residual_apac"] - d.groupby(["sector", "DOY"])["residual_apac"].transform("mean")

present = set(d["sector"])
all_codes = [s for s in SECTORS if s in present]                    # chapter order, circumpolar last
sector_codes = [s for s in all_codes if "circumpolar" not in s.lower()]
lab = lambda s: SECTOR_LABELS.get(s, s)


# ESTIMATOR (changed 2026-09-19). Earlier versions pooled all days of a
# period into one ACF and bootstrapped that by resampling years. On the
# real data the bootstrap distribution of the pooled e-folding time sat
# systematically above the full-sample value (Weddell 1988-2015: point
# 22.1 d, percentile CI [22.1, 29.1]; the reflected interval then put the
# point on the other bound), so neither interval was usable. The estimator
# here is instead the MEAN of the per-year ACFs: each year's ACF from its
# own days (exact-lag pairs within the year, so gaps and year boundaries
# never pair), averaged across the period's years. That is what the prose
# claims to measure -- how long a departure persists within a season -- and
# the bootstrap of a mean over years is well-behaved. The per-year e-folding
# times also give a spread and a Welch test between periods.


def yearly_acf(a, max_lag):
    """Rows = years, cols = lags 0..max_lag; NaN where a year has < 30 pairs."""
    out = []
    years = []
    for y, b in a.groupby("Year"):
        x = np.full(367, np.nan)
        x[b["DOY"].values.astype(int)] = b["ra"].values
        row = np.full(max_lag + 1, np.nan); row[0] = 1.0
        for L in range(1, max_lag + 1):
            u, v = x[:-L], x[L:]
            ok = np.isfinite(u) & np.isfinite(v)
            if ok.sum() > 30:
                row[L] = np.corrcoef(u[ok], v[ok])[0, 1]
        out.append(row); years.append(int(y))
    return np.array(years), np.array(out)


def efold(acf):
    thr = 1 / np.e
    for L in range(1, len(acf)):
        if np.isnan(acf[L]):
            return np.nan
        if acf[L] < thr:
            return (L - 1) + (acf[L - 1] - thr) / (acf[L - 1] - acf[L])
    return np.nan


rows, curves, per_year = [], {}, {}
rng = np.random.default_rng(1)
for sec in all_codes:
    a = d[d["sector"] == sec]
    for pname, sel, _ in PERIODS:
        b = a[sel(a["Year"])]
        yrs, M = yearly_acf(b, MAX_LAG)
        acf = np.nanmean(M, axis=0)
        curves[(sec, pname)] = acf
        ef = efold(acf)
        ef_by_year = np.array([efold(r) for r in M])
        per_year[(sec, pname)] = ef_by_year
        boots = np.array([efold(np.nanmean(M[rng.integers(0, len(yrs), len(yrs))], axis=0))
                          for _ in range(N_BOOT)])
        boots = boots[np.isfinite(boots)]
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 10 else (np.nan, np.nan))
        rows.append(dict(sector=lab(sec), period=pname, n_years=len(yrs),
                         sd=b["ra"].std(ddof=1),
                         rho1=acf[1], rho7=acf[7], rho14=acf[14],
                         efold_days=ef, efold_lo=lo, efold_hi=hi,
                         efold_by_year_mean=np.nanmean(ef_by_year),
                         efold_by_year_se=np.nanstd(ef_by_year, ddof=1) / np.sqrt(np.isfinite(ef_by_year).sum()),
                         n_boot=len(boots)))
        print(f"  {lab(sec):24s} {pname}: sd {rows[-1]['sd']:.3f}  rho1 {acf[1]:.3f}  rho7 {acf[7]:.2f}  "
              f"rho14 {acf[14]:.2f}  e-fold {ef:5.1f} d [{lo:4.1f}, {hi:4.1f}]  "
              f"(per-year mean {np.nanmean(ef_by_year):.1f} ± {rows[-1]['efold_by_year_se']:.1f})")
    p = welch_p(per_year[(sec, PERIODS[0][0])], per_year[(sec, PERIODS[1][0])])
    print(f"  {'':24s} per-year e-folding, {PERIODS[0][0]} vs {PERIODS[1][0]}: Welch p = {p:.3f}")
    rows[-1]["welch_p_vs_pre"] = p

tab = pd.DataFrame(rows)
os.makedirs(TABLES_DIR, exist_ok=True)
tab.to_csv(os.path.join(TABLES_DIR, "t34h_raw_anomaly_persistence.csv"), index=False)

# ── cross-sector structure of the anomaly level (dipole check) ──────────────
print("\n== cross-sector correlation of the deseasonalised raw anomaly, five sectors ==")
W = (d[d["sector"].isin(sector_codes)]
     .pivot_table(index="Date", columns="sector", values="ra")).dropna()
W.columns = [lab(c) for c in W.columns]
W_year = W.index.year


def var_ratio(w):
    return w.sum(axis=1).var(ddof=1) / w.var(ddof=1).sum()


ratio_boot = {}
comp_rows = []
for pname, sel, _ in PERIODS:
    w = W[sel(W_year)]
    yrs = np.unique(W_year[sel(W_year)])
    C = w.corr(); off = C.values[np.triu_indices_from(C.values, k=1)]
    ratio = var_ratio(w)
    by_year = {y: w[W_year[sel(W_year)] == y] for y in yrs}
    boots = np.array([var_ratio(pd.concat([by_year[y] for y in rng.choice(yrs, len(yrs), replace=True)]))
                      for _ in range(N_BOOT)])
    ratio_boot[pname] = boots
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n  {pname} (n = {len(w)} days, {len(yrs)} years): mean pairwise r = {off.mean():+.3f}, "
          f"var(sum)/sum(var) = {ratio:.2f} [{lo:.2f}, {hi:.2f}]  (<1 = compensating)")
    print(C.round(2).to_string())
    comp_rows.append(dict(period=pname, n_days=len(w), n_years=len(yrs), mean_pairwise_r=off.mean(),
                          var_ratio=ratio, var_ratio_lo=lo, var_ratio_hi=hi))
diff = ratio_boot[PERIODS[1][0]] - ratio_boot[PERIODS[0][0]]
d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
print(f"\n  change in var(sum)/sum(var), {PERIODS[1][0]} minus {PERIODS[0][0]}: "
      f"{comp_rows[1]['var_ratio'] - comp_rows[0]['var_ratio']:+.2f} [{d_lo:+.2f}, {d_hi:+.2f}]  "
      f"(bootstrap over years; interval excluding 0 = compensation changed)")
comp_rows.append(dict(period="difference", var_ratio=comp_rows[1]["var_ratio"] - comp_rows[0]["var_ratio"],
                      var_ratio_lo=d_lo, var_ratio_hi=d_hi))
pd.DataFrame(comp_rows).to_csv(os.path.join(TABLES_DIR, "t34i_raw_anomaly_compensation.csv"), index=False)

# ── figure: 2 x 3, ACF pre vs post per sector ───────────────────────────────
# Style: direct labels in panel (a) instead of a legend, grey ink for
# everything that isn't data, no tick clutter, bold sector titles.
INK = "0.35"
bold = ch3_style.bold_font_properties(size=10.5)
ncol = 3
nrow = int(np.ceil(len(all_codes) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.9 * nrow), sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
lags = np.arange(MAX_LAG + 1)
for k, sec in enumerate(all_codes):
    ax = axes[k]
    ax.axhline(1 / np.e, color="0.75", lw=0.8, ls=(0, (2, 2)), zorder=1)
    for pname, _, col in PERIODS:
        acf = curves[(sec, pname)]
        ax.plot(lags, acf, color=col, lw=2.2, zorder=3)
        ef = tab[(tab.sector == lab(sec)) & (tab.period == pname)]["efold_days"].iloc[0]
        if np.isfinite(ef):
            ax.plot([ef, ef], [-0.1, 1 / np.e], color=col, lw=1.0, ls=":", zorder=2)
    # "(a)  Weddell": bold, black, left-aligned, as in Figs 3-8
    ax.set_title(f"({chr(97 + k)})  {TITLE_LABEL.get(sec, lab(sec))}", loc="left", pad=6,
                 fontproperties=ch3_style.bold_font_properties(size=11), color="0.1")
    ax.set_xlim(0, MAX_LAG); ax.set_ylim(-0.1, 1)
    ax.set_xticks([0, 20, 40, 60]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=8, colors=INK, length=3)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(0.8)
    ax.spines[["top", "right"]].set_visible(False)
    if k % ncol == 0:
        ax.set_ylabel("autocorrelation of\nraw anomaly", fontsize=9, color=INK)
    if k >= len(all_codes) - ncol:
        ax.set_xlabel("lag (days)", fontsize=9, color=INK)
    if k == 0:  # direct labels, once
        for (pname, _, col), y in zip(PERIODS, (0.80, 0.66)):
            ax.text(0.98, y, pname, transform=ax.transAxes, ha="right", va="center",
                    color=col, fontproperties=ch3_style.bold_font_properties(size=9))
        ax.text(MAX_LAG - 1, 1 / np.e + 0.03, "1/e", ha="right", va="bottom", color="0.55", fontsize=8)
        ax.text(0.98, 0.58, "dotted: e-folding time", transform=ax.transAxes, ha="right",
                va="center", color=INK, fontsize=8)
for k in range(len(all_codes), len(axes)):
    axes[k].set_visible(False)
fig.tight_layout()
out = os.path.join(OUTPUT_DIR, FIG_NAME)
fig.savefig(out, dpi=200)
plt.close(fig)
print(f"\nwrote {out}\nwrote t34h_raw_anomaly_persistence.csv")