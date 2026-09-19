#!/usr/bin/env python3
"""
fig_08_volatility_trend.py -- REVAMP of fig_08_volatility_raw_vs_residual.py
(Frida's call, 2026-09-19, following Handcock's email): "it doesn't have to
be a post-2016 thing... it can just be a trend."

The original Fig 8 asked "did day-to-day volatility jump at 2016." This asks
the more general question: has day-to-day volatility been drifting over the
WHOLE record, with no boundary assumed -- same spirit as fig_s33_variance_trend.py
(observed timing/amplitude) and check_component_share_trend.py (attribution
shares), applied here to day-to-day volatility.

Unlike the original Fig 8, this reads its numbers directly from
t34c_volatility_gamlss_trend.csv (01_fit_apac.R SECTION 7b) rather than going
through the ch3_numbers.csv ledger -- one less place for staleness to creep
in, since that ledger's "3.4c" section was the thing that went stale after
the 05_volatility_gamlss.R duplication-bug fix. If you want the step-change
(post-2016) version instead of/alongside this, that's t34c_volatility_gamlss_post2016.csv
(SECTION 7), same script, same shape.

Two response variables, same as Section 7/7b, plotted together per sector:
  dSIE            day-to-day tendency, Extent(t) - Extent(t-1), consecutive
                  days only. Removes level/trend/amplitude/phase by
                  construction -- the cleanest proxy for short-lag ("weather"
                  + retrieval) noise.
  residual_apac   Extent - fitted_apac, the leftover after removing the
                  fitted trend+amplitude+phase curve for that year. Retains
                  anything the smooth per-year Beta-warped seasonal cycle
                  doesn't capture -- including slower (multi-day to
                  multi-week) departures that dSIE differencing would net out
                  to ~zero. If these two trend differently, that's itself the
                  finding: it says WHERE the extra dispersion is showing up
                  (adjacent-day noise vs. a growing mismatch between the
                  actual annual-cycle shape and the smooth fit), not just
                  whether it's growing.

Both responses come from a gamlss sigma-trend model with a cyclic day-of-year
spline and a sensor term held fixed, so a season effect or an SSM/I-to-SSMIS
retrieval-noise difference can't masquerade as a real trend (see 01_fit_apac.R
SECTION 7b docstring). Season and sensor fixed; not corrected for multiple
comparisons across the 6 sectors -- apply the same 5-sector Bonferroni
convention used elsewhere in this chapter before citing any one sector.

Inputs
    results/ch3/tables/t34c_volatility_gamlss_trend.csv
Outputs
    results/ch3/figures/fig8_volatility_trend.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR, OUTPUT_DIR, SECTOR_ORDER_BY_LONGITUDE, SECTOR_LABELS
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure

TREND_CSV = os.path.join(TABLES_DIR, "t34c_volatility_gamlss_trend.csv")

vol = pd.read_csv(TREND_CSV)
needed = {"sector", "response", "pct_change_per_decade", "boot_lo_pct", "boot_hi_pct"}
miss = needed - set(vol.columns)
if miss:
    sys.exit(f"{TREND_CSV} lacks {miss}; columns are {sorted(vol.columns)}")

vol["kind"] = np.where(vol["response"] == "dSIE", "raw dSIE",
                        np.where(vol["response"] == "residual_apac", "residual (APAC)", "?"))

sector_order = [SECTOR_LABELS[s] for s in SECTOR_ORDER_BY_LONGITUDE if s in SECTOR_LABELS] + ["Circumpolar"]
vol["sector_label"] = vol["sector"].map(SECTOR_LABELS)
sector_order = [s for s in sector_order if s in vol["sector_label"].unique()]
if not sector_order:
    sys.exit(f"no rows in {TREND_CSV} matched a known sector label; sector values are {sorted(vol['sector'].unique())}")

fig, ax = plt.subplots(figsize=(7.2, 4.8))
y_raw = np.arange(len(sector_order))[::-1] * 2 + 0.32
y_res = np.arange(len(sector_order))[::-1] * 2 - 0.32

def _err_or_flag(row, label):
    """xerr for errorbar(), clipped at 0 with a console warning if the point
    estimate falls outside its own bootstrap CI -- which should never happen
    for a well-behaved bootstrap and is a sign the fit didn't converge on some
    resamples (see 01_fit_apac.R SECTION 7b). Clipping (not skipping) keeps
    the dot visible so the problem is seen, not hidden."""
    lo = row["pct_change_per_decade"] - row["boot_lo_pct"]
    hi = row["boot_hi_pct"] - row["pct_change_per_decade"]
    if lo < 0 or hi < 0:
        print(f"  WARNING: {label} point estimate ({row['pct_change_per_decade']:+.2f}%/decade) "
              f"falls OUTSIDE its own bootstrap CI [{row['boot_lo_pct']:+.2f}, {row['boot_hi_pct']:+.2f}] "
              f"-- do not trust this CI, likely non-converged gamlss fits in the bootstrap. Clipped for display only.")
    return [max(lo, 0)], [max(hi, 0)]

for sector, yr, yres in zip(sector_order, y_raw, y_res):
    row_raw = vol[(vol["sector_label"] == sector) & (vol["kind"] == "raw dSIE")]
    row_res = vol[(vol["sector_label"] == sector) & (vol["kind"] == "residual (APAC)")]
    if row_raw.empty or row_res.empty:
        print(f"  WARNING: missing dSIE or residual_apac row for {sector}; skipping")
        continue
    r_raw = row_raw.iloc[0]
    r_res = row_res.iloc[0]
    xlo, xhi = _err_or_flag(r_raw, f"{sector} dSIE")
    ax.errorbar(r_raw["pct_change_per_decade"], yr, xerr=[xlo, xhi],
                fmt="o", ms=8, color="#C0392B", capsize=4, lw=1.8, zorder=3)
    xlo, xhi = _err_or_flag(r_res, f"{sector} residual_apac")
    ax.errorbar(r_res["pct_change_per_decade"], yres, xerr=[xlo, xhi],
                fmt="o", ms=8, color="#7f8c8d", capsize=4, lw=1.8, zorder=3)

ax.axvline(0, color="k", lw=1.0, ls="--", zorder=1, label="no trend (0%/decade)")
yticks = np.arange(len(sector_order))[::-1] * 2
ax.set_yticks(yticks)
ax.set_yticklabels(sector_order)
ax.set_xlabel("Day-to-day volatility trend, % change per decade (bootstrap 95% CI)")
ax.set_title(
    "Day-to-day SIE volatility, continuous trend (not pre/post-2016)\n"
    "(red = raw dSIE, gray = APAC residual — season+sensor fixed)",
    fontsize=9.5)
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#C0392B", markersize=8, label="raw dSIE"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f8c8d", markersize=8, label="APAC residual"),
    plt.Line2D([0], [0], color="k", lw=1.0, ls="--", label="no trend (0%/decade)"),
]
ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="lower right")
fig.tight_layout()
out8 = os.path.join(OUTPUT_DIR, "fig8_volatility_trend.png")
fig.savefig(out8, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out8}")
for sector in sector_order:
    r_raw = vol[(vol["sector_label"] == sector) & (vol["kind"] == "raw dSIE")].iloc[0]
    r_res = vol[(vol["sector_label"] == sector) & (vol["kind"] == "residual (APAC)")].iloc[0]
    print(f"  {sector:16s} raw dSIE {r_raw['pct_change_per_decade']:+.2f}%/decade "
          f"[{r_raw['boot_lo_pct']:+.2f},{r_raw['boot_hi_pct']:+.2f}]  "
          f"residual {r_res['pct_change_per_decade']:+.2f}%/decade "
          f"[{r_res['boot_lo_pct']:+.2f},{r_res['boot_hi_pct']:+.2f}]")