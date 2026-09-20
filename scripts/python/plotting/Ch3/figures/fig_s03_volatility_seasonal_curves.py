#!/usr/bin/env python3
"""
fig_s03_volatility_seasonal_curves.py -- Supplement to Sect. 3.4.

Seasonal structure of day-to-day volatility, pre-2016 vs 2016+, per sector:
the fitted sigma (standard deviation) from the separate-period gamlss fits
in 01_fit_apac.R SECTION 7 / 07_volatility_only.R, as a function of
calendar day of year. This is the direct analogue of Handcock and Raphael
(2020) Fig. 3 (volatility over the day of the cycle, by sensor era), with
the split by 2016 instead of by sensor -- the sensor term is held at SSMIS
in both periods so the two curves are comparable.

Two responses, two figures (their sigma scales differ by an order of
magnitude, so they don't share axes):
    fig_s03a_volatility_seasonal_dSIE.png         day-to-day tendency
    fig_s03b_volatility_seasonal_raw_anomaly.png  raw APAC anomaly

Inputs
    results/ch3/tables/t34c_volatility_seasonal_curves.csv
        columns: sector, response, DOY, sigma_pre, sigma_post, season
Outputs
    results/ch3/figures/fig_s03a_volatility_seasonal_dSIE.png
    results/ch3/figures/fig_s03b_volatility_seasonal_raw_anomaly.png

Note the x axis is CALENDAR day of year (1 = 1 Jan), not day of the cycle
as in Handcock and Raphael (their day 0 = Julian day 50). The minimum is
near day 50 and the maximum near day 260 here.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR, OUTPUT_DIR, SECTOR_ORDER_BY_LONGITUDE, SECTOR_LABELS
import ch3_style  # font + spines; bold_font_properties() for real bold

CSV = os.path.join(TABLES_DIR, "t34c_volatility_seasonal_curves.csv")
cv = pd.read_csv(CSV)
needed = {"sector", "response", "DOY", "sigma_pre", "sigma_post"}
miss = needed - set(cv.columns)
if miss:
    sys.exit(f"{CSV} lacks {miss}; columns are {sorted(cv.columns)}")

# dataviz palette slots 1-2 (validated adjacent pair): blue = pre-2016, orange = 2016+
COLOR_PRE = "#2a78d6"
COLOR_POST = "#eb6834"
BREAK = 2016
END_LABEL = os.environ.get("END_LABEL", "2023")   # last year of the post period, for the label only

sector_codes = [s for s in SECTOR_ORDER_BY_LONGITUDE if s in SECTOR_LABELS]
circ = [s for s in cv["sector"].unique() if "circumpolar" in s.lower()]
sector_codes = [s for s in sector_codes if s in set(cv["sector"])] + circ
if not sector_codes:
    sys.exit(f"no known sectors in {CSV}; sector values are {sorted(cv['sector'].unique())}")

MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

bold = ch3_style.bold_font_properties(size=10.5)


def draw(response, ylabel, out_name):
    d = cv[cv["response"] == response]
    if d.empty:
        print(f"  WARNING: no rows for response == {response!r}; skipping {out_name}")
        return
    ncol = 3
    nrow = int(np.ceil(len(sector_codes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.9 * nrow),
                             sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    # Common y scale for the five sectors; circumpolar (their sum, several
    # times larger) gets its own so it doesn't squash the others.
    non_circ = [s for s in sector_codes if "circumpolar" not in s.lower()]
    dd = d[d["sector"].isin(non_circ)]
    y_top = 1.05 * max(dd["sigma_pre"].max(), dd["sigma_post"].max())
    INK = "0.35"
    label_pre, label_post = f"1988–{BREAK - 1}", f"{BREAK}–{END_LABEL}"
    for k, sec in enumerate(sector_codes):
        ax = axes[k]
        a = d[d["sector"] == sec].sort_values("DOY")
        ax.plot(a["DOY"], a["sigma_pre"], color=COLOR_PRE, lw=2.2)
        ax.plot(a["DOY"], a["sigma_post"], color=COLOR_POST, lw=2.2)
        ax.set_title(SECTOR_LABELS.get(sec, sec), pad=4, fontproperties=bold)
        ax.text(0.02, 0.96, f"({chr(97 + k)})", transform=ax.transAxes, va="top", color=INK,
                fontproperties=ch3_style.bold_font_properties(size=9))
        ax.set_xticks(MONTH_STARTS)
        ax.set_xticklabels(MONTH_LABELS)
        ax.set_xlim(1, 365)
        ax.tick_params(labelsize=8, colors=INK, length=3)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if "circumpolar" not in sec.lower():
            ax.set_ylim(0, y_top)
        else:
            ax.set_ylim(bottom=0)
        if k % ncol == 0 or "circumpolar" in sec.lower():
            ax.set_ylabel(ylabel, fontsize=9, color=INK)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if k == 0:  # direct labels, once, in the lower right (Feb-Mar minimum leaves it empty)
            ax.text(0.98, 0.20, label_pre, transform=ax.transAxes, ha="right", va="bottom",
                    color=COLOR_PRE, fontproperties=ch3_style.bold_font_properties(size=9))
            ax.text(0.98, 0.08, label_post, transform=ax.transAxes, ha="right", va="bottom",
                    color=COLOR_POST, fontproperties=ch3_style.bold_font_properties(size=9))
    for k in range(len(sector_codes), len(axes)):
        axes[k].set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
    # console summary: full-year and seasonal post/pre ratios, same as SECTION 7 prints
    for sec in sector_codes:
        a = d[d["sector"] == sec]
        r = a["sigma_post"].mean() / a["sigma_pre"].mean()
        print(f"  {SECTOR_LABELS.get(sec, sec):24s} post/pre sigma ratio (full year) {r:.2f}")


draw("dSIE", "standard deviation of\nday-to-day change (10$^6$ km$^2$)",
     "fig_s03a_volatility_seasonal_dSIE.png")
draw("residual_apac", "standard deviation of\nraw anomaly (10$^6$ km$^2$)",
     "fig_s03b_volatility_seasonal_raw_anomaly.png")