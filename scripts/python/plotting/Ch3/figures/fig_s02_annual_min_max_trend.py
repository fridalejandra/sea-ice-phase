#!/usr/bin/env python3
"""
fig_s02_annual_min_max_trend.py -- CANDIDATE Fig. S2 (or S3 -- see note
below): annual summer-minimum and winter-maximum SIE, by sector, with a
linear trend per panel.

Built 2026-09-18 from the render Frida pasted into chat -- that image had
no surviving script, so this is a rebuild against the real daily SIE data,
not a guess at the numbers. It answers a DIFFERENT question than either the
day-of-year timing trend (checked separately, straight from
annual_params.csv's max_doy_raw/min_doy_raw -- WHEN the extrema occur) or
compute_rolling_diagnostics.py's "Diagnostic 3" (how strongly phase/
amplitude track the total anomaly). This one asks: is the SIZE of the
extent at the two extrema itself trending -- is the summer minimum getting
lower, is the winter maximum getting higher or lower.

DIFFERENT FROM THE PASTED REFERENCE IMAGE IN TWO WAYS, both deliberate:
  1. The reference image was titled "1979-2023" and only ran to 2023. That
     matches the same stale ch3_config.YEAR_MAX=2023 bug fixed today (see
     ch3_config.py's 2026-09-18 note) -- the record actually runs to 2025,
     so this version uses the full 1979-2025 span. Expect the slopes and
     significance stars to shift slightly from what was pasted.
  2. The reference image showed 5 sectors (no circumpolar). This version
     adds Circumpolar as a 6th panel row, matching the convention used
     everywhere else in this chapter (Fig. S1, the attribution figures,
     etc.) -- drop it back to 5 if you want to match the original exactly.

DATA SOURCE: the daily SIE_<sector> columns are RAW observed daily extent
(million km^2), not the APAC fit -- annual_params.csv has amplitude and
timing but not the raw extent value at the extrema, so this reads the daily
series directly. Path below (DAILY_RAW_CSV) is a best guess at where this
lives in your real repo (matching the filename you uploaded) -- confirm/fix
if it's actually somewhere else; added to ch3_config.py as a new constant
since nothing pre-existing pointed to it.

CAVEAT carried over from every other script that touches 1979-1987 in this
chapter: sampling is every-other-day (not daily) until mid-1987, so the
true annual min/max can be missed by up to ~1 day's worth of change in
those years -- small, but not zero, and worth a caption footnote if this
figure ships.

Trend test: ordinary least squares (scipy.stats.linregress) of the annual
extremum value against Year, per sector, per extremum. Asterisk marks
p<0.05, uncorrected -- if this goes in as a real supplementary figure
alongside the other multi-sector trend tests in this chapter, it should get
the same 5-sector Bonferroni treatment they get (see
check_component_share_trend.py's convention), not bare p<0.05 per panel.
Not applied here since this is a first draft of the figure, not yet decided
as manuscript-bound.

NUMBERING: Frida called this "maybe supplementary 2 or 3" -- flagging
before this goes further: two OTHER figures are also waiting on an S2/S3
slot with no decision yet -- the old fig07a_circumpolar_2016.png panel
(displaced when Fig. S1 was replaced) and plot_fig7_sectors.R's six-panel
sector-anatomy grid (displaced when Fig. 3 was assigned to
fig03_attribution_by_cycle.png). All three are now competing for the same
one or two supplementary slots -- worth sorting out in one pass rather than
assigning them one at a time and hitting another collision like the fig03_
naming one.

Writes results/ch3/figures/fig_s02_annual_min_max_trend.png and
results/ch3/tables/t_s02_annual_min_max_trend.csv. Paste the console
output back.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import (
    DAILY_RAW_CSV, OUTPUT_DIR, TABLES_DIR, SECTORS, SECTOR_LABELS, SECTOR_COLORS,
    YEAR_MIN, YEAR_MAX, BREAK_YEAR,
)
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure

daily = pd.read_csv(DAILY_RAW_CSV, parse_dates=["time"])
daily["Year"] = daily["time"].dt.year
daily = daily[(daily["Year"] >= YEAR_MIN) & (daily["Year"] <= YEAR_MAX)]

rows = []
fig, axes = plt.subplots(len(SECTORS), 2, figsize=(9, 2.1 * len(SECTORS)), sharex=True)

for i, sector_key in enumerate(SECTORS):
    label = SECTOR_LABELS[sector_key]
    color = SECTOR_COLORS[sector_key]
    g = daily.groupby("Year")[sector_key].agg(["min", "max"]).reset_index()

    for j, (col, title) in enumerate([("min", "Summer minimum"), ("max", "Winter maximum")]):
        ax = axes[i, j]
        years, vals = g["Year"].values, g[col].values
        slope, intercept, r, p, se = linregress(years, vals)
        star = "*" if p < 0.05 else ""
        rows.append(dict(sector=label, extremum=col, n=len(years),
                          slope_Mkm2_per_decade=slope * 10, p_value=p, r_value=r))

        post = years >= BREAK_YEAR
        ax.axvspan(BREAK_YEAR, YEAR_MAX, color="#F8D7DA", alpha=0.6, zorder=0)
        ax.scatter(years, vals, s=18, color=color, zorder=3, edgecolor="none")
        xx = np.array([years.min(), years.max()])
        ax.plot(xx, intercept + slope * xx, color=color, lw=1.3, ls="--", zorder=2)
        ax.text(0.03, 0.92, f"{slope * 10:+.3f} Mkm²/decade{star}",
                transform=ax.transAxes, fontsize=8, va="top", color="#333")

        if i == 0:
            ax.set_title(title, fontsize=11, fontweight="bold")
        if j == 0:
            ax.set_ylabel(f"{label}\nMkm²", fontsize=9)
        if i == len(SECTORS) - 1:
            ax.set_xlabel("Year")

fig.suptitle(f"Annual minimum and maximum SIE by sector ({YEAR_MIN}-{YEAR_MAX})",
             fontweight="bold", y=1.0)
fig.tight_layout()
out = os.path.join(OUTPUT_DIR, "fig_s02_annual_min_max_trend.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)

out_tab = pd.DataFrame(rows)
out_tab_path = os.path.join(TABLES_DIR, "t_s02_annual_min_max_trend.csv")
out_tab.to_csv(out_tab_path, index=False)

pd.set_option("display.width", 160)
print(f"wrote {out}")
print(f"wrote {out_tab_path}")
print("\n(* = p<0.05, uncorrected -- apply 5-sector Bonferroni before citing as significant)")
print(out_tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
