#!/usr/bin/env python3
"""
fig_07_rolling_phase_amp.py -- Fig. 7: rolling correlation between the observed
timing anomalies and the observed amplitude anomaly, by sector (Sect. 3.4).

Statistic: TRAILING 10-year Spearman rho, plotted at the window's last year,
identical to t33_phase_amp_rolling10.csv from ch3_stats.py. The whole-era
values printed in each panel (1979-2015 and 2016-2025) are what the text
quotes; the curve is orientation. The shaded band is the p = 0.05 threshold
for the FULL record and is far too generous for a 10-year window.

Layout (2026-09-21 facelift): no figure title; panel titles "(a) Weddell" in
bold black with no sector colours (nothing else in the chapter colours by
sector); one figure-level legend; the day of MAXIMUM -- the chapter's primary
timing variable -- is the solid line, the minimum the dashed one; the era
values sit in their own band under the curves, bold where p < 0.05 for that
era's own n (37 and 10 years).

Reads   annual_params.csv (period == FULL, via ch3_data.load_annual)
Writes  results/ch3/figures/fig07_rolling_phase_amp.png
        results/ch3/tables/t33_phase_amp_era_split.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ch3_data as D
import ch3_style
from ch3_config import SECTORS, SECTOR_LABELS, TABLES_DIR, ROLL_SHORT, BREAK_YEAR
from ch3_plot import sector_grid, sig_band, year_axis, save

INK = "0.35"
# (label, column, line style, colour, width)
PAIRS = [
    ("day of maximum", "max_doy_raw_anom", "-",  "#2a78d6", 2.2),
    ("day of minimum", "min_doy_raw_anom", "--", "0.45",    1.8),
]
TITLE_LABEL = {"SIE_Amundsen_Bellingshausen": "Amundsen-Bellingshausen",
               "SIE_circumpolar": "Circumpolar total"}


def r_critical(n, alpha=0.05):
    """Two-tailed critical r for the full record (falls back to n = 47 value)."""
    if n < 4:
        return np.nan
    try:
        from scipy import stats
        t = stats.t.ppf(1 - alpha / 2, n - 2)
        return float(t / np.sqrt(t ** 2 + n - 2))
    except ImportError:
        print("  WARNING: scipy unavailable -- using the n=47 threshold (0.288).")
        return 0.288


print("fig07 -- rolling r(timing, amplitude)")
annual = D.load_annual(period="FULL")
D.summary(annual=annual)
dup = annual.duplicated(["sector", "Year"], keep=False)
if dup.any():
    raise SystemExit(f"annual_params has {int(dup.sum())} rows sharing a (sector, Year) after the FULL filter")

YR_MIN, YR_MAX = int(annual["Year"].min()), int(annual["Year"].max())
ERA_PRE = f"{YR_MIN}–{BREAK_YEAR - 1}"
ERA_POST = f"{BREAK_YEAR}–{YR_MAX}"
N_FULL = int(annual.groupby("sector")["Year"].nunique().max())
RCRIT = r_critical(N_FULL)
N_PRE = int(annual[annual["Year"] < BREAK_YEAR].groupby("sector")["Year"].nunique().max())
N_POST = int(annual[annual["Year"] >= BREAK_YEAR].groupby("sector")["Year"].nunique().max())
RC_PRE, RC_POST = r_critical(N_PRE), r_critical(N_POST)
print(f"  eras: {ERA_PRE} (n={N_PRE}, r_crit={RC_PRE:.2f}) / {ERA_POST} (n={N_POST}, r_crit={RC_POST:.2f});"
      f"   full record n={N_FULL}, band r_crit={RCRIT:.3f}")

roll = {lab: D.rolling_corr(annual, col, "amplitude_raw_anom", ROLL_SHORT, method="spearman")
        for lab, col, _, _, _ in PAIRS}

fig, axmap = sector_grid(2, 3, figsize=(11.0, 7.2), sharey=True)
title_font = ch3_style.bold_font_properties(size=11)
records = []
axes = list(axmap.values())
ncol = 3

for k, sec in enumerate(SECTORS):
    ax = axmap[sec]
    sig_band(ax, RCRIT)
    for lab, col, ls, color, lw in PAIRS:
        g = roll[lab]
        g = g[g["sector"] == sec].sort_values("Year")
        ax.plot(g["Year"], g["value"], color=color, lw=lw, ls=ls, label=lab)
    # break marker only over the curve band, so it does not cut through the table
    ax.vlines(BREAK_YEAR, -1, 1, color="#D4537E", lw=1.4, ls="--", alpha=0.9, zorder=2)
    year_axis(ax, YR_MIN, YR_MAX)
    # the curves live in [-1, 1]; the band below -1 holds the era table
    ax.set_ylim(-1.55, 1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.spines["left"].set_bounds(-1, 1)

    # panel title: letter + sector, bold, black, left-aligned (no sector colour);
    # sector_grid() set a coloured centred title -- clear it
    ax.set_title("")
    ax.set_title(f"({chr(97 + k)})  {TITLE_LABEL.get(sec, SECTOR_LABELS[sec])}",
                 loc="left", fontproperties=title_font, color="0.1", pad=8)
    ax.set_xlabel("year" if k // ncol == 1 else "", color=INK, fontsize=9.5)
    ax.set_ylabel(f"{ROLL_SHORT}-yr trailing Spearman ρ" if k % ncol == 0 else "",
                  color=INK, fontsize=9.5)
    ax.tick_params(colors=INK, labelsize=9, length=3)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)

    # whole-era split: what the text quotes
    a = annual[annual["sector"] == sec]
    pre, post = a[a["Year"] < BREAK_YEAR], a[a["Year"] >= BREAK_YEAR]
    rows = []
    for lab, col, _, _, _ in PAIRS:
        r_pre = pre[col].corr(pre["amplitude_raw_anom"], method="spearman")
        r_post = post[col].corr(post["amplitude_raw_anom"], method="spearman")
        rows.append((lab, r_pre, r_post))
        records += [dict(sector=sec, pair=f"{lab}-amplitude", era=ERA_PRE, spearman_rho=r_pre, n=len(pre)),
                    dict(sector=sec, pair=f"{lab}-amplitude", era=ERA_POST, spearman_rho=r_post, n=len(post))]
    # small table in the band below the curves: x in axes fraction, y in data units
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    small = ch3_style.bold_font_properties(size=8.5)
    ax.text(0.58, -1.12, ERA_PRE, transform=tr, ha="center", va="top", fontsize=8.5, color=INK)
    ax.text(0.86, -1.12, ERA_POST, transform=tr, ha="center", va="top", fontsize=8.5, color=INK)
    for j, (lab, rp, rq) in enumerate(rows):
        y = -1.28 - 0.14 * j
        ax.text(0.03, y, lab, transform=tr, ha="left", va="top", fontsize=8.5, color="0.15")
        # bold = p < 0.05 for THAT era's n (37 and 10 years), not the full-record band
        for xf, val, rc in ((0.58, rp, RC_PRE), (0.86, rq, RC_POST)):
            txt = "0.00" if abs(val) < 0.005 else f"{val:+.2f}"
            ax.text(xf, y, txt, transform=tr, ha="center", va="top", fontsize=8.5,
                    color="0.1", fontproperties=small if abs(val) >= rc else None)

# one legend for the whole figure, above the panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "fig07_rolling_phase_amp.png", sync=False)

tab = pd.DataFrame(records)
os.makedirs(TABLES_DIR, exist_ok=True)
tab.to_csv(os.path.join(TABLES_DIR, "t33_phase_amp_era_split.csv"), index=False)

print("\n" + "=" * 70)
print(f"Whole-era Spearman rho  (|rho| > {RCRIT:.2f} is p<0.05 at n={N_FULL})")
print("=" * 70)
for pair, g in tab.groupby("pair", sort=False):
    print(f"\n{pair}")
    w = g.pivot(index="sector", columns="era", values="spearman_rho")
    w = w.reindex(SECTORS).rename(index=SECTOR_LABELS)
    print(w[[ERA_PRE, ERA_POST]].round(3).to_string())
print("\nwrote t33_phase_amp_era_split.csv")