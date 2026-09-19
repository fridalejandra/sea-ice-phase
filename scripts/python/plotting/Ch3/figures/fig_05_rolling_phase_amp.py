"""
fig03_rolling_phase_amp_corr.py
===============================
Rolling correlation between observed timing and amplitude anomalies, by sector.

This is the chapter's answer to Q2: have timing and amplitude STAYED
independent of one another? A pooled whole-record correlation cannot answer
that — a relationship that is positive before 2016 and negative after averages
to roughly zero, which is exactly the case this figure exists to detect. So
the statistic is a rolling window, and the inference is the pre/post split.

Statistic: TRAILING 10-year Spearman rho, plotted at the window's last year,
identical to t33_phase_amp_rolling10.csv from ch3_stats.py. Spearman is the
chapter's convention for these (ch3_data.rolling_corr's default); Pearson
values are NOT comparable and should not be mixed into the same sentence.

Both timing metrics are shown. The observed minimum date is measured two to
four times more precisely than the maximum (Fig. S1), so it carries more power;
the maximum is the chapter's primary variable and the one §3.5 pairs use.

The shaded band is the p=0.05 threshold for the FULL record and is far too
generous for a 10-year window — orientation only. The inference in the text
comes from the pooled whole-era split and leave-one-year-out (ch3_numbers.md
§3.3), not from the curve.

Outputs
    results/ch3/figures/fig03_rolling_phase_amp_corr.png
    results/ch3/tables/t33_phase_amp_era_split.csv
"""
import os
import numpy as np
import pandas as pd

import ch3_data as D
from ch3_config import SECTORS, SECTOR_LABELS, SECTOR_COLORS, TABLES_DIR, \
    ROLL_SHORT, BREAK_YEAR
from ch3_plot import sector_grid, mark_break, sig_band, year_axis, \
    panel_letters, save

PAIRS = [
    ("min date", "min_doy_raw_anom", "-",  1.0),
    ("max date", "max_doy_raw_anom", "--", 0.65),
]


def r_critical(n, alpha=0.05):
    """Two-tailed critical Pearson r. Falls back to the n=47 value if scipy
    is unavailable, rather than silently using a threshold for the wrong n."""
    if n < 4:
        return np.nan
    try:
        from scipy import stats
        t = stats.t.ppf(1 - alpha / 2, n - 2)
        return float(t / np.sqrt(t ** 2 + n - 2))
    except ImportError:
        print("  WARNING: scipy unavailable — using the n=47 threshold (0.288).")
        return 0.288


print("fig03 — rolling r(timing, amplitude)")
annual = D.load_annual(period="FULL")
D.summary(annual=annual)

YR_MIN, YR_MAX = int(annual["Year"].min()), int(annual["Year"].max())
ERA_PRE  = f"{YR_MIN}–{BREAK_YEAR - 1}"
ERA_POST = f"{BREAK_YEAR}–{YR_MAX}"
N_FULL = int(annual.groupby("sector")["Year"].nunique().max())
RCRIT = r_critical(N_FULL)
print(f"  eras: {ERA_PRE} / {ERA_POST}   n={N_FULL}   r_crit={RCRIT:.3f}")

roll = {lab: D.rolling_corr(annual, col, "amplitude_raw_anom",
                            ROLL_SHORT, method="spearman")
        for lab, col, _, _ in PAIRS}

fig, axmap = sector_grid(2, 3, figsize=(11.0, 6.5), sharey=True)
records = []

for sec in SECTORS:
    ax = axmap[sec]
    c = SECTOR_COLORS[sec]
    sig_band(ax, RCRIT)

    for lab, col, ls, alpha in PAIRS:
        g = roll[lab]
        g = g[g["sector"] == sec].sort_values("Year")
        ax.plot(g["Year"], g["value"], color=c, lw=2.0, ls=ls, alpha=alpha,
                label=f"r({lab}, amplitude)")

    mark_break(ax)
    year_axis(ax, YR_MIN, YR_MAX)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(f"{ROLL_SHORT}-yr trailing Spearman ρ", fontsize=9)

    # whole-era split — this, not the curve, is what the text quotes
    a = annual[annual["sector"] == sec]
    pre, post = a[a["Year"] < BREAK_YEAR], a[a["Year"] >= BREAK_YEAR]
    lines = []
    for lab, col, _, _ in PAIRS:
        r_pre  = pre[col].corr(pre["amplitude_raw_anom"],   method="spearman")
        r_post = post[col].corr(post["amplitude_raw_anom"], method="spearman")
        lines.append(f"{lab}–amp   {ERA_PRE} {r_pre:+.2f}    {ERA_POST} {r_post:+.2f}")
        records += [
            dict(sector=sec, pair=f"{lab}-amplitude", era=ERA_PRE,
                 spearman_rho=r_pre, n=len(pre)),
            dict(sector=sec, pair=f"{lab}-amplitude", era=ERA_POST,
                 spearman_rho=r_post, n=len(post)),
        ]
    ax.text(0.02, 0.04, "\n".join(lines), transform=ax.transAxes,
            fontsize=7, color="#616161", linespacing=1.5)

axmap[SECTORS[0]].legend(fontsize=7.5, loc="upper left", frameon=False)
panel_letters(list(axmap.values()))
fig.suptitle("Rolling correlation between observed timing and amplitude anomalies",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "fig05_rolling_phase_amp_corr.png", sync=False)

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