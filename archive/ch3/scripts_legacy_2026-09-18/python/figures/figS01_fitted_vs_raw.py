"""
figS1_timing_recovery.py — Fig. S1, referenced from Sect. 2.2.3.
=================================================================
Replaces figS01_fitted_vs_observed.py and validation_phase_amplitude.py.
Delete both once this runs; three scripts for one supplementary figure is
how the version confusion started.

The point of the figure is the CONTRAST BETWEEN THE TWO ROWS, not either
row alone:

  row 1  day of MAXIMUM   fitted vs observed — poor agreement, and the
                          fitted spread is a fraction of the observed
  row 2  day of MINIMUM   fitted vs observed — near-perfect agreement

Same model, same years, same sectors. The only difference is that the
minimum is a sharp extremum and the September maximum is a plateau on
which the day of the highest value is decided by small fluctuations.
So this is not model failure: a flat maximum carries no identifiable
timing signal. That is the claim Sect. 2.2.3 makes and this is its
evidence.

Plotted as anomalies (departures from the record mean), matching the
chapter's convention everywhere else. Correlation and SD are identical
either way; anomalies put zero at the centre, which makes the vertical
compression in row 1 read at a glance.

Amplitude is deliberately NOT a panel here. Fitted and observed amplitude
agree at r = 1.00 by construction — the adjusted curve is scaled to those
same observed extremes — so a scatter of it is a diagonal line carrying no
information. Sect. 2.2.2 states it; it does not need a picture.

Outputs
    results/ch3/figures/figS1_timing_recovery.png
    results/ch3/tables/tS1_timing_recovery.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ch3_data as D
from ch3_config import SECTORS, SECTOR_LABELS, SECTOR_COLORS, TABLES_DIR
from ch3_plot import save

# (row label, observed column, fitted column)
ROWS = [
    ("Day of maximum", "max_doy_raw_anom", "max_doy_anom"),
    ("Day of minimum", "min_doy_raw_anom", "min_doy_anom"),
]

print("figS1 — timing recovery")
annual = D.load_annual(period="FULL")
D.summary(annual=annual)

need = {c for _, o, f in ROWS for c in (o, f)}
missing = sorted(need - set(annual.columns))
if missing:
    raise SystemExit(
        f"annual_params.csv is missing {missing}.\n"
        f"Columns present: {sorted(annual.columns)}\n"
        "Rerun R/ch3/01_fit_apac.R."
    )

fig, axes = plt.subplots(len(ROWS), len(SECTORS),
                         figsize=(2.25 * len(SECTORS), 2.55 * len(ROWS)),
                         squeeze=False)

records = []
letters = iter("abcdefghijklmnopqrstuvwxyz")

for i, (row_label, obs_col, fit_col) in enumerate(ROWS):
    for j, sec in enumerate(SECTORS):
        ax = axes[i][j]
        g = annual[annual["sector"] == sec][[obs_col, fit_col]].dropna()
        x, y = g[obs_col].values, g[fit_col].values
        col = SECTOR_COLORS[sec]

        # common square limits so the 1:1 line is a true 45 degrees and the
        # vertical squash in row 1 is not an artefact of unequal axes
        lim = np.nanmax(np.abs(np.concatenate([x, y]))) * 1.15
        ax.plot([-lim, lim], [-lim, lim], color="#BDBDBD", lw=0.9, zorder=1)
        ax.axhline(0, color="#EEEEEE", lw=0.8, zorder=0)
        ax.axvline(0, color="#EEEEEE", lw=0.8, zorder=0)
        ax.scatter(x, y, s=20, color=col, alpha=0.8,
                   edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")

        r = np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan
        sd_obs, sd_fit = x.std(ddof=1), y.std(ddof=1)
        records.append(dict(sector=sec, metric=row_label, r=r, n=len(x),
                            sd_observed=sd_obs, sd_fitted=sd_fit,
                            sd_ratio=sd_fit / sd_obs if sd_obs else np.nan))

        # panel letter top-left; stats bottom-right. They cannot collide,
        # and a positively-correlated cloud leaves both corners empty.
        ax.text(0.04, 0.95, f"({next(letters)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold",
                color="#424242")
        ax.text(0.96, 0.05,
                f"r = {r:+.2f}\nSD  fit {sd_fit:.1f}  obs {sd_obs:.1f} d",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.2, color="#424242", linespacing=1.4)

        ax.tick_params(labelsize=7, length=2, color="#BDBDBD")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#BDBDBD")

        if i == 0:
            ax.set_title(SECTOR_LABELS[sec], fontsize=10,
                         fontweight="bold", color=col, pad=8)
        if j == 0:
            ax.set_ylabel(f"{row_label}\nfitted anomaly (days)", fontsize=8.5)
        if i == len(ROWS) - 1:
            ax.set_xlabel("observed anomaly (days)", fontsize=8.5)

fig.suptitle("Fitted versus observed phase anomalies, by sector",
             fontsize=12.5, fontweight="bold", y=0.995)
fig.text(0.5, 0.945,
         "Day of maximum (top row) and day of minimum (bottom row), 1979–2025. "
         "The minimum anchors the phase coordinate; the maximum is generated by the fit.",
         ha="center", fontsize=8.8, color="#616161")
fig.tight_layout(rect=[0, 0, 1, 0.925])

save(fig, "figS1_timing_recovery.png", sync=False)

tab = pd.DataFrame(records)
os.makedirs(TABLES_DIR, exist_ok=True)
tab.to_csv(os.path.join(TABLES_DIR, "tS1_timing_recovery.csv"), index=False)

print("\n" + "=" * 68)
for metric, g in tab.groupby("metric", sort=False):
    print(f"\n{metric}")
    print(f"  r        : {g['r'].min():+.2f} to {g['r'].max():+.2f}")
    print(f"  SD ratio : {g['sd_ratio'].min():.2f} to {g['sd_ratio'].max():.2f} "
          f"(fitted / observed)")
    for _, row in g.iterrows():
        print(f"    {SECTOR_LABELS[row.sector]:<18s} r={row.r:+.3f}  "
              f"SD fit {row.sd_fitted:5.1f}  obs {row.sd_observed:5.1f}  "
              f"ratio {row.sd_ratio:.2f}  n={row.n}")
print("=" * 68)
print("\nwrote tS1_timing_recovery.csv")