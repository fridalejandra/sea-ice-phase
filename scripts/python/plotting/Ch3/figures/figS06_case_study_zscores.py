"""
figS06_case_study_zscores.py
============================
2016 and 2023 as z-scores of the three observed metrics, by sector.
The compact statement of the chapter's central contrast: 2016 is a timing
event, 2023 an amplitude event, and the SIE anomaly alone cannot tell them
apart.

z-scores are computed against the full 1980–2023 record for that sector.
"""

import numpy as np
import matplotlib.pyplot as plt

import ch3_data as D
from ch3_config import SECTORS, SECTOR_LABELS
from ch3_plot import panel_letters, save

print("figS06 — 2016 vs 2023 z-scores")
annual = D.load_annual()
D.summary(annual=annual)

METRICS = [("min_doy_raw_anom", "Minimum\ndate"),
           ("max_doy_raw_anom", "Maximum\ndate"),
           ("amplitude_raw_anom", "Amplitude")]
YEARS = [2016, 2023]

z = annual.copy()
for col, _ in METRICS:
    z[col + "_z"] = z.groupby("sector")[col].transform(D.zscore)

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
y = np.arange(len(SECTORS)); h = 0.26
colors = ["#C77B26", "#378ADD", "#D4537E"]

for ax, yr in zip(axes, YEARS):
    for k, ((col, lab), colr) in enumerate(zip(METRICS, colors)):
        vals = [float(z[(z.sector == s) & (z.Year == yr)][col + "_z"].iloc[0])
                if len(z[(z.sector == s) & (z.Year == yr)]) else np.nan
                for s in SECTORS]
        ax.barh(y + (k - 1) * h, vals, h, color=colr, alpha=0.85,
                label=lab.replace("\n", " "))

    ax.axvline(0, color="#2C2C2A", lw=0.9)
    for t in (-2, 2):
        ax.axvline(t, color="#888888", lw=0.7, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([SECTOR_LABELS[s] for s in SECTORS], fontsize=10)
    ax.set_xlabel("z-score (vs 1980–2023)", fontsize=10)
    ax.set_title(str(yr), fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

axes[0].legend(fontsize=8.5, loc="lower left", frameon=False)
panel_letters(axes)
fig.suptitle("2016 is a timing event; 2023 is an amplitude event",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "figS06_case_study_zscores.png")
