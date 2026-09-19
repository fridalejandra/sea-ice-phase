#!/usr/bin/env python3
"""
compute_rolling_diagnostics.py -- rebuilds the "rolling 10-year Spearman r
with SIE anomaly" diagnostic (the old script is gone, only the PNG
survives). For each sector, plots two rolling-window lines:

  phase-SIE r  = rolling Spearman r(max_doy_raw_anom, SIE anomaly)
  amp-SIE r    = rolling Spearman r(amplitude_raw_anom, SIE anomaly)

over a ROLL_SHORT-year window (ch3_config.ROLL_SHORT, currently 10 --
matches the old figure's title and confirms this window length was
already a real part of your framework, just the script that drew it
was lost).

RESOLVED 2026-09-18 (Frida's call): "SIE anomaly" and the phase/amplitude
inputs should all be raw (undetrended) values -- confirming the assumption
this script already made: sie_annual minus its own mean, UNDETRENDED, paired
with amplitude_raw_anom and max_doy_raw_anom. No source change needed; this
note is kept for history rather than as an open question.

Also fixed 2026-09-18: ch3_config.py's YEAR_MAX was hardcoded to 2023 (the
record now runs to 2025), which silently stopped this script's rolling
window two years early -- the version of this figure pasted into chat was
very likely missing its last two windows. Re-run this script now that
YEAR_MAX is fixed to see the corrected tail.

This is a different question from Figs. 5/6: those ask whether phase and
amplitude are correlated with EACH OTHER; this asks how strongly each one
individually tracks the overall extent anomaly, and whether that's been
changing -- a rolling-window, time-resolved version of the §3.1
attribution finding (that amplitude carries more of the anomaly than
phase does, in every sector but King Haakon).

Same caveat as any rolling-window figure (already in your §2.3): adjacent
windows share 9 of 10 years, so the line is smooth by construction and
this is for visual orientation, not a substitute for the pre/post-2016
split test.

NOT YET ASSIGNED a manuscript figure/table number -- still writes the
placeholder names fig12_rolling_component_sie.png /
t38_rolling_component_sie.csv from before Figs 1-11 were locked in. Numbers
1-11 are all taken (see run_all.py); this would need a fresh number or a
supplementary slot once Frida decides whether/where it goes in the text.

Writes results/ch3/figures/fig12_rolling_component_sie.png and
results/ch3/tables/t38_rolling_component_sie.csv. Paste the console
output back.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import (
    ANNUAL_CSV, OUTPUT_DIR, TABLES_DIR, ROLL_SHORT, BREAK_YEAR,
    YEAR_MIN, YEAR_MAX, SECTORS_COMPUTE, SECTOR_ORDER_BY_LONGITUDE,
    SECTOR_LABELS, COMPONENT_COLORS,
)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"] == "FULL"]

# annual_params.csv stores sectors under the raw SIE_-prefixed keys (e.g.
# "SIE_Weddell"), not the short display labels -- confirmed by this run's
# console output. Iterate over the raw keys and map to display labels for
# titles only.
raw_sectors_present = set(ann["sector"].unique())
sector_keys = [s for s in SECTOR_ORDER_BY_LONGITUDE if s in raw_sectors_present]
sector_keys += [s for s in raw_sectors_present if s not in sector_keys]

rows = []
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
axes = axes.ravel()

for ax, sector_key in zip(axes, sector_keys):
    sector = SECTOR_LABELS.get(sector_key, sector_key)
    a = ann[ann["sector"] == sector_key].sort_values("Year").reset_index(drop=True)
    a["sie_anom"] = a["sie_annual"] - a["sie_annual"].mean()

    window_ends, r_phase, r_amp = [], [], []
    for end_year in range(YEAR_MIN + ROLL_SHORT - 1, YEAR_MAX + 1):
        win = a[(a["Year"] > end_year - ROLL_SHORT) & (a["Year"] <= end_year)]
        if len(win) < ROLL_SHORT:
            continue
        rp, _ = spearmanr(win["max_doy_raw_anom"], win["sie_anom"])
        ra, _ = spearmanr(win["amplitude_raw_anom"], win["sie_anom"])
        window_ends.append(end_year)
        r_phase.append(rp)
        r_amp.append(ra)
        rows.append(dict(sector=sector, window_end=end_year, r_phase_sie=rp, r_amp_sie=ra))

    ax.axhspan(-0.4, 0.4, color="0.92", zorder=0)
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.axvline(BREAK_YEAR, color="#C0392B", lw=1.2, ls="--", zorder=1, label=str(BREAK_YEAR))
    ax.plot(window_ends, r_phase, color=COMPONENT_COLORS["Phase"], lw=1.8, marker="o", ms=3, label="phase-SIE r")
    ax.plot(window_ends, r_amp, color=COMPONENT_COLORS["Amplitude"], lw=1.8, marker="o", ms=3, label="amp-SIE r")
    ax.set_ylim(-1, 1)
    ax.set_title(sector, fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

for ax in axes[3:]:
    ax.set_xlabel(f"Year (end of {ROLL_SHORT}-yr window)")
fig.suptitle(f"Rolling {ROLL_SHORT}-year Spearman r: each component vs. annual SIE anomaly", fontweight="bold", y=1.00)
fig.tight_layout()
out = os.path.join(OUTPUT_DIR, "fig12_rolling_component_sie.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)

out_tab = pd.DataFrame(rows)
out_tab_path = os.path.join(TABLES_DIR, "t38_rolling_component_sie.csv")
out_tab.to_csv(out_tab_path, index=False)

print(f"wrote {out}")
print(f"wrote {out_tab_path}  ({len(out_tab)} rows)")
for sector_key in sector_keys:
    sector = SECTOR_LABELS.get(sector_key, sector_key)
    sub = out_tab[out_tab["sector"] == sector]
    pre = sub[sub["window_end"] < BREAK_YEAR]
    post = sub[sub["window_end"] >= BREAK_YEAR]
    print(f"  {sector:16s} phase-SIE mean: pre={pre['r_phase_sie'].mean():+.2f} post={post['r_phase_sie'].mean():+.2f}"
          f"   amp-SIE mean: pre={pre['r_amp_sie'].mean():+.2f} post={post['r_amp_sie'].mean():+.2f}")
