#!/usr/bin/env python3
"""
fig_s01_fitted_vs_observed.py -- Fig. S1 (restyled): fitted against observed
amplitude and day of maximum, by sector.

Layout: two rows x six sectors.
  top row     amplitude (fitted vs observed anomaly, 10^6 km^2)       blue
  bottom row  day of maximum (fitted vs observed anomaly, days)       orange
Each panel: 1:1 line, r, and for the day of maximum the fitted/observed
variance ratio (the "quarter to a half" in Sect. 2.2.3). The day of minimum
(r = 0.99-1.00 everywhere) is not drawn; its r values are printed and go in
the caption.

Columns (confirmed 2026-09-18 against annual_params.csv, anomaly scale on both sides):
  observed  amplitude_raw_anom, max_doy_raw_anom, min_doy_raw_anom
  fitted    amplitude_anom,     max_doy_anom,     min_doy_anom

Reads  ANNUAL_CSV (period == FULL)
Writes results/ch3/figures/fig_s01_fitted_vs_observed.png
       results/ch3/tables/t_s01_fitted_vs_observed_fit_quality.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import SECTORS, ANNUAL_CSV, TABLES_DIR, OUTPUT_DIR
import ch3_style

TITLE = {"SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "Amundsen-Bellingshausen",
         "SIE_Ross": "Ross", "SIE_East_Antarctica": "East Antarctica",
         "SIE_King_Haakon": "King Haakon", "SIE_circumpolar": "Circumpolar total"}
ROWS = [("amplitude_raw_anom", "amplitude_anom", "Amplitude", "10$^6$ km$^2$", "#2a78d6"),
        ("max_doy_raw_anom", "max_doy_anom", "Day of maximum", "days", "#eb6834")]
INK = "0.35"

a = pd.read_csv(ANNUAL_CSV)
if "period" in a.columns:
    a = a[a["period"] == "FULL"]
need = [c for r in ROWS for c in r[:2]] + ["min_doy_raw_anom", "min_doy_anom"]
miss = [c for c in need if c not in a.columns]
if miss:
    sys.exit(f"annual_params lacks {miss}; columns are {sorted(a.columns)}")

# ---- fit-quality table (all three quantities) ----
rows = []
for sec in SECTORS:
    s = a[a.sector == sec]
    for obs_c, fit_c, q in [("amplitude_raw_anom", "amplitude_anom", "amplitude"),
                            ("max_doy_raw_anom", "max_doy_anom", "day_of_max"),
                            ("min_doy_raw_anom", "min_doy_anom", "day_of_min")]:
        j = s[[obs_c, fit_c]].dropna()
        r, p = pearsonr(j[obs_c], j[fit_c])
        rows.append(dict(sector=TITLE.get(sec, sec), quantity=q, n=len(j), r=r, p=p,
                         rmse=float(np.sqrt(np.mean((j[obs_c] - j[fit_c]) ** 2))),
                         var_ratio_fit_over_obs=float(j[fit_c].var() / j[obs_c].var())))
fq = pd.DataFrame(rows)
os.makedirs(TABLES_DIR, exist_ok=True)
fq.to_csv(os.path.join(TABLES_DIR, "t_s01_fitted_vs_observed_fit_quality.csv"), index=False)

# ---- figure ----
n = len(SECTORS)
fig, axes = plt.subplots(2, n, figsize=(2.25 * n, 5.0))
for i, (obs_c, fit_c, name, unit, col) in enumerate(ROWS):
    for k, sec in enumerate(SECTORS):
        ax = axes[i, k]
        j = a[a.sector == sec][[obs_c, fit_c]].dropna()
        x, y = j[obs_c].values, j[fit_c].values
        lim = 1.08 * np.nanmax(np.abs(np.r_[x, y]))
        ax.plot([-lim, lim], [-lim, lim], color="0.7", lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.axhline(0, color="0.9", lw=0.6, zorder=0)
        ax.axvline(0, color="0.9", lw=0.6, zorder=0)
        ax.scatter(x, y, s=14, color=col, alpha=0.85, lw=0, zorder=3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        r = pearsonr(x, y)[0]
        txt = f"r = {r:.2f}"
        if i == 1:
            txt += f"\nvariance ratio {np.var(y, ddof=1) / np.var(x, ddof=1):.2f}"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", fontsize=7.5, color="0.15")
        ax.tick_params(labelsize=7, colors=INK, length=2.5)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.set_title(f"({chr(97 + k)})  {TITLE.get(sec, sec)}", loc="left",
                         fontproperties=ch3_style.bold_font_properties(size=8.5), color="0.1")
        if k == 0:
            ax.set_ylabel(f"{name}, fitted\n({unit})", fontsize=8.5, color=INK)
        ax.set_xlabel(f"observed ({unit})", fontsize=7.5, color=INK)
fig.tight_layout(w_pad=0.8, h_pad=1.2)
out = os.path.join(OUTPUT_DIR, "fig_s01_fitted_vs_observed.png")
fig.savefig(out, dpi=250, bbox_inches="tight")
plt.close(fig)

print(f"wrote {out}")
pv = fq.pivot_table(index="sector", columns="quantity", values=["r", "var_ratio_fit_over_obs"])
print(pv.round(2).to_string())
d = fq[fq.quantity == "day_of_min"]["r"]
m = fq[fq.quantity == "day_of_max"]
print(f"\nfor the caption: day of minimum r = {d.min():.2f}-{d.max():.2f}; "
      f"day of maximum r = {m.r.min():.2f}-{m.r.max():.2f}, "
      f"variance ratio {m.var_ratio_fit_over_obs.min():.2f}-{m.var_ratio_fit_over_obs.max():.2f}")