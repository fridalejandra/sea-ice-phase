"""
RMSE validation figure — two panels, two time periods.

Top panel:    1979–2018  (matches Handcock & Raphael 2020 Table 1)
Bottom panel: 1979–2023  (full record extension)


H&R 2020 circumpolar reference values are annotated on the top panel as
dashed horizontal lines so the reader can confirm we reproduce the paper.

Computes RMSE directly from daily_fitted.csv for both periods — no
separate pipeline run needed.

H&R 2020 Table 1 reference values (circumpolar, 1979-2018):
    Amplitude-adjusted: 55.2%
    Phase-adjusted:     63.9%
    Amp + Phase:        77.3%
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from ch3_style import (
    apply_style,
    SECTORS_NO_CIRC, SECTOR_LABELS,
    stroke, save_fig, DEFAULT_OUTPUT_DIR,
)

apply_style()

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
DAILY_CSV  = os.path.join(DATA_DIR, "daily_fitted.csv")

# H&R 2020 Table 1 reference values — circumpolar, 1979-2018
HR2020 = {
    "Invariant":   28.7,
    "Amplitude":   55.2,
    "Phase":       63.9,
    "Amp + Phase": 77.3,
}

PERIODS = {
    "1979–2018": (1979, 2018),
    "1979–2023": (1979, 2023),
}

ALL_SECTORS = SECTORS_NO_CIRC + ["SIE_circumpolar"]

MODELS = [
    ("fitted_invariant", "Invariant",   "#B4B2A9"),
    ("fitted_amp",       "Amplitude",   "#378ADD"),
    ("fitted_phase",     "Phase",       "#D4537E"),
    ("fitted_apac",      "Amp + Phase", "#1D9E75"),
]


# --- Load -----------------------------------------------------------------

print("Loading daily fitted data...")
daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
daily["Year"] = daily["Date"].dt.year
print(f"  {len(daily)} rows | sectors: {sorted(daily['sector'].unique())}")

# Compute full-record DOY means per sector — used as traditional baseline
trad_means = {}
for sector in ALL_SECTORS:
    sec_data = daily[daily["sector"] == sector]
    trad_means[sector] = sec_data.groupby("DOY")["Extent"].mean()

# --- Compute RMSE improvement per sector and period -----------------------

def compute_improvements(daily_sub, trad_means_full):
    # Use full-record DOY means as the traditional baseline — matches H&R 2020
    trad_mean = daily_sub["DOY"].map(trad_means_full)
    rmse_trad = np.sqrt(np.mean((daily_sub["Extent"] - trad_mean) ** 2))
    if rmse_trad == 0:
        return {label: np.nan for _, label, _ in MODELS}
    out = {}
    for col, label, _ in MODELS:
        if col not in daily_sub.columns:
            out[label] = np.nan
            continue
        rmse = np.sqrt(np.mean((daily_sub["Extent"] - daily_sub[col]) ** 2))
        out[label] = round(100 * (1 - rmse**2 / rmse_trad**2), 1)
    return out


print("Computing RMSE improvements...")
records = []
for sector in ALL_SECTORS:
    sec_data = daily[daily["sector"] == sector]
    for period_label, (yr_min, yr_max) in PERIODS.items():
        sub = sec_data[sec_data["Year"].between(yr_min, yr_max)].copy()
        if len(sub) < 100:
            continue
        for label, pct in compute_improvements(sub, trad_means[sector]).items():
            records.append({
                "sector_label": SECTOR_LABELS.get(sector, sector),
                "period"      : period_label,
                "model"       : label,
                "pct_imp"     : pct,
            })

df = pd.DataFrame(records)
print(f"  {len(df)} records")


# --- Plot -----------------------------------------------------------------

sector_order = [SECTOR_LABELS[s] for s in ALL_SECTORS]
x     = np.arange(len(sector_order))
bar_w = 0.18

fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharey=False)
fig.subplots_adjust(hspace=0.40)

for ax_idx, (period_label, _) in enumerate(PERIODS.items()):
    ax        = axes[ax_idx]
    period_df = df[df["period"] == period_label]

    for m_idx, (_, model_label, color) in enumerate(MODELS):
        offsets = x + (m_idx - 1) * bar_w
        vals    = [
            float(period_df[(period_df["sector_label"] == s) &
                            (period_df["model"] == model_label)]["pct_imp"].values[0])
            if len(period_df[(period_df["sector_label"] == s) &
                             (period_df["model"] == model_label)]) > 0
            else np.nan
            for s in sector_order
        ]

        bars = ax.bar(offsets, vals, bar_w,
                      color=color, edgecolor="white",
                      label=model_label, zorder=3)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{val:.0f}%",
                        ha="center", va="bottom",
                        fontsize=7.5, color=color, fontweight="bold",
                        path_effects=stroke())

    # H&R 2020 reference lines on top panel only
    if ax_idx == 0:
        ref_colors = {"Invariant": "#B4B2A9",
                      "Amplitude": "#378ADD",
                      "Phase": "#D4537E",
                      "Amp + Phase": "#1D9E75"}
        for model_label, ref_val in HR2020.items():
            ax.axhline(ref_val, color=ref_colors[model_label],
                       lw=1.0, ls="--", alpha=0.5, zorder=2)
            ax.text(len(sector_order) - 0.2, ref_val + 0.5,
                    f"H&R2020: {ref_val}%",
                    fontsize=7.5, color=ref_colors[model_label],
                    ha="right", va="bottom",
                    path_effects=stroke())

    ax.axhline(0, color="grey", lw=0.7, ls="--", zorder=0)

    # Circumpolar separator
    ax.axvline(len(sector_order) - 1 - 0.5,
               color="#B4B2A9", lw=0.8, ls=":", zorder=1, alpha=0.7)
    ax.text(len(sector_order) - 1, -4, "circumpolar",
            ha="center", va="top", fontsize=8,
            color="#B4B2A9", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(sector_order, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("RMSE improvement over\ntraditional seasonal cycle (%)",
                  fontsize=10)
    ax.set_title(period_label, fontsize=12, fontweight="bold")
    ax.set_ylim(bottom=-6,
                top=max(df["pct_imp"].dropna().max() * 1.15, 85))

    if ax_idx == 0:
        ax.legend(title="Model", fontsize=9, loc="upper left",
                  title_fontsize=9)

fig.suptitle(
    "Sequential RMSE Improvement by Sector\n"
    "Validating against Handcock & Raphael (2020) and extending to 2023",
    fontsize=12, fontweight="bold", y=1.01,
)

save_fig(fig, "fig_rmse_validation.png", OUTPUT_DIR, gdrive_sync=True)
print("fig_rmse_validation.png saved.")