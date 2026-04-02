"""
Ch3 Presentation Figures
========================
Produces all figures for the departmental seminar from the APAC
pipeline outputs.

Figures:
  1. phase_variability_by_sector.png   — std dev of max/min DOY anom per sector
  2. phase_anomaly_timeseries.png      — max_doy_anom time series all sectors
  3. amplitude_anomaly_timeseries.png  — amplitude_anom time series all sectors
  4. season_length_timeseries.png      — growth + retreat season length per year
  5. rate_of_change.png                — advance + retreat rates per year
  6. rmse_improvement.png              — amplitude-adjusted vs invariant per sector
  7. rolling_variance.png              — 10-yr rolling std of max_doy_anom
  8. phase_vs_amplitude.png            — side-by-side std dev comparison

Usage on cluster:
    cd /user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3
    python plot_ch3_figures.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures"

ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")
DAILY_CSV  = os.path.join(DATA_DIR, "daily_fitted.csv")
RMSE_CSV   = os.path.join(DATA_DIR, "rmse_summary.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STYLE
# =============================================================================

plt.rcParams.update({
    "font.family"      : "sans-serif",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.linewidth"   : 0.8,
    "axes.labelsize"   : 12,
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
    "xtick.labelsize"  : 10,
    "ytick.labelsize"  : 10,
    "legend.fontsize"  : 10,
    "legend.frameon"   : False,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
    "savefig.facecolor": "white",
})

SECTOR_COLORS = {
    "SIE_Weddell"                : "#2196F3",
    "SIE_Amundsen_Bellingshausen": "#F44336",
    "SIE_Ross"                   : "#4CAF50",
    "SIE_East_Antarctica"        : "#FF9800",
    "SIE_King_Haakon"            : "#9C27B0",
}

SECTOR_LABELS = {
    "SIE_Weddell"                : "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross"                   : "Ross",
    "SIE_East_Antarctica"        : "East Antarctica",
    "SIE_King_Haakon"            : "King Haakon",
}

SECTORS = list(SECTOR_COLORS.keys())

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data...")
annual = pd.read_csv(ANNUAL_CSV, parse_dates=["min_date", "max_date"])
daily  = pd.read_csv(DAILY_CSV,  parse_dates=["Date"])
rmse   = pd.read_csv(RMSE_CSV)

for col in ["min_doy_anom", "max_doy_anom", "amplitude_anom",
            "amplitude", "min_doy", "max_doy"]:
    if col in annual.columns:
        annual[col] = pd.to_numeric(annual[col], errors="coerce")

yr_min = int(annual["Year"].min())
yr_max = int(annual["Year"].max())
print(f"  Annual : {len(annual)} rows | {yr_min}-{yr_max}")
print(f"  Daily  : {len(daily)} rows")
print(f"  RMSE   : {len(rmse)} rows")

# =============================================================================
# HELPERS
# =============================================================================

def zero_line(ax):
    ax.axhline(0, color="grey", lw=0.7, ls="--", zorder=0)

def shade2016(ax):
    ax.axvspan(2016.5, yr_max + 0.5, color="grey", alpha=0.07, zorder=0)

def xlim(ax):
    ax.set_xlim(yr_min - 0.5, yr_max + 0.5)

def bar_labels(ax, bars, vals, fmt="{:.1f}"):
    for bar, val in zip(bars, vals):
        if pd.notna(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(v for v in vals if pd.notna(v)) * 0.02,
                    fmt.format(val),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    print(f"  -> {name}")
    plt.close(fig)

# =============================================================================
# FIG 1 — Phase variability by sector
# =============================================================================
print("\nFig 1: Phase variability by sector")
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, col, title in zip(
    axes,
    ["max_doy_anom", "min_doy_anom"],
    ["Melt onset timing\n(max DOY anomaly std dev)",
     "Freeze onset timing\n(min DOY anomaly std dev)"]
):
    vals   = [annual[annual["sector"]==s][col].dropna().std() for s in SECTORS]
    labels = [SECTOR_LABELS[s] for s in SECTORS]
    colors = [SECTOR_COLORS[s] for s in SECTORS]
    bars   = ax.bar(labels, vals, color=colors, width=0.6, edgecolor="white")
    bar_labels(ax, bars, vals, "{:.1f}d")
    ax.set_ylabel("Std dev (days)")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.tick_params(axis="x", rotation=25)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

fig.suptitle("Phase Variability by Sector  (1979-2023)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "1_phase_variability_by_sector.png")

# =============================================================================
# FIG 2 — Max DOY anomaly time series
# =============================================================================
print("Fig 2: Phase anomaly time series")
fig, ax = plt.subplots(figsize=(13, 5))

for sec in SECTORS:
    sub = annual[annual["sector"]==sec].sort_values("Year")
    ax.plot(sub["Year"], sub["max_doy_anom"],
            color=SECTOR_COLORS[sec], lw=1.6,
            label=SECTOR_LABELS[sec], zorder=3)

zero_line(ax)
shade2016(ax)
ax.set_xlabel("Year")
ax.set_ylabel("Max DOY anomaly (days)\n<- Earlier    |    Later ->")
ax.set_title("Timing of Sea Ice Maximum - Annual Anomaly by Sector",
             fontweight="bold")
ax.legend(ncol=2, loc="upper left")
xlim(ax)
fig.tight_layout()
save(fig, "2_phase_anomaly_timeseries.png")

# =============================================================================
# FIG 3 — Amplitude anomaly time series
# =============================================================================
print("Fig 3: Amplitude anomaly time series")
fig, ax = plt.subplots(figsize=(13, 5))

for sec in SECTORS:
    sub = annual[annual["sector"]==sec].sort_values("Year")
    ax.plot(sub["Year"], sub["amplitude_anom"],
            color=SECTOR_COLORS[sec], lw=1.6,
            label=SECTOR_LABELS[sec], zorder=3)

zero_line(ax)
shade2016(ax)
ax.set_xlabel("Year")
ax.set_ylabel("Amplitude anomaly (million km2)\n<- Smaller    |    Larger ->")
ax.set_title("Seasonal Amplitude Anomaly by Sector", fontweight="bold")
ax.legend(ncol=2, loc="upper left")
xlim(ax)
fig.tight_layout()
save(fig, "3_amplitude_anomaly_timeseries.png")

# =============================================================================
# FIG 4 — Season length time series
# =============================================================================
print("Fig 4: Season length")
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

for sec in SECTORS:
    sub = annual[annual["sector"]==sec].sort_values("Year").copy()
    sub["growth_days"]  = (sub["max_date"] - sub["min_date"]).dt.days
    sub["next_min"]     = sub["min_date"].shift(-1)
    sub["retreat_days"] = (sub["next_min"] - sub["max_date"]).dt.days
    c = SECTOR_COLORS[sec]
    l = SECTOR_LABELS[sec]
    axes[0].plot(sub["Year"], sub["growth_days"],  color=c, lw=1.5, label=l)
    axes[1].plot(sub["Year"], sub["retreat_days"], color=c, lw=1.5, label=l)

for ax, title in zip(axes,
    ["Growth Season Length  (days: min to max)",
     "Retreat Season Length  (days: max to following min)"]):
    shade2016(ax)
    ax.set_ylabel("Days")
    ax.set_title(title, fontweight="bold")
    ax.legend(ncol=3, loc="upper left", fontsize=9)

axes[1].set_xlabel("Year")
xlim(axes[0])
fig.suptitle("Growth and Retreat Season Length by Sector",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "4_season_length_timeseries.png")

# =============================================================================
# FIG 5 — Rate of change
# =============================================================================
print("Fig 5: Rate of change")

def compute_rates(daily_df, annual_df):
    records = []
    for sec in SECTORS:
        d = daily_df[daily_df["sector"]==sec].sort_values(["Year","Date"]).copy()
        a = annual_df[annual_df["sector"]==sec][["Year","min_doy","max_doy"]]
        d["dSIE"] = d.groupby("Year")["fitted_amp"].diff()
        for _, row in a.iterrows():
            yr = int(row["Year"])
            yd = d[d["Year"]==yr]
            if len(yd) < 20:
                continue
            adv = yd[(yd["DOY"] >= row["min_doy"]) & (yd["DOY"] <= row["max_doy"])]
            ret = yd[(yd["DOY"] > row["max_doy"]) | (yd["DOY"] < row["min_doy"])]
            records.append({
                "sector"      : sec,
                "Year"        : yr,
                "advance_rate": adv["dSIE"].mean(),
                "retreat_rate": ret["dSIE"].mean(),
            })
    return pd.DataFrame(records)

rate_df = compute_rates(daily, annual)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for sec in SECTORS:
    sub = rate_df[rate_df["sector"]==sec].sort_values("Year")
    c = SECTOR_COLORS[sec]
    l = SECTOR_LABELS[sec]
    axes[0].plot(sub["Year"], sub["advance_rate"], color=c, lw=1.5, label=l)
    axes[1].plot(sub["Year"], sub["retreat_rate"], color=c, lw=1.5, label=l)

for ax, title in zip(axes,
    ["Mean Daily Advance Rate (million km2/day)",
     "Mean Daily Retreat Rate (million km2/day)"]):
    zero_line(ax)
    shade2016(ax)
    ax.set_ylabel("million km2/day")
    ax.set_title(title, fontweight="bold")
    ax.legend(ncol=3, loc="upper left", fontsize=9)

axes[1].set_xlabel("Year")
xlim(axes[0])
fig.suptitle("Rate of Sea Ice Advance and Retreat by Sector",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "5_rate_of_change.png")

# =============================================================================
# FIG 6 — RMSE improvement from rmse_summary.csv
# =============================================================================
print("Fig 6: RMSE improvement")

rmse_ordered = (pd.DataFrame({"sector": SECTORS,
                               "label": [SECTOR_LABELS[s] for s in SECTORS]})
                .merge(rmse, on="sector", how="left"))

fig, ax = plt.subplots(figsize=(8, 5))
colors = [SECTOR_COLORS[s] for s in rmse_ordered["sector"]]
bars   = ax.bar(rmse_ordered["label"], rmse_ordered["pct_imp_amp"],
                color=colors, width=0.6, edgecolor="white")
bar_labels(ax, bars, rmse_ordered["pct_imp_amp"].tolist(), "{:.1f}%")
ax.set_ylabel("MSE improvement over invariant cycle (%)")
ax.set_title("Amplitude Adjustment Improves Fit by Sector\n"
             "(Handcock & Raphael 2020, Table 1 analogue)",
             fontweight="bold")
ax.set_ylim(0, rmse_ordered["pct_imp_amp"].max() * 1.25)
ax.tick_params(axis="x", rotation=20)
for lbl in ax.get_xticklabels():
    lbl.set_ha("right")
fig.tight_layout()
save(fig, "6_rmse_improvement_by_sector.png")

# =============================================================================
# FIG 7 — Rolling 10-year variance
# =============================================================================
print("Fig 7: Rolling variance")
fig, ax = plt.subplots(figsize=(13, 5))

for sec in SECTORS:
    sub  = annual[annual["sector"]==sec].sort_values("Year").set_index("Year")
    roll = sub["max_doy_anom"].rolling(10, center=True, min_periods=6).std()
    ax.plot(roll.index, roll.values,
            color=SECTOR_COLORS[sec], lw=2,
            label=SECTOR_LABELS[sec], zorder=3)

shade2016(ax)
ax.set_xlabel("Year")
ax.set_ylabel("10-year rolling std dev (days)")
ax.set_title("Has Phase Variability Changed Over Time?\n"
             "10-year Rolling Std Dev of Max DOY Anomaly",
             fontweight="bold")
ax.legend(ncol=2, loc="upper left")
ax.set_xlim(yr_min + 4, yr_max - 4)
fig.tight_layout()
save(fig, "7_rolling_variance_phase.png")

# =============================================================================
# FIG 8 — Phase vs amplitude side by side
# =============================================================================
print("Fig 8: Phase vs amplitude")

phase_stds = [annual[annual["sector"]==s]["max_doy_anom"].dropna().std()
              for s in SECTORS]
amp_stds   = [annual[annual["sector"]==s]["amplitude_anom"].dropna().std()
              for s in SECTORS]
labels     = [SECTOR_LABELS[s] for s in SECTORS]
colors     = [SECTOR_COLORS[s] for s in SECTORS]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, vals, ylabel, title, fmt in zip(
    axes,
    [phase_stds, amp_stds],
    ["Std dev of max DOY anomaly (days)",
     "Std dev of amplitude anomaly (million km2)"],
    ["Phase Variability\n(timing of maximum)",
     "Amplitude Variability\n(size of seasonal cycle)"],
    ["{:.1f}d", "{:.2f}"]
):
    bars = ax.bar(labels, vals, color=colors, width=0.6, edgecolor="white")
    bar_labels(ax, bars, vals, fmt)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.tick_params(axis="x", rotation=25)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

fig.suptitle("What Varies More - Timing or Magnitude?",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "8_phase_vs_amplitude_variability.png")

# =============================================================================
# DONE
# =============================================================================
print(f"\nAll 8 figures saved to:\n  {OUTPUT_DIR}")