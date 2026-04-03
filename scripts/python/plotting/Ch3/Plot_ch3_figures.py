"""
Plot_ch3_figures.py
===================
All Chapter 3 figures for departmental seminar.

Figures:
  1.  sector_sie_anomaly.png           — sector SIE anomaly timeseries
  2.  phase_anomaly_timeseries.png     — phase anomaly, fixed baseline, decade colours
  3.  amplitude_anomaly_timeseries.png — amplitude anomaly, fixed baseline, decade colours
  4.  season_length_timeseries.png     — growth + retreat season length anomaly
  5.  rate_of_change.png               — advance + retreat rates
  6.  rmse_improvement_by_sector.png   — RMSE comparison
  7.  rolling_variance_phase_amplitude.png — 10-yr rolling std dev
  8.  phase_vs_amplitude_variability.png   — std dev comparison bars
  bridge_phase_as_retreat_advance.png  — Himmich-style retreat/advance
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import uniform_filter1d
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data"
SIE_CSV    = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Sea_Ice_Sheets/SIE_daily_sector_and_circumpolar_million_km2.csv"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures"

ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")
DAILY_CSV  = os.path.join(DATA_DIR, "daily_fitted.csv")
RMSE_CSV   = os.path.join(DATA_DIR, "rmse_summary.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STYLE
# =============================================================================

plt.rcParams.update({
    "font.family"      : "Nimbus Sans",
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

# Decade colours for Figs 2 & 3
def decade_color(year):
    if year < 1990:   return "#888780"
    elif year < 2000: return "#378ADD"
    elif year < 2010: return "#1D9E75"
    elif year < 2016: return "#BA7517"
    else:             return "#D4537E"

DECADE_LEGEND = [
    ("#888780", "1980s"),
    ("#378ADD", "1990s"),
    ("#1D9E75", "2000s"),
    ("#BA7517", "2010–2015"),
    ("#D4537E", "2016+"),
]

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

# Fixed 1979-2000 baseline anomalies
baselines_doy = (annual[annual["Year"].between(1979, 2000)]
                 .groupby("sector")["max_doy"].median())
baselines_amp = (annual[annual["Year"].between(1979, 2000)]
                 .groupby("sector")["amplitude"].median())
annual["max_doy_anom_fixed"]   = (annual["max_doy"] -
                                   annual["sector"].map(baselines_doy))
annual["amplitude_anom_fixed"] = (annual["amplitude"] -
                                   annual["sector"].map(baselines_amp))
print("Fixed baseline anomalies computed")

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

def stroke():
    return [pe.withStroke(linewidth=2, foreground="white")]

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    print(f"  -> {name}")
    plt.close(fig)

# =============================================================================
# FIG 1 — Sector SIE anomaly timeseries
# =============================================================================
print("\nFig 1: Sector SIE anomaly timeseries")

sie = pd.read_csv(SIE_CSV)
sie["Date"] = pd.to_datetime(sie["Date"], format="%m/%d/%y")
sie["Year"] = sie["Date"].dt.year
sie["DOY"]  = sie["Date"].dt.dayofyear
sie = sie[sie["Year"].between(1979, 2022)].sort_values("Date").reset_index(drop=True)

for col in SECTORS:
    sie[col] = uniform_filter1d(
        sie[col].fillna(method="ffill").values, size=5, mode="nearest")

clim_sie = (sie[sie["Year"].between(1979, 2000)]
            .groupby("DOY")[SECTORS].mean())

for col in SECTORS:
    sie[f"{col}_anom"] = sie[col] - sie["DOY"].map(clim_sie[col])

annual_anom = (sie.groupby("Year")
               [[f"{col}_anom" for col in SECTORS]]
               .mean().reset_index())

fig, axes = plt.subplots(1, 5, figsize=(18, 5),
                         sharey=False, sharex=True)

for ax, (sec_col, sec_label) in zip(axes, SECTOR_LABELS.items()):
    anom_col = f"{sec_col}_anom"
    yrs  = annual_anom["Year"].values
    vals = annual_anom[anom_col].values

    ax.fill_between(yrs, vals, 0,
                    where=vals >= 0, color="#378ADD",
                    alpha=0.7, linewidth=0, zorder=3)
    ax.fill_between(yrs, vals, 0,
                    where=vals < 0, color="#D4537E",
                    alpha=0.7, linewidth=0, zorder=3)
    ax.axhline(0, color="#2C2C2A", lw=0.8, zorder=4)
    ax.axvline(2016, color="#2C2C2A", lw=1.0, ls="--", alpha=0.5, zorder=4)

    post_mean = annual_anom[annual_anom["Year"] >= 2016][anom_col].mean()
    ax.text(0.97, 0.04, f"Post-2016: {post_mean:+.2f} Mkm²",
            transform=ax.transAxes, fontsize=8,
            color="#D4537E", ha="right", path_effects=stroke())

    ax.set_title(sec_label, fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(labelsize=10)
    ax.set_xlim(1977, 2024)

    if ax == axes[0]:
        ax.set_ylabel("SIE anomaly (million km²)", fontsize=11, labelpad=8)
    ax.set_xlabel("Year", fontsize=10, color="#5F5E5A")

fig.suptitle("Annual Mean SIE Anomaly by Sector — 1979–2000 Baseline",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "1_sector_sie_anomaly.png")

# =============================================================================
# FIG 2 — Phase anomaly timeseries (fixed baseline, decade colours)
# =============================================================================
print("Fig 2: Phase anomaly timeseries")

def plot_anomaly_panel(var, ylabel, title, outfile):
    fig, axes = plt.subplots(1, 5, figsize=(18, 5),
                             sharey=False, sharex=True)

    for ax, (sec_col, sec_label) in zip(axes, SECTOR_LABELS.items()):
        sub  = annual[annual["sector"] == sec_col].sort_values("Year")
        yrs  = sub["Year"].values
        vals = sub[var].values

        ax.axhline(0, color="#B4B2A9", lw=0.8, ls="--", zorder=1)
        ax.axvline(2016, color="#D4537E", lw=1.2, ls="--",
                   alpha=0.7, zorder=2)

        for yr, val in zip(yrs, vals):
            ax.scatter(yr, val, color=decade_color(yr),
                       s=50, zorder=4, edgecolors="white", linewidth=0.5)

        if len(vals) >= 5:
            smooth = uniform_filter1d(vals, size=5, mode="nearest")
            ax.plot(yrs, smooth, color="#2C2C2A", lw=2.0,
                    zorder=3, alpha=0.85)

        post_mean = sub[sub["Year"] >= 2016][var].mean()
        ax.text(0.97, 0.04,
                f"Post-2016: {post_mean:+.1f} days" if "doy" in var
                else f"Post-2016: {post_mean:+.3f} Mkm²",
                transform=ax.transAxes, fontsize=8,
                color="#D4537E", ha="right", path_effects=stroke())

        ax.set_title(sec_label, fontsize=12, fontweight="bold", pad=8)
        ax.tick_params(labelsize=10)
        ax.set_xlim(1977, 2024)

        if ax == axes[0]:
            ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
        ax.set_xlabel("Year", fontsize=10, color="#5F5E5A")

    handles = [plt.scatter([], [], color=c, s=50,
                           edgecolors="white", linewidth=0.5, label=l)
               for c, l in DECADE_LEGEND]
    handles.append(Line2D([0], [0], color="#2C2C2A", lw=2.0,
                          label="5-yr running mean"))

    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=10, bbox_to_anchor=(0.5, -0.05), frameon=False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, outfile)

plot_anomaly_panel(
    var     = "max_doy_anom_fixed",
    ylabel  = "Phase anomaly (days)\n← Ahead of phase  |  Behind phase →",
    title   = "Timing of Sea Ice Maximum — Anomaly from 1979–2000 Baseline",
    outfile = "2_phase_anomaly_timeseries.png"
)

# =============================================================================
# FIG 3 — Amplitude anomaly timeseries (fixed baseline, decade colours)
# =============================================================================
print("Fig 3: Amplitude anomaly timeseries")

plot_anomaly_panel(
    var     = "amplitude_anom_fixed",
    ylabel  = "Amplitude anomaly (million km²)\n← Smaller  |  Larger →",
    title   = "Seasonal Amplitude — Anomaly from 1979–2000 Baseline",
    outfile = "3_amplitude_anomaly_timeseries.png"
)

# =============================================================================
# FIG 4 — Season length timeseries
# =============================================================================
print("Fig 4: Season length")

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey="row")

for col, sec in enumerate(SECTORS):
    sub = annual[annual["sector"]==sec].sort_values("Year").copy()
    sub["growth_days"]  = (sub["max_date"] - sub["min_date"]).dt.days
    sub["next_min"]     = sub["min_date"].shift(-1)
    sub["retreat_days"] = (sub["next_min"] - sub["max_date"]).dt.days

    for row, metric in enumerate(["growth_days", "retreat_days"]):
        ax   = axes[row, col]
        vals = sub[metric].values
        yrs  = sub["Year"].values
        anom = vals - np.nanmean(vals)
        std  = np.nanstd(vals)

        ax.axhspan(-std, std, color="grey", alpha=0.15, zorder=0)
        ax.plot(yrs, anom, color="black", lw=0.8, zorder=1)

        for yr, val in zip(yrs, anom):
            c = "#F44336" if val > 0 else "#2196F3"
            ax.scatter(yr, val, color=c, s=14, zorder=3, linewidths=0)

        ax.axhline(0, color="grey", lw=0.6, ls="--", zorder=0)
        shade2016(ax)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(yr_min - 1, yr_max + 1)

        if col == 0:
            label = ("Growth season\nanomaly (days)"
                     if row == 0 else "Retreat season\nanomaly (days)")
            ax.set_ylabel(label, fontsize=9)
        if row == 0:
            ax.set_title(SECTOR_LABELS[sec], fontweight="bold", fontsize=10)
        if row == 1:
            ax.set_xlabel("Year", fontsize=8)

fig.suptitle("Growth and Retreat Season Length Anomaly by Sector",
             fontsize=13, fontweight="bold", y=1.01)
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
    ["Mean Daily Advance Rate (million km²/day)",
     "Mean Daily Retreat Rate (million km²/day)"]):
    zero_line(ax)
    shade2016(ax)
    ax.set_ylabel("million km²/day")
    ax.set_title(title, fontweight="bold")
    ax.legend(ncol=3, loc="upper left", fontsize=9)

axes[1].set_xlabel("Year")
xlim(axes[0])
fig.suptitle("Rate of Sea Ice Advance and Retreat by Sector",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "5_rate_of_change.png")

# =============================================================================
# FIG 6 — RMSE improvement
# =============================================================================
print("Fig 6: RMSE improvement")

rmse_ordered = (pd.DataFrame({"sector": SECTORS,
                               "label": [SECTOR_LABELS[s] for s in SECTORS]})
                .merge(rmse, on="sector", how="left"))

fig, ax = plt.subplots(figsize=(8, 5))
vals   = rmse_ordered["pct_imp_amp"].tolist()
labels = rmse_ordered["label"].tolist()

colors_rmse = []
hatches     = []
for sec, val in zip(rmse_ordered["sector"], vals):
    if val >= 0:
        colors_rmse.append(SECTOR_COLORS[sec])
        hatches.append("")
    else:
        colors_rmse.append("#cccccc")
        hatches.append("///")

bars = ax.bar(labels, vals, color=colors_rmse, width=0.6,
              edgecolor="white", hatch=hatches)

for bar, val in zip(bars, vals):
    ypos = (bar.get_height() + max(abs(v) for v in vals) * 0.02
            if val >= 0
            else bar.get_height() - max(abs(v) for v in vals) * 0.06)
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

zero_line(ax)
ax.set_ylabel("MSE improvement over invariant cycle (%)")
ax.set_title("Amplitude Adjustment: Improvement by Sector\n"
             "Grey/hatched = degradation (Handcock & Raphael 2020)",
             fontweight="bold")
yabs = max(abs(v) for v in vals)
ax.set_ylim(-yabs * 1.3, yabs * 1.3)
ax.tick_params(axis="x", rotation=20)
for lbl in ax.get_xticklabels():
    lbl.set_ha("right")
fig.tight_layout()
save(fig, "6_rmse_improvement_by_sector.png")

# =============================================================================
# FIG 7 — Rolling variance small multiples
# =============================================================================
print("Fig 7: Rolling variance small multiples")

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey="row", sharex=True)

row_vars   = ["max_doy_anom", "amplitude_anom"]
row_labels = ["10-yr rolling std dev of phase (days)",
              "10-yr rolling std dev of amplitude (million km²)"]
row_titles = ["Phase Variability Over Time",
              "Amplitude Variability Over Time"]

for row, (var, ylabel, row_title) in enumerate(
        zip(row_vars, row_labels, row_titles)):
    for col, sec in enumerate(SECTORS):
        ax   = axes[row, col]
        sub  = annual[annual["sector"] == sec].sort_values("Year").set_index("Year")
        roll = sub[var].rolling(10, center=True, min_periods=6).std()

        ax.plot(roll.index, roll.values,
                color=SECTOR_COLORS[sec], lw=2, zorder=3)
        ax.fill_between(roll.index, roll.values,
                        color=SECTOR_COLORS[sec], alpha=0.15, zorder=2)
        shade2016(ax)
        ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=1)
        ax.set_xlim(yr_min + 4, yr_max - 4)

        if row == 0:
            ax.set_title(SECTOR_LABELS[sec], fontweight="bold", fontsize=11)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if row == 1:
            ax.set_xlabel("Year", fontsize=9)
            ax.tick_params(axis="x", rotation=30)

for row, row_title in enumerate(row_titles):
    axes[row, -1].annotate(
        row_title, xy=(1.02, 0.5), xycoords="axes fraction",
        rotation=270, va="center", ha="left", fontsize=10, fontweight="bold"
    )

fig.suptitle("Has Variability Changed Over Time? Phase and Amplitude",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "7_rolling_variance_phase_amplitude.png")

# =============================================================================
# FIG 8 — Phase vs amplitude variability bars
# =============================================================================
print("Fig 8: Phase vs amplitude variability")

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
     "Std dev of amplitude anomaly (million km²)"],
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

fig.suptitle("What Varies More — Timing or Magnitude?",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "8_phase_vs_amplitude_variability.png")

# =============================================================================
# BRIDGE — Phase anomaly as retreat/advance language
# =============================================================================
print("Bridge fig: phase anomaly as retreat/advance")

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey="row", sharex=True)

row_vars   = ["min_doy_anom", "max_doy_anom"]
row_titles = ["Retreat timing anomaly\n(negative = earlier retreat)",
              "Advance timing anomaly\n(positive = later advance)"]

for row, (var, row_title) in enumerate(zip(row_vars, row_titles)):
    for col, sec in enumerate(SECTORS):
        ax  = axes[row, col]
        sub = annual[annual["sector"] == sec].sort_values("Year")
        yrs = sub["Year"].values
        vals = sub[var].values

        ax.plot(yrs, vals, color="grey", lw=0.8, alpha=0.5, zorder=1)

        roll  = pd.Series(vals, index=yrs).rolling(6, center=True, min_periods=4)
        rmean = roll.mean()
        rstd  = roll.std()

        ax.fill_between(yrs, rmean - rstd, rmean + rstd,
                        color="grey", alpha=0.15, zorder=2)

        for y, m in zip(yrs, rmean):
            if pd.isna(m):
                continue
            color = "#E8A020" if m > 0 else "#3A7DC9"
            ax.scatter(y, m, color=color, s=18, zorder=4)

        ax.plot(yrs, rmean, color="black", lw=1.2, zorder=3)
        ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=1)
        shade2016(ax)

        if row == 0:
            ax.set_title(SECTOR_LABELS[sec], fontweight="bold", fontsize=11)
        if col == 0:
            ax.set_ylabel("Anomaly (days)", fontsize=9)
        if row == 1:
            ax.set_xlabel("Year", fontsize=9)
            ax.tick_params(axis="x", rotation=30)

for row, row_title in enumerate(row_titles):
    axes[row, -1].annotate(
        row_title, xy=(1.02, 0.5), xycoords="axes fraction",
        rotation=270, va="center", ha="left", fontsize=9
    )

fig.suptitle(
    "Earlier Retreat, Later Advance?\n"
    "APAC Phase Anomalies Capture the Known Post-2016 Signal",
    fontsize=13, fontweight="bold", y=1.02
)
fig.tight_layout()
save(fig, "bridge_phase_as_retreat_advance.png")

# =============================================================================
# DONE
# =============================================================================
print(f"\nAll figures saved to:\n  {OUTPUT_DIR}")