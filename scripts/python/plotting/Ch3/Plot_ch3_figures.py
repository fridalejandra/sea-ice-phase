"""
Plot_ch3_figures.py
===================
All Chapter 3 figures for departmental seminar.

Figures:
  1.  sector_sie_anomaly.png               — sector SIE anomaly timeseries
  2.  phase_anomaly_timeseries.png         — phase anomaly, fixed baseline, decade colours
  3.  amplitude_anomaly_timeseries.png     — amplitude anomaly, fixed baseline, decade colours
  2_3. phase_amplitude_selected.png        — combined phase + amplitude, selected sectors
  case_study_2023.png                      — 2023 record minimum case study
  4.  season_length_timeseries.png         — growth + retreat season length anomaly
  5.  rate_of_change.png                   — advance + retreat rates
  6.  rmse_improvement_by_sector.png       — RMSE comparison
  7.  rolling_variance_phase_amplitude.png — 10-yr rolling std dev
  8.  phase_vs_amplitude_variability.png   — std dev comparison bars
  bridge_phase_as_retreat_advance.png      — Himmich-style retreat/advance
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

# Fixed 1979-2010 baseline anomalies
baselines_doy = (annual[annual["Year"].between(1979, 2010)]
                 .groupby("sector")["max_doy_fitted"].median())
baselines_amp = (annual[annual["Year"].between(1979, 2010)]
                 .groupby("sector")["amplitude_fitted"].median())
annual["max_doy_anom_fixed"]   = (annual["max_doy_fitted"] -
                                   annual["sector"].map(baselines_doy))
annual["amplitude_anom_fixed"] = (annual["amplitude_fitted"] -
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
# FIG 1 — Sector SIE anomaly timeseries + circumpolar (daily)
# =============================================================================
print("\nFig 1: Sector SIE anomaly timeseries")

sie = pd.read_csv(SIE_CSV)
sie["Date"]  = pd.to_datetime(sie["Date"], format="%m/%d/%y")
sie["Year"]  = sie["Date"].dt.year
sie["DOY"]   = sie["Date"].dt.dayofyear
sie["Month"] = sie["Date"].dt.month
sie = sie[sie["Year"].between(1979, 2022)].sort_values("Date").reset_index(drop=True)

ALL_COLS = SECTORS + ["SIE_circumpolar"]

for col in ALL_COLS:
    sie[col] = uniform_filter1d(
        sie[col].fillna(method="ffill").values, size=5, mode="nearest")

# 1979-2010 baseline climatology
clim_sie = (sie[sie["Year"].between(1979, 2010)]
            .groupby("DOY")[ALL_COLS].mean())

for col in ALL_COLS:
    sie[f"{col}_anom"] = sie[col] - sie["DOY"].map(clim_sie[col])

# Annual mean for pre/post mean lines
annual_anom = (sie.groupby("Year")
               [[f"{col}_anom" for col in ALL_COLS]]
               .mean().reset_index())

PLOT_COLS = {
    "SIE_Weddell":                 ("Weddell",        "#2196F3"),
    "SIE_Amundsen_Bellingshausen": ("ABS",             "#F44336"),
    "SIE_Ross":                    ("Ross",            "#4CAF50"),
    "SIE_East_Antarctica":         ("East Antarctica", "#FF9800"),
    "SIE_King_Haakon":             ("King Haakon",     "#9C27B0"),
    "SIE_circumpolar":             ("Circumpolar",     "#2C2C2A"),
}

fig, axes = plt.subplots(1, 6, figsize=(22, 5),
                         sharey=False, sharex=True)

for ax, (sec_col, (sec_label, sec_color)) in zip(axes, PLOT_COLS.items()):
    anom_col = f"{sec_col}_anom"

    # Daily values for fill
    dates = sie["Date"].values
    vals  = sie[anom_col].values

    ax.fill_between(dates, vals, 0,
                    where=vals >= 0, color="#378ADD",
                    alpha=0.8, linewidth=0, zorder=3)
    ax.fill_between(dates, vals, 0,
                    where=vals < 0, color="#D4537E",
                    alpha=0.8, linewidth=0, zorder=3)

    ax.axhline(0, color="#2C2C2A", lw=0.8, zorder=4)

    # 2016 vertical line
    ax.axvline(pd.Timestamp("2016-01-01"), color="#2C2C2A",
               lw=1.2, ls="--", alpha=0.7, zorder=5)

    # Pre/post mean lines from annual means
    ann_vals = annual_anom[anom_col].values
    ann_yrs  = annual_anom["Year"].values
    pre_mean  = np.nanmean(ann_vals[ann_yrs < 2016])
    post_mean = np.nanmean(ann_vals[ann_yrs >= 2016])

    ax.hlines(pre_mean,
              pd.Timestamp("1979-01-01"),
              pd.Timestamp("2015-12-31"),
              colors="#888780", linewidth=1.5,
              linestyle="-", zorder=5)
    ax.hlines(post_mean,
              pd.Timestamp("2016-01-01"),
              pd.Timestamp("2022-12-31"),
              colors="#D4537E", linewidth=1.5,
              linestyle="-", zorder=5)

    ax.text(0.97, 0.04,
            f"Post-2016: {post_mean:+.2f} Mkm²",
            transform=ax.transAxes, fontsize=8,
            color="#D4537E", ha="right",
            path_effects=stroke())

    if sec_col == "SIE_circumpolar":
        ax.spines["left"].set_linewidth(2.0)
        ax.spines["left"].set_color("#B4B2A9")
        ax.set_title(sec_label, fontsize=12,
                     fontweight="bold", pad=8,
                     color=sec_color, style="italic")
    else:
        ax.set_title(sec_label, fontsize=12,
                     fontweight="bold", pad=8)

    ax.tick_params(labelsize=9)
    ax.set_xlim(pd.Timestamp("1979-01-01"),
                pd.Timestamp("2023-01-01"))

    if ax == axes[0]:
        ax.set_ylabel("SIE anomaly (million km²)",
                      fontsize=11, labelpad=8)
    ax.set_xlabel("Year", fontsize=10, color="#5F5E5A")

fig.suptitle(
    "Daily SIE Anomaly by Sector — 1979–2010 Baseline  |  5-day running mean",
    fontsize=14, fontweight="bold", y=1.01
)
fig.tight_layout()
save(fig, "1_sector_sie_anomaly.png")

# =============================================================================
# FIG 2, 3 & COMBINED — Phase and amplitude anomaly figures
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
    title   = "Timing of Sea Ice Maximum — Anomaly from 1979–2010 Baseline",
    outfile = "2_phase_anomaly_timeseries.png"
)

print("Fig 3: Amplitude anomaly timeseries")
plot_anomaly_panel(
    var     = "amplitude_anom_fixed",
    ylabel  = "Amplitude anomaly (million km²)\n← Smaller  |  Larger →",
    title   = "Seasonal Amplitude — Anomaly from 1979–2010 Baseline",
    outfile = "3_amplitude_anomaly_timeseries.png"
)

# =============================================================================
# FIG 2+3 COMBINED — Phase and amplitude, selected sectors
# 2 rows x 3 columns: Weddell, Ross, King Haakon
# =============================================================================
print("Fig 2+3 combined: selected sectors")

SELECTED_SECTORS = {
    "SIE_Weddell":     "Weddell",
    "SIE_Ross":        "Ross",
    "SIE_King_Haakon": "King Haakon",
}

SELECTED_COLORS = {
    "SIE_Weddell":     "#2196F3",
    "SIE_Ross":        "#4CAF50",
    "SIE_King_Haakon": "#9C27B0",
}

fig, axes = plt.subplots(2, 3, figsize=(16, 8),
                         sharex=True, sharey="row")

row_vars    = ["max_doy_anom_fixed", "amplitude_anom_fixed"]
row_ylabels = [
    "Phase anomaly (days)\n← Ahead of phase  |  Behind phase →",
    "Amplitude anomaly (million km²)\n← Smaller  |  Larger →"
]
row_titles  = ["Phase", "Amplitude"]

for row, (var, ylabel) in enumerate(zip(row_vars, row_ylabels)):
    for col, (sec_col, sec_label) in enumerate(SELECTED_SECTORS.items()):
        ax    = axes[row, col]
        sub   = annual[annual["sector"] == sec_col].sort_values("Year")
        yrs   = sub["Year"].values
        vals  = sub[var].values
        color = SELECTED_COLORS[sec_col]

        ax.axhline(0, color="#B4B2A9", lw=0.8, ls="--", zorder=1)
        ax.axvline(2016, color="#D4537E", lw=1.2, ls="--",
                   alpha=0.7, zorder=2)

        for yr, val in zip(yrs, vals):
            ax.scatter(yr, val, color=decade_color(yr),
                       s=50, zorder=4, edgecolors="white", linewidth=0.5)

        if len(vals) >= 5:
            smooth = uniform_filter1d(vals, size=5, mode="nearest")
            ax.plot(yrs, smooth, color=color, lw=2.5,
                    zorder=3, alpha=0.9)

        post_mean = sub[sub["Year"] >= 2016][var].mean()
        ax.text(0.97, 0.04,
                f"Post-2016: {post_mean:+.1f} days" if "doy" in var
                else f"Post-2016: {post_mean:+.3f} Mkm²",
                transform=ax.transAxes, fontsize=9,
                color="#D4537E", ha="right", path_effects=stroke())

        if row == 0:
            ax.set_title(sec_label, fontsize=13,
                         fontweight="bold", pad=10, color=color)

        if col == 0:
            ax.set_ylabel(ylabel, fontsize=10, labelpad=8)

        if col == 2:
            ax.text(1.03, 0.5, row_titles[row],
                    transform=ax.transAxes,
                    fontsize=11, fontweight="bold",
                    color="#2C2C2A", va="center", rotation=270)

        if row == 1:
            ax.set_xlabel("Year", fontsize=10, color="#5F5E5A")

        ax.tick_params(labelsize=10)
        ax.set_xlim(1977, 2024)

handles = [plt.scatter([], [], color=c, s=50,
                       edgecolors="white", linewidth=0.5, label=l)
           for c, l in DECADE_LEGEND]
handles.append(Line2D([0], [0], color="#2C2C2A", lw=2.5,
                       label="5-yr running mean"))

fig.legend(handles=handles, loc="lower center", ncol=6,
           fontsize=10, bbox_to_anchor=(0.5, -0.04), frameon=False)

fig.suptitle(
    "Phase and Amplitude Anomaly — Selected Sectors\n"
    "Anomaly from 1979–2010 Baseline",
    fontsize=14, fontweight="bold", y=1.01
)
fig.tight_layout(rect=[0, 0.04, 0.97, 1])
save(fig, "2_3_phase_amplitude_selected.png")
# =============================================================================
# CASE STUDY — 2023 record minimum
# =============================================================================
print("Case study: 2023")

case_year = 2023
case_data = annual[annual["Year"] == case_year].copy()
case_data = case_data.set_index("sector")

sector_order_case = ["SIE_Weddell", "SIE_Amundsen_Bellingshausen",
                     "SIE_Ross", "SIE_East_Antarctica", "SIE_King_Haakon"]
labels_case = [SECTOR_LABELS[s] for s in sector_order_case]
colors_case = [SECTOR_COLORS[s] for s in sector_order_case]

phase_vals = [float(case_data.loc[s, "max_doy_anom"]) for s in sector_order_case]
amp_vals   = [float(case_data.loc[s, "amplitude_anom"]) for s in sector_order_case]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, vals, title, ylabel, fmt in zip(
    axes,
    [phase_vals, amp_vals],
    ["Phase anomaly — 2023",
     "Amplitude anomaly — 2023"],
    ["Days (negative = ahead of phase)",
     "Million km² (negative = smaller cycle)"],
    ["{:+.0f}d", "{:+.2f}"]
):
    bars = ax.bar(labels_case, vals,
                  color=colors_case,
                  width=0.6, edgecolor="white", zorder=3)

    ax.axhline(0, color="#2C2C2A", lw=0.8, zorder=4)

    for bar, val in zip(bars, vals):
        ypos = (bar.get_height() + max(abs(v) for v in vals) * 0.03
                if val >= 0
                else bar.get_height() - max(abs(v) for v in vals) * 0.06)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                fmt.format(val),
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                color="#2C2C2A", path_effects=stroke())

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.tick_params(axis="x", rotation=20, labelsize=11)
    ax.tick_params(axis="y", labelsize=10)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    yabs = max(abs(v) for v in vals)
    ax.set_ylim(-yabs * 1.4, yabs * 1.4)

fig.suptitle(
    "2023 — Record Minimum: Same Signal, Different Mechanisms\n"
    "APAC phase and amplitude anomalies by sector",
    fontsize=13, fontweight="bold", y=1.02
)
fig.tight_layout()
save(fig, "case_study_2023.png")

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
            else bar.get_height() - (max(abs(v) for v in vals) * 0.14))
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

zero_line(ax)
ax.set_ylabel("RMSE improvement over invariant cycle (%)")
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

        if sec == "SIE_King_Haakon":
            roll = roll[roll.index >= 1988]

        ax.plot(roll.index, roll.values,
                color=SECTOR_COLORS[sec], lw=2, zorder=3)
        ax.fill_between(roll.index, roll.values,
                        color=SECTOR_COLORS[sec], alpha=0.15, zorder=2)
        shade2016(ax)
        ax.set_xlim(yr_min + 4, yr_max - 4)

        if row == 0:
            ax.set_title(SECTOR_LABELS[sec], fontweight="bold", fontsize=11)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if row == 1:
            ax.set_xlabel("Year", fontsize=9)
            ax.tick_params(axis="x", rotation=30)

for row, row_title in enumerate(row_titles):
    axes[row, 0].set_ylabel(row_title, fontsize=9)

fig.suptitle("Has Variability Changed Over Time? Phase and Amplitude",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "7_rolling_variance_phase_amplitude.png")

# =============================================================================
# FIG 8 — Phase vs amplitude variability — pre/post 2016 grouped bars
# =============================================================================
print("Fig 8: Phase vs amplitude variability pre/post 2016")

pre  = annual[annual["Year"] <  2016]
post = annual[annual["Year"] >= 2016]

phase_pre  = [pre[pre["sector"]==s]["max_doy_anom"].dropna().std()  for s in SECTORS]
phase_post = [post[post["sector"]==s]["max_doy_anom"].dropna().std() for s in SECTORS]
amp_pre    = [pre[pre["sector"]==s]["amplitude_anom"].dropna().std()  for s in SECTORS]
amp_post   = [post[post["sector"]==s]["amplitude_anom"].dropna().std() for s in SECTORS]

labels = [SECTOR_LABELS[s] for s in SECTORS]
colors = [SECTOR_COLORS[s] for s in SECTORS]

x      = np.arange(len(SECTORS))
width  = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, pre_vals, post_vals, ylabel, title, fmt in zip(
    axes,
    [phase_pre,  amp_pre],
    [phase_post, amp_post],
    ["Std dev of phase anomaly (days)",
     "Std dev of amplitude anomaly (million km²)"],
    ["Phase Variability\n(timing of maximum)",
     "Amplitude Variability\n(size of seasonal cycle)"],
    ["{:.1f}d", "{:.2f}"]
):
    bars_pre  = ax.bar(x - width/2, pre_vals,  width,
                       color=colors, alpha=1.0,
                       edgecolor="white", label="1979–2015", zorder=3)
    bars_post = ax.bar(x + width/2, post_vals, width,
                       color=colors, alpha=0.45,
                       edgecolor="white", label="2016–2022",
                       hatch="///", zorder=3)

    # Value labels
    for bar, val in zip(bars_pre, pre_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(pre_vals) * 0.02,
                fmt.format(val),
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                path_effects=stroke())

    for bar, val in zip(bars_post, post_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(pre_vals) * 0.02,
                fmt.format(val),
                ha="center", va="bottom",
                fontsize=9, color="#5F5E5A",
                path_effects=stroke())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylim(0, max(max(pre_vals), max(post_vals)) * 1.3)
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Note about post-2016 sample size
fig.text(0.5, -0.02,
         "Note: post-2016 period spans only 7 years (2016–2022) — "
         "std dev estimates are less stable",
         ha="center", fontsize=9, color="#5F5E5A",
         style="italic")

fig.suptitle("Has Variability Changed? Pre vs Post 2016",
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