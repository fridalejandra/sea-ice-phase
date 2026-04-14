"""
rolling_window_correlations.py

Computes trailing rolling-window Pearson correlations between APAC
phase/amplitude anomalies and atmospheric indices to test whether
atmospheric sensitivity of the Antarctic sea ice seasonal cycle
has changed over time.

KEY CHANGE FROM THE ORIGINAL VERSION
------------------------------------
This version uses TRAILING windows rather than CENTERED windows.

Original behavior:
- With a 15-year centered window and data ending in 2023,
  the final valid centre year is 2016.
- That made the plot look like it stopped in 2016.

New behavior:
- Each rolling value is assigned to the WINDOW END YEAR.
- A 15-year window ending in 2023 uses 2009–2023.
- The x-axis therefore runs through 2023.

INPUTS:
- master_index_detrended.csv  : detrended atmospheric indices
- annual_params.csv           : APAC phase and amplitude anomalies by sector

OUTPUTS:
- rolling_window_correlations.png
- rolling_window_all.csv

Author: cleaned and updated for Frida A. Perez dissertation Chapter 3
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# =============================================================================
# 0. PATHS
# =============================================================================

FIGURES_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/"
ANNUAL_CSV = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data/annual_params.csv"
INDEX_CSV = os.path.join(FIGURES_DIR, "master_index_detrended.csv")
OUTPUT_DIR = FIGURES_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEAR_MIN = 1979
YEAR_MAX = 2023
REGIME_SHIFT_YEAR = 2016
WINDOW = 15  # trailing window length in years

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("Loading data...")

idx = pd.read_csv(INDEX_CSV)
print(f"  Index table: {idx.shape} — years {idx['Year'].min()}–{idx['Year'].max()}")

annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(YEAR_MIN, YEAR_MAX)].copy()

print(f"  APAC table: {annual.shape} — years {annual['Year'].min()}–{annual['Year'].max()}")

SECTORS = {
    "SIE_Weddell": "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross": "Ross",
    "SIE_East_Antarctica": "East Antarctica",
    "SIE_King_Haakon": "King Haakon",
}

# =============================================================================
# 2. DETREND APAC VARIABLES
# =============================================================================

def detrend_series(x, y):
    """
    Remove linear trend from y as a function of x.
    Returns detrended y values.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return y

    slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])

    y_dt = y.copy()
    y_dt[mask] = y[mask] - (slope * x[mask] + intercept)
    return y_dt


annual_dt_list = []

for sec_col in SECTORS:
    sec = annual[annual["sector"] == sec_col].copy().sort_values("Year")
    x = sec["Year"].values.astype(float)

    for var in ["max_doy_anom", "amplitude_anom"]:
        sec[var] = detrend_series(x, sec[var].values.astype(float))

    annual_dt_list.append(sec)

annual_dt = pd.concat(annual_dt_list, ignore_index=True)
print("Detrending complete")

# =============================================================================
# 3. CORRELATION FUNCTIONS
# =============================================================================

def rolling_corr(apac_df, sector_col, apac_var, idx_df, idx_col,
                 window=WINDOW, year_min=YEAR_MIN, year_max=YEAR_MAX):
    """
    Compute Pearson r in a trailing rolling window of length `window`.

    For each end year t, compute the correlation over:
        [t - window + 1, t]

    Returns a DataFrame with columns:
        year, window_start, window_end, r, p, n, sig
    """
    sec_data = (
        apac_df[apac_df["sector"] == sector_col][["Year", apac_var]]
        .dropna()
        .copy()
    )

    results = []

    for end_year in range(year_min + window - 1, year_max + 1):
        start_year = end_year - window + 1

        sec_win = sec_data[sec_data["Year"].between(start_year, end_year)]
        idx_win = idx_df[idx_df["Year"].between(start_year, end_year)][["Year", idx_col]].dropna()

        merged = sec_win.merge(idx_win, on="Year", how="inner").dropna()

        if len(merged) < 8:
            continue

        r, p = stats.pearsonr(merged[idx_col].values, merged[apac_var].values)

        results.append({
            "year": end_year,
            "window_start": start_year,
            "window_end": end_year,
            "r": round(r, 3),
            "p": round(p, 4),
            "n": len(merged),
            "sig": "*" if p < 0.05 else ("." if p < 0.10 else ""),
        })

    return pd.DataFrame(results)


def full_record_corr(apac_df, sector_col, apac_var, idx_df, idx_col):
    """
    Compute full-record Pearson r for reference.
    """
    sec_data = apac_df[apac_df["sector"] == sector_col][["Year", apac_var]].dropna()
    idx_data = idx_df[["Year", idx_col]].dropna()

    merged = sec_data.merge(idx_data, on="Year", how="inner").dropna()

    if len(merged) < 5:
        return np.nan, np.nan

    r, p = stats.pearsonr(merged[idx_col].values, merged[apac_var].values)
    return round(r, 3), round(p, 4)

# =============================================================================
# 4. DEFINE PAIRS TO PLOT
# =============================================================================

PAIRS = [
    {
        "sector_col": "SIE_East_Antarctica",
        "sector_label": "East Antarctica",
        "apac_var": "amplitude_anom",
        "apac_label": "Amplitude anomaly",
        "idx_col": "SAM_annual",
        "idx_label": "SAM annual",
        "color": "#185FA5",
        "hypothesis": "Expect decline post-2016 if amplitude decoupling",
    },
    {
        "sector_col": "SIE_Ross",
        "sector_label": "Ross",
        "apac_var": "amplitude_anom",
        "apac_label": "Amplitude anomaly",
        "idx_col": "ZW3R_annual",
        "idx_label": "ZW3 annual",
        "color": "#1D9E75",
        "hypothesis": "Expect decline post-2016 if ocean dominates Ross",
    },
    {
        "sector_col": "SIE_Amundsen_Bellingshausen",
        "sector_label": "ABS",
        "apac_var": "max_doy_anom",
        "apac_label": "Phase anomaly",
        "idx_col": "ASL_DJF",
        "idx_label": "ASL DJF",
        "color": "#D85A30",
        "hypothesis": "Expect stability — phase still atmospherically forced",
    },
]

# =============================================================================
# 5. COMPUTE ROLLING CORRELATIONS
# =============================================================================

print(f"\nComputing rolling correlations (trailing window = {WINDOW} years)...")

all_results = []

for pair in PAIRS:
    df = rolling_corr(
        annual_dt,
        pair["sector_col"],
        pair["apac_var"],
        idx,
        pair["idx_col"],
        window=WINDOW,
        year_min=YEAR_MIN,
        year_max=YEAR_MAX,
    )

    df["sector"] = pair["sector_label"]
    df["apac_var"] = pair["apac_label"]
    df["index"] = pair["idx_col"]
    df["idx_label"] = pair["idx_label"]
    df["color"] = pair["color"]

    pair["rolling"] = df

    r_full, p_full = full_record_corr(
        annual_dt,
        pair["sector_col"],
        pair["apac_var"],
        idx,
        pair["idx_col"],
    )

    pair["r_full"] = r_full
    pair["p_full"] = p_full

    all_results.append(df)

    print(
        f"  {pair['sector_label']} {pair['apac_label']} ~ {pair['idx_label']}: "
        f"full r={r_full}, p={p_full}, {len(df)} windows"
    )

all_df = pd.concat(all_results, ignore_index=True)
all_csv_path = os.path.join(OUTPUT_DIR, "rolling_window_all.csv")
all_df.to_csv(all_csv_path, index=False)
print(f"\nRolling window results saved: {all_csv_path}")
print(f"  Total rows: {len(all_df)}")

# =============================================================================
# 6. FIGURE
# =============================================================================

print("\nGenerating figure...")

n_panels = len(PAIRS)
fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.2 * n_panels), sharex=True)

if n_panels == 1:
    axes = [axes]

fig.subplots_adjust(hspace=0.12, top=0.94, bottom=0.12, left=0.08, right=0.97)

for ax, pair in zip(axes, PAIRS):
    df = pair["rolling"]
    r_full = pair["r_full"]
    color = pair["color"]

    if df.empty:
        ax.text(0.5, 0.5, "No valid rolling windows", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        ax.set_axis_off()
        continue

    n_win = df["n"].values
    r_vals = df["r"].values

    # Post-2016 shaded region
    ax.axvspan(REGIME_SHIFT_YEAR, YEAR_MAX + 0.5, alpha=0.08,
               color="#E24B4A", zorder=0)

    # Zero reference line
    ax.axhline(0, color="#888780", linewidth=0.5,
               linestyle="-", alpha=0.4, zorder=1)

    # Full-record reference line
    ax.axhline(r_full, color=color, linewidth=1.2, linestyle="--",
               alpha=0.55, zorder=2)

    # 95% CI using Fisher z-transform
    ci_lower = np.full(len(r_vals), np.nan)
    ci_upper = np.full(len(r_vals), np.nan)

    for i, (r_i, n_i) in enumerate(zip(r_vals, n_win)):
        if n_i > 4 and np.isfinite(r_i) and abs(r_i) < 1:
            z_i = np.arctanh(np.clip(r_i, -0.9999, 0.9999))
            se = 1.0 / np.sqrt(n_i - 3)
            ci_lower[i] = np.tanh(z_i - 1.96 * se)
            ci_upper[i] = np.tanh(z_i + 1.96 * se)

    ax.fill_between(
        df["year"].values,
        ci_lower,
        ci_upper,
        color=color,
        alpha=0.12,
        zorder=2,
        label="95% CI",
    )

    # Rolling line
    ax.plot(df["year"], df["r"], color=color, linewidth=2.0, zorder=3)

    # Significance markers
    sig = df[df["sig"] == "*"]
    nonsig = df[df["sig"] != "*"]

    ax.scatter(sig["year"], sig["r"], color=color, s=30, zorder=4, alpha=0.85)
    ax.scatter(
        nonsig["year"],
        nonsig["r"],
        color=color,
        s=20,
        zorder=4,
        alpha=0.4,
        facecolors="none",
        edgecolors=color,
        linewidths=0.8,
    )

    # Axis formatting
    ax.set_ylim(-0.90, 0.90)
    ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax.set_ylabel("Pearson r", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    panel_title = f"{pair['sector_label']} {pair['apac_label'].lower()} ~ {pair['idx_label']}"
    ax.text(
        0.01,
        0.05,
        panel_title,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        color=color,
        clip_on=False,
    )

legend_elements = [
    Line2D([0], [0], color="gray", linewidth=2, label=f"{WINDOW}-yr rolling r"),
    Line2D([0], [0], color="gray", linewidth=1.2, linestyle="--", label="Full-record r"),
    Patch(facecolor="gray", alpha=0.2, label="95% CI"),
    Line2D([0], [0], marker="o", color="gray", markersize=5, linewidth=0, label="p < 0.05"),
    Line2D([0], [0], marker="o", color="gray", markersize=5, linewidth=0,
           markerfacecolor="none", label="p ≥ 0.05"),
    Patch(facecolor="#E24B4A", alpha=0.15, label="Post-2016"),
]

axes[-1].legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.24),
    ncol=3,
    fontsize=8.5,
    frameon=False,
)

axes[-1].set_xlabel("Window end year", fontsize=10)
axes[-1].set_xlim(YEAR_MIN + WINDOW - 1, YEAR_MAX)

ticks = list(range(1995, YEAR_MAX + 1, 5))
if YEAR_MAX not in ticks:
    ticks.append(YEAR_MAX)
axes[-1].set_xticks(sorted(ticks))

fig.suptitle(
    "Rolling window correlations: atmospheric sensitivity of the seasonal cycle\n"
    f"(trailing window = {WINDOW} years, shaded = 95% CI, filled circles = p < 0.05)",
    fontsize=11
)

outpath = os.path.join(OUTPUT_DIR, "rolling_window_correlations.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Figure saved: {outpath}")

# =============================================================================
# 7. SUMMARY TABLE — pre vs post 2016
# =============================================================================

print("\n=== Rolling window summary: pre vs post 2016 ===")
print("  Using window end year to define pre/post 2016")
print(f"{'Pair':<45} {'Pre-2016 mean r':>15} {'Post-2016 mean r':>16} {'Change':>8}")
print("-" * 90)

for pair in PAIRS:
    df = pair["rolling"].copy()

    pre = df[df["year"] <= REGIME_SHIFT_YEAR]["r"].mean()
    post = df[df["year"] > REGIME_SHIFT_YEAR]["r"].mean()

    label = f"{pair['sector_label']} {pair['apac_label'].lower()[:3]} ~ {pair['idx_label']}"

    pre_str = f"{pre:+.3f}" if pd.notna(pre) else "n/a"
    post_str = f"{post:+.3f}" if pd.notna(post) else "n/a"
    change_str = f"{post - pre:+.3f}" if pd.notna(pre) and pd.notna(post) else "n/a"

    print(f"  {label:<43} {pre_str:>15} {post_str:>16} {change_str:>8}")

print("\nDone.")