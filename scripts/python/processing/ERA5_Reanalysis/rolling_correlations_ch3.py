"""
rolling_window_correlations.py

Computes rolling window Pearson correlations between APAC phase/amplitude
anomalies and atmospheric indices to test whether atmospheric sensitivity
of the Antarctic sea ice seasonal cycle has changed over time.

This is the key analysis for Chapter 3, Option 3:
"Has the sensitivity of the Antarctic sea ice seasonal cycle to
atmospheric variability changed in recent decades?"

APPROACH:
- For a sliding window of W years, compute r between a chosen index and
  a chosen APAC variable
- Plot r as a function of the window centre year
- A declining r after 2016 is evidence of weakening atmospheric control
- The full-record r is shown as a horizontal reference line

INPUTS:
- master_index_detrended.csv  : detrended atmospheric indices (from main script)
- annual_params.csv           : APAC phase and amplitude anomalies by sector

OUTPUTS:
- rolling_window_correlations.png  : main proposal figure (3 panels)
- rolling_window_all.csv           : full rolling window results table

Author: generated for Frida A. Perez dissertation Chapter 3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import os

# =============================================================================
# 0. PATHS — update these to match your directory structure
# =============================================================================

FIGURES_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/"
ANNUAL_CSV  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data/annual_params.csv"
INDEX_CSV   = os.path.join(FIGURES_DIR, "master_index_detrended.csv")
OUTPUT_DIR  = FIGURES_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEAR_MIN = 1979
YEAR_MAX = 2023
REGIME_SHIFT_YEAR = 2016  # vertical line marking the post-2016 period

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("Loading data...")

# Detrended atmospheric indices (saved by main correlations script)
idx = pd.read_csv(INDEX_CSV)
print(f"  Index table: {idx.shape} — years {idx['Year'].min()}–{idx['Year'].max()}")

# APAC annual parameters — phase and amplitude anomalies by sector
annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(YEAR_MIN, YEAR_MAX)]

# Map sector column names to readable labels
SECTORS = {
    "SIE_Weddell":                 "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross":                    "Ross",
    "SIE_East_Antarctica":         "East Antarctica",
    "SIE_King_Haakon":             "King Haakon",
}

# =============================================================================
# 2. DETREND APAC VARIABLES
# =============================================================================
# We detrend here as well to match what the main script does.
# Even though the master index CSV is already detrended, the APAC annual
# params need to be detrended separately.

def detrend_series(x, y):
    """
    Remove linear trend from y as a function of x.
    Returns detrended y values.

    This isolates interannual variability by removing the long-term
    linear drift — necessary so that shared trends don't produce
    spurious correlations.
    """
    mask = ~np.isnan(y)
    if mask.sum() < 5:
        return y
    slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
    return y - (slope * x + intercept)


# Detrend phase and amplitude for each sector
annual_dt = []
for sec_col, sec_label in SECTORS.items():
    sec = annual[annual["sector"] == sec_col].copy().sort_values("Year")
    x = sec["Year"].values.astype(float)
    for var in ["max_doy_anom", "amplitude_anom"]:
        sec[var] = detrend_series(x, sec[var].values.astype(float))
    annual_dt.append(sec)
annual_dt = pd.concat(annual_dt)

print("Detrending complete")

# =============================================================================
# 3. ROLLING WINDOW CORRELATION FUNCTION
# =============================================================================

def rolling_corr(apac_df, sector_col, apac_var, idx_df, idx_col,
                 window=15, year_min=YEAR_MIN, year_max=YEAR_MAX):
    """
    Compute Pearson r in a sliding window of W years.

    HOW IT WORKS:
    For each centre year t, we take years [t - W//2, t + W//2] and compute
    the correlation between the APAC variable and the atmospheric index
    within that window. The result is a time series of r values showing
    how the relationship has evolved.

    Parameters:
        apac_df    : DataFrame with APAC annual parameters (detrended)
        sector_col : sector column name e.g. "SIE_East_Antarctica"
        apac_var   : "max_doy_anom" (phase) or "amplitude_anom"
        idx_df     : DataFrame with detrended atmospheric indices
        idx_col    : index column name e.g. "SAM_annual"
        window     : number of years in each window (default 15)

    Returns:
        DataFrame with columns: centre_year, r, p, n
    """
    half = window // 2
    sec_data = apac_df[apac_df["sector"] == sector_col][["Year", apac_var]].dropna()

    results = []
    for centre in range(year_min + half, year_max - half + 1):
        yr_start = centre - half
        yr_end   = centre + half

        # Slice the window
        sec_win = sec_data[sec_data["Year"].between(yr_start, yr_end)]
        idx_win = idx_df[idx_df["Year"].between(yr_start, yr_end)][["Year", idx_col]].dropna()

        # Merge on year — only keep years where both variables have data
        merged = sec_win.merge(idx_win, on="Year", how="inner").dropna()

        if len(merged) < 8:
            # Skip windows with too few data points for a meaningful test
            continue

        r, p = stats.pearsonr(merged[idx_col].values, merged[apac_var].values)
        results.append({
            "centre_year": centre,
            "r":           round(r, 3),
            "p":           round(p, 4),
            "n":           len(merged),
            "sig":         "*" if p < 0.05 else ("." if p < 0.10 else ""),
        })

    return pd.DataFrame(results)


def full_record_corr(apac_df, sector_col, apac_var, idx_df, idx_col):
    """
    Compute the full-record Pearson r for reference on the rolling plot.
    Returns (r, p).
    """
    sec_data = apac_df[apac_df["sector"] == sector_col][["Year", apac_var]].dropna()
    merged = sec_data.merge(idx_df[["Year", idx_col]].dropna(), on="Year", how="inner").dropna()
    if len(merged) < 5:
        return np.nan, np.nan
    r, p = stats.pearsonr(merged[idx_col].values, merged[apac_var].values)
    return round(r, 3), round(p, 4)


# =============================================================================
# 4. DEFINE PAIRS TO PLOT
# =============================================================================
# These are the three physically motivated pairs from the main analysis:
#
# Panel 1: SAM annual ~ EA amplitude (r=0.47 full record)
#   → SAM+ strengthens westerlies → Ekman transport → more ice grows
#   → If sensitivity is weakening, this r should decline post-2016
#
# Panel 2: ZW3R annual ~ Ross amplitude (r=-0.41 full record)
#   → ZW3 controls cold air export from Ross → sets amplitude
#   → Ross amplitude most affected by post-2016 ocean warming
#
# Panel 3: ASL DJF ~ ABS phase (r=-0.36 full record)
#   → Deep ASL drives early retreat timing in ABS
#   → Phase is still atmospherically forced — expect r to remain stable

PAIRS = [
    {
        "sector_col":  "SIE_East_Antarctica",
        "sector_label": "East Antarctica",
        "apac_var":    "amplitude_anom",
        "apac_label":  "Amplitude anomaly",
        "idx_col":     "SAM_annual",
        "idx_label":   "SAM annual",
        "color":       "#185FA5",   # blue
        "hypothesis":  "Expect decline post-2016 if amplitude decoupling",
    },
    {
        "sector_col":  "SIE_Ross",
        "sector_label": "Ross",
        "apac_var":    "amplitude_anom",
        "apac_label":  "Amplitude anomaly",
        "idx_col":     "ZW3R_annual",
        "idx_label":   "ZW3 annual",
        "color":       "#1D9E75",   # teal
        "hypothesis":  "Expect decline post-2016 if ocean dominates Ross",
    },
    {
        "sector_col":  "SIE_Amundsen_Bellingshausen",
        "sector_label": "ABS",
        "apac_var":    "max_doy_anom",
        "apac_label":  "Phase anomaly",
        "idx_col":     "ASL_DJF",
        "idx_label":   "ASL DJF",
        "color":       "#D85A30",   # coral
        "hypothesis":  "Expect stability — phase still atmospherically forced",
    },
]

WINDOW = 12  # years — smaller window gives more post-2016 data points

# =============================================================================
# 5. COMPUTE ROLLING CORRELATIONS
# =============================================================================

print(f"\nComputing rolling window correlations (window = {WINDOW} years)...")
all_results = []

for pair in PAIRS:
    df = rolling_corr(
        annual_dt, pair["sector_col"], pair["apac_var"],
        idx, pair["idx_col"], window=WINDOW
    )
    df["sector"]     = pair["sector_label"]
    df["apac_var"]   = pair["apac_label"]
    df["index"]      = pair["idx_col"]
    df["idx_label"]  = pair["idx_label"]
    df["color"]      = pair["color"]
    pair["rolling"]  = df

    r_full, p_full = full_record_corr(
        annual_dt, pair["sector_col"], pair["apac_var"],
        idx, pair["idx_col"]
    )
    pair["r_full"] = r_full
    pair["p_full"] = p_full
    all_results.append(df)

    print(f"  {pair['sector_label']} {pair['apac_label']} ~ {pair['idx_label']}: "
          f"full r={r_full}, p={p_full}, {len(df)} windows")

# Save full results
all_df = pd.concat(all_results)
all_df.to_csv(os.path.join(OUTPUT_DIR, "rolling_window_all.csv"), index=False)
print(f"\nRolling window results saved: {len(all_df)} rows")

# =============================================================================
# 6. FIGURE
# =============================================================================
# Three-panel figure showing rolling window r over time for each pair.
# Layout:
#   - Each panel: rolling r (coloured line) + full-record r (dashed)
#   - Shaded region: post-2016 period (the new regime)
#   - Horizontal zero line for reference
#   - Significance markers where p < 0.05

print("\nGenerating figure...")

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
fig.subplots_adjust(hspace=0.08)

for ax, pair in zip(axes, PAIRS):
    df = pair["rolling"]
    r_full = pair["r_full"]
    color  = pair["color"]

    # --- Post-2016 shaded region ---
    # Shade from 2016 to end of record to highlight the new regime
    ax.axvspan(REGIME_SHIFT_YEAR, YEAR_MAX + 1, alpha=0.08,
               color="#E24B4A", zorder=0, label="Post-2016")

    # --- Zero reference line ---
    ax.axhline(0, color="#888780", linewidth=0.5,
               linestyle="-", alpha=0.4, zorder=1)

    # --- Full-record r (horizontal dashed line) ---
    # This is the correlation computed over all 44 years — the reference value
    ax.axhline(r_full, color=color, linewidth=1.2, linestyle="--",
               alpha=0.6, zorder=2,
               label=f"Full record r = {r_full}")

    # --- Rolling r line ---
    ax.plot(df["centre_year"], df["r"], color=color, linewidth=2.0,
            zorder=3, label=f"{WINDOW}-yr rolling r")

    # --- Mark windows where p < 0.05 with filled circles ---
    sig = df[df["sig"] == "*"]
    ax.scatter(sig["centre_year"], sig["r"], color=color, s=30,
               zorder=4, alpha=0.8)

    # --- Mark windows where p >= 0.05 with open circles ---
    nonsig = df[df["sig"] != "*"]
    ax.scatter(nonsig["centre_year"], nonsig["r"], color=color, s=20,
               zorder=4, alpha=0.4, facecolors="none",
               edgecolors=color, linewidths=0.8)

    # --- Axis formatting ---
    ax.set_ylim(-0.85, 0.85)
    ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax.set_ylabel("Pearson r", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Panel label ---
    # e.g. "East Antarctica amplitude ~ SAM annual"
    panel_title = (f"{pair['sector_label']} {pair['apac_label'].lower()} "
                   f"~ {pair['idx_label']}")
    ax.text(0.02, 0.93, panel_title, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top",
            color=color)

    # --- Legend (first panel only) ---
    if ax == axes[0]:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Line2D([0], [0], color="gray", linewidth=2, label=f"{WINDOW}-yr rolling r"),
            Line2D([0], [0], color="gray", linewidth=1.2, linestyle="--",
                   label="Full-record r"),
            Line2D([0], [0], marker="o", color="gray", markersize=5,
                   linewidth=0, label="p < 0.05"),
            Line2D([0], [0], marker="o", color="gray", markersize=5,
                   linewidth=0, markerfacecolor="none", label="p ≥ 0.05"),
            Patch(facecolor="#E24B4A", alpha=0.15, label="Post-2016"),
        ]
        ax.legend(handles=legend_elements, loc="upper right",
                  fontsize=8, framealpha=0.9)

# --- X axis ---
axes[-1].set_xlabel("Window centre year", fontsize=10)
axes[-1].set_xlim(YEAR_MIN + WINDOW // 2 - 1, YEAR_MAX - WINDOW // 2 + 1)
axes[-1].set_xticks(range(1990, 2020, 5))

# --- Overall title ---
fig.suptitle(
    "Rolling window correlations: atmospheric sensitivity of the seasonal cycle\n"
    f"(window = {WINDOW} years, filled circles = p < 0.05)",
    fontsize=11, y=0.98
)

outpath = os.path.join(OUTPUT_DIR, "rolling_window_correlations.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved: {outpath}")

# =============================================================================
# 7. SUMMARY TABLE — pre vs post 2016
# =============================================================================
# For each pair, compare the mean rolling r before and after 2016
# This gives a quantitative summary of how much each relationship has changed

print("\n=== Rolling window summary: pre vs post 2016 ===")
print(f"  Using window end year (centre + {WINDOW//2}) to define pre/post 2016")
print(f"{'Pair':<45} {'Pre-2016 mean r':>15} {'Post-2016 mean r':>16} {'Change':>8}")
print("-" * 90)

for pair in PAIRS:
    df = pair["rolling"].copy()
    # Use window end year = centre + half window
    df["end_year"] = df["centre_year"] + WINDOW // 2
    pre  = df[df["end_year"] <= REGIME_SHIFT_YEAR]["r"].mean()
    post = df[df["end_year"] >  REGIME_SHIFT_YEAR]["r"].mean()
    label = (f"{pair['sector_label']} {pair['apac_label'].lower()[:3]} "
             f"~ {pair['idx_label']}")
    change_str = f"{post-pre:+.3f}" if not np.isnan(post) else "  n/a"
    pre_str  = f"{pre:+.3f}"  if not np.isnan(pre)  else "   n/a"
    post_str = f"{post:+.3f}" if not np.isnan(post) else "    n/a"
    print(f"  {label:<43} {pre_str:>15} {post_str:>16} {change_str:>8}")