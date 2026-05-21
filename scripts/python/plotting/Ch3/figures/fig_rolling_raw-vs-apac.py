"""
fig_rolling_raw_vs_apac.py

Advisor figure: raw vs APAC rolling correlations side by side for selected pairs.
Shows where the APAC decomposition changes the story vs raw phase/amplitude.

Layout: N rows × 2 columns (left=raw, right=APAC)
Same y-axis scale per row for direct visual comparison.

Pairs selected based on largest raw vs APAC divergence:
  Phase:
    1. EA ~ SAM RET         (raw=+0.494 vs APAC=-0.069) — sign flip, strongest raw result vanishes
    2. ABS ~ Nino34 DJF     (raw=-0.025 vs APAC=-0.393) — near zero becomes strong
    3. Ross ~ Nino34 RET    (raw=-0.050 vs APAC=-0.244) — same pattern
    4. Weddell ~ SAM RET    (raw=+0.106 vs APAC=+0.251) — APAC strengthens
  Amplitude:
    5. King Haakon ~ Nino34 annual (raw=-0.420 vs APAC=-0.415) — stable, contrast case
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
import os

# =============================================================================
# 0. PATHS
# =============================================================================

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data/"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch3/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")
INDEX_CSV  = os.path.join(DATA_DIR, "master_index_detrended.csv")

YEAR_MIN          = 1979
YEAR_MAX          = 2023
REGIME_SHIFT_YEAR = 2016
WINDOW            = 10

# =============================================================================
# 1. PAIRS TO PLOT
# =============================================================================
# Each entry defines one row (one pair).
# raw_var and apac_var are the column names in annual_params.csv.
# idx_col is the column name in master_index_detrended.csv.
# r_raw and r_apac are full-record r from correlations_output.csv (for annotation).

PAIRS = [
    {
        "label":     "EA phase ~ SAM RET",
        "sector":    "SIE_East_Antarctica",
        "raw_var":   "max_doy_raw_anom",
        "apac_var":  "max_doy_anom",
        "idx_col":   "SAM_RET",
        "color":     "#FF9800",
        "r_raw":     +0.494,
        "r_apac":    -0.069,
        "var_type":  "phase",
    },
    {
        "label":     "ABS phase ~ Niño3.4 DJF",
        "sector":    "SIE_Amundsen_Bellingshausen",
        "raw_var":   "max_doy_raw_anom",
        "apac_var":  "max_doy_anom",
        "idx_col":   "Nino34_DJF",
        "color":     "#F44336",
        "r_raw":     -0.025,
        "r_apac":    -0.393,
        "var_type":  "phase",
    },
    {
        "label":     "Ross phase ~ Niño3.4 RET",
        "sector":    "SIE_Ross",
        "raw_var":   "max_doy_raw_anom",
        "apac_var":  "max_doy_anom",
        "idx_col":   "Nino34_RET",
        "color":     "#4CAF50",
        "r_raw":     -0.050,
        "r_apac":    -0.244,
        "var_type":  "phase",
    },
    {
        "label":     "Weddell phase ~ SAM RET",
        "sector":    "SIE_Weddell",
        "raw_var":   "max_doy_raw_anom",
        "apac_var":  "max_doy_anom",
        "idx_col":   "SAM_RET",
        "color":     "#2196F3",
        "r_raw":     +0.106,
        "r_apac":    +0.251,
        "var_type":  "phase",
    },
    {
        "label":     "King Haakon amplitude ~ Niño3.4 annual",
        "sector":    "SIE_King_Haakon",
        "raw_var":   "amplitude_raw_anom",
        "apac_var":  "amplitude_anom",
        "idx_col":   "Nino34_annual",
        "color":     "#9C27B0",
        "r_raw":     -0.420,
        "r_apac":    -0.415,
        "var_type":  "amplitude",
    },
]

# =============================================================================
# 2. LOAD & DETREND
# =============================================================================

print("Loading data...")
idx    = pd.read_csv(INDEX_CSV)
annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(YEAR_MIN, YEAR_MAX)]

def detrend_series(x, y):
    mask = ~np.isnan(y)
    if mask.sum() < 5:
        return y
    slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
    return y - (slope * x + intercept)

# Detrend all relevant variables per sector
annual_dt = []
for sec_col in annual["sector"].unique():
    sec = annual[annual["sector"] == sec_col].copy().sort_values("Year")
    x   = sec["Year"].values.astype(float)
    for var in ["max_doy_anom", "amplitude_anom",
                "max_doy_raw_anom", "amplitude_raw_anom"]:
        if var in sec.columns:
            sec[var] = detrend_series(x, sec[var].values.astype(float))
    annual_dt.append(sec)
annual_dt = pd.concat(annual_dt)
print("Detrending complete")

# =============================================================================
# 3. ROLLING CORRELATION
# =============================================================================

def rolling_corr(apac_df, sector_col, var_col, idx_df, idx_col, window=WINDOW):
    half     = window // 2
    sec_data = apac_df[apac_df["sector"] == sector_col][["Year", var_col]].dropna()
    results  = []
    for centre in range(YEAR_MIN + half, YEAR_MAX - half + 1):
        sec_win = sec_data[sec_data["Year"].between(centre - half, centre + half)]
        idx_win = idx_df[idx_df["Year"].between(centre - half, centre + half)][["Year", idx_col]].dropna()
        merged  = sec_win.merge(idx_win, on="Year", how="inner").dropna()
        if len(merged) < 8:
            continue
        r, p = stats.pearsonr(merged[idx_col].values, merged[var_col].values)
        results.append({
            "centre_year": centre,
            "r":  round(r, 3),
            "p":  round(p, 4),
            "n":  len(merged),
            "sig": "*" if p < 0.05 else ("." if p < 0.10 else ""),
        })
    return pd.DataFrame(results)

# =============================================================================
# 4. COMPUTE ALL ROLLING CORRELATIONS
# =============================================================================

print(f"\nComputing rolling correlations (window={WINDOW} yr)...")
for pair in PAIRS:
    pair["df_raw"]  = rolling_corr(annual_dt, pair["sector"], pair["raw_var"],
                                   idx, pair["idx_col"])
    pair["df_apac"] = rolling_corr(annual_dt, pair["sector"], pair["apac_var"],
                                   idx, pair["idx_col"])
    print(f"  {pair['label']}: {len(pair['df_raw'])} windows (raw), "
          f"{len(pair['df_apac'])} windows (APAC)")

# =============================================================================
# 5. FIGURE
# =============================================================================

def plot_panel(ax, df, r_full, color, title, show_ylabel=False):
    """Draw one rolling correlation panel."""
    if df.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        return

    n_win  = df["n"].values
    r_vals = df["r"].values

    # Post-2016 shade
    ax.axvspan(REGIME_SHIFT_YEAR, YEAR_MAX + 1,
               alpha=0.08, color="#E24B4A", zorder=0)

    # Zero line
    ax.axhline(0, color="#888780", linewidth=0.5, alpha=0.4, zorder=1)

    # Full-record r
    ax.axhline(r_full, color=color, linewidth=1.2,
               linestyle="--", alpha=0.6, zorder=2)

    # 95% CI (Fisher z)
    ci_lower = np.full_like(r_vals, np.nan)
    ci_upper = np.full_like(r_vals, np.nan)
    for i, (r_i, n_i) in enumerate(zip(r_vals, n_win)):
        if n_i > 4 and not np.isnan(r_i):
            z_i         = np.arctanh(np.clip(r_i, -0.9999, 0.9999))
            se          = 1.0 / np.sqrt(n_i - 3)
            ci_lower[i] = np.tanh(z_i - 1.96 * se)
            ci_upper[i] = np.tanh(z_i + 1.96 * se)

    ax.fill_between(df["centre_year"], ci_lower, ci_upper,
                    color=color, alpha=0.12, zorder=2)

    # Rolling r line
    ax.plot(df["centre_year"], df["r"],
            color=color, linewidth=2.0, zorder=3)

    # Significance markers
    sig    = df[df["sig"] == "*"]
    nonsig = df[df["sig"] != "*"]
    ax.scatter(sig["centre_year"],    sig["r"],
               color=color, s=28, zorder=4, alpha=0.85)
    ax.scatter(nonsig["centre_year"], nonsig["r"],
               color=color, s=18, zorder=4, alpha=0.4,
               facecolors="none", edgecolors=color, linewidths=0.8)

    # Full-record r annotation
    ax.text(0.98, 0.05, f"full r={r_full:+.3f}",
            transform=ax.transAxes, fontsize=8,
            ha="right", va="bottom", color=color, alpha=0.85)

    # Panel title
    ax.set_title(title, fontsize=9, fontweight="bold", color=color, pad=4)

    # Axes
    ax.set_ylim(-0.95, 0.95)
    ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("Pearson r", fontsize=9)

n_rows = len(PAIRS)
fig, axes = plt.subplots(n_rows, 2,
                          figsize=(11, 3.0 * n_rows),
                          sharex=True)
fig.subplots_adjust(hspace=0.25, wspace=0.08,
                    top=0.93, bottom=0.10, left=0.08, right=0.97)

# Column headers
axes[0, 0].annotate("Raw", xy=(0.5, 1.12), xycoords="axes fraction",
                    ha="center", fontsize=11, fontweight="bold", color="#333333")
axes[0, 1].annotate("APAC", xy=(0.5, 1.12), xycoords="axes fraction",
                    ha="center", fontsize=11, fontweight="bold", color="#333333")

for i, pair in enumerate(PAIRS):
    ax_raw  = axes[i, 0]
    ax_apac = axes[i, 1]

    var_label = "phase" if pair["var_type"] == "phase" else "amplitude"

    plot_panel(ax_raw,  pair["df_raw"],  pair["r_raw"],
               pair["color"], f"{pair['label']} — raw {var_label}",
               show_ylabel=(i == n_rows // 2))
    plot_panel(ax_apac, pair["df_apac"], pair["r_apac"],
               pair["color"], f"{pair['label']} — APAC {var_label}",
               show_ylabel=False)

    # Row label on far left
    axes[i, 0].text(-0.10, 0.5, pair["label"],
                    transform=axes[i, 0].transAxes,
                    fontsize=7.5, color="#5F5E5A",
                    rotation=90, va="center", ha="center")

# X axis
for ax in axes[-1, :]:
    ax.set_xlabel("Window centre year", fontsize=9)
    ax.set_xticks(range(1990, 2022, 5))
    ax.set_xlim(YEAR_MIN + WINDOW // 2 - 1, YEAR_MAX - WINDOW // 2 + 1)

# Title
fig.suptitle(
    f"Raw vs APAC rolling correlations: selected pairs ({WINDOW}-yr Pearson r, {YEAR_MIN}–{YEAR_MAX})",
    fontsize=10, fontweight="bold", y=0.97
)

# Panel letters
letters = "abcdefghij"
for idx_ax, ax in enumerate(axes.flat):
    ax.text(-0.01, 1.02, f"({letters[idx_ax]})",
            transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="right",
            color="#333333")

# Legend
legend_elements = [
    Line2D([0], [0], color="gray", linewidth=2,   label=f"{WINDOW}-yr rolling r"),
    Line2D([0], [0], color="gray", linewidth=1.2, linestyle="--", label="Full-record r"),
    Patch(facecolor="gray", alpha=0.2,            label="95% CI"),
    Line2D([0], [0], marker="o", color="gray", markersize=5,
           linewidth=0,                           label="p < 0.05"),
    Line2D([0], [0], marker="o", color="gray", markersize=5,
           linewidth=0, markerfacecolor="none",   label="p ≥ 0.05"),
    Patch(facecolor="#E24B4A", alpha=0.15,        label="Post-2016"),
]
fig.legend(handles=legend_elements,
           loc="lower center", bbox_to_anchor=(0.5, 0.01),
           ncol=6, fontsize=8.5, frameon=False)

outpath = os.path.join(OUTPUT_DIR, "fig_rolling_raw_vs_apac.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nFigure saved: {outpath}")

# =============================================================================
# 6. SUMMARY TABLE
# =============================================================================

half = WINDOW // 2
print(f"\n=== Pre vs post-{REGIME_SHIFT_YEAR} comparison: raw vs APAC ===")
print(f"{'Pair':<40} {'':6} {'Pre r':>7} {'Post r':>7} {'Change':>8}")
print("-" * 75)

for pair in PAIRS:
    for label, df, r_full in [
        ("raw",  pair["df_raw"],  pair["r_raw"]),
        ("APAC", pair["df_apac"], pair["r_apac"]),
    ]:
        if df.empty:
            continue
        df2 = df.copy()
        df2["end_year"] = df2["centre_year"] + half
        pre  = df2[df2["end_year"] <= REGIME_SHIFT_YEAR]["r"].mean()
        post = df2[df2["end_year"] >  REGIME_SHIFT_YEAR]["r"].mean()
        chg  = post - pre
        print(f"  {pair['label']:<40} {label:<6} "
              f"{pre:>+7.3f} {post:>+7.3f} {chg:>+8.3f}")
    print()

print("Done.")

# =============================================================================
# 7. SYNC TO GOOGLE DRIVE
# =============================================================================
import subprocess

GDRIVE_DEST = "gdrive:results/Ch3_Figures/"
print(f"\nSyncing to {GDRIVE_DEST}")
result = subprocess.run(
    ["rclone", "copy", outpath, GDRIVE_DEST],
    capture_output=True, text=True
)
if result.returncode == 0:
    print(f"  ✓ fig_rolling_raw_vs_apac.png")
else:
    print(f"  ✗ {result.stderr.strip()}")