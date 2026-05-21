"""
rolling_window_correlations.py  (v2 — updated May 2026)

Computes 15-year rolling Pearson correlations between APAC phase/amplitude
anomalies and atmospheric indices to test stationarity of atmosphere-ice
relationships.

CHANGES FROM v1:
- YEAR_MAX updated to 2025
- Pairs now selected automatically: top 3 per sector from correlations_output.csv
  ranked by: (1) survived any FDR method, then (2) absolute Pearson r
- ADV/RET shoulder season indices included
- 15 panels total (5 sectors × top 3)

INPUTS:
  correlations_output.csv     — 700 pairs with FDR flags
  annual_params.csv           — APAC phase and amplitude anomalies by sector
  master_index_detrended.csv  — detrended indices including ADV/RET seasons

OUTPUTS:
  rolling_window_top3.png     — 15-panel figure (5 sectors × 3 pairs)
  rolling_window_top3.csv     — full rolling window results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
import os
import subprocess

# =============================================================================
# 0. PATHS
# =============================================================================

FIGURES_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch3/figures/"
DATA_DIR    = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data/"
OUTPUT_DIR  = FIGURES_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

ANNUAL_CSV  = os.path.join(DATA_DIR, "annual_params.csv")
INDEX_CSV   = os.path.join(DATA_DIR, "master_index_detrended.csv")
CORR_CSV    = os.path.join(DATA_DIR, "correlations_output.csv")

YEAR_MIN          = 1979
YEAR_MAX          = 2025
REGIME_SHIFT_YEAR = 2016
WINDOW            = 15
TOP_N             = 3   # pairs per sector

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("Loading data...")
idx    = pd.read_csv(INDEX_CSV)
annual = pd.read_csv(ANNUAL_CSV)
corrs  = pd.read_csv(CORR_CSV)

annual = annual[annual["Year"].between(YEAR_MIN, YEAR_MAX)]
print(f"  Index:   {idx.shape}  years {idx['Year'].min()}–{idx['Year'].max()}")
print(f"  Annual:  {annual.shape}")
print(f"  Corrs:   {corrs.shape}")

# =============================================================================
# 2. SELECT TOP-3 PAIRS PER SECTOR
# =============================================================================
# Ranking logic:
#   1. Survived any FDR method (boot_sig OR perm_sig OR pearson_sig) — True first
#   2. Within each tier, rank by absolute Pearson r descending
#   Take top 3 per sector

print("\nSelecting top-3 pairs per sector...")

# Derive any_fdr flag (sig column is empty in the CSV)
for col in ["boot_sig", "perm_sig", "pearson_sig"]:
    corrs[col] = corrs[col].astype(str).str.strip().str.lower() == "true"

corrs["any_fdr"]  = corrs["boot_sig"] | corrs["perm_sig"] | corrs["pearson_sig"]
corrs["abs_r"]    = corrs["pearson_r"].abs()

# Sort: any_fdr descending (True=1 > False=0), then abs_r descending
corrs_sorted = corrs.sort_values(
    ["sector_label", "any_fdr", "abs_r"],
    ascending=[True, False, False]
)

top_pairs = (
    corrs_sorted
    .groupby("sector_label", sort=False)
    .head(TOP_N)
    .reset_index(drop=True)
)

print(f"\nSelected {len(top_pairs)} pairs:")
print(top_pairs[["sector_label", "variable", "var_type", "index", "season",
                  "pearson_r", "any_fdr"]].to_string(index=False))

# =============================================================================
# 3. SECTOR COLOURS
# =============================================================================

SECTOR_COLORS = {
    "Weddell":         "#2196F3",
    "ABS":             "#F44336",
    "Ross":            "#4CAF50",
    "East Antarctica": "#FF9800",
    "King Haakon":     "#9C27B0",
}

# =============================================================================
# 4. DETREND APAC VARIABLES
# =============================================================================

def detrend_series(x, y):
    mask = ~np.isnan(y)
    if mask.sum() < 5:
        return y
    slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
    return y - (slope * x + intercept)

SECTORS = {
    "SIE_Weddell":                 "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross":                    "Ross",
    "SIE_East_Antarctica":         "East Antarctica",
    "SIE_King_Haakon":             "King Haakon",
}

annual_dt = []
for sec_col, sec_label in SECTORS.items():
    sec = annual[annual["sector"] == sec_col].copy().sort_values("Year")
    x   = sec["Year"].values.astype(float)
    for var in ["max_doy_anom", "amplitude_anom",
                "max_doy_raw_anom", "amplitude_raw_anom"]:
        if var in sec.columns:
            sec[var] = detrend_series(x, sec[var].values.astype(float))
    annual_dt.append(sec)

annual_dt = pd.concat(annual_dt)
print("\nDetrending complete")

# =============================================================================
# 5. ROLLING CORRELATION FUNCTIONS
# =============================================================================

def rolling_corr(apac_df, sector_col, apac_var, idx_df, idx_col,
                 window=WINDOW):
    half     = window // 2
    sec_data = apac_df[apac_df["sector"] == sector_col][["Year", apac_var]].dropna()

    results = []
    for centre in range(YEAR_MIN + half, YEAR_MAX - half + 1):
        sec_win = sec_data[sec_data["Year"].between(centre - half, centre + half)]
        idx_win = idx_df[idx_df["Year"].between(centre - half, centre + half)][["Year", idx_col]].dropna()
        merged  = sec_win.merge(idx_win, on="Year", how="inner").dropna()
        if len(merged) < 8:
            continue
        r, p = stats.pearsonr(merged[idx_col].values, merged[apac_var].values)
        results.append({
            "centre_year": centre,
            "r":  round(r, 3),
            "p":  round(p, 4),
            "n":  len(merged),
            "sig": "*" if p < 0.05 else ("." if p < 0.10 else ""),
        })
    return pd.DataFrame(results)


def full_record_corr(apac_df, sector_col, apac_var, idx_df, idx_col):
    sec_data = apac_df[apac_df["sector"] == sector_col][["Year", apac_var]].dropna()
    merged   = sec_data.merge(idx_df[["Year", idx_col]].dropna(), on="Year", how="inner").dropna()
    if len(merged) < 5:
        return np.nan, np.nan
    r, p = stats.pearsonr(merged[idx_col].values, merged[apac_var].values)
    return round(r, 3), round(p, 4)

# =============================================================================
# 6. COMPUTE ROLLING CORRELATIONS FOR ALL TOP PAIRS
# =============================================================================

# Map var_type from corrs CSV to actual column in annual_params
VAR_MAP = {
    "amplitude_apac": "amplitude_anom",
    "amplitude_raw":  "amplitude_raw_anom",
    "phase_apac":     "max_doy_anom",
    "phase_raw":      "max_doy_raw_anom",
}

# Build index column name from index + season
# e.g. index="SAM", season="ADV" → "SAM_ADV"
#      index="Nino34", season="annual" → "Nino34_annual"
def make_idx_col(index, season):
    return f"{index}_{season}"

print(f"\nComputing rolling correlations (window={WINDOW} yr, YEAR_MAX={YEAR_MAX})...")
all_results = []
pair_data   = []

for _, row in top_pairs.iterrows():
    sec_col    = row["sector"]
    sec_label  = row["sector_label"]
    apac_var   = VAR_MAP.get(row["var_type"], "amplitude_anom")
    idx_col    = make_idx_col(row["index"], row["season"])
    r_full_csv = row["pearson_r"]
    any_fdr    = row["any_fdr"]

    if idx_col not in idx.columns:
        print(f"  SKIP — {idx_col} not in index CSV")
        continue
    if apac_var not in annual_dt.columns:
        print(f"  SKIP — {apac_var} not in annual CSV")
        continue

    df = rolling_corr(annual_dt, sec_col, apac_var, idx, idx_col)
    if df.empty:
        print(f"  SKIP — no windows for {sec_label} {apac_var} ~ {idx_col}")
        continue

    r_full, p_full = full_record_corr(annual_dt, sec_col, apac_var, idx, idx_col)

    df["sector"]    = sec_label
    df["apac_var"]  = apac_var
    df["idx_col"]   = idx_col
    all_results.append(df)

    pair_data.append({
        "sec_col":    sec_col,
        "sec_label":  sec_label,
        "apac_var":   apac_var,
        "idx_col":    idx_col,
        "var_type":   row["var_type"],
        "season":     row["season"],
        "r_full":     r_full,
        "p_full":     p_full,
        "r_full_csv": r_full_csv,
        "any_fdr":    any_fdr,
        "color":      SECTOR_COLORS.get(sec_label, "#666666"),
        "rolling":    df,
    })

    fdr_tag = "FDR✓" if any_fdr else "    "
    print(f"  {fdr_tag} {sec_label:<18} {apac_var:<22} ~ {idx_col:<18}  "
          f"full r={r_full:+.3f}  {len(df)} windows")

# Save
all_df = pd.concat(all_results)
all_df.to_csv(os.path.join(OUTPUT_DIR, "rolling_window_top3.csv"), index=False)
print(f"\nSaved rolling results: {len(all_df)} rows")

# =============================================================================
# 7. FIGURE — 15 panels arranged as 5 rows × 3 columns (one row per sector)
# =============================================================================
# Layout: rows = sectors (Weddell, ABS, Ross, EA, KH)
#         cols = rank 1, 2, 3 within sector
# Each panel: rolling r line + CI band + full-record dashed + 2016 marker

print("\nGenerating figure...")

sector_order = ["Weddell", "ABS", "Ross", "East Antarctica", "King Haakon"]
n_rows = len(sector_order)
n_cols = TOP_N

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(5.5 * n_cols, 3.2 * n_rows),
                          sharex=True)
fig.subplots_adjust(hspace=0.18, wspace=0.12,
                    top=0.94, bottom=0.09, left=0.07, right=0.98)

# Index pair_data by sector for easy lookup
from collections import defaultdict
sector_pairs = defaultdict(list)
for p in pair_data:
    sector_pairs[p["sec_label"]].append(p)

for row_i, sec_label in enumerate(sector_order):
    pairs_this_sector = sector_pairs.get(sec_label, [])

    for col_i in range(n_cols):
        ax = axes[row_i, col_i]

        if col_i >= len(pairs_this_sector):
            ax.set_visible(False)
            continue

        pair   = pairs_this_sector[col_i]
        df     = pair["rolling"]
        r_full = pair["r_full"]
        color  = pair["color"]
        n_win  = df["n"].values

        # Post-2016 shade
        ax.axvspan(REGIME_SHIFT_YEAR, YEAR_MAX + 1,
                   alpha=0.08, color="#E24B4A", zorder=0)

        # Zero line
        ax.axhline(0, color="#888780", linewidth=0.5,
                   linestyle="-", alpha=0.4, zorder=1)

        # Full-record r
        if not np.isnan(r_full):
            ax.axhline(r_full, color=color, linewidth=1.2,
                       linestyle="--", alpha=0.55, zorder=2)

        # 95% CI band (Fisher z)
        r_vals   = df["r"].values
        ci_lower = np.full_like(r_vals, np.nan)
        ci_upper = np.full_like(r_vals, np.nan)
        for i, (r_i, n_i) in enumerate(zip(r_vals, n_win)):
            if n_i > 4 and not np.isnan(r_i):
                z_i        = np.arctanh(np.clip(r_i, -0.9999, 0.9999))
                se         = 1.0 / np.sqrt(n_i - 3)
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

        # FDR badge
        if pair["any_fdr"]:
            ax.text(0.98, 0.95, "FDR✓",
                    transform=ax.transAxes, fontsize=7,
                    color=color, alpha=0.7, ha="right", va="top",
                    fontweight="bold")

        # Panel label
        season_tag = f" [{pair['season']}]" if pair["season"] != "annual" else ""
        var_short  = "amp" if "amplitude" in pair["apac_var"] else "phase"
        panel_lbl  = f"{sec_label} {var_short} ~ {pair['idx_col']}{season_tag}"
        ax.text(0.02, 0.05, panel_lbl,
                transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom",
                color=color, clip_on=False)

        # Full-record r annotation
        if not np.isnan(r_full):
            ax.text(0.98, 0.05, f"r={r_full:+.2f}",
                    transform=ax.transAxes, fontsize=7.5,
                    ha="right", va="bottom", color=color, alpha=0.8)

        # Axes formatting
        ax.set_ylim(-0.95, 0.95)
        ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        if col_i == 0:
            ax.set_ylabel("Pearson r", fontsize=9)
        if row_i == n_rows - 1:
            ax.set_xlabel("Window centre year", fontsize=9)
            ax.set_xticks(range(1990, 2022, 5))
            ax.set_xlim(YEAR_MIN + WINDOW // 2 - 1,
                        YEAR_MAX - WINDOW // 2 + 1)

        # Column header (rank label) on top row
        if row_i == 0:
            ax.set_title(f"Rank {col_i + 1}", fontsize=9,
                         color="#5F5E5A", pad=4)

# Overall title
fig.suptitle(
    f"Stationarity of atmosphere–sea ice relationships: "
    f"top-{TOP_N} pairs per sector ({WINDOW}-yr rolling Pearson r, {YEAR_MIN}–{YEAR_MAX})",
    fontsize=11, fontweight="bold", y=0.97
)

# Shared legend
legend_elements = [
    Line2D([0], [0], color="gray", linewidth=2,
           label=f"{WINDOW}-yr rolling r"),
    Line2D([0], [0], color="gray", linewidth=1.2,
           linestyle="--", label="Full-record r"),
    Patch(facecolor="gray", alpha=0.2, label="95% CI"),
    Line2D([0], [0], marker="o", color="gray", markersize=5,
           linewidth=0, label="p < 0.05"),
    Line2D([0], [0], marker="o", color="gray", markersize=5,
           linewidth=0, markerfacecolor="none", label="p ≥ 0.05"),
    Patch(facecolor="#E24B4A", alpha=0.15, label="Post-2016"),
]
fig.legend(handles=legend_elements,
           loc="lower center", bbox_to_anchor=(0.5, 0.01),
           ncol=6, fontsize=8.5, frameon=False)

outpath = os.path.join(OUTPUT_DIR, "rolling_window_top3.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Figure saved: {outpath}")

# =============================================================================
# 8. SUMMARY TABLE — pre vs post 2016 mean r
# =============================================================================

print(f"\n=== Pre vs post-{REGIME_SHIFT_YEAR} rolling r ===")
half = WINDOW // 2
print(f"{'Pair':<55} {'Pre mean r':>10} {'Post mean r':>11} {'Change':>8} {'FDR':>5}")
print("-" * 95)

for p in pair_data:
    df       = p["rolling"].copy()
    df["end_year"] = df["centre_year"] + half
    pre      = df[df["end_year"] <= REGIME_SHIFT_YEAR]["r"].mean()
    post     = df[df["end_year"] >  REGIME_SHIFT_YEAR]["r"].mean()
    season_t = f"[{p['season']}]" if p["season"] != "annual" else ""
    label    = f"{p['sec_label']} {p['apac_var'][:3]} ~ {p['idx_col']} {season_t}"
    fdr_t    = "✓" if p["any_fdr"] else ""
    pre_s    = f"{pre:+.3f}"  if not np.isnan(pre)  else "  n/a"
    post_s   = f"{post:+.3f}" if not np.isnan(post) else "   n/a"
    chg_s    = f"{post-pre:+.3f}" if (not np.isnan(pre) and not np.isnan(post)) else "  n/a"
    print(f"  {label:<53} {pre_s:>10} {post_s:>11} {chg_s:>8} {fdr_t:>5}")

# =============================================================================
# 9. SYNC TO GOOGLE DRIVE
# =============================================================================

GDRIVE_DEST = "gdrive:results/Ch3_Figures/"
figures = [
    os.path.join(OUTPUT_DIR, "rolling_window_top3.png"),
    os.path.join(OUTPUT_DIR, "rolling_window_top3.csv"),
]

print(f"\nSyncing to {GDRIVE_DEST}")
for fpath in figures:
    if os.path.exists(fpath):
        result = subprocess.run(
            ["rclone", "copy", fpath, GDRIVE_DEST],
            capture_output=True, text=True
        )
        fname = os.path.basename(fpath)
        status = "✓" if result.returncode == 0 else f"✗ {result.stderr.strip()}"
        print(f"  {status} {fname}")

print("\nDone.")