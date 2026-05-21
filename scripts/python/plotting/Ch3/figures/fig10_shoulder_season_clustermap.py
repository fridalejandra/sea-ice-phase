"""
fig_shoulder_season_clustermap.py
==================================
Seaborn clustermap showing shoulder season (ADV/RET) correlations between
atmospheric indices and sea ice phase/amplitude anomalies.

Two figures:
  fig_shoulder_clustermap_apac.png  — APAC anomaly correlations
  fig_shoulder_clustermap_raw.png   — Raw anomaly correlations

Each figure has two panels:
  Left  — Phase anomaly ~ atmospheric indices (ADV + RET seasons only)
  Right — Amplitude anomaly ~ atmospheric indices (ADV + RET seasons only)

Non-significant cells (p >= 0.05) are shown as white.
Dendrogram clusters similar correlation patterns across sectors and indices.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch3/figures"
GDRIVE     = "gdrive:sea-ice-phase/results/Ch3_Figures/"
CORR_CSV   = os.path.join(DATA_DIR, "correlations_output.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
SECTOR_ORDER = ["Weddell", "ABS", "Ross", "East Antarctica", "King Haakon"]
SECTOR_COLORS = {
    "Weddell"        : "#2196F3",
    "ABS"            : "#F44336",
    "Ross"           : "#4CAF50",
    "East Antarctica": "#FF9800",
    "King Haakon"    : "#9C27B0",
}
VLIM = 0.55
CMAP = "RdBu_r"

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading correlations...")
df = pd.read_csv(CORR_CSV)
print(f"  {len(df)} rows | var_types: {df['var_type'].unique()}")

# Filter to shoulder seasons only
df_shoulder = df[df["season"].isin(["ADV", "RET"])].copy()
df_shoulder["col_label"] = df_shoulder["index"] + "\n" + df_shoulder["season"]
print(f"  Shoulder season pairs: {len(df_shoulder)}")


# ── Build masked r matrix ─────────────────────────────────────────────────────
def build_masked_matrix(var_type):
    """
    Returns r matrix where non-significant cells are NaN (shown as white).
    Rows = sectors, columns = index×season combinations.
    """
    sub = df_shoulder[df_shoulder["var_type"] == var_type].copy()

    # Get all columns
    all_cols = sorted(sub["col_label"].unique(),
                      key=lambda x: (x.split("\n")[0],
                                     {"ADV": 0, "RET": 1}.get(
                                         x.split("\n")[1], 99)))

    r_mat   = pd.DataFrame(np.nan, index=SECTOR_ORDER, columns=all_cols)
    sig_mat = pd.DataFrame("",    index=SECTOR_ORDER, columns=all_cols)

    for _, row in sub.iterrows():
        sec = row["sector_label"]
        col = row["col_label"]
        if sec in SECTOR_ORDER and col in all_cols:
            r_mat.loc[sec, col]   = row["pearson_r"]
            sig_mat.loc[sec, col] = row["sig"]

    # Mask non-significant cells (p >= 0.05) — set to NaN for white display
    mask_insig = sig_mat.map(lambda x: x == "" or x == ".")
    r_masked   = r_mat.copy()
    r_masked[mask_insig] = np.nan

    return r_mat.astype(float), r_masked.astype(float), sig_mat


# ── Draw clustermap ───────────────────────────────────────────────────────────
def draw_clustermap(var_type, title, outfile):

    r_full, r_masked, sig_mat = build_masked_matrix(var_type)

    # Fill NaN with 0 for clustering purposes only
    r_for_clustering = r_full.fillna(0)

    # Row colors — sector colors
    row_colors = pd.Series(
        [SECTOR_COLORS[s] for s in r_full.index],
        index=r_full.index, name="Sector"
    )

    norm = mcolors.TwoSlopeNorm(vmin=-VLIM, vcenter=0, vmax=VLIM)
    cmap_with_nan = plt.get_cmap(CMAP).copy()
    cmap_with_nan.set_bad(color="#F5F5F5")  # NaN cells → light grey

    # First compute linkage from full matrix
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    row_dist = pdist(r_for_clustering.values, metric="euclidean")
    col_dist = pdist(r_for_clustering.values.T, metric="euclidean")
    row_link = linkage(row_dist, method="average")
    col_link = linkage(col_dist, method="average")

    g = sns.clustermap(
        r_masked,
        row_linkage      = row_link,
        col_linkage      = col_link,
        row_cluster      = True,
        col_cluster      = True,
        cmap             = cmap_with_nan,
        norm             = norm,
        row_colors       = row_colors,
        figsize          = (10, 5),
        linewidths       = 0.5,
        linecolor        = "white",
        dendrogram_ratio = (0.15, 0.15),
        cbar_pos         = (0.02, 0.85, 0.03, 0.12),
        annot            = r_full.round(2),
        annot_kws        = {"size": 9},
        fmt              = ".2f",
        mask             = r_masked.isna(),
    )

    # Style
    g.ax_heatmap.set_xlabel("Index × Shoulder Season", fontsize=10)
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(axis="x", labelsize=9, rotation=0)
    g.ax_heatmap.tick_params(axis="y", labelsize=10, rotation=0)

    # Colorbar label
    g.cax.set_ylabel("Pearson r", fontsize=8, rotation=90)
    g.cax.tick_params(labelsize=8)

    # Title
    g.fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    # Sector legend
    handles = [Patch(facecolor=c, label=s)
               for s, c in SECTOR_COLORS.items()]
    g.fig.legend(handles=handles, loc="lower center", ncol=5,
                 fontsize=8.5, frameon=False,
                 bbox_to_anchor=(0.5, -0.04))

    # Note on significance
    g.fig.text(0.5, -0.06,
               "Only significant correlations shown (p < 0.05). "
               "White cells are non-significant. "
               "Clustering based on significant r values.",
               ha="center", fontsize=8, color="#5F5E5A", style="italic")

    g.fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {outfile}")
    plt.close()

    ret = os.system(f'rclone copy "{outfile}" "{GDRIVE}"')
    if ret == 0:
        print(f"Synced → {GDRIVE}")


# ── Run ───────────────────────────────────────────────────────────────────────
print("\nBuilding APAC clustermap...")

# Phase APAC
r_phase, r_phase_m, sig_phase = build_masked_matrix("phase_apac")
r_amp,   r_amp_m,   sig_amp   = build_masked_matrix("amplitude_apac")

print(f"  Phase columns: {list(r_phase.columns)}")
print(f"  Amplitude columns: {list(r_amp.columns)}")

draw_clustermap(
    var_type = "phase_apac",
    title    = "Shoulder season correlations — APAC phase anomaly vs atmospheric indices",
    outfile  = os.path.join(OUTPUT_DIR, "fig10_shoulder_clustermap_phase_apac.png")
)

draw_clustermap(
    var_type = "amplitude_apac",
    title    = "Shoulder season correlations — APAC amplitude anomaly vs atmospheric indices",
    outfile  = os.path.join(OUTPUT_DIR, "fig10_shoulder_clustermap_amp_apac.png")
)

print("\nBuilding raw clustermap...")
draw_clustermap(
    var_type = "phase_raw",
    title    = "Shoulder season correlations — Raw phase anomaly vs atmospheric indices",
    outfile  = os.path.join(OUTPUT_DIR, "fig10_shoulder_clustermap_phase_raw.png")
)

draw_clustermap(
    var_type = "amplitude_raw",
    title    = "Shoulder season correlations — Raw amplitude anomaly vs atmospheric indices",
    outfile  = os.path.join(OUTPUT_DIR, "fig10_shoulder_clustermap_amp_raw.png")
)

print("\nDone.")