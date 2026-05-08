"""
Correlation heatmap — phase and amplitude anomalies vs atmospheric indices.

Two stacked heatmaps (phase on top, amplitude below), one cell per
sector × index/season combination. Only index/season columns where at least
one sector reaches p < 0.10 are shown — this keeps the figure focused on
combinations with some physical signal rather than displaying 20 columns
of noise.

Niño3.4 is included if any sector shows p < 0.10 for any season.

Diverging blue-red colour scale centred at zero. Cell values show Pearson r.
Significance annotated as * (p<0.05 raw) or ** (survives FDR correction).
A dot (.) marks p<0.10.

Loads correlations_output.csv produced by compute_atmospheric_correlations.py.
Run that script first if the CSV doesn't exist yet.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

from ch3_style import (
    apply_style,
    SECTORS_NO_CIRC, SECTOR_LABELS,
    stroke, save_fig, DEFAULT_OUTPUT_DIR,
)

apply_style()

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
CORR_CSV   = os.path.join(DATA_DIR, "correlations_output.csv")

# Threshold for a column to be included — at least one sector must clear this
# for that index/season combination to appear in the figure.
INCLUDE_THRESHOLD = 0.10   # raw p-value

# Colour scale — symmetric around zero, clipped at ±0.6
CMAP  = "RdBu_r"
VLIM  = 0.6


# --- Load -----------------------------------------------------------------

print("Loading correlation results...")
df = pd.read_csv(CORR_CSV)
print(f"  {len(df)} rows")

# Sector display order — matches other Ch3 figures
SECTOR_ORDER = [SECTOR_LABELS[s] for s in SECTORS_NO_CIRC]


# --- Filter to significant index/season columns ---------------------------

def get_significant_cols(var_type, threshold=INCLUDE_THRESHOLD):
    """
    For a given variable type (phase or amplitude), return the list of
    index_season column labels where at least one sector has p < threshold.
    Sorted by index then season for readability.
    """
    sub = df[df["var_type"] == var_type].copy()
    sub["col_label"] = sub["index"] + "\n" + sub["season"]

    # Keep columns where any sector clears the threshold
    sig_cols = (sub.groupby("col_label")["pearson_p"]
                .min()
                .loc[lambda x: x < threshold]
                .index.tolist())

    # Sort: by index name first, then by season in a logical order
    season_order = {"annual": 0, "DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}
    def sort_key(label):
        parts  = label.split("\n")
        idx    = parts[0]
        season = parts[1] if len(parts) > 1 else "annual"
        return (idx, season_order.get(season, 99))

    return sorted(sig_cols, key=sort_key)


# --- Build heatmap matrix -------------------------------------------------

def build_matrix(var_type, col_labels):
    """
    Returns:
        r_mat   : sectors × cols matrix of Pearson r values
        sig_mat : sectors × cols matrix of significance strings
    """
    sub = df[df["var_type"] == var_type].copy()
    sub["col_label"] = sub["index"] + "\n" + sub["season"]

    r_mat   = pd.DataFrame(index=SECTOR_ORDER, columns=col_labels, dtype=float)
    sig_mat = pd.DataFrame(index=SECTOR_ORDER, columns=col_labels, dtype=str)

    for _, row in sub.iterrows():
        sec = row["sector_label"]
        col = row["col_label"]
        if sec in SECTOR_ORDER and col in col_labels:
            r_mat.loc[sec, col]   = row["pearson_r"]
            sig_mat.loc[sec, col] = row["sig"]

    return r_mat.astype(float), sig_mat


# --- Draw a single heatmap panel ------------------------------------------

def draw_heatmap(ax, r_mat, sig_mat, title, show_xticklabels=False):
    n_rows, n_cols = r_mat.shape
    norm  = mcolors.TwoSlopeNorm(vmin=-VLIM, vcenter=0, vmax=VLIM)
    cmap  = plt.get_cmap(CMAP)

    for i, sector in enumerate(r_mat.index):
        for j, col in enumerate(r_mat.columns):
            r   = r_mat.loc[sector, col]
            sig = sig_mat.loc[sector, col]

            if pd.isna(r):
                color = "#F5F5F5"
            else:
                color = cmap(norm(r))

            rect = FancyBboxPatch(
                (j + 0.05, i + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="white", linewidth=1.5,
                zorder=2,
            )
            ax.add_patch(rect)

            if not pd.isna(r):
                # Text colour: white on dark cells, dark on light cells
                brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
                txt_color  = "white" if brightness < 0.5 else "#2C2C2A"

                label = f"{r:+.2f}{sig}"
                ax.text(j + 0.5, i + 0.5, label,
                        ha="center", va="center",
                        fontsize=9, color=txt_color,
                        fontweight="bold" if "**" in sig or "*" in sig else "normal",
                        path_effects=stroke(lw=1.5, foreground="none")
                        if brightness > 0.3 else [])

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(r_mat.index[::-1] if False else r_mat.index,
                       fontsize=10)
    ax.invert_yaxis()

    if show_xticklabels:
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels(r_mat.columns, fontsize=9,
                           ha="center", va="top")
    else:
        ax.set_xticks([])

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    # Light grid lines between cells
    for x in range(n_cols + 1):
        ax.axvline(x, color="white", lw=0.5, zorder=1)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", lw=0.5, zorder=1)

    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(length=0)

    return norm, cmap


# --- Main figure ----------------------------------------------------------

print("Identifying significant index/season columns...")
phase_cols = get_significant_cols("phase")
amp_cols   = get_significant_cols("amplitude")

# Use the union so both panels have the same columns — easier to compare
all_cols   = sorted(set(phase_cols) | set(amp_cols),
                    key=lambda x: (x.split("\n")[0],
                                   {"annual":0,"DJF":1,"MAM":2,
                                    "JJA":3,"SON":4}.get(
                                        x.split("\n")[1] if "\n" in x
                                        else "annual", 99)))

print(f"  Phase significant cols:     {len(phase_cols)}")
print(f"  Amplitude significant cols: {len(amp_cols)}")
print(f"  Union (displayed):          {len(all_cols)}")
print(f"  Columns: {all_cols}")

r_phase,   sig_phase   = build_matrix("phase",     all_cols)
r_amp,     sig_amp     = build_matrix("amplitude", all_cols)

n_rows = len(SECTOR_ORDER)
n_cols = len(all_cols)

# Figure height scales with number of rows; width with columns
fig_w = max(10, n_cols * 1.4)
fig_h = n_rows * 0.9 * 2 + 2.5   # two panels + legend space

fig, (ax_phase, ax_amp) = plt.subplots(
    2, 1,
    figsize=(fig_w, fig_h),
    gridspec_kw={"hspace": 0.35}
)

norm, cmap = draw_heatmap(ax_phase, r_phase, sig_phase,
                           "Phase anomaly\nvs atmospheric indices",
                           show_xticklabels=False)

draw_heatmap(ax_amp, r_amp, sig_amp,
             "Amplitude anomaly\nvs atmospheric indices",
             show_xticklabels=True)

# Colourbar
sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.25, -0.03, 0.50, 0.025])
cbar    = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Pearson r  |  ** FDR p<0.05   * p<0.05   . p<0.10",
               fontsize=9)
cbar.ax.tick_params(labelsize=8)

fig.suptitle(
    "Atmospheric drivers of sea ice phase and amplitude\n"
    "Columns shown where at least one sector reaches p < 0.10",
    fontsize=12, fontweight="bold", y=1.01,
)

save_fig(fig, "fig_correlation_heatmap.png", OUTPUT_DIR)
print("fig_correlation_heatmap.png saved.")