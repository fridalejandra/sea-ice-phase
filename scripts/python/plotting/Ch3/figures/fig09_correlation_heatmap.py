"""
fig_correlation_heatmap.py
==========================
Figure 10 in Ch3
Two correlation heatmap figures:

  fig_correlation_heatmap_apac.png
      (a) Phase APAC anomaly vs atmospheric indices
      (b) Amplitude APAC anomaly vs atmospheric indices

  fig_correlation_heatmap_raw.png
      (a) Phase raw anomaly vs atmospheric indices
      (b) Amplitude raw anomaly vs atmospheric indices

Panels are side by side (not stacked) for direct phase/amplitude comparison.
Cell values show Pearson r. Significance: * p<0.05, ** FDR p<0.05.
Only index/season columns where at least one sector reaches p<0.10 are shown.
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

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
GDRIVE     = "gdrive:My Drive/sea-ice-phase/results/Ch3_Figures"
CORR_CSV   = os.path.join(DATA_DIR, "correlations_output.csv")

INCLUDE_THRESHOLD = 0.05
CMAP  = "RdBu_r"
VLIM  = 0.5

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading correlation results...")
df = pd.read_csv(CORR_CSV)
print(f"  {len(df)} rows | var_types: {df['var_type'].unique()}")

SECTOR_ORDER = [SECTOR_LABELS[s] for s in SECTORS_NO_CIRC]

SEASON_ORDER = {"annual": 0, "DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}

def sort_key(label):
    parts  = label.split("\n")
    idx    = parts[0]
    season = parts[1] if len(parts) > 1 else "annual"
    return (idx, SEASON_ORDER.get(season, 99))


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_significant_cols(var_types, threshold=INCLUDE_THRESHOLD):
    sub = df[df["var_type"].isin(var_types)].copy()
    sub = sub[~sub["season"].isin(["ADV", "RET"])]  # exclude shoulder seasons
    sub["col_label"] = sub["index"] + "\n" + sub["season"]
    sig_cols = (sub.groupby("col_label")["pearson_p"]
                .min()
                .loc[lambda x: x < threshold]
                .index.tolist())
    return sorted(sig_cols, key=sort_key)


def build_matrix(var_type, col_labels):
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


def draw_heatmap(ax, r_mat, sig_mat, title, show_xticklabels=True):
    n_rows, n_cols = r_mat.shape
    norm = mcolors.TwoSlopeNorm(vmin=-VLIM, vcenter=0, vmax=VLIM)
    cmap = plt.get_cmap(CMAP)

    for i, sector in enumerate(r_mat.index):
        for j, col in enumerate(r_mat.columns):
            r   = r_mat.loc[sector, col]
            sig = sig_mat.loc[sector, col]

            color = "#F5F5F5" if pd.isna(r) else cmap(norm(r))

            rect = FancyBboxPatch(
                (j + 0.05, i + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="white", linewidth=1.5,
                zorder=2,
            )
            ax.add_patch(rect)

            if not pd.isna(r):
                brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
                txt_color  = "white" if brightness < 0.5 else "#2C2C2A"
                sig_str    = sig if isinstance(sig, str) else ""
                ax.text(j + 0.5, i + 0.5, f"{r:+.2f}{sig_str}",
                        ha="center", va="center",
                        fontsize=12, color=txt_color,
                        fontweight="bold" if "*" in sig_str else "normal")

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(r_mat.index, fontsize=13)
    ax.invert_yaxis()

    if show_xticklabels:
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels(r_mat.columns, fontsize=12,
                           ha="center", va="top")
    else:
        ax.set_xticks([])

    ax.set_title(title, fontsize=18, fontweight="bold", pad=8)

    for x in range(n_cols + 1):
        ax.axvline(x, color="white", lw=0.5, zorder=1)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", lw=0.5, zorder=1)

    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(length=0)

    return norm, cmap


# ── Figure builder ────────────────────────────────────────────────────────────
def make_figure(phase_var, amp_var, outfile, suptitle):
    # Get union of significant columns across phase and amplitude
    all_cols = sorted(
        set(get_significant_cols([phase_var])) |
        set(get_significant_cols([amp_var])),
        key=sort_key
    )
    print(f"  {outfile}: {len(all_cols)} columns")

    r_phase, sig_phase = build_matrix(phase_var, all_cols)
    r_amp,   sig_amp   = build_matrix(amp_var,   all_cols)

    n_rows = len(SECTOR_ORDER)
    n_cols = len(all_cols)

    cell_w = 1.3
    cell_h = 1.3
    fig_w  = max(10, n_cols * cell_w + 2.0)
    fig_h  = n_rows * cell_h + 3.5

    fig, (ax_phase, ax_amp) = plt.subplots(
        1, 2,
        figsize=(fig_w * 1.8, fig_h),
    )

    norm, cmap = draw_heatmap(ax_phase, r_phase, sig_phase,
                               "(a)  Phase anomaly",
                               show_xticklabels=True)
    # Hide y-axis labels on right panel
    draw_heatmap(ax_amp, r_amp, sig_amp,
                 "(b)  Amplitude anomaly",
                 show_xticklabels=True)
    ax_amp.set_yticks([])
    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, -0.04, 0.50, 0.025])
    cbar    = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Pearson r  |  ** FDR p<0.05   * p<0.05",
                   fontsize=14)
    cbar.ax.tick_params(labelsize=15)

  #  fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.08, right=0.98, wspace=0.08, bottom=0.12, top=0.95)
    save_fig(fig, outfile, OUTPUT_DIR)

    ret = os.system(f'rclone copy "{os.path.join(OUTPUT_DIR, outfile)}" "{GDRIVE}"')
    if ret == 0:
        print(f"  Synced → {GDRIVE}")
    else:
        print(f"  WARNING: rclone failed")


# ── Run ───────────────────────────────────────────────────────────────────────
print("\nFigure 1: APAC anomaly correlations")
make_figure(
    phase_var = "phase_apac",
    amp_var   = "amplitude_apac",
    outfile   = "fig_correlation_heatmap_apac.png",
    suptitle  = "Correlation between atmospheric indices and APAC phase and amplitude anomalies",
)

print("\nFigure 2: Raw anomaly correlations")
make_figure(
    phase_var = "phase_raw",
    amp_var   = "amplitude_raw",
    outfile   = "fig_correlation_heatmap_raw.png",
    suptitle  = "Correlation between atmospheric indices and raw phase and amplitude anomalies",
)

print("\nDone.")