"""
figs_monthly_lagged_heatmap.py
==============================
Monthly lagged correlation heatmaps (phase and amplitude) with block bootstrap
CI width panels. Style matches Fig 10 (existing seasonal correlation heatmap).

Figure layout — four lettered panels:
    (a) Phase anomaly vs atmospheric indices        [r heatmap]
    (b) Phase — block bootstrap 95 % CI width       [CI panel]
    (c) Amplitude anomaly vs atmospheric indices    [r heatmap]
    (d) Amplitude — block bootstrap 95 % CI width   [CI panel]

Rows    = index × sector  (SAM/ZW3/ASL/Niño3.4  ×  EA/Ross/ABS/Weddell/KH)
Columns = calendar month of atmospheric index (Jan–Dec)

Cell content (r panels):
    Numeric r value centred in each cell (e.g. +0.47)
    ** p_fdr  < 0.05   (Benjamini–Hochberg FDR, matches Fig 10 convention)
    *  p_raw  < 0.05   (uncorrected)
    Stars superscripted top-right of r value

CI width panels (below each r panel):
    Cell colour = ci_high − ci_low   (Greys: darker = wider = less precise)
    Numeric CI width printed in each cell

Input
-----
    processed/monthly_cross_correlations.csv
    Required columns:
        index, sector, variable, month,
        r, p_raw, p_fdr, ci_low, ci_high

Output
------
    figures/figS02_monthly_lagged_heatmap.pdf  +  .png  (300 dpi)

Usage
-----
    python figS02_monthly_lagged_heatmap.py
    python figS02_monthly_lagged_heatmap.py --input processed/monthly_cross_correlations.csv --outdir figures/
"""

import argparse
import os
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, Normalize

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_INPUT  = pathlib.Path("/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data/monthly_cross_correlations.csv")
DEFAULT_OUTDIR = pathlib.Path("/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch3/figures")

INDEX_ORDER   = ["SAM", "ZW3R", "ASL", "Nino34"]
SECTOR_ORDER  = ["East Antarctica", "Ross", "ABS", "Weddell", "King Haakon"]

INDEX_LABELS  = {"SAM": "SAM", "ZW3R": "ZW3", "ASL": "ASL", "Nino34": "Niño3.4"}
SECTOR_LABELS = {"East Antarctica": "East Antarctica", "Ross": "Ross",
                 "ABS": "ABS", "Weddell": "Weddell", "King Haakon": "King Haakon"}

MONTH_LABELS  = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

VMAX_R   = 0.5     # r colour scale cap (matches Fig 10 ±0.4 range, slight expansion)
CMAP_R   = "RdBu_r"
CMAP_CI  = "Greys"
ALPHA_NS = 0.30    # white overlay opacity for non-significant cells

# Layout
FIG_WIDTH   = 16.0   # inches
ROW_H       = 0.42   # inches per data row (r panels)
CI_ROW_H    = 0.30   # inches per data row (CI panels, more compact)
TOP_PAD     = 0.55   # inches
BOT_PAD     = 0.65   # inches
INTRA_GAP   = 0.20   # between r panel and its CI panel
INTER_GAP   = 0.55   # between CI panel and next r panel
LEFT_FRAC   = 0.13
RIGHT_FRAC  = 0.85


# ── Helpers ────────────────────────────────────────────────────────────────────
def row_keys():
    return [(idx, sec) for idx in INDEX_ORDER for sec in SECTOR_ORDER]


def load_data(path):
    required = {"index", "sector", "variable", "month",
                "r", "p_raw", "p_fdr", "ci_low", "ci_high"}
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing columns: {missing}\n"
            "Run compute_monthly_lagged_correlations.py first."
        )
    df["variable"] = df["variable"].str.lower().str.strip()
    df["index"]    = df["index"].str.strip()
    df["sector"]   = df["sector"].str.strip()
    return df


def build_matrices(df, variable):
    sub  = df[df["variable"] == variable]
    keys = row_keys()
    n    = len(keys)

    r_mat   = np.full((n, 12), np.nan)
    sig_mat = np.zeros((n, 12), dtype=int)
    ci_mat  = np.full((n, 12), np.nan)

    for i, (idx, sec) in enumerate(keys):
        cell = sub[(sub["index"] == idx) & (sub["sector"] == sec)]
        if cell.empty:
            continue
        cell = cell.set_index("month")
        for m in range(1, 13):
            j = m - 1
            if m not in cell.index:
                continue
            row = cell.loc[m]
            r_mat[i, j]  = row["r"]
            ci_mat[i, j] = row["ci_high"] - row["ci_low"]
            if   row["p_fdr"] < 0.05:
                sig_mat[i, j] = 2
            elif row["p_raw"] < 0.05:
                sig_mat[i, j] = 1

    row_labels  = [SECTOR_LABELS.get(sec, sec) for _, sec in keys]
    group_edges = [i for i, (idx, _) in enumerate(keys)
                   if i == 0 or keys[i-1][0] != idx]

    return r_mat, sig_mat, ci_mat, row_labels, group_edges


# ── Panel drawing ──────────────────────────────────────────────────────────────
def fs_for(n_rows):
    if n_rows <= 10: return 7.0
    if n_rows <= 16: return 6.5
    return 6.0


def _index_group_labels(ax, group_edges, n_rows, n_cols, fontsize=7.5):
    """Draw index group labels on the right margin."""
    for g, edge in enumerate(group_edges):
        end = group_edges[g + 1] if g + 1 < len(group_edges) else n_rows
        mid = (edge + end - 1) / 2
        ax.text(n_cols - 0.5 + 0.4, mid,
                INDEX_LABELS.get(INDEX_ORDER[g], INDEX_ORDER[g]),
                va="center", ha="left", fontsize=fontsize,
                fontweight="bold", clip_on=False)


def _group_dividers(ax, group_edges):
    for edge in group_edges[1:]:
        ax.axhline(edge - 0.5, color="white", linewidth=1.8, zorder=4)


def _minor_grid(ax, n_rows, n_cols):
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, zorder=5)
    ax.tick_params(which="minor", length=0)


def draw_r_panel(ax, r_mat, sig_mat, row_labels, group_edges,
                 title, show_xlabels=True):
    n_rows, n_cols = r_mat.shape
    fs = fs_for(n_rows)

    norm = TwoSlopeNorm(vmin=-VMAX_R, vcenter=0, vmax=VMAX_R)
    cmap = plt.get_cmap(CMAP_R).copy()
    cmap.set_bad("#f0f0f0")

    im = ax.imshow(r_mat, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    for i in range(n_rows):
        for j in range(n_cols):
            r_val = r_mat[i, j]
            sig   = sig_mat[i, j]
            if np.isnan(r_val):
                continue
            if sig == 0:
                ax.add_patch(mpl.patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    color="white", alpha=ALPHA_NS, zorder=2, linewidth=0
                ))
            txt   = f"{r_val:+.2f}"
            stars = "**" if sig == 2 else ("*" if sig == 1 else "")
            color = "white" if abs(r_val) > 0.32 else "black"
            ax.text(j, i, txt + stars, ha="center", va="center",
                    fontsize=fs, color=color, zorder=3,
                    fontweight="bold" if sig > 0 else "normal")

    _group_dividers(ax, group_edges)
    _index_group_labels(ax, group_edges, n_rows, n_cols)
    _minor_grid(ax, n_rows, n_cols)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(MONTH_LABELS if show_xlabels else [], fontsize=7.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5, loc="left")

    return im


def draw_ci_panel(ax, ci_mat, row_labels, group_edges,
                  title, show_xlabels=True):
    n_rows, n_cols = ci_mat.shape
    fs = fs_for(n_rows) - 0.5

    vmax_ci = np.nanpercentile(ci_mat, 95)
    norm    = Normalize(vmin=0, vmax=vmax_ci)
    cmap    = plt.get_cmap(CMAP_CI).copy()
    cmap.set_bad("#f8f8f8")

    im = ax.imshow(ci_mat, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    for i in range(n_rows):
        for j in range(n_cols):
            v = ci_mat[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > 0.65 * vmax_ci else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=fs, color=color, zorder=3)

    _group_dividers(ax, group_edges)
    _index_group_labels(ax, group_edges, n_rows, n_cols, fontsize=7.0)
    _minor_grid(ax, n_rows, n_cols)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(MONTH_LABELS if show_xlabels else [], fontsize=7.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7.0)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_title(title, fontsize=8.5, fontstyle="italic", pad=4, loc="left")

    return im


# ── Figure assembly ────────────────────────────────────────────────────────────
def make_figure(input_path, outdir):
    df   = load_data(input_path)
    keys = row_keys()
    n    = len(keys)

    r_ph, sig_ph, ci_ph, row_labels, group_edges = build_matrices(df, "phase")
    r_am, sig_am, ci_am, _,          _           = build_matrices(df, "amplitude")

    # ── Compute figure height from panel heights ───────────────────────────
    h_r  = n * ROW_H
    h_ci = n * CI_ROW_H
    heights_in = [h_r, h_ci, h_r, h_ci]
    total_h    = (TOP_PAD + BOT_PAD
                  + heights_in[0] + INTRA_GAP
                  + heights_in[1] + INTER_GAP
                  + heights_in[2] + INTRA_GAP
                  + heights_in[3])

    fig = plt.figure(figsize=(FIG_WIDTH, total_h))

    # ── Position axes manually (top → bottom) ─────────────────────────────
    w     = RIGHT_FRAC - LEFT_FRAC
    y_top = 1.0 - TOP_PAD / total_h
    gaps  = [INTRA_GAP, INTER_GAP, INTRA_GAP]

    axes = []
    y = y_top
    for k, h in enumerate(heights_in):
        h_frac = h / total_h
        ax = fig.add_axes([LEFT_FRAC, y - h_frac, w, h_frac])
        axes.append(ax)
        y -= h_frac
        if k < len(gaps):
            y -= gaps[k] / total_h

    # ── Draw panels ───────────────────────────────────────────────────────
    im_r_ph  = draw_r_panel( axes[0], r_ph,  sig_ph,  row_labels, group_edges,
                             title="(a)  Phase anomaly vs atmospheric indices",
                             show_xlabels=False)
    im_ci_ph = draw_ci_panel(axes[1], ci_ph,           row_labels, group_edges,
                             title="(b)  Phase — block bootstrap 95 % CI width",
                             show_xlabels=False)
    im_r_am  = draw_r_panel( axes[2], r_am,  sig_am,  row_labels, group_edges,
                             title="(c)  Amplitude anomaly vs atmospheric indices",
                             show_xlabels=False)
    im_ci_am = draw_ci_panel(axes[3], ci_am,           row_labels, group_edges,
                             title="(d)  Amplitude — block bootstrap 95 % CI width",
                             show_xlabels=True)

    # ── Colorbars ─────────────────────────────────────────────────────────
    cb_x     = RIGHT_FRAC + 0.015
    cb_w     = 0.013

    # Shared r colorbar spanning panels (a) and (c)
    pos_a = axes[0].get_position()
    pos_c = axes[2].get_position()
    cb_r_ax = fig.add_axes([cb_x, pos_c.y0, cb_w, pos_a.y1 - pos_c.y0])
    cb_r = fig.colorbar(im_r_ph, cax=cb_r_ax, extend="both")
    cb_r.set_label("Pearson  r", fontsize=8, labelpad=5)
    cb_r.set_ticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    cb_r.ax.tick_params(labelsize=7.5)
    # Significance note inside r colorbar
    cb_r.ax.text(0.5, -0.07, "** FDR p<0.05\n* p<0.05",
                 transform=cb_r.ax.transAxes,
                 fontsize=6.5, va="top", ha="center", color="#444")

    # CI colorbar — phase (b)
    pos_b = axes[1].get_position()
    cb_ci_ph_ax = fig.add_axes([cb_x, pos_b.y0, cb_w, pos_b.height])
    cb_ci_ph = fig.colorbar(im_ci_ph, cax=cb_ci_ph_ax)
    cb_ci_ph.set_label("95 % CI width", fontsize=7, labelpad=4)
    cb_ci_ph.ax.tick_params(labelsize=7)

    # CI colorbar — amplitude (d)
    pos_d = axes[3].get_position()
    cb_ci_am_ax = fig.add_axes([cb_x, pos_d.y0, cb_w, pos_d.height])
    cb_ci_am = fig.colorbar(im_ci_am, cax=cb_ci_am_ax)
    cb_ci_am.set_label("95 % CI width", fontsize=7, labelpad=4)
    cb_ci_am.ax.tick_params(labelsize=7)

    # ── Save ──────────────────────────────────────────────────────────────
    outdir.mkdir(parents=True, exist_ok=True)
    stem  = "figS02_monthly_lagged_heatmap"
    fpath = outdir / f"{stem}.png"
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"Saved → {fpath}")
    plt.close(fig)

    gdrive_dest = "gdrive:sea-ice-phase/results/Ch3_Figures/"
    ret = os.system(f'rclone copy "{fpath}" "{gdrive_dest}"')
    if ret == 0:
        print(f"Synced → {gdrive_dest}/{stem}.png")
    else:
        print(f"WARNING: rclone failed (exit code {ret})")
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monthly lagged correlation heatmap + CI width panels"
    )
    parser.add_argument("--input",  type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    make_figure(args.input, args.outdir)