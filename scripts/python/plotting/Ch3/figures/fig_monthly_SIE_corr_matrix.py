"""
fig_monthly_sie_correlation_matrix.py
======================================
Monthly SIE anomaly correlation matrix for all 5 sectors.

For each sector, computes the mean SIE anomaly per calendar month per year
(observed - DOY climatological mean), then builds a 12x12 Spearman
correlation matrix across months.

Layout: 2x3 grid of subplots (5 sectors + 1 shared colorbar panel)

Input:
    SIE_daily_sector_and_circumpolar_million_km2.csv

Output:
    fig_monthly_sie_correlation_matrix.png
    Auto-rcloned to gdrive:My Drive/sea-ice-phase/results/Ch3_Figures

Usage:
    python fig_monthly_sie_correlation_matrix.py
"""

import os
import pathlib
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_FILE = pathlib.Path(
    "/user/geog/falejandraperez/sea-ice-phase/scripts/R/observations/"
    "SIE_daily_sector_and_circumpolar_million_km2.csv"
)
OUTDIR     = pathlib.Path(
    "/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch3/figures"
)
GDRIVE     = "gdrive:My Drive/sea-ice-phase/results/Ch3_Figures"

YEAR_MIN = 1979
YEAR_MAX = 2023

SECTOR_COLS = [
    "SIE_East_Antarctica",
    "SIE_Ross",
    "SIE_Amundsen_Bellingshausen",
    "SIE_Weddell",
    "SIE_King_Haakon",
]
SECTOR_LABELS = {
    "SIE_East_Antarctica"         : "East Antarctica",
    "SIE_Ross"                    : "Ross",
    "SIE_Amundsen_Bellingshausen" : "ABS",
    "SIE_Weddell"                 : "Weddell",
    "SIE_King_Haakon"             : "King Haakon",
}

MONTH_LABELS = ["J", "F", "M", "A", "M", "J",
                "J", "A", "S", "O", "N", "D"]
VMAX  = 1.0
CMAP  = "RdBu_r"


# ── Load and prepare data ─────────────────────────────────────────────────────
def load_monthly_anomalies(path):
    df = pd.read_csv(path)
    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DOY"]   = df["Date"].dt.dayofyear
    df = df[df["Year"].between(YEAR_MIN, YEAR_MAX)]

    # Compute DOY climatological mean for each sector
    monthly_anom = {}
    for sec in SECTOR_COLS:
        if sec not in df.columns:
            print(f"  WARNING: {sec} not found in data")
            continue
        d = df[["Year", "Month", "DOY", sec]].copy()
        d[sec] = pd.to_numeric(d[sec], errors="coerce")

        # DOY climatology
        doy_mean = d.groupby("DOY")[sec].mean()
        d["doy_mean"] = d["DOY"].map(doy_mean)
        d["anomaly"]  = d[sec] - d["doy_mean"]

        # Monthly mean anomaly per year
        mon = (d.groupby(["Year", "Month"])["anomaly"]
               .mean()
               .reset_index()
               .pivot(index="Year", columns="Month", values="anomaly"))
        mon.columns = mon.columns.astype(int)
        monthly_anom[sec] = mon

    return monthly_anom


# ── Build correlation matrix ──────────────────────────────────────────────────
def correlation_matrix(mon_df):
    months = list(range(1, 13))
    r_mat = np.full((12, 12), np.nan)
    p_mat = np.full((12, 12), np.nan)
    for i, m1 in enumerate(months):
        for j, m2 in enumerate(months):
            if m1 not in mon_df.columns or m2 not in mon_df.columns:
                continue
            paired = mon_df[[m1, m2]].dropna()
            if len(paired) < 10:
                continue
            r, p = spearmanr(paired[m1], paired[m2])
            r_mat[i, j] = r
            p_mat[i, j] = p
    return r_mat, p_mat


# ── Draw one heatmap panel ────────────────────────────────────────────────────
def draw_panel(ax, r_mat, p_mat, title):
    n = 12
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("#f0f0f0")

    im = ax.imshow(r_mat, aspect="auto", cmap=cmap,
                   vmin=-VMAX, vmax=VMAX, interpolation="nearest")

    # Significance markers — * p<0.05
    for i in range(n):
        for j in range(n):
            if np.isnan(r_mat[i, j]):
                continue
            r_val = r_mat[i, j]
            color = "white" if abs(r_val) > 0.5 else "black"
            marker = "*" if p_mat[i, j] < 0.05 else ""
            # r value
            ax.text(j, i, f"{r_val:.2f}", ha="center", va="center",
                    fontsize=5.5, color=color, zorder=3)
            if marker:
                ax.text(j + 0.38, i - 0.28, marker,
                        ha="center", va="center",
                        fontsize=5, color=color,
                        fontweight="bold", zorder=4)

    # Diagonal highlight
    for k in range(n):
        ax.add_patch(mpl.patches.Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            fill=False, edgecolor="#444444",
            linewidth=0.8, zorder=5
        ))

    ax.set_xticks(range(n))
    ax.set_xticklabels(MONTH_LABELS, fontsize=7.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(MONTH_LABELS, fontsize=7.5)
    ax.tick_params(length=0)

    # Minor grid
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, zorder=4)
    ax.tick_params(which="minor", length=0)

    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    return im


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    monthly_anom = load_monthly_anomalies(INPUT_FILE)
    print(f"  Loaded {len(monthly_anom)} sectors")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Figure layout: 2 rows x 3 cols, last cell = colorbar ─────────────────
    fig, axes = plt.subplots(
        2, 3,
        figsize=(14, 9),
        gridspec_kw={"hspace": 0.38, "wspace": 0.22,
                     "left": 0.05, "right": 0.88,
                     "top": 0.93, "bottom": 0.06}
    )
    axes_flat = axes.flatten()

    # Hide last panel — used for colorbar
    axes_flat[5].set_visible(False)

    letters = "abcde"
    last_im = None

    for k, sec in enumerate(SECTOR_COLS):
        ax = axes_flat[k]
        if sec not in monthly_anom:
            ax.set_visible(False)
            continue

        r_mat, p_mat = correlation_matrix(monthly_anom[sec])
        title = f"({letters[k]})  {SECTOR_LABELS[sec]}"
        last_im = draw_panel(ax, r_mat, p_mat, title)

        # x/y labels only on edge panels
        if k in (3, 4):
            ax.set_xlabel("Month", fontsize=8)
        if k in (0, 3):
            ax.set_ylabel("Month", fontsize=8)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.65])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label("Spearman  r", fontsize=9, labelpad=6)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=8)

    fig.text(0.05, 0.01,
             "* p < 0.05   |   Monthly mean SIE anomaly (observed − DOY climatology)",
             fontsize=7.5, color="#555555")

    fig.suptitle(
        "Monthly SIE anomaly correlations by sector  (1979–2023)",
        fontsize=12, fontweight="bold"
    )

    # ── Save + rclone ─────────────────────────────────────────────────────────
    stem  = "fig_monthly_sie_correlation_matrix"
    fpath = OUTDIR / f"{stem}.png"
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"Saved → {fpath}")
    plt.close(fig)

    ret = os.system(f'rclone copy "{fpath}" "{GDRIVE}"')
    if ret == 0:
        print(f"Synced → {GDRIVE}/{stem}.png")
    else:
        print(f"WARNING: rclone failed (exit code {ret})")
    print("Done.")


if __name__ == "__main__":
    main()