"""
fig_monthly_lagged_lineplot.py
==============================
Monthly lagged correlation line plots with block bootstrap CI shading.

For each variable (amplitude / phase), produces a figure with one subplot
per atmospheric index (SAM, ZW3, ASL, Niño3.4). Each subplot shows one
line per sector across calendar months (Jan–Dec), with shaded 95% CI bands.

Significance markers on data points:
    **  p_fdr < 0.05
    *   p_raw < 0.05

Two output figures:
    fig_monthly_lagged_amplitude.png
    fig_monthly_lagged_phase.png

Input:
    monthly_cross_correlations.csv  (from compute_monthly_lagged_correlations.py)

Usage:
    python fig_monthly_lagged_lineplot.py
    python fig_monthly_lagged_lineplot.py --input /path/to/monthly_cross_correlations.csv --outdir figures/
"""

import argparse
import os
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = pathlib.Path("/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data/monthly_cross_correlations.csv")
DEFAULT_OUTDIR = pathlib.Path("/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch3/figures")
GDRIVE_DEST    = "gdrive:My Drive/sea-ice-phase/results/Ch3_Figures"

# ── Display order & styling ───────────────────────────────────────────────────
INDEX_ORDER  = ["SAM", "ZW3R", "ASL", "Nino34"]
SECTOR_ORDER = ["East Antarctica", "Ross", "ABS", "Weddell", "King Haakon"]

INDEX_LABELS  = {"SAM": "SAM", "ZW3R": "ZW3", "ASL": "ASL", "Nino34": "Niño3.4"}
SECTOR_LABELS = {
    "East Antarctica": "East Antarctica",
    "Ross"           : "Ross",
    "ABS"            : "ABS",
    "Weddell"        : "Weddell",
    "King Haakon"    : "King Haakon",
}

SECTOR_COLORS = {
    "East Antarctica": "#e6194b",   # red
    "Ross"           : "#4363d8",   # blue
    "ABS"            : "#f58231",   # orange
    "Weddell"        : "#3cb44b",   # green
    "King Haakon"    : "#911eb4",   # purple
}

VAR_TITLES = {
    "amplitude": "Monthly lagged correlations — Amplitude anomaly",
    "phase"    : "Monthly lagged correlations — Phase anomaly",
}
VAR_YLABELS = {
    "amplitude": "Spearman  r  (amplitude anomaly)",
    "phase"    : "Spearman  r  (phase anomaly)",
}

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS       = list(range(1, 13))

FIG_WIDTH  = 13.0
FIG_HEIGHT = 8.0


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(path):
    required = {"index", "sector", "variable", "month",
                "r", "p_raw", "p_fdr", "ci_low", "ci_high"}
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df["variable"] = df["variable"].str.lower().str.strip()
    df["index"]    = df["index"].str.strip()
    df["sector"]   = df["sector"].str.strip()
    return df


# ── Figure builder ────────────────────────────────────────────────────────────
def make_figure(df, variable, outdir):
    indices = [i for i in INDEX_ORDER if i in df["index"].unique()]
    n       = len(indices)

    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        gridspec_kw={"hspace": 0.45, "wspace": 0.32,
                     "left": 0.07, "right": 0.97,
                     "top": 0.88, "bottom": 0.10}
    )
    axes = axes.flatten()

    # Hide unused subplots
    for k in range(n, len(axes)):
        axes[k].set_visible(False)

    letters = "abcdefghijklmnopqrstuvwxyz"

    for k, idx in enumerate(indices):
        ax      = axes[k]
        idx_df  = df[(df["index"] == idx) & (df["variable"] == variable)]
        idx_lbl = INDEX_LABELS.get(idx, idx)

        for sec in SECTOR_ORDER:
            sec_df = idx_df[idx_df["sector"] == sec].set_index("month")
            if sec_df.empty:
                continue

            r      = np.array([sec_df.loc[m, "r"]      if m in sec_df.index else np.nan for m in MONTHS])
            ci_lo  = np.array([sec_df.loc[m, "ci_low"]  if m in sec_df.index else np.nan for m in MONTHS])
            ci_hi  = np.array([sec_df.loc[m, "ci_high"] if m in sec_df.index else np.nan for m in MONTHS])
            p_raw  = np.array([sec_df.loc[m, "p_raw"]   if m in sec_df.index else 1.0    for m in MONTHS])
            p_fdr  = np.array([sec_df.loc[m, "p_fdr"]   if m in sec_df.index else 1.0    for m in MONTHS])

            color = SECTOR_COLORS.get(sec, "#888888")
            x     = np.array(MONTHS)

            # Line
            ax.plot(x, r, color=color, linewidth=1.4,
                    label=SECTOR_LABELS.get(sec, sec), zorder=3)

            # Shaded CI band
            valid = ~np.isnan(r) & ~np.isnan(ci_lo) & ~np.isnan(ci_hi)
            if valid.sum() > 1:
                ax.fill_between(x[valid], ci_lo[valid], ci_hi[valid],
                                color=color, alpha=0.12, zorder=2)

            # Significance markers
            for m_i, m in enumerate(MONTHS):
                if np.isnan(r[m_i]):
                    continue
                if p_fdr[m_i] < 0.05:
                    ax.text(m, r[m_i] + 0.03, "**", ha="center", va="bottom",
                            fontsize=6, color=color, fontweight="bold", zorder=4)
                elif p_raw[m_i] < 0.05:
                    ax.text(m, r[m_i] + 0.03, "*", ha="center", va="bottom",
                            fontsize=6, color=color, fontweight="bold", zorder=4)

        # Reference line at r=0
        ax.axhline(0, color="#999999", linewidth=0.7, linestyle="--", zorder=1)

        # Axes formatting
        ax.set_xlim(0.5, 12.5)
        ax.set_xticks(MONTHS)
        ax.set_xticklabels(MONTH_LABELS, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylabel("Spearman  r", fontsize=8)
        ax.set_title(idx_lbl, fontsize=10, fontweight="bold", pad=4)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        # r=±0.5 reference lines (visual guide)
        for ref in (-0.4, 0.4):
            ax.axhline(ref, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)

        # Panel label
        ax.text(-0.01, 1.04, f"({letters[k]})",
                transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="right")

    # ── Shared legend ─────────────────────────────────────────────────────────
    handles = [
        mpl.lines.Line2D([0], [0], color=SECTOR_COLORS[s], linewidth=2,
                         label=SECTOR_LABELS[s])
        for s in SECTOR_ORDER if s in df["sector"].unique()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.01))

    # Significance note
    fig.text(0.97, 0.01, "** FDR p<0.05   * p<0.05",
             fontsize=7.5, ha="right", va="bottom", color="#555555")

    fig.suptitle(VAR_TITLES[variable], fontsize=12, fontweight="bold")

    # ── Save + rclone ─────────────────────────────────────────────────────────
    stem  = f"fig_monthly_lagged_{variable}"
    fpath = outdir / f"{stem}.png"
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"  Saved → {fpath}")
    plt.close(fig)

    ret = os.system(f'rclone copy "{fpath}" "{GDRIVE_DEST}"')
    if ret == 0:
        print(f"  Synced → {GDRIVE_DEST}/{stem}.png")
    else:
        print(f"  WARNING: rclone failed (exit code {ret})")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(input_path, outdir):
    print("Loading data...")
    df = load_data(input_path)
    print(f"  {len(df)} rows | indices: {df['index'].unique()} | sectors: {df['sector'].unique()}")

    outdir.mkdir(parents=True, exist_ok=True)

    for variable in ["amplitude", "phase"]:
        print(f"\nPlotting {variable}...")
        make_figure(df, variable, outdir)

    print("\nDone.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monthly lagged correlation line plots with CI shading"
    )
    parser.add_argument("--input",  type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    main(args.input, args.outdir)