#!/usr/bin/env python3
"""
fig_11_component_comparison_heatmap.py — Figure 11: for each of the seven
pre-specified pairs, the same index and season correlated against four
different summaries of the sea ice: raw seasonal SIE, amplitude, phase, and
residual volatility (§3.3/§4.3). This is the figure that shows what
decomposition adds beyond raw extent, pair by pair.

Split out of build_ch3_figures.py (2026-09-18) so each manuscript figure has
its own script, named to match: fig_##_name.py. Figure logic unchanged from
that file's Fig-11 block.

The East Antarctica-SAM_RET pair's raw_sie cell will show "n/a" — that's
correct, not a bug: annual_params.csv has no retreat-season SIE column, so
there is no like-for-like raw-extent value for that one pair (see the
chapter draft's changelog, ninth pass, for the full diagnosis). Whether to
compute a proper retreat-season SIE or just caption this gap is still an
open decision; this script already handles the NaN gracefully either way.

Run this from the same directory as ch3_config.py
(scripts/python/plotting/Ch3/figures/), after compute_fig11_component_comparison.py
(renamed 2026-09-18 from compute_component_comparison.py) has produced
t37_component_comparison.csv.

Reads:
    results/ch3/tables/t37_component_comparison.csv (skipped if absent)
Writes:
    results/ch3/figures/fig11_component_comparison_heatmap.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR, OUTPUT_DIR
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure

comp_path = os.path.join(TABLES_DIR, "t37_component_comparison.csv")
if not os.path.exists(comp_path):
    print(f"SKIPPING Fig 11 — {comp_path} not found. "
          f"Run compute_component_comparison.py first, then re-run this script.")
else:
    comp = pd.read_csv(comp_path)
    cols = ["r_raw_sie", "r_amplitude", "r_phase", "r_residual"]
    pcols = ["p_raw_sie", "p_amplitude", "p_phase", "p_residual"]
    col_labels = ["Raw seasonal\nSIE (detrended)", "Amplitude", "Phase\n(day of max/min)", "Residual\nvolatility (SD)"]

    comp["row_label"] = comp["sector"] + " · " + comp["index"]
    mat = comp.set_index("row_label")[cols].astype(float)
    pmat = comp.set_index("row_label")[pcols].astype(float)
    pmat.columns = cols

    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(mat) + 1.6))
    vmax = np.nanmax(np.abs(mat.values))
    vmax = max(vmax, 0.3)
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            r = mat.values[i, j]
            p = pmat.values[i, j]
            if np.isnan(r):
                txt = "n/a"
            else:
                star = "*" if (not np.isnan(p) and p < 0.05) else ""
                txt = f"{r:+.2f}{star}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if (not np.isnan(r) and abs(r) > vmax * 0.55) else "black", fontsize=9)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=9)
    ax.set_title(
        "Correlation with the same index and season, by measurement type\n"
        "(* p<.05, uncorrected, single comparison)",
        fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    out11 = os.path.join(OUTPUT_DIR, "fig11_component_comparison_heatmap.png")
    fig.savefig(out11, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out11}")
    print(comp[["sector", "index", "season", "r_raw_sie", "r_amplitude", "r_phase", "r_residual"]].to_string(index=False))