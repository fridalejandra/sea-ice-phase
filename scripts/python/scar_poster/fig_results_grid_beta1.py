"""
fig_results_grid_beta1.py
Panel 1: Sensitivity (beta1) -- baseline wind coupling, own colorbar.
Panels 2-3: Response after 2016 (beta3, Net + Convergence) -- shared colorbar.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SECTORS = ["Amundsen-Bellingshausen", "Weddell", "King Haakon VII",
           "East Antarctica", "Ross-Amundsen"]
SHORT = {"Amundsen-Bellingshausen": "ABS", "Weddell": "WS",
         "King Haakon VII": "KH", "East Antarctica": "EA",
         "Ross-Amundsen": "RA"}
SEASONS = ["DJF", "MAM", "JJA", "SON"]

BETA1_CSV = "wind_divergence_binary_test.csv"
PANELS_BETA3 = [
    ("wind_divergence_binary_test.csv",              "Net"),
    ("wind_divergence_binary_test_div_negative.csv", "Convergence"),
]

OUT = "fig_results_grid_beta1.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def grid_from(csv, value_col):
    df = pd.read_csv(csv)
    vals = np.full((len(SECTORS), len(SEASONS)), np.nan)
    sig = np.zeros_like(vals, dtype=bool)
    for i, s in enumerate(SECTORS):
        for j, sea in enumerate(SEASONS):
            r = df[(df.sector == s) & (df.season == sea)]
            if len(r) == 0:
                continue
            vals[i, j] = r.iloc[0][value_col]
            sig[i, j] = bool(r.iloc[0].get("significant_fdr", False))
    return vals, sig


def draw_panel(ax, vals, sig, vmax, title, show_ylabels):
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(SECTORS)):
        for j in range(len(SEASONS)):
            if sig[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                            edgecolor="k", lw=2.2))
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=16, fontweight="bold")
    ax.set_xticks(range(len(SEASONS)))
    ax.set_xticklabels(SEASONS, fontsize=11)
    ax.set_yticks(range(len(SECTORS)))
    ax.set_yticklabels([SHORT[s] for s in SECTORS] if show_ylabels else [], fontsize=11)
    ax.set_title(title, fontsize=13.5, fontweight="bold")
    return im


def main():
    beta1_vals, beta1_sig = grid_from(BETA1_CSV, "wind_coef")
    vmax_beta1 = np.nanpercentile(np.abs(beta1_vals[np.isfinite(beta1_vals)]), 98)

    beta3_data = []
    beta3_all = []
    for csv, title in PANELS_BETA3:
        if not os.path.exists(csv):
            print(f"[skip] {csv} not found")
            continue
        vals, sig = grid_from(csv, "interaction_coef")
        beta3_data.append((vals, sig, title))
        beta3_all.append(vals[np.isfinite(vals)])
    vmax_beta3 = np.percentile(np.abs(np.concatenate(beta3_all)), 98)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # no significance stars on the baseline panel -- it's descriptive,
    # not a hypothesis test; stars belong only to the beta3 (response) panels
    beta1_sig_blank = np.zeros_like(beta1_sig, dtype=bool)
    im1 = draw_panel(axes[0], beta1_vals, beta1_sig_blank, vmax_beta1,
                     "Sensitivity (\u03b21)\nbaseline, pre-2016", show_ylabels=True)
    cb1 = fig.colorbar(im1, ax=axes[0], orientation="vertical", fraction=0.046, pad=0.04)
    cb1.set_label("\u03b21 (day\u207b\u00b9 per unit wind)", fontsize=9.5)

    im_last = None
    for k, (vals, sig, title) in enumerate(beta3_data):
        ax = axes[k + 1]
        im_last = draw_panel(ax, vals, sig, vmax_beta3,
                             f"{title}\nresponse after 2016 (\u03b23)",
                             show_ylabels=False)

    cb2 = fig.colorbar(im_last, ax=axes[1:].tolist(), orientation="vertical",
                       fraction=0.03, pad=0.03)
    cb2.set_label("\u03b23 (day\u207b\u00b9 per unit wind stress)", fontsize=9.5)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()
