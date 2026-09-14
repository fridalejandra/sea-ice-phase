"""
fig08_correlation_heatmap.py
============================
Index × season correlation heatmap, one panel per sector, for the OBSERVED
metrics only.

METRIC POLICY — the reason this figure filters
----------------------------------------------
correlations_output.csv contains four var_type values:

    phase_apac      from max_doy_anom       — argmax of the FITTED curve
    amplitude_apac  from amplitude_anom     — fitted amplitude
    phase_raw       from max_doy_raw_anom   — OBSERVED maximum date
    amplitude_raw   from amplitude_raw_anom — OBSERVED max minus min

The chapter uses the observed pair. max_doy_fitted is dominated by the fixed
s(DOY) term: its SD is 2-4x smaller than observed, it correlates with the
observed maximum at only r = 0.02-0.46, and in Ross it correlates with the
observed MINIMUM at r = -0.74. Correlations computed against it are
correlations against a compressed, partly spurious series.

This script therefore plots ONLY phase_raw and amplitude_raw. Set
SHOW_FITTED = True to produce the fitted-metric version for comparison in
the supplement — it is not the main-text figure.

If min_doy_raw_anom is present (it is not in the current pipeline — no
correlation script has ever used the minimum date), it is included
automatically.

Significance: uses pearson_sig, the Benjamini-Hochberg FDR flag computed
within each (var_type x index) group by compute_atmospheric_correlations.py.
Cells failing FDR are drawn faded; a dot marks those that pass.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import ch3_data as D
from ch3_config import CORR_CSV, SECTORS, SECTOR_LABELS
from ch3_plot import panel_letters, save

SHOW_FITTED = False          # True -> supplementary fitted-metric version

# var_type values to plot, in row-block order, with display names
VAR_OBSERVED = [
    # "phase_raw" is the pre-patch name; "phase_max_raw" the post-patch name
    # (renamed for symmetry once phase_min_raw was added). Both accepted.
    ("phase_max_raw", "Max date"),
    ("phase_raw",     "Max date"),
    ("phase_min_raw", "Min date"),
    ("amplitude_raw", "Amplitude"),
]
VAR_FITTED = [
    ("phase_apac",     "Max date (fitted)"),
    ("amplitude_apac", "Amplitude (fitted)"),
]
VAR_OPTIONAL = []

SEASON_ORDER = ["annual", "DJF", "MAM", "JJA", "SON", "ADV", "RET"]
INDEX_ORDER  = ["SAM", "ZW3R", "ZW3G", "ASL", "Nino34"]
INDEX_LABELS = {"SAM": "SAM", "ZW3R": "ZW3 (R)", "ZW3G": "ZW3 (G)",
                "ASL": "ASL", "Nino34": "Niño3.4"}

print("fig08 — index correlation heatmap (observed metrics)")

corr = D.load_correlations(
    CORR_CSV,
    required_cols=["sector", "var_type", "index", "season", "pearson_r"])

sig_col = "pearson_sig" if "pearson_sig" in corr.columns else None
if sig_col is None:
    print("  NOTE: no pearson_sig column — significance marks omitted.")

wanted = VAR_FITTED if SHOW_FITTED else VAR_OBSERVED
present = set(corr["var_type"].unique())
wanted = [(v, lab) for v, lab in wanted if v in present]
wanted += [(v, lab) for v, lab in VAR_OPTIONAL if v in present]

if not wanted:
    raise SystemExit(
        f"None of the expected var_type values found. Present: {sorted(present)}\n"
        "Rerun compute_atmospheric_correlations.py against annual_params_B.csv."
    )
print(f"  plotting var_type: {[v for v, _ in wanted]}")

# Row labels: one row per (variable, index) pair
rows = [(v, lab, idx) for v, lab in wanted for idx in INDEX_ORDER
        if idx in corr["index"].unique()]
seasons = [s for s in SEASON_ORDER if s in corr["season"].unique()]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
norm = TwoSlopeNorm(vmin=-0.7, vcenter=0.0, vmax=0.7)

for ax, sec in zip(axes.ravel(), SECTORS):
    sub = corr[corr["sector"] == sec]
    if len(sub) == 0:
        # correlation pipeline covers the 5 sectors, not circumpolar
        ax.set_visible(False)
        continue

    M = np.full((len(rows), len(seasons)), np.nan)
    S = np.zeros_like(M, dtype=bool)

    for i, (vt, _, idx_name) in enumerate(rows):
        for j, season in enumerate(seasons):
            cell = sub[(sub["var_type"] == vt) &
                       (sub["index"] == idx_name) &
                       (sub["season"] == season)]
            if len(cell):
                M[i, j] = cell["pearson_r"].iloc[0]
                if sig_col:
                    S[i, j] = bool(cell[sig_col].iloc[0])

    # faded base layer, full-opacity for FDR-significant cells
    ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto", alpha=0.45)
    masked = np.where(S, M, np.nan)
    ax.imshow(masked, cmap="RdBu_r", norm=norm, aspect="auto")

    for i in range(len(rows)):
        for j in range(len(seasons)):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if abs(M[i, j]) > 0.42 else "#2C2C2A",
                    fontweight="bold" if S[i, j] else "normal")

    # block separators between variables
    nidx = len([r for r in rows if r[0] == rows[0][0]])
    for k in range(nidx, len(rows), nidx):
        ax.axhline(k - 0.5, color="#2C2C2A", lw=1.4)

    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([INDEX_LABELS.get(r[2], r[2]) for r in rows], fontsize=8)
    ax.set_title(SECTOR_LABELS[sec], fontsize=11, fontweight="bold")
    ax.set_ylim(len(rows) - 0.5, -1.1)   # headroom so panel letters clear row 0

    # variable-block labels down the left edge
    for b, (vt, lab) in enumerate(wanted):
        y = b * nidx + (nidx - 1) / 2
        ax.text(-0.5 - (len(seasons) * 0.14), y, lab, rotation=90,
                va="center", ha="center", transform=ax.transData,
                fontsize=8.5, fontweight="bold", color="#555555",
                clip_on=False)

sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                    fraction=0.02, pad=0.02)
cbar.set_label("Pearson r", fontsize=10)

panel_letters(axes, y=0.995)
kind = "fitted (SUPPLEMENT — artifact metrics)" if SHOW_FITTED else "observed"
fig.suptitle(f"Index–metric correlations by season and sector ({kind}).  "
             "Bold and full colour = passes FDR.",
             fontsize=12, fontweight="bold")

save(fig, "fig08_correlation_heatmap.png"
     if not SHOW_FITTED else "figS0X_correlation_heatmap_fitted.png")
