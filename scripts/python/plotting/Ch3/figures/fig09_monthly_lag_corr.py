"""
fig09_monthly_lag_corr.py
=========================
Monthly lag correlations: which month's atmosphere relates to the annual
cycle metric, by index and sector.

Reads monthly_cross_correlations.csv (from compute_monthly_lagged_
correlations.py). One row per index; x-axis is calendar month; one line per
sector.

⚠️ METRIC WARNING — READ BEFORE USING THIS FIGURE
--------------------------------------------------
As currently written, compute_monthly_lagged_correlations.py sets

    APAC_VARS = {"max_doy_anom": "phase", "amplitude_anom": "amplitude"}

Both are FITTED quantities. max_doy_anom is the anomaly of the argmax of the
fitted curve — the artifact the chapter does not use (SD 2-4x smaller than
observed; r with observed max 0.02-0.46; in Ross r with the observed MINIMUM
is -0.74). Correlations against it are not correlations against observed
timing.

To fix, in compute_monthly_lagged_correlations.py:
    APAC_VARS = {"max_doy_raw_anom": "phase_max",
                 "min_doy_raw_anom": "phase_min",
                 "amplitude_raw_anom": "amplitude"}
and point ANNUAL_CSV at annual_params_B.csv, then rerun.

This script checks the `variable` values it finds and prints a loud warning
if they look like the fitted set. Set ALLOW_FITTED = True to plot anyway
(e.g. to reproduce the old figure for comparison).

Significance: the source script stores p_raw (uncorrected). With 5 sectors x
5 indices x 12 months x 2 variables = 600 tests, uncorrected p < 0.05 means
little on its own. Filled markers use FDR within each (sector, variable,
index) group, computed here.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests

import ch3_data as D
from ch3_config import MONTHLY_XC_CSV, SECTOR_COLORS, SECTOR_LABELS, R_SIG_44
from ch3_plot import sig_band, panel_letters, save

ALLOW_FITTED = False

FITTED_VARS = {"phase", "amplitude"}          # what the old pipeline emits
OBSERVED_HINTS = ("raw", "_max", "_min")      # what the fixed pipeline emits

INDEX_ORDER  = ["SAM", "ZW3R", "ASL", "Nino34"]
INDEX_LABELS = {"SAM": "SAM", "ZW3R": "ZW3R", "ASL": "ASL", "Nino34": "Niño3.4"}
MONTH_LABELS = list("JFMAMJJASOND")

# Advance / retreat shading
ADV_MONTHS = [3, 4, 5, 6, 7, 8]
RET_MONTHS = [10, 11, 12, 1]

print("fig09 — monthly lag correlations")

xc = D.load_correlations(
    MONTHLY_XC_CSV,
    required_cols=["sector", "variable", "index", "month", "r"])

# --- metric provenance check --------------------------------------------------
vars_found = set(xc["variable"].unique())
looks_observed = any(any(h in v for h in OBSERVED_HINTS) for v in vars_found)
if not looks_observed and vars_found & FITTED_VARS:
    msg = (f"variable values {sorted(vars_found)} look like the FITTED set.\n"
           "  These are argmax-of-fitted-curve artifacts, not observed timing.\n"
           "  Fix APAC_VARS in compute_monthly_lagged_correlations.py and rerun.")
    if not ALLOW_FITTED:
        raise SystemExit("STOP: " + msg + "\n  (set ALLOW_FITTED=True to override)")
    print("  WARNING: " + msg)

# which variable to plot — prefer an observed amplitude, else whatever exists
pref = [v for v in vars_found if "amplitude" in v]
VAR = pref[0] if pref else sorted(vars_found)[0]
print(f"  plotting variable = '{VAR}'  (available: {sorted(vars_found)})")

sub_all = xc[xc["variable"] == VAR].copy()

# --- FDR within (sector, index) -----------------------------------------------
if "p_raw" in sub_all.columns:
    out = []
    for _, g in sub_all.groupby(["sector", "index"]):
        g = g.copy()
        _, padj, _, _ = multipletests(g["p_raw"].values, alpha=0.05,
                                      method="fdr_bh")
        g["p_fdr"] = padj
        g["sig"] = padj < 0.05
        out.append(g)
    sub_all = __import__("pandas").concat(out)
else:
    sub_all["sig"] = sub_all["r"].abs() > R_SIG_44
    print("  NOTE: no p_raw column — falling back to |r| threshold.")

indices = [i for i in INDEX_ORDER if i in sub_all["index"].unique()]
sectors = [s for s in SECTOR_COLORS if SECTOR_LABELS[s] in sub_all["sector"].unique()]
# the correlation pipeline labels sectors by short name
label_to_key = {SECTOR_LABELS[k]: k for k in SECTOR_COLORS}

fig, axes = plt.subplots(len(indices), 1, figsize=(11, 3.0 * len(indices)),
                         sharex=True)
axes = np.atleast_1d(axes)

for ax, idx_name in zip(axes, indices):
    for m in ADV_MONTHS:
        ax.axvspan(m - 0.5, m + 0.5, color="#4CAF50", alpha=0.07, zorder=0)
    for m in RET_MONTHS:
        ax.axvspan(m - 0.5, m + 0.5, color="#F44336", alpha=0.07, zorder=0)

    sig_band(ax, R_SIG_44)

    g_idx = sub_all[sub_all["index"] == idx_name]
    for sec_label, g in g_idx.groupby("sector"):
        key = label_to_key.get(sec_label)
        if key is None:
            continue
        c = SECTOR_COLORS[key]
        g = g.sort_values("month")
        ax.plot(g["month"], g["r"], color=c, lw=1.5, alpha=0.85, zorder=2)
        s = g["sig"].values.astype(bool)
        ax.scatter(g["month"][s], g["r"][s], color=c, s=45, zorder=4,
                   edgecolors="white", linewidth=0.5)
        ax.scatter(g["month"][~s], g["r"][~s], s=25, zorder=3,
                   facecolors="none", edgecolors=c, linewidth=0.8, alpha=0.5)

    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_ylabel(f"{INDEX_LABELS.get(idx_name, idx_name)}\nPearson r", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

axes[-1].set_xlabel("Month of atmospheric index", fontsize=10)

sector_handles = [Line2D([0], [0], color=SECTOR_COLORS[k], lw=2,
                         label=SECTOR_LABELS[k])
                  for k in sectors]
extra = [
    mpatches.Patch(facecolor="#4CAF50", alpha=0.3, label="Advance (Mar–Aug)"),
    mpatches.Patch(facecolor="#F44336", alpha=0.3, label="Retreat (Oct–Jan)"),
    mpatches.Patch(facecolor="#888888", alpha=0.2, label="|r| below p=0.05, n=44"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555",
           markersize=8, label="passes FDR"),
]
fig.legend(handles=sector_handles + extra, loc="lower center", ncol=5,
           fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

panel_letters(axes)
fig.suptitle(f"Monthly lag correlations: annual {VAR} vs monthly index",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0.05, 1, 0.97])
save(fig, f"fig09_monthly_lag_corr_{VAR}.png")
