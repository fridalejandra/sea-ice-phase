#!/usr/bin/env python3
"""
fig_s01_fitted_vs_observed.py -- Supplementary Figure 1: how well the APAC
fit's own per-year parameters track the directly observed scalars, for
amplitude and for timing (phase), all six sectors (five + circumpolar).

REPLACES the previous Fig. S1 (`fig07a_circumpolar_2016.png`, from
plot_fig7_sectors.R -- the circumpolar-only panel digitizing HR20's
published Fig. 7a and comparing our reproduction against it). That
panel's own validation numbers (RMS differences 0.03/0.49/1.10, trend/
amplitude/phase; 2016 phase trough -8.6 vs -7.8) stay fully described in
prose in Sec 2.2.5 -- they don't need a citation to a figure. This new
figure is a different, broader check: not "does our reproduction match
HR20's paper" but "does the fitted model's own amplitude/timing agree with
the amplitude/timing you'd read directly off the data," for every sector,
not just circumpolar. Frida confirmed 2026-09-18: this replaces Fig. S1
(the old panel is not kept as a separate supplementary figure); Sec 2.2.5's
"(Fig. S1)" citation on the old panel needs removing to match (see the
note left in ch3_paper_draft_v2.md at that spot, and the note in
run_all.py's docstring).

This backs up specific claims already written in Sec 3.2: "the day of
minimum ... fitted and observed dates agree almost exactly (r >= 0.99)"
and "the fitted timing has a quarter to a half of the variance of the
observed timing." CONFIRMED 2026-09-18 against Frida's real
annual_params.csv: day-of-minimum fitted-vs-observed r = 0.99-1.00 in
every sector (matches "r >= 0.99" exactly), day-of-maximum r = 0.11-0.53
(much weaker, consistent with Sec 3.2's own framing that the maximum is
the harder-to-fit extremum), amplitude r = 1.00 in every sector. The
variance-ratio claim ("a quarter to a half") hasn't been checked yet --
this script's own t_s01_fitted_vs_observed_fit_quality.csv table has a
var_ratio_fit_over_obs column for exactly that comparison.

COLUMN NAMES: CONFIRMED 2026-09-18 against Frida's real annual_params.csv
(she uploaded it after the first version of this script picked the wrong
ones). See the FIT_AMP/FIT_MAXDOY/FIT_MINDOY block below for the specific
diagnosis -- the first version paired an absolute-scale fitted column
against an anomaly-scale observed column, which produces a misleadingly
near-perfect-looking but offset scatter, not a real validation.

Layout: two stacked blocks, each a 2x3 grid (one panel per sector, matching
the ncol=3 convention used elsewhere in this chapter, e.g.
fig03_attribution_annual.py). Top block: fitted vs. observed amplitude.
Bottom block: fitted vs. observed timing, day-of-max and day-of-min
overlaid in the same panel (filled = day of max, open = day of min),
since both are the same units (days) and both bear on the "phase" claim.
Each panel gets a 1:1 reference line and its Pearson r.

Inputs:
    ANNUAL_CSV (period == "FULL")
Outputs:
    results/ch3/figures/fig_s01_fitted_vs_observed.png
    results/ch3/tables/t_s01_fitted_vs_observed_fit_quality.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import SECTORS, SECTOR_LABELS, SECTOR_COLORS, ANNUAL_CSV, TABLES_DIR, OUTPUT_DIR
from ch3_plot import save
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure


a = pd.read_csv(ANNUAL_CSV)
if "period" in a.columns:
    a = a[a["period"] == "FULL"]
a = a[a.sector.isin(SECTORS)]
print(f"annual_params.csv: {a.Year.min()}-{a.Year.max()}, {a.Year.nunique()} years, "
      f"{a.sector.nunique()} sectors, period filtered to FULL")
print(f"\nall columns in {os.path.basename(ANNUAL_CSV)} (for reference if the "
      f"guesses below are wrong):\n  {sorted(a.columns)}")

# ---- observed columns: confirmed real names from earlier this session ----
OBS_AMP = "amplitude_raw_anom"
OBS_MAXDOY = "max_doy_raw_anom"
OBS_MINDOY = "min_doy_raw_anom"
for c, lab in [(OBS_AMP, "observed amplitude"), (OBS_MAXDOY, "observed max DOY"),
               (OBS_MINDOY, "observed min DOY")]:
    if c not in a.columns:
        sys.exit(f"expected observed column {c!r} ({lab}) not found -- "
                  f"columns are {sorted(a.columns)}")

# ---- fitted columns: CONFIRMED 2026-09-18 against Frida's real annual_params.csv ----
# The first version of this script guessed "amplitude_fitted" / "max_doy_fitted" /
# "min_doy_fitted" for the fitted side -- those columns are real, but they're on
# the ABSOLUTE scale (actual day-of-year 1-365, actual amplitude in km^2), while
# the observed columns above (*_raw_anom) are on the ANOMALY scale (deviation from
# each sector's own climatology, centered near zero). Pairing an absolute-scale
# fitted value against an anomaly-scale observed value produces a near-perfect but
# WRONG scatter -- r close to 1.0, but offset from the 1:1 line by a large,
# roughly-constant amount, which is exactly the "something is off" plot Frida
# flagged. The real annual_params.csv has a SEPARATE set of anomaly-scale fitted
# columns -- amplitude_anom / max_doy_anom / min_doy_anom -- verified directly
# against the uploaded file to sit on the same scale as, and correlate sensibly
# with, the *_raw_anom observed columns (r=1.00 for amplitude in every sector;
# r=0.99-1.00 for day of minimum, matching the "r >= 0.99" already written in
# Sec 3.2; r=0.11-0.53 for day of maximum, consistent with Sec 3.2's own framing
# that timing agreement is much weaker for the maximum than the minimum). These
# are the correct fitted-side columns, not the "_fitted" ones.
FIT_AMP = "amplitude_anom"
FIT_MAXDOY = "max_doy_anom"
FIT_MINDOY = "min_doy_anom"
for c, lab in [(FIT_AMP, "fitted amplitude"), (FIT_MAXDOY, "fitted day of maximum"),
               (FIT_MINDOY, "fitted day of minimum")]:
    if c not in a.columns:
        sys.exit(f"expected fitted column {c!r} ({lab}) not found -- "
                  f"columns are {sorted(a.columns)}")

# ---- per-sector fit-quality table ----
rows = []
for sec in SECTORS:
    s = a[a.sector == sec].dropna(subset=[OBS_AMP, FIT_AMP, OBS_MAXDOY, FIT_MAXDOY,
                                           OBS_MINDOY, FIT_MINDOY])
    for obs_col, fit_col, quantity in [(OBS_AMP, FIT_AMP, "amplitude"),
                                        (OBS_MAXDOY, FIT_MAXDOY, "day_of_max"),
                                        (OBS_MINDOY, FIT_MINDOY, "day_of_min")]:
        obs, fit = s[obs_col].values, s[fit_col].values
        r, p = pearsonr(obs, fit)
        rmse = float(np.sqrt(np.mean((obs - fit) ** 2)))
        var_ratio = float(np.var(fit, ddof=1) / np.var(obs, ddof=1))
        rows.append(dict(sector=SECTOR_LABELS[sec], quantity=quantity, n=len(s),
                          r=r, p=p, rmse=rmse, var_ratio_fit_over_obs=var_ratio))
fitq = pd.DataFrame(rows)
os.makedirs(TABLES_DIR, exist_ok=True)
fitq.to_csv(os.path.join(TABLES_DIR, "t_s01_fitted_vs_observed_fit_quality.csv"), index=False)

pd.set_option("display.width", 160)
print("\n" + "=" * 100)
print("fitted vs. observed fit quality, per sector (backs the Sec 3.2 claims -- "
      "check these numbers against what's currently written there)")
print("=" * 100)
print(fitq.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---- figure: two stacked 2x3 blocks ----
ncol = 3
nrow_ = int(np.ceil(len(SECTORS) / ncol))


def scatter_panel(ax, obs, fit, color, marker="o", filled=True, label=None):
    face = color if filled else "none"
    ax.scatter(obs, fit, s=22, marker=marker, facecolors=face, edgecolors=color,
               linewidths=1.0, alpha=0.85, zorder=4, label=label)


fig, axes = plt.subplots(2 * nrow_, ncol, figsize=(4.6 * ncol, 3.0 * 2 * nrow_))
axes = np.atleast_2d(axes)
amp_axes = axes[:nrow_].ravel()
phase_axes = axes[nrow_:].ravel()

for k, sec in enumerate(SECTORS):
    s = a[a.sector == sec]
    color = SECTOR_COLORS.get(sec, "0.4")

    # -- amplitude block --
    ax = amp_axes[k]
    obs, fit = s[OBS_AMP].values, s[FIT_AMP].values
    lo, hi = np.nanmin([obs.min(), fit.min()]), np.nanmax([obs.max(), fit.max()])
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.6", lw=0.9,
            ls="--", zorder=1)
    scatter_panel(ax, obs, fit, color)
    r, _ = pearsonr(obs, fit)
    ax.text(0.04, 0.94, f"r = {r:.2f}", transform=ax.transAxes, fontsize=8.5,
            va="top", color="0.25")
    ax.set_title(SECTOR_LABELS[sec], fontsize=10.5, color=color, pad=4)
    ax.tick_params(labelsize=8)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    if k % ncol == 0:
        ax.set_ylabel("fitted amplitude\n(10$^6$ km$^2$)", fontsize=8.5)

    # -- phase block: day of max (filled) + day of min (open), same panel --
    ax = phase_axes[k]
    obs_max, fit_max = s[OBS_MAXDOY].values, s[FIT_MAXDOY].values
    obs_min, fit_min = s[OBS_MINDOY].values, s[FIT_MINDOY].values
    lo = np.nanmin([obs_max.min(), fit_max.min(), obs_min.min(), fit_min.min()])
    hi = np.nanmax([obs_max.max(), fit_max.max(), obs_min.max(), fit_min.max()])
    pad = 0.05 * (hi - lo) if hi > lo else 5
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.6", lw=0.9,
            ls="--", zorder=1)
    scatter_panel(ax, obs_max, fit_max, color, marker="o", filled=True,
                  label="day of max" if k == 0 else None)
    scatter_panel(ax, obs_min, fit_min, color, marker="o", filled=False,
                  label="day of min" if k == 0 else None)
    r_max, _ = pearsonr(obs_max, fit_max)
    r_min, _ = pearsonr(obs_min, fit_min)
    ax.text(0.04, 0.94, f"r$_{{max}}$={r_max:.2f}  r$_{{min}}$={r_min:.2f}",
            transform=ax.transAxes, fontsize=7.8, va="top", color="0.25")
    ax.tick_params(labelsize=8)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    if k % ncol == 0:
        ax.set_ylabel("fitted day of year\n(anomaly, days)", fontsize=8.5)
    if k == 0:
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")

for k in range(len(SECTORS), len(amp_axes)):
    amp_axes[k].set_visible(False)
    phase_axes[k].set_visible(False)

fig.text(0.5, 1.0, "Fitted vs. observed amplitude", ha="center", va="top",
         fontsize=11, fontweight="bold")
fig.text(0.5, 0.50, "Fitted vs. observed timing (day of max / day of min)",
         ha="center", va="top", fontsize=11, fontweight="bold")
for ax in amp_axes:
    ax.set_xlabel("observed amplitude (10$^6$ km$^2$)", fontsize=8)
for ax in phase_axes:
    ax.set_xlabel("observed day of year (anomaly, days)", fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.subplots_adjust(hspace=0.55, wspace=0.30)
save(fig, "fig_s01_fitted_vs_observed.png", sync=False)
print("\nwrote fig_s01_fitted_vs_observed.png, t_s01_fitted_vs_observed_fit_quality.csv")