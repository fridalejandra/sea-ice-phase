#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS05_dynamic_parameter_sweep.py

Supplementary figure: sensitivity of the dynamic method to its own parameters.

Each sweep run changes one setting from the baseline (Q_FS = 0.70,
Q_MS = 0.30, rate = 2% per day) and is compared against it pixel by pixel
and year by year. Panels show the cumulative distribution of the absolute
difference in detected date for (a) Freeze Start and (b) Melt Start.

Also prints median and percentile shifts, and the change in the number of
detections, which is the quantity the rate threshold actually controls.
"""

import os
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ============================== EDIT-ME ==============================
ROOT      = "/user/geog/falejandraperez/sea-ice-phase"
BASE      = f"{ROOT}/data/SMMR_phase/dynamic/k5_q70"
SWEEP     = f"{ROOT}/data/SMMR_phase_sweepv3/dynamic"
OUT       = f"{ROOT}/results/Ch2_Figures/FigS07_dynamic_parameter_sweep.png"

RUNS = [
    ("k5_q60_s02", "Percentiles 0.60 / 0.40", "#4c72b0"),
    ("k5_q80_s02", "Percentiles 0.80 / 0.20", "#dd8452"),
    ("k5_q70_s01", "Rate 1% per day",         "#55a868"),
    ("k5_q70_s04", "Rate 4% per day",         "#c44e52"),
]
XMAX = 30
# =====================================================================


def load_stack(d, phase):
    pat = re.compile(rf"^(?:.*_)?{phase}_(\d{{4}})\.nc$")
    if not os.path.isdir(d):
        return None, None
    files = sorted(f for f in os.listdir(d) if pat.match(f))
    if not files:
        return None, None
    yrs = [int(pat.match(f).group(1)) for f in files]
    arr = np.stack([xr.open_dataset(os.path.join(d, f))[phase].values
                    for f in files])
    return np.array(yrs), arr


def main():
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), dpi=300, sharey=True)

    print("=" * 64)
    print("  Dynamic parameter sweep, differences from baseline")
    print("=" * 64)

    for ax, phase, label in zip(axes, ["FS", "MS"],
                                ["Freeze Start", "Melt Start"]):
        yb, base = load_stack(f"{BASE}/{phase}", phase)
        if base is None:
            raise SystemExit(f"baseline missing at {BASE}/{phase}")
        print(f"\n  {label}: baseline {base.shape[0]} years, "
              f"{int(np.isfinite(base).sum()):,} detections")

        for sub, name, colour in RUNS:
            ys, arr = load_stack(f"{SWEEP}/{sub}/{phase}", phase)
            if arr is None:
                print(f"    [skip] {name}: no outputs at {SWEEP}/{sub}/{phase}")
                continue
            n = min(base.shape[0], arr.shape[0])
            d = np.abs(arr[:n] - base[:n])
            d = d[np.isfinite(d)]
            if d.size == 0:
                print(f"    [skip] {name}: no overlapping detections")
                continue
            med = np.median(d)
            p90, p95 = np.percentile(d, [90, 95])
            n_base = int(np.isfinite(base[:n]).sum())
            n_run  = int(np.isfinite(arr[:n]).sum())
            print(f"    {name:<26s} median={med:4.1f}  p90={p90:5.1f}  "
                  f"p95={p95:5.1f}  detections {n_run:,} "
                  f"({100*(n_run-n_base)/max(n_base,1):+.1f}%)")

            xs = np.sort(d)
            ys_cdf = np.arange(1, xs.size + 1) / xs.size
            ax.step(xs, ys_cdf, where="post", color=colour,
                    linewidth=1.3, label=name)

        ax.set_xlim(0, XMAX)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlabel("|Δ date| (days)", fontsize=8,
                      fontweight="bold", color="0.35")
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Cumulative fraction of detections", fontsize=8,
                       fontweight="bold", color="0.35")
    for ax, lab in zip(axes, ["a", "b"]):
        ax.text(-0.14, 1.06, f"({lab})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=12, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower right",
                   handlelength=1.6, labelspacing=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"\n  saved -> {OUT}")
    os.system(f"rclone copy {OUT} gdrive:sea-ice-phase/results/Ch2_Figures")
    print("=" * 64)


if __name__ == "__main__":
    main()
