#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS04_detection_frequency.py

Supplementary figure: how often each pixel yields a valid phase date.

For every pixel, counts the fraction of years in which the detection
criterion produces a date. A pixel yields a date only when the search
window contains a persistent run in the target state preceded by a
completed persistent run in the opposite state, so pixels that do not
undergo a seasonal transition drop out year by year rather than being
removed by a mask.

Panels (a, b): static method, Freeze Start and Melt Start.
Panels (c, d): dynamic method, same.
The black contour marks the 80% threshold used to define active pixels.
"""

import os
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================== EDIT-ME ==============================
ROOT     = "/user/geog/falejandraperez/sea-ice-phase"
PHASE    = f"{ROOT}/data/SMMR_phase"
SECTORS  = f"{ROOT}/data/canonical_sectors.nc"
OUT      = f"{ROOT}/results/Ch2_Figures/FigS04_detection_frequency.png"
STATIC_SUB  = "thr15_k5"
DYNAMIC_SUB = "k5_q70"
ACTIVE_FRAC = 0.80
# =====================================================================


def load_counts(method, sub, phase):
    d = f"{PHASE}/{method}/{sub}/{phase}"
    pat = re.compile(rf"^{phase}_(\d{{4}})\.nc$")
    files = sorted(f for f in os.listdir(d) if pat.match(f))
    if not files:
        raise SystemExit(f"no files in {d}")
    arr = np.stack([xr.open_dataset(os.path.join(d, f))[phase].values
                    for f in files])
    frac = np.isfinite(arr).sum(axis=0) / arr.shape[0]
    return frac, arr.shape[0], xr.open_dataset(os.path.join(d, files[0]))


def main():
    sec_ds = xr.open_dataset(SECTORS)
    ocean = sec_ds["valid_ocean"].values.astype(bool)

    panels = []
    for method, sub in [("static", STATIC_SUB), ("dynamic", DYNAMIC_SUB)]:
        for phase in ["FS", "MS"]:
            frac, nyr, tmpl = load_counts(method, sub, phase)
            panels.append((method, phase, frac, nyr, tmpl))
            n_act = int(((frac >= ACTIVE_FRAC) & ocean).sum())
            n_zero = int(((frac == 0) & ocean).sum())
            print(f"  {method:>7s} {phase}: {nyr} yrs, "
                  f"active(>={ACTIVE_FRAC:.0%}) = {n_act:,}, never = {n_zero:,}")

    x = panels[0][4].x
    y = panels[0][4].y
    proj = ccrs.SouthPolarStereo()
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 6.6),
                             subplot_kw={"projection": proj}, dpi=300)
    axes = axes.ravel()

    cmap = plt.get_cmap("YlGnBu")
    im = None
    for ax, (method, phase, frac, nyr, _) in zip(axes, panels):
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=3)
        ax.coastlines(linewidth=0.3, zorder=4)
        ax.gridlines(linewidth=0.3, color="0.7", alpha=0.5)

        f = np.where(ocean, frac * 100.0, np.nan)
        im = ax.pcolormesh(x, y, f, transform=proj, cmap=cmap,
                           vmin=0, vmax=100, shading="auto", zorder=1)
        ax.contour(x, y, np.where(ocean, frac, np.nan),
                   levels=[ACTIVE_FRAC], colors="k",
                   linewidths=0.6, transform=proj, zorder=2)

        label = "Freeze Start" if phase == "FS" else "Melt Start"
        ax.set_title(f"{method.capitalize()}, {label}",
                     fontsize=9, fontweight="bold")

    for ax, lab in zip(axes, ["a", "b", "c", "d"]):
        ax.text(0.02, 0.98, f"({lab})", transform=ax.transAxes,
                ha="left", va="top", fontsize=12, fontweight="bold", zorder=10)

    fig.subplots_adjust(right=0.86, wspace=0.05, hspace=0.12)
    cax = fig.add_axes([0.88, 0.20, 0.02, 0.60])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel("Years with a valid date (%)", fontsize=8,
                     fontweight="bold", color="0.35")
    cb.ax.tick_params(labelsize=7)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved -> {OUT}")
    os.system(f"rclone copy {OUT} gdrive:sea-ice-phase/results/Ch2_Figures")


if __name__ == "__main__":
    main()
