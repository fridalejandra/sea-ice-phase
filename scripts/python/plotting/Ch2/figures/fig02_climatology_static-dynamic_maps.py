#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig02_climatology_static_dynamic_maps.py

Ch2 Fig. 2: climatological Freeze Start (a-c) and Melt Start (d-f),
static / dynamic / dynamic - static, 1979-2024 (1986-87 absent; 44 years).

Changes from fig04-5_climatology_static_dynamic_maps.py
- Reads the v3 climatology + anomaly files in data/anomalies/SMMR/ (the same
  source as ch2_results_numbers.py) instead of per-year folders. Valid-year
  counts come from the anomaly files' year dimension.
- One 2x3 figure with panel letters (a)-(f), built here rather than in the
  shared plot_utils.plot_phase_comparison_map (left untouched, so other
  figures are unaffected).
- Sector boundaries from canonical_sectors.nc on every panel; sector labels
  on (a) and (d).
- Active80 footprint (the pixels behind every Results statistic) outlined.
- Difference colourbar +/-DIFF_VLIM days with extend arrows.
- Month ticks along the top of the timing colourbars.
- Data transform uses the NSIDC south polar stereographic definition
  (EPSG:3412: true scale at 70S, Hughes 1980 ellipsoid). Set
  USE_NSIDC_CRS = False to reproduce the previous plain SouthPolarStereo().

Display masking is unchanged: per-method floor of >= DISPLAY_MIN_YEARS valid
years within valid_ocean (rationale in the fig04-5 docstring). Difference
panels reduce to the joint footprint via NaN propagation.
"""

import sys
import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.python.plotting.Ch2.utils.ch2_fig_utils import (  # noqa: E402
    set_mpl_defaults,
    format_fig_name,
    get_fig_path,
    save_and_upload,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ANOM_DIR = PROJECT_ROOT / "data" / "anomalies" / "SMMR"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"
REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"
SUBFOLDER = ""
FIG_NUM = 2

YEAR_START, YEAR_END = 1979, 2024
DISPLAY_MIN_YEARS = 10          # display floor for single-method panels
MIN_FRAC_ACTIVE = 0.80          # active80, outlined only
DIFF_VLIM = 20                  # days; values beyond shown by extend arrows
SHOW_ACTIVE80_OUTLINE = True
USE_NSIDC_CRS = True

SECTORS = {1: "ABS", 2: "WS", 3: "KHV", 4: "EA", 5: "RS"}
SECTOR_LABEL_LAT = -53.0        # labels sit in the open-ocean ring

TAGS = {
    "static": {"FS": "FS_static_thr15_k5", "MS": "MS_static_thr15_k5"},
    "dynamic": {"FS": "FS_dynamic_k5_q70", "MS": "MS_dynamic_k5_q70"},
}

# timing axes: FS in calendar DOY (1-based), MS in days since Aug 15
PHASE_CFG = {
    "FS": dict(label="Freeze start (day of year)", vmin=46, vmax=273,
               ref=dt.date(2001, 1, 1), offset=1,
               months=[(2001, m) for m in range(3, 10)]),
    "MS": dict(label="Melt start (days since Aug 15)", vmin=0, vmax=210,
               ref=dt.date(2001, 8, 15), offset=0,
               months=[(2001, 9), (2001, 10), (2001, 11), (2001, 12),
                       (2002, 1), (2002, 2), (2002, 3)]),
}

MAP_CRS = ccrs.SouthPolarStereo()
if USE_NSIDC_CRS:
    DATA_CRS = ccrs.Stereographic(
        central_latitude=-90, central_longitude=0, true_scale_latitude=-70,
        globe=ccrs.Globe(ellipse=None, semimajor_axis=6378273.0,
                         semiminor_axis=6356889.449),
    )
else:
    DATA_CRS = ccrs.SouthPolarStereo()


# ---------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------
def _first_var(path: Path, names: list[str]) -> xr.DataArray:
    with xr.open_dataset(path, decode_times=False) as ds:
        for n in names:
            if n in ds:
                return ds[n].load()
        raise KeyError(f"None of {names} in {path}. Vars={list(ds.data_vars)}")


def load(phase: str, method: str):
    """Climatology field + per-pixel count of valid years."""
    tag = TAGS[method][phase]
    csfx = ["_clim_dsa", "_clim"] if phase == "MS" else ["_clim"]
    asfx = ["_anom_dsa", "_anom"] if phase == "MS" else ["_anom"]
    clim = _first_var(ANOM_DIR / f"{tag}_climatology.nc", [tag + s for s in csfx])
    anom = _first_var(ANOM_DIR / f"{tag}_anomalies.nc", [tag + s for s in asfx])
    anom = anom.sel(year=slice(YEAR_START, YEAR_END))
    nvalid = np.isfinite(anom.values).sum(axis=0)
    return clim, nvalid, int(anom.sizes["year"])


# ---------------------------------------------------------------------
# DECORATION
# ---------------------------------------------------------------------
def sector_label_lons(x, y, sector_id, valid_ocean) -> dict[int, float]:
    """Circular-mean longitude of each sector's ocean pixels."""
    X, Y = np.meshgrid(x, y)
    lon = ccrs.PlateCarree().transform_points(DATA_CRS, X, Y)[..., 0]
    out = {}
    for s in SECTORS:
        sel = (sector_id == s) & valid_ocean
        if not sel.any():
            continue
        a = np.deg2rad(lon[sel])
        out[s] = float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))
    return out


def draw_sectors(ax, x, y, sector_id, valid_ocean, label_lons=None):
    sid = np.where(valid_ocean, sector_id, np.nan)
    ax.contour(x, y, sid, levels=[1.5, 2.5, 3.5, 4.5], colors="0.15",
               linewidths=0.6, transform=DATA_CRS, zorder=4)
    if label_lons:
        for s, lon in label_lons.items():
            ax.text(lon, SECTOR_LABEL_LAT, SECTORS[s], transform=ccrs.PlateCarree(),
                    ha="center", va="center", fontsize=7, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))


def add_month_ticks(cbar, cfg):
    pos, lab = [], []
    for yy, mm in cfg["months"]:
        d = dt.date(yy, mm, 1)
        p = (d - cfg["ref"]).days + cfg["offset"]
        if cfg["vmin"] <= p <= cfg["vmax"]:
            pos.append(p)
            lab.append(d.strftime("%b"))
    top = cbar.ax.secondary_xaxis("top")
    top.set_xticks(pos)
    top.set_xticklabels(lab, fontsize=7)
    top.tick_params(length=2, pad=1)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    set_mpl_defaults()

    with xr.open_dataset(SECTOR_FILE) as ds_mask:
        valid_ocean = ds_mask["valid_ocean"].astype(bool).values
        sector_id = ds_mask["sector_id"].values.astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.8),
                             subplot_kw=dict(projection=MAP_CRS),
                             constrained_layout=True)
    letters = "abcdef"
    label_lons = None

    for r, phase in enumerate(["FS", "MS"]):
        cfg = PHASE_CFG[phase]
        fields, nv = {}, {}
        for m in ["static", "dynamic"]:
            clim, nvalid, n_years = load(phase, m)
            assert clim.shape == valid_ocean.shape, "grid mismatch with sector file"
            floor = (nvalid >= DISPLAY_MIN_YEARS) & valid_ocean
            fields[m] = np.where(floor, clim.values, np.nan)
            nv[m] = nvalid
            x, y = clim["x"].values, clim["y"].values

        active = ((nv["static"] / n_years >= MIN_FRAC_ACTIVE)
                  & (nv["dynamic"] / n_years >= MIN_FRAC_ACTIVE) & valid_ocean)
        diff = fields["dynamic"] - fields["static"]

        if label_lons is None:
            label_lons = sector_label_lons(x, y, sector_id, valid_ocean)

        d_all = diff[np.isfinite(diff)]
        d_act = diff[active & np.isfinite(diff)]
        print(f"{phase}: n_years={n_years}, active80={int(active.sum())}, "
              f"displayed diff pixels={d_all.size}")
        print(f"  diff p1/p50/p99 (displayed): "
              f"{np.percentile(d_all, [1, 50, 99]).round(1)}; "
              f"|diff|>{DIFF_VLIM}: {np.mean(np.abs(d_all) > DIFF_VLIM):.1%} displayed, "
              f"{np.mean(np.abs(d_act) > DIFF_VLIM):.1%} active80")

        panels = [("Static", fields["static"], False),
                  ("Dynamic", fields["dynamic"], False),
                  ("Difference", diff, True)]

        for c, (name, data, is_diff) in enumerate(panels):
            ax = axes[r, c]
            if is_diff:
                cmap, vmin, vmax, extend = "RdBu_r", -DIFF_VLIM, DIFF_VLIM, "both"
            else:
                cmap, vmin, vmax, extend = "viridis", cfg["vmin"], cfg["vmax"], "neither"

            im = ax.pcolormesh(x, y, data, transform=DATA_CRS, cmap=cmap,
                               vmin=vmin, vmax=vmax, shading="auto", zorder=1)
            ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="0.6", zorder=2)
            ax.coastlines(linewidth=0.4, zorder=3)
            ax.gridlines(linewidth=0.3, color="0.5", alpha=0.5, linestyle="--")

            draw_sectors(ax, x, y, sector_id, valid_ocean,
                         label_lons=label_lons if c == 0 else None)
            if SHOW_ACTIVE80_OUTLINE:
                ax.contour(x, y, active.astype(float), levels=[0.5], colors="k",
                           linewidths=0.5, linestyles="--", transform=DATA_CRS, zorder=5)

            ax.set_title(f"{phase} {name}", fontsize=11, fontweight="bold")
            ax.text(0.02, 0.98, f"({letters[3 * r + c]})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=12, fontweight="bold", zorder=7)

            cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                                pad=0.04, shrink=0.8, extend=extend)
            cbar.ax.grid(False)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=8)
            if is_diff:
                cbar.set_label("Days (dynamic − static)", fontsize=9)
            else:
                cbar.set_label(cfg["label"], fontsize=9)
                add_month_ticks(cbar, cfg)

    fig_name = format_fig_name(
        num=FIG_NUM,
        short=f"climatology_FS_MS_static_vs_dynamic_{YEAR_START}-{YEAR_END}_minN{DISPLAY_MIN_YEARS}",
    )
    out_path = get_fig_path(project_root=PROJECT_ROOT, subfolder=SUBFOLDER, fig_name=fig_name)
    save_and_upload(fig, out_path, remote_root=REMOTE_ROOT, remote_subdir="")


if __name__ == "__main__":
    main()