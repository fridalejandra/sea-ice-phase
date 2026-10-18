#!/usr/bin/env python3
"""
ch2_consolidation_map.py

Consolidation failure = a pixel-year where the static method detects Freeze
Start (ice returns past 15%) but the dynamic method does not (ice does not
reach the pixel's consolidated level at the required rate).

Rate is CONDITIONAL on a static detection:
    P(dynamic fails | static detects FS)
so it is not confounded by changes in how often ice melts out in summer.

Outputs
  - table: conditional rate by sector x duration zone, 1988-2015 vs
    2016-2024, change with year-bootstrap 95% CI
  - map: pixel rates pre, post and post - pre over the baseline footprint,
    long-season zone (> 8.5 mo) outlined, sector boundaries drawn
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.python.plotting.Ch2.utils.ch2_fig_utils import (  # noqa: E402
    set_mpl_defaults, format_fig_name, get_fig_path, save_and_upload)

ANOM = PROJECT_ROOT / "data" / "anomalies" / "SMMR"
SECT = PROJECT_ROOT / "data" / "canonical_sectors.nc"
OUT = PROJECT_ROOT / "results"
REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"

SECTORS = {1: "ABS", 2: "WS", 3: "KHV", 4: "EA", 5: "RAS"}
TAGS = {("FS", "static"): "FS_static_thr15_k5", ("FS", "dynamic"): "FS_dynamic_k5_q70",
        ("MS", "static"): "MS_static_thr15_k5", ("MS", "dynamic"): "MS_dynamic_k5_q70"}
PRE, POST = (1988, 2015), (2016, 2024)
BASE_FRAC, MONTH, AUG15_DOY = 0.5, 30.4, 227
LB, UB = 4.0, 8.5
MIN_DET = {"pre": 10, "post": 3}   # min static detections for a pixel rate to be mapped
NBOOT = 2000
rng = np.random.default_rng(42)

DATA_CRS = ccrs.Stereographic(
    central_latitude=-90, central_longitude=0, true_scale_latitude=-70,
    globe=ccrs.Globe(ellipse=None, semimajor_axis=6378273.0, semiminor_axis=6356889.449))
MAP_CRS = ccrs.SouthPolarStereo()


def _var(path, names):
    with xr.open_dataset(path, decode_times=False) as ds:
        for n in names:
            if n in ds:
                return ds[n].load()
    raise KeyError(f"{names} not in {path}")


def load_dates(phase, method):
    t = TAGS[(phase, method)]
    cs = [f"{t}_clim_dsa", f"{t}_clim"] if phase == "MS" else [f"{t}_clim"]
    an = [f"{t}_anom_dsa", f"{t}_anom"] if phase == "MS" else [f"{t}_anom"]
    clim = _var(ANOM / f"{t}_climatology.nc", cs)
    anom = _var(ANOM / f"{t}_anomalies.nc", an)
    return clim.values[None] + anom.values, anom["year"].values, clim["x"].values, clim["y"].values


def boot_change(pre, post):
    pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
    if len(pre) < 3 or len(post) < 3:
        return (np.nan,) * 5
    bs = (rng.choice(post, (NBOOT, len(post))).mean(1)
          - rng.choice(pre, (NBOOT, len(pre))).mean(1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pre.mean(), post.mean(), post.mean() - pre.mean(), lo, hi


def main():
    with xr.open_dataset(SECT) as m:
        ocean = m["valid_ocean"].astype(bool).values
        sid = m["sector_id"].values

    date = {}
    for key in TAGS:
        date[key], years, x, y = load_dates(*key)
    per = {"pre": (years >= PRE[0]) & (years <= PRE[1]),
           "post": (years >= POST[0]) & (years <= POST[1])}
    valid = {k: np.isfinite(v) for k, v in date.items()}
    foot = ocean & np.all([valid[k][per["pre"]].mean(0) >= BASE_FRAC for k in valid], axis=0)

    dur = np.nanmean(((date[("MS", "static")] + AUG15_DOY) - date[("FS", "static")])[per["pre"]], axis=0) / MONTH
    zones = {"short": foot & (dur < LB), "mid": foot & (dur >= LB) & (dur <= UB), "long": foot & (dur > UB)}

    det = valid[("FS", "static")]
    fail = det & ~valid[("FS", "dynamic")]

    rows = []
    for s in list(SECTORS) + [0]:
        lab = SECTORS.get(s, "ALL")
        for zname, zmask in zones.items():
            sel = zmask & ((sid == s) if s else True)
            if not sel.any():
                continue
            nd = det[:, sel].sum(1).astype(float)
            nf = fail[:, sel].sum(1).astype(float)
            rate_y = np.where(nd > 0, nf / np.maximum(nd, 1), np.nan)
            b = boot_change(rate_y[per["pre"]], rate_y[per["post"]])
            star = "*" if (b[3] > 0 or b[4] < 0) else " "
            rows.append(dict(sector=lab, zone=zname, n_pix=int(sel.sum()),
                             pre_pct=round(100 * b[0], 1), post_pct=round(100 * b[1], 1),
                             change_pp=f"{100*b[2]:+.1f} [{100*b[3]:+.1f},{100*b[4]:+.1f}]{star}"))
    df = pd.DataFrame(rows)
    print("\n=== Conditional consolidation failure: P(dynamic misses FS | static detects FS), % ===")
    print(df.to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ch2_consolidation_failure_conditional.csv", index=False)

    rate = {}
    for p, ip in per.items():
        nd = det[ip].sum(0)
        rate[p] = np.where(foot & (nd >= MIN_DET[p]), fail[ip].sum(0) / np.maximum(nd, 1), np.nan)
    delta = rate["post"] - rate["pre"]

    set_mpl_defaults()
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), subplot_kw=dict(projection=MAP_CRS),
                             constrained_layout=True)
    panels = [("1988–2015", rate["pre"], "YlOrRd", 0, 0.4, "neither", "Failure rate"),
              ("2016–2024", rate["post"], "YlOrRd", 0, 0.4, "max", "Failure rate"),
              ("Change (post − pre)", delta, "RdBu_r", -0.1, 0.1, "both", "Δ failure rate")]
    long_mask = np.where(foot, (dur > UB).astype(float), np.nan)
    sid_plot = np.where(ocean, sid, np.nan)
    from scipy.ndimage import distance_transform_edt
    _land = xr.open_dataset(PROJECT_ROOT / "data" / "landmask_sic.nc")["land"].values.astype(bool)
    coast100 = np.where(foot, (distance_transform_edt(~_land) * 25.0 <= 100).astype(float), np.nan)
    for ax, (title, data, cmap, vmin, vmax, ext, clab), letter in zip(axes, panels, "abc"):
        im = ax.pcolormesh(x, y, data, transform=DATA_CRS, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto", zorder=1)
        ax.set_extent([-180, 180, -90, -52], ccrs.PlateCarree())
        ax.contour(x, y, sid_plot, levels=[1.5, 2.5, 3.5, 4.5], colors="0.15",
                   linewidths=0.6, transform=DATA_CRS, zorder=3)
        ax.contour(x, y, long_mask, levels=[0.5], colors="k", linewidths=0.8,
                   linestyles="--", transform=DATA_CRS, zorder=4)
        ax.contour(x, y, coast100, levels=[0.5], colors="0.3", linewidths=0.6,
                   linestyles=":", transform=DATA_CRS, zorder=4)
        ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="0.6", zorder=4.5)
        ax.coastlines(linewidth=0.4, zorder=4.6)
        ax.gridlines(linewidth=0.3, color="0.5", alpha=0.5, linestyle="--")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold", zorder=7)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.8, extend=ext)
        cb.ax.grid(False)
        cb.outline.set_visible(False)
        cb.set_label(clab, fontsize=9)
    fig.suptitle("Consolidation failure: static detects Freeze Start, dynamic does not "
                 "(dashed: long-season zone, > 8.5 mo; dotted: 100 km from coast)", fontsize=10)

    fig_name = format_fig_name(num=99, short="consolidation_failure_map_pre_post_v2")
    out_path = get_fig_path(project_root=PROJECT_ROOT, subfolder="", fig_name=fig_name)
    save_and_upload(fig, out_path, remote_root=REMOTE_ROOT, remote_subdir="")


if __name__ == "__main__":
    main()
