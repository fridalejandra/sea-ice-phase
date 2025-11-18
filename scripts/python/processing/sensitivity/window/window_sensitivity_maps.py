#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
window_sensitivity_maps.py

Outputs under OUT_DIR (local, for your records) and mirrored to Google Drive:
  - maps/:       mean/std/q95 per-pixel maps for {MS,FS} × {3v5,7v5}
                  + facet maps (FS/MS × 3v5/7v5) for mean/std/q95
  - cdf/:        CDF(|Δ|) per metric (active pixels only, circumpolar)
  - distributions/: signed-Δ histograms per metric (active pixels only)
  - trends/:     annual mean(|Δ|) trend plots (circumpolar + sectors)
  - joint/:      hexbin joint histogram of per-pixel mean |Δ₃₋₅| vs |Δ₇₋₅|
  - summary_stats.csv (sectoral + circumpolar statistics of maps)
  - trend_timeseries.csv (annual mean(|Δ|) by region)
  - summary_meta.json (year coverage)
"""

from pathlib import Path
import os, glob, re, json, subprocess, shlex

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
from matplotlib.colors import LogNorm

# =========================
# CONFIG — EDIT THIS BLOCK
# =========================

# Version/provenance tag (used in local file names only; Drive folder is fixed below)
VERSION_TAG  = "static_v2_slopeH"
SENSOR       = "SMMR"   # "SMMR" or "AMSRE"
THRESH_PCT   = 15       # threshold percentage

# Input: where FS/MS NetCDFs (k=3/5/7) live
INPUT_ROOT   = f"/user/geog/falejandraperez/sea-ice-phase/results/{SENSOR}_phase"

# Local outputs (kept for your archive); Drive upload path is separate
OUT_ROOT     = f"/user/geog/falejandraperez/sea-ice-phase/results/{VERSION_TAG}/window_sensitivity/{SENSOR}_thr{THRESH_PCT}"

# Canonical sectors/masks
CANONICAL    = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"

CFG = {
    "SENSOR": SENSOR,
    "THRESH_PCT": THRESH_PCT,
    "INPUT_ROOT": INPUT_ROOT,
    "CANONICAL":  CANONICAL,

    "OUT_DIR": OUT_ROOT,
    "PERIOD": 366,

    # colorbar ranges
    "MEAN_VMAX": 10,
    "STD_VMAX": 8,
    "Q95_VMAX": 10,

    # "active" pixels must have valid phase DOY in ≥ this fraction of years (baseline k=5)
    "ACTIVE_MIN_FRAC": 0.30,

    "DPI": 180,

    # subdirectories (local)
    "SUBDIRS": {"cdf": "cdf", "dist": "distributions"},

    # rclone upload → FIXED Drive destination
    "RCLONE": {
        "enabled": True,
        "remote": "gdrive",  # EXACT name from `rclone listremotes` (no colon)
        "dst_dir": "sea-ice-phase/results/Ch2_Figures/slope_window_sensitivity",
        "extra_flags": ["--transfers=8", "--checkers=8", "--fast-list"],
        "dry_run": False
    }
}

# Sector ID → name (must match canonical_sectors.nc)
SECTOR_ID_TO_NAME = {
    1: "Amundsen–Bellingshausen",
    2: "Weddell",
    3: "King Haakon VII",
    4: "East Antarctic",
    5: "Ross–Amundsen",
}

# Meridians (deg East) used to draw sector lines
SECTOR_BOUNDARIES_E = [71, 162, 250, 290, 346]

# ======================
# UTILITIES
# ======================
year_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    m = year_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def rclone_copy(local_path, cfg=CFG):
    """Upload a file to Google Drive under .../slope_window_sensitivity/<subfolder>,
    where <subfolder> is the local parent directory name (maps, cdf, distributions, trends, joint)."""
    rc = cfg["RCLONE"]
    if not rc.get("enabled", False):
        return
    remote   = rc["remote"]                      # e.g., "gdrive"
    dst_root = rc["dst_dir"].lstrip("/")         # "sea-ice-phase/results/Ch2_Figures/slope_window_sensitivity"
    sub = Path(local_path).parent.name           # subfolder name
    dst = f"{remote}:{dst_root}/{sub}"

    cmd = ["rclone", "copy", str(local_path), dst, "-vv"] + rc.get("extra_flags", [])
    if rc.get("dry_run"):
        cmd.insert(1, "--dry-run")

    print("[rclone cmd]", " ".join(shlex.quote(c) for c in cmd))
    res = subprocess.run(cmd, text=True, capture_output=True)
    print("[rclone stdout]\n", res.stdout[-1200:])
    print("[rclone stderr]\n", res.stderr[-1200:])
    if res.returncode != 0:
        raise RuntimeError(f"rclone failed (code {res.returncode}). See logs above.")

def open_phase_dict(metric, kdays, cfg=CFG):
    """Return {year: DataArray} for metric ('MS'/'FS') and window (3/5/7) with var name == metric."""
    subdir = f"{metric}_thr{cfg['THRESH_PCT']}_k{kdays}"
    folder = os.path.join(cfg["INPUT_ROOT"], subdir)
    files = sorted(glob.glob(os.path.join(folder, f"{metric}_*.nc")))
    out = {}
    for f in files:
        yr = parse_year(f)
        if yr is None:
            continue
        try:
            with xr.open_dataset(f) as ds:
                da = ds[metric].load()
            out[yr] = da
        except Exception as e:
            print(f"Skip {f}: {e}")
    if not out:
        raise FileNotFoundError(f"No phase files in {folder}")
    return out

def align_years(dicts):
    common = set.intersection(*[set(d.keys()) for d in dicts])
    years = sorted(common)
    if not years:
        raise ValueError("No overlapping years across inputs.")
    return years

def stack_years(d, years, name):
    arrs = [d[y].expand_dims(year=[y]) for y in years]
    out = xr.concat(arrs, dim="year").rename(name)
    return out

def wrapped_abs_diff(a, b, period):
    """Absolute wrapped difference |Δ| between two day-of-year arrays."""
    return np.abs(((a - b + period//2) % period) - (period//2))

# ======================
# ACTIVE PIXEL MASK
# ======================
def compute_active_mask(metric, cano, cfg=CFG):
    """
    Active = pixels that have a valid DOY for baseline (k=5) in >= ACTIVE_MIN_FRAC of years.
    Returns (active_mask bool DataArray (y,x), years list).
    """
    d5 = open_phase_dict(metric, 5, cfg)
    years = sorted(d5.keys())
    A5 = stack_years(d5, years, f"{metric}_k5")  # (year,y,x)
    valid_count = A5.notnull().sum(dim="year")
    min_years = max(1, int(np.floor(cfg["ACTIVE_MIN_FRAC"] * len(years))))
    active = valid_count >= min_years
    active = active & cano["valid_ocean"].astype(bool)
    return active.astype(bool), years

# ======================
# CORE MAP COMPUTE
# ======================
def compute_maps_for_metric(metric, cano, cfg=CFG, active_mask=None):
    """
    Returns per-pixel maps for the metric (MS/FS):
      { "3v5": {"mean": DA, "std": DA, "q95": DA},
        "7v5": {"mean": DA, "std": DA, "q95": DA},
        "years": [int,...] }
    """
    d3 = open_phase_dict(metric, 3, cfg)
    d5 = open_phase_dict(metric, 5, cfg)
    d7 = open_phase_dict(metric, 7, cfg)
    years = align_years([d3, d5, d7])

    A3 = stack_years(d3, years, f"{metric}_k3")  # (year,y,x)
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")

    valid = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7))
    if active_mask is not None:
        valid = valid & active_mask.astype(bool)
    A3 = A3.where(valid); A5 = A5.where(valid); A7 = A7.where(valid)

    period = cfg["PERIOD"]
    D35 = xr.apply_ufunc(wrapped_abs_diff, A3, A5, period, dask="allowed")
    D75 = xr.apply_ufunc(wrapped_abs_diff, A7, A5, period, dask="allowed")

    mean35 = D35.mean(dim="year", skipna=True)
    std35  = D35.std(dim="year",  skipna=True)
    q95_35 = D35.quantile(0.95,   dim="year", skipna=True)

    mean75 = D75.mean(dim="year", skipna=True)
    std75  = D75.std(dim="year",  skipna=True)
    q95_75 = D75.quantile(0.95,   dim="year", skipna=True)

    vo = cano["valid_ocean"].astype(bool)
    for da in [mean35, std35, q95_35, mean75, std75, q95_75]:
        da.values[~vo.values] = np.nan

    return {
        "3v5": {"mean": mean35, "std": std35, "q95": q95_35},
        "7v5": {"mean": mean75, "std": std75, "q95": q95_75},
        "years": years
    }

# ======================
# FLAT DELTAS FOR CDF & DISTRIBUTION
# ======================
def compute_flat_deltas(metric, active_mask, cfg=CFG):
    """
    Returns flattened signed deltas (Δ₃₋₅, Δ₇₋₅) and absolute deltas for active pixels across all years.
    """
    d3 = open_phase_dict(metric, 3, cfg)
    d5 = open_phase_dict(metric, 5, cfg)
    d7 = open_phase_dict(metric, 7, cfg)
    years = align_years([d3, d5, d7])

    A3 = stack_years(d3, years, f"{metric}_k3")
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")

    valid = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7)) & active_mask.astype(bool)

    # wrapped signed deltas (−183..+183)
    period = cfg["PERIOD"]
    D35s = xr.apply_ufunc(lambda a,b: ((a-b + period//2) % period) - (period//2), A3, A5, dask="allowed")
    D75s = xr.apply_ufunc(lambda a,b: ((a-b + period//2) % period) - (period//2), A7, A5, dask="allowed")

    d35 = D35s.where(valid).values.ravel()
    d75 = D75s.where(valid).values.ravel()
    m = np.isfinite(d35) & np.isfinite(d75)
    d35 = d35[m]; d75 = d75[m]
    return d35, d75, np.abs(d35), np.abs(d75)

# ======================
# CARTOPY PLOTTING HELPERS
# ======================
def draw_sector_meridians(ax):
    lats = np.linspace(-90, -45, 256)
    for bE in SECTOR_BOUNDARIES_E:
        lon = bE if bE <= 180 else bE - 360
        lons = np.full_like(lats, lon, dtype=float)
        ax.plot(lons, lats, transform=ccrs.Geodetic(),
                color="k", linewidth=0.6, alpha=0.8, zorder=6)

def _round_polar_axes(ax):
    theta = np.linspace(0, 2*np.pi, 200)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

def plot_map_cartopy(da, cano, title, out_png, vmin, vmax, cmap="viridis", white_under=True, cfg=CFG):
    """Single-panel map. Kept for completeness; you can ignore these PNGs if you only care
    about the new facet maps."""
    proj = ccrs.SouthPolarStereo()
    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw={"projection": proj})

    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="0.2", linewidth=0.4, zorder=2)
    ax.coastlines(resolution="110m", color="0.2", linewidth=0.5, zorder=3)

    lon = cano["lon"]; lat = cano["lat"]
    cmap_obj = plt.get_cmap(cmap).copy()
    if white_under:
        cmap_obj.set_under("white")
    eps = 1e-6 if vmin == 0 else 0.0

    pc = ax.pcolormesh(lon, lat, da, transform=ccrs.PlateCarree(),
                       vmin=vmin + eps, vmax=vmax, cmap=cmap_obj, zorder=4)

    cb = plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.02, shrink=0.85)
    cb.set_label(title.split("•")[0].strip(), fontsize=9)

    draw_sector_meridians(ax)
    ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
    _round_polar_axes(ax)

    # no figure title (you'll add titles in the manuscript)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=cfg["DPI"])
    plt.close()
    print(f"[OK] wrote {out_png}")
    rclone_copy(out_png, cfg)

# ======================
# FACET MAPS (FS/MS × 3v5/7v5)
# ======================
def facet_maps_stat(maps_by_metric, cano, stat_key, out_png, cfg=CFG):
    """
    Create 2×2 facet map for a given stat_key in {"mean","std","q95"}.

    rows:  FS (row 0), MS (row 1)
    cols:  3v5 (col 0), 7v5 (col 1)
    """
    metrics = ["FS", "MS"]
    difftypes = ["3v5", "7v5"]

    if stat_key == "mean":
        vmax = cfg["MEAN_VMAX"]; cbar_label = "mean |Δ| (days)"
        cmap = "viridis"
    elif stat_key == "std":
        vmax = cfg["STD_VMAX"]; cbar_label = "std |Δ| (days)"
        cmap = "magma"
    elif stat_key == "q95":
        vmax = cfg["Q95_VMAX"]; cbar_label = "95th percentile |Δ| (days)"
        cmap = "plasma"
    else:
        raise ValueError("stat_key must be one of 'mean','std','q95'")

    proj = ccrs.SouthPolarStereo()
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(7.0, 7.0),
        subplot_kw={"projection": proj}
    )

    lon = cano["lon"]; lat = cano["lat"]
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_under("white")
    vmin = 0.0
    eps = 1e-6 if vmin == 0 else 0.0

    mlist = []
    for i, metric in enumerate(metrics):
        for j, difftype in enumerate(difftypes):
            ax = axes[i, j]
            da = maps_by_metric[metric][difftype][stat_key]

            ax.add_feature(cfeature.LAND, facecolor="lightgray",
                           edgecolor="0.2", linewidth=0.3, zorder=2)
            ax.coastlines(resolution="110m", color="0.2", linewidth=0.4, zorder=3)

            m = ax.pcolormesh(
                lon, lat, da,
                transform=ccrs.PlateCarree(),
                vmin=vmin + eps, vmax=vmax,
                cmap=cmap_obj, zorder=4
            )
            mlist.append(m)

            draw_sector_meridians(ax)
            ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
            _round_polar_axes(ax)

            # small panel label (metric + diff), not a big title
            ax.text(
                0.02, 0.97,
                f"{metric} • {difftype}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8,
                color="0.1",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5)
            )

    # shared colorbar along bottom
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    cb = fig.colorbar(mlist[0], cax=cax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=cfg["DPI"])
    plt.close()
    print(f"[OK] wrote {out_png}")
    rclone_copy(out_png, cfg)

# ======================
# MAP STATS → CSV
# ======================
def summarize_to_rows(da, cano, metric, difftype, statname):
    rows = []
    sid = cano["sector_id"].astype(np.int16)
    vo  = cano["valid_ocean"].astype(bool)

    # circumpolar
    vals = da.where(vo).values
    vals = vals[np.isfinite(vals)]
    if vals.size:
        rows.append({"Metric": metric, "DiffType": difftype, "Sector": "Circumpolar",
                     "Stat": statname,
                     "MedianDays": float(np.nanmedian(vals)),
                     "P95Days": float(np.nanpercentile(vals, 95)),
                     "MeanDays": float(np.nanmean(vals)),
                     "Npixels": int(vals.size)})

    # per sector
    for k, name in SECTOR_ID_TO_NAME.items():
        v = da.where((sid == k) & vo).values
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        rows.append({"Metric": metric, "DiffType": difftype, "Sector": name,
                     "Stat": statname,
                     "MedianDays": float(np.nanmedian(v)),
                     "P95Days": float(np.nanpercentile(v, 95)),
                     "MeanDays": float(np.nanmean(v)),
                     "Npixels": int(v.size)})
    return rows

# ======================
# CDF & DISTRIBUTION PLOTS (CIRCUMPOLAR)
# ======================
def plot_cdf_pairs(metric, abs35, abs75, out_png, cfg=CFG, max_x=30):
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("paper")

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    v35 = abs35[abs35 <= max_x]; v75 = abs75[abs75 <= max_x]

    sns.ecdfplot(v35, ax=ax, label="3 vs 5-day window", lw=1.8)
    sns.ecdfplot(v75, ax=ax, label="7 vs 5-day window", lw=1.8)

    ax.set_xlim(0, max_x); ax.set_ylim(0, 1)
    ax.set_xlabel("Absolute timing difference (days)", fontsize=10)
    ax.set_ylabel("Cumulative fraction of pixels", fontsize=10)
    ax.tick_params(labelsize=9)

    # legend below/right to avoid covering curves
    leg = ax.legend(
        title="Window comparison",
        fontsize=8,
        title_fontsize=9,
        loc="lower right",
        frameon=True
    )
    leg.get_frame().set_alpha(0.9)

    # no title – you'll add context in the manuscript
    plt.tight_layout()
    plt.savefig(out_png, dpi=cfg["DPI"])
    plt.close()
    rclone_copy(out_png, cfg)
    print(f"[OK] wrote {out_png}")

def plot_distribution_pairs(metric, d35, d75, out_png, cfg=CFG):
    bins = np.arange(-30, 31, 1)
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 6.0), sharex=True, sharey=True)
    panels = [(d35, "Δ(3–5)"), (d75, "Δ(7–5)"), (None, ""), (None, "")]
    for (data, label), ax in zip(panels, axs.ravel()):
        if data is None:
            ax.axis("off"); continue
        data = np.asarray(data)
        data = data[np.isfinite(data)]
        if data.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color="0.5", transform=ax.transAxes, fontsize=9)
            ax.grid(False); continue
        ax.hist(data, bins=bins, density=True, alpha=0.9)
        med = float(np.nanmedian(data))
        q25, q75 = np.nanpercentile(data, [25, 75])
        pct5 = 100.0 * np.mean(np.abs(data) > 5)
        ax.axvline(med, color="k", lw=1)
        ax.axvline(q25, color="k", lw=1, ls=":")
        ax.axvline(q75, color="k", lw=1, ls=":")
        ax.set_title(
            f"{metric} {label}\nmedian {med:+.1f} d | IQR {q25:+.1f}–{q75:+.1f} d | %|Δ|>5: {pct5:.1f}%",
            fontsize=9
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    for ax in axs[-1,:]:
        if ax.has_data():
            ax.set_xlabel("Δ timing (days)", fontsize=10)
    fig.supylabel("Density", fontsize=10)
    # no suptitle
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(out_png, dpi=cfg["DPI"])
    plt.close()
    rclone_copy(out_png, cfg)
    print(f"[OK] wrote {out_png}")

# ======================
# TEMPORAL TRENDS & JOINT HISTOGRAM
# ======================
def compute_year_series_for_metric(metric, cano, cfg=CFG, active_mask=None):
    """Returns DataFrame: year, Metric, DiffType, Region, MeanAbsDiff (active pixels only if mask supplied)."""
    d3 = open_phase_dict(metric, 3, cfg)
    d5 = open_phase_dict(metric, 5, cfg)
    d7 = open_phase_dict(metric, 7, cfg)
    years = align_years([d3, d5, d7])

    A3 = stack_years(d3, years, f"{metric}_k3")
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")

    valid = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7))
    if active_mask is not None:
        valid = valid & active_mask.astype(bool)

    period = cfg["PERIOD"]
    D35 = xr.apply_ufunc(wrapped_abs_diff, A3, A5, period, dask="allowed")
    D75 = xr.apply_ufunc(wrapped_abs_diff, A7, A5, period, dask="allowed")
    D35 = D35.where(valid); D75 = D75.where(valid)

    vo  = cano["valid_ocean"].astype(bool)
    sid = cano["sector_id"].astype(np.int16)

    rows = []
    for yi, yr in enumerate(D35["year"].values):
        # circumpolar
        v35 = D35.isel(year=yi).where(vo).values
        v75 = D75.isel(year=yi).where(vo).values
        v35 = v35[np.isfinite(v35)]; v75 = v75[np.isfinite(v75)]
        if v35.size:
            rows.append({"year": int(yr), "Metric": metric, "DiffType": "3v5",
                         "Region": "Circumpolar", "MeanAbsDiff": float(np.nanmean(v35))})
        if v75.size:
            rows.append({"year": int(yr), "Metric": metric, "DiffType": "7v5",
                         "Region": "Circumpolar", "MeanAbsDiff": float(np.nanmean(v75))})

        # per sector
        for k, name in SECTOR_ID_TO_NAME.items():
            mask_k = ((sid == k) & vo).values
            a35 = D35.isel(year=yi).values[mask_k]
            a75 = D75.isel(year=yi).values[mask_k]
            a35 = a35[np.isfinite(a35)]; a75 = a75[np.isfinite(a75)]
            if a35.size:
                rows.append({"year": int(yr), "Metric": metric, "DiffType": "3v5",
                             "Region": name, "MeanAbsDiff": float(np.nanmean(a35))})
            if a75.size:
                rows.append({"year": int(yr), "Metric": metric, "DiffType": "7v5",
                             "Region": name, "MeanAbsDiff": float(np.nanmean(a75))})
    return pd.DataFrame(rows)

def _set_trend_axes_ticks(ax, years):
    # a bit more granular ticks, but not crazy
    if len(years) > 25:
        step = 5
    else:
        step = 2
    ax.set_xticks(np.arange(years[0], years[-1] + 1, step))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=9)

def plot_trend_lines(df, out_png, title, cfg=CFG, per_sector=False):
    years = sorted(df["year"].unique())

    if not per_sector:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        for difftype, style, color in [("3v5", "-", "C0"), ("7v5", "--", "C1")]:
            sub = df[(df["Region"]=="Circumpolar") & (df["DiffType"]==difftype)]
            sub = sub.groupby("year")["MeanAbsDiff"].mean().reindex(years)
            ax.plot(years, sub.values, style, lw=2, label=difftype, color=color)

        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("mean(|Δ|) (days)", fontsize=10)
        _set_trend_axes_ticks(ax, years)
        ax.grid(True, alpha=0.3)

        # Legend above plot so it doesn't cover lines
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            title="Diff",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            fontsize=9,
            title_fontsize=10
        )

        # no title on figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_png, dpi=cfg["DPI"])
        plt.close()
        rclone_copy(out_png, cfg)
        print(f"[OK] wrote {out_png}")
        return

    # per-sector faceted trends
    sectors = list(SECTOR_ID_TO_NAME.values())
    n = len(sectors); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols*3.0, nrows*2.4),
        sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    for i, name in enumerate(sectors):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        for difftype, style, color in [("3v5", "-", "C0"), ("7v5", "--", "C1")]:
            sub = df[(df["Region"]==name) & (df["DiffType"]==difftype)]
            sub = sub.groupby("year")["MeanAbsDiff"].mean().reindex(years)
            ax.plot(years, sub.values, style, lw=1.6, label=difftype, color=color)
        ax.set_title(name, fontsize=8)
        ax.grid(True, alpha=0.3)
        _set_trend_axes_ticks(ax, years)

    for j in range(n, nrows*ncols):
        r, c = divmod(j, ncols); axes[r, c].set_visible(False)

    fig.supxlabel("Year", fontsize=10)
    fig.supylabel("mean(|Δ|) (days)", fontsize=10)

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=9,
        title="Diff",
        title_fontsize=10
    )

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(out_png, dpi=cfg["DPI"])
    plt.close()
    rclone_copy(out_png, cfg)
    print(f"[OK] wrote {out_png}")

def plot_joint_hist(da_mean35, da_mean75, cano, metric, out_png, cfg=CFG):
    vo = cano["valid_ocean"].astype(bool).values
    x = da_mean35.where(vo).values.ravel()
    y = da_mean75.where(vo).values.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size:
        slope, intercept = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0,1]
    else:
        slope = intercept = r = np.nan

    fig, ax = plt.subplots(figsize=(5.1, 5.0))
    hb = ax.hexbin(x, y, gridsize=60, norm=LogNorm(), mincnt=1)
    cb = plt.colorbar(hb, ax=ax); cb.set_label("pixel count (log)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    lim = max(6, float(np.nanmax([x.max() if x.size else 0, y.max() if y.size else 0])))
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="1:1")
    xx = np.linspace(0, lim, 100)
    ax.plot(xx, slope*xx + intercept, color="C1", lw=1.6, label=f"fit: y={slope:.2f}x+{intercept:.2f}")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("mean(|Δ₃₋₅|) (days)", fontsize=10)
    ax.set_ylabel("mean(|Δ₇₋₅|) (days)", fontsize=10)
    ax.tick_params(labelsize=9)
    # no title
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=cfg["DPI"])
    plt.close()
    rclone_copy(out_png, cfg)
    print(f"[OK] wrote {out_png}")

# ======================
# MAIN
# ======================
def main(cfg=CFG):
    out_dir = Path(cfg["OUT_DIR"])
    maps_dir = out_dir / "maps"
    cdf_dir  = out_dir / cfg["SUBDIRS"]["cdf"]
    dist_dir = out_dir / cfg["SUBDIRS"]["dist"]
    trends_dir = out_dir / "trends"
    joint_dir  = out_dir / "joint"
    for d in [out_dir, maps_dir, cdf_dir, dist_dir, trends_dir, joint_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cano = xr.open_dataset(cfg["CANONICAL"]).load()  # lat, lon, lonE, valid_ocean, sector_id, area_m2

    # Precompute active masks per metric (based on k=5 baseline)
    active_masks = {}
    for metric in ["MS", "FS"]:
        amask, _ = compute_active_mask(metric, cano, cfg)
        active_masks[metric] = amask

    # Collect rows for summary CSV
    all_rows = []

    # MAPS + STATS + CDF/DISTRIBUTION
    maps_by_metric = {}

    for metric in ["MS", "FS"]:
        maps = compute_maps_for_metric(metric, cano, cfg, active_mask=active_masks[metric])
        maps_by_metric[metric] = maps

        # single-panel maps (you can ignore these if you just want the facets)
        for difftype, group in [("3v5", maps["3v5"]), ("7v5", maps["7v5"])]:
            plot_map_cartopy(
                group["mean"], cano,
                title=f"mean(|Δ|) days • {metric} • {difftype}",
                out_png=maps_dir / f"{VERSION_TAG}_mean_absdiff_{metric}_{difftype}.png",
                vmin=0, vmax=cfg["MEAN_VMAX"], cmap="viridis", white_under=True, cfg=cfg
            )

            plot_map_cartopy(
                group["std"], cano,
                title=f"std(|Δ|) days • {metric} • {difftype}",
                out_png=maps_dir / f"{VERSION_TAG}_std_absdiff_{metric}_{difftype}.png",
                vmin=0, vmax=cfg["STD_VMAX"], cmap="magma", white_under=True, cfg=cfg
            )

            plot_map_cartopy(
                group["q95"], cano,
                title=f"q95(|Δ|) days • {metric} • {difftype}",
                out_png=maps_dir / f"{VERSION_TAG}_q95_absdiff_{metric}_{difftype}.png",
                vmin=0, vmax=cfg["Q95_VMAX"], cmap="plasma", white_under=True, cfg=cfg
            )

            # stats
            all_rows += summarize_to_rows(group["mean"], cano, metric, difftype, "mean_absdiff")
            all_rows += summarize_to_rows(group["std"],  cano, metric, difftype, "std_absdiff")
            all_rows += summarize_to_rows(group["q95"],  cano, metric, difftype, "q95_absdiff")

        # CDFs + distributions on active pixels (circumpolar)
        d35, d75, a35, a75 = compute_flat_deltas(metric, active_masks[metric], cfg)
        plot_cdf_pairs(
            metric, a35, a75,
            out_png=cdf_dir / f"{VERSION_TAG}_cdf_absdiff_{metric}.png",
            cfg=cfg
        )
        plot_distribution_pairs(
            metric, d35, d75,
            out_png=dist_dir / f"{VERSION_TAG}_dist_signed_delta_{metric}.png",
            cfg=cfg
        )

    # Faceted FS/MS maps: 2×2 panels (row = metric, col = diff type)
    facet_maps_stat(
        maps_by_metric,
        cano,
        "mean",
        out_png=maps_dir / f"{VERSION_TAG}_facet_mean_absdiff_FS_MS.png",
        cfg=cfg,
    )
    facet_maps_stat(
        maps_by_metric,
        cano,
        "std",
        out_png=maps_dir / f"{VERSION_TAG}_facet_std_absdiff_FS_MS.png",
        cfg=cfg,
    )
    facet_maps_stat(
        maps_by_metric,
        cano,
        "q95",
        out_png=maps_dir / f"{VERSION_TAG}_facet_q95_absdiff_FS_MS.png",
        cfg=cfg,
    )

    # Write map summary CSV
    df = pd.DataFrame(
        all_rows,
        columns=["Metric","DiffType","Sector","Stat","MedianDays","P95Days","MeanDays","Npixels"]
    )
    csv_path = out_dir / f"{VERSION_TAG}_summary_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] wrote {csv_path}")

    # Upload CSV to Drive root (no subfolder inference here)
    try:
        rc = CFG["RCLONE"]
        dst = f"{rc['remote']}:{rc['dst_dir']}"
        subprocess.run(
            ["rclone", "copy", str(csv_path), dst, "-vv"] + rc.get("extra_flags", []),
            check=False
        )
    except Exception as e:
        print(f"[rclone] CSV upload skipped: {e}")

    # TEMPORAL TRENDS + JOINT HISTOGRAMS (active pixels)
    trend_rows = []
    for metric in ["MS", "FS"]:
        df_year = compute_year_series_for_metric(
            metric, cano, cfg, active_mask=active_masks[metric]
        )
        trend_rows.append(df_year)

        # circumpolar trend
        df_circ = df_year[df_year["Region"]=="Circumpolar"]
        plot_trend_lines(
            df_circ,
            out_png=trends_dir / f"{VERSION_TAG}_trend_circ_{metric}.png",
            title=f"Window sensitivity trends • {cfg['SENSOR']} • {metric}",
            cfg=cfg,
            per_sector=False
        )

        # per-sector trends
        plot_trend_lines(
            df_year,
            out_png=trends_dir / f"{VERSION_TAG}_trend_sectors_{metric}.png",
            title=f"Window sensitivity trends • {cfg['SENSOR']} • {metric}",
            cfg=cfg,
            per_sector=True
        )

        # joint histogram (per-pixel mean |Δ|) using maps for this metric
        maps = maps_by_metric[metric]
        plot_joint_hist(
            maps["3v5"]["mean"],
            maps["7v5"]["mean"],
            cano,
            metric,
            out_png=joint_dir / f"{VERSION_TAG}_joint_mean_absdiff_{metric}.png",
            cfg=cfg
        )

    # Write trend series CSV
    df_trend = pd.concat(trend_rows, ignore_index=True)
    ts_path = out_dir / f"{VERSION_TAG}_trend_timeseries.csv"
    df_trend.to_csv(ts_path, index=False)
    print(f"[OK] wrote {ts_path}")
    try:
        rc = CFG["RCLONE"]
        dst = f"{rc['remote']}:{rc['dst_dir']}"
        subprocess.run(
            ["rclone", "copy", str(ts_path), dst, "-vv"] + rc.get("extra_flags", []),
            check=False
        )
    except Exception as e:
        print(f"[rclone] trend CSV upload skipped: {e}")

    # META JSON ABOUT YEAR COVERAGE
    try:
        maps_ms  = maps_by_metric["MS"]
        maps_fs  = maps_by_metric["FS"]
        years_ms = maps_ms["years"]
        years_fs = maps_fs["years"]
    except Exception:
        years_ms = years_fs = []

    meta = {
        "version_tag": VERSION_TAG,
        "sensor": SENSOR,
        "threshold_pct": THRESH_PCT,
        "years_MS": list(map(int, years_ms)) if years_ms else [],
        "years_FS": list(map(int, years_fs)) if years_fs else []
    }
    meta_path = out_dir / f"{VERSION_TAG}_summary_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] wrote {meta_path}")
    try:
        rc = CFG["RCLONE"]
        dst = f"{rc['remote']}:{rc['dst_dir']}"
        subprocess.run(
            ["rclone", "copy", str(meta_path), dst, "-vv"] + rc.get("extra_flags", []),
            check=False
        )
    except Exception as e:
        print(f"[rclone] meta upload skipped: {e}")

    print("\nDone: window_sensitivity_maps.py")

if __name__ == "__main__":
    main()
