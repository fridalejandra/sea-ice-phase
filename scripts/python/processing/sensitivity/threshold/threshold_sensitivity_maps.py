#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
threshold_sensitivity_maps.py

Compares detection timing between thresholds at fixed k (persistence),
holding the slope gate and detection windows constant (as in the slope+H static run).

Outputs under OUT_DIR:
  - maps/:         mean/std/q95 per-pixel maps for {MS,FS} × {thr10v15, thr30v15, thr10v30}
  - cdf/:          CDF(|Δ|) per metric (active pixels only)
  - distributions/: signed-Δ histograms per metric (active pixels only)
  - trends/:       annual mean(|Δ|) trend plots (circumpolar + sectors)
  - joint/:        hexbin joint histogram of per-pixel mean |Δ10–15| vs |Δ15–30|
  - summary_stats.csv (sectoral + circumpolar statistics of maps)
  - trend_timeseries.csv (annual mean(|Δ|) by region)
  - summary_meta.json (year coverage)
"""

from pathlib import Path
import os, glob, re, json, subprocess
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
from matplotlib.colors import LogNorm

# =========================
# CONFIG — EDIT THIS BLOCK
# =========================
VERSION_TAG   = "static_v2_slopeH"     # identifies slope + persistence version
SENSOR        = "SMMR"                 # "SMMR" or "AMSRE"
K_FIXED       = 5                      # we compare thresholds at fixed k
THRESHOLDS    = [10, 15, 30]           # percent thresholds available on disk
BASE_REMOTE   = "gdrive:sea-ice-phase" # your Google Drive rclone remote

# Input FS/MS (produced by your slope+H detector) — this is the same root you showed
INPUT_ROOT    = f"/user/geog/falejandraperez/sea-ice-phase/results/{SENSOR}_phase"

# Canonical grid + masks
CANONICAL     = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"

# Outputs (local) + Drive (rclone)
OUT_ROOT      = f"/user/geog/falejandraperez/sea-ice-phase/results/{VERSION_TAG}/threshold_sensitivity/{SENSOR}_k{K_FIXED}"
REMOTE_PATH   = f"{BASE_REMOTE}/Results/Static2/slope_threshold_sensitivity/{VERSION_TAG}/{SENSOR}_k{K_FIXED}"

# Circular day-of-year period and plotting ranges (match window script)
PERIOD        = 366
MEAN_VMAX     = 10
STD_VMAX      = 8
Q95_VMAX      = 10
DPI           = 180
MAX_X         = 30  # for CDF/plots

# Active-pixel mask based on baseline threshold (thr=15) at k=5
ACTIVE_FROM_THR = 15
ACTIVE_MIN_FRAC = 0.30

# rclone upload settings
RCLONE = {
    "enabled": True,
    "remote": BASE_REMOTE.split(":")[0],  # "gdrive"
    "dst_dir": REMOTE_PATH,
    "extra_flags": ["--transfers=8","--checkers=8","--fast-list"],
    "dry_run": False
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

def rclone_copy(local_path):
    rc = RCLONE
    if not rc.get("enabled", False):
        return
    dst = f"{rc['remote']}:{rc['dst_dir']}"
    cmd = ["rclone", "copy", str(local_path), dst] + rc.get("extra_flags", [])
    if rc.get("dry_run"): cmd.insert(1, "--dry-run")
    print("[rclone]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!! rclone failed: {e}")

def open_phase_dict(metric, thr_pct, kdays=K_FIXED):
    """Return {year: DataArray} for metric ('MS'/'FS') and given threshold (10/15/30) at fixed kdays."""
    subdir = f"{metric}_thr{thr_pct}_k{kdays}"
    folder = os.path.join(INPUT_ROOT, subdir)
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
    out = xr.concat(arrs, dim="year").rename(name)  # (year,y,x)
    return out

def wrapped_abs_diff(a, b, period=PERIOD):
    """Absolute wrapped difference |Δ| between two day-of-year arrays."""
    return np.abs(((a - b + period//2) % period) - (period//2))

# canonical grid
def load_canonical():
    cano = xr.open_dataset(CANONICAL).load()
    return cano

# ======================
# ACTIVE PIXEL MASK
# ======================
def compute_active_mask(metric, cano):
    """
    Active = pixels that have a valid DOY for baseline (thr=15, k=5) in >= ACTIVE_MIN_FRAC of years.
    Returns bool DataArray (y,x).
    """
    d5 = open_phase_dict(metric, ACTIVE_FROM_THR, kdays=K_FIXED)
    years = sorted(d5.keys())
    A5 = stack_years(d5, years, f"{metric}_thr{ACTIVE_FROM_THR}_k{K_FIXED}")
    valid_count = A5.notnull().sum(dim="year")
    min_years = max(1, int(np.floor(ACTIVE_MIN_FRAC * len(years))))
    active = valid_count >= min_years
    active = active & cano["valid_ocean"].astype(bool)
    return active.astype(bool), years

# ======================
# Δ MAPS BETWEEN THRESHOLDS
# ======================
def compute_maps_for_metric_pair(metric, thrA, thrB, cano, active_mask=None):
    """
    Returns per-pixel maps for |Δ(thrA−thrB)|:
      { "mean": DA, "std": DA, "q95": DA, "years": [int,...] }
    """
    dA = open_phase_dict(metric, thrA, kdays=K_FIXED)
    dB = open_phase_dict(metric, thrB, kdays=K_FIXED)
    years = align_years([dA, dB])

    A = stack_years(dA, years, f"{metric}_thr{thrA}")
    B = stack_years(dB, years, f"{metric}_thr{thrB}")

    valid = (~np.isnan(A)) & (~np.isnan(B))
    if active_mask is not None:
        valid = valid & active_mask.astype(bool)

    D = xr.apply_ufunc(wrapped_abs_diff, A, B, PERIOD, dask="allowed")
    D = D.where(valid)

    meanD = D.mean(dim="year", skipna=True)
    stdD  = D.std(dim="year",  skipna=True)
    q95D  = D.quantile(0.95,   dim="year", skipna=True)

    vo = cano["valid_ocean"].astype(bool)
    for da in [meanD, stdD, q95D]:
        da.values[~vo.values] = np.nan

    return {
        "mean": meanD,
        "std": stdD,
        "q95": q95D,
        "years": years
    }

# ======================
# FLAT DELTAS FOR CDF & DISTRIBUTION
# ======================
def compute_flat_deltas_pair(metric, thrA, thrB, cano, active_mask=None):
    dA = open_phase_dict(metric, thrA, kdays=K_FIXED)
    dB = open_phase_dict(metric, thrB, kdays=K_FIXED)
    years = align_years([dA, dB])
    A = stack_years(dA, years, f"{metric}_thr{thrA}")
    B = stack_years(dB, years, f"{metric}_thr{thrB}")

    valid = (~np.isnan(A)) & (~np.isnan(B))
    if active_mask is not None:
        valid = valid & active_mask.astype(bool)

    # wrapped signed deltas (−183..+183)
    D_signed = xr.apply_ufunc(lambda a,b: ((a-b + PERIOD//2) % PERIOD) - (PERIOD//2), A, B, dask="allowed")
    d = D_signed.where(valid).values.ravel()
    d = d[np.isfinite(d)]
    return d, np.abs(d)

# ======================
# CARTOPY PLOTTING
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

def plot_map_cartopy(da, cano, title, out_png, vmin, vmax, cmap="viridis", white_under=True):
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

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=DPI)
    plt.close()
    print(f"[OK] wrote {out_png}")
    rclone_copy(out_png)

# ======================
# MAP STATS → CSV
# ======================
def summarize_to_rows(da, cano, metric, pair_label, statname):
    rows = []
    sid = cano["sector_id"].astype(np.int16)
    vo  = cano["valid_ocean"].astype(bool)

    # circumpolar
    vals = da.where(vo).values
    vals = vals[np.isfinite(vals)]
    if vals.size:
        rows.append({"Metric": metric, "Pair": pair_label, "Sector": "Circumpolar",
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
        rows.append({"Metric": metric, "Pair": pair_label, "Sector": name,
                     "Stat": statname,
                     "MedianDays": float(np.nanmedian(v)),
                     "P95Days": float(np.nanpercentile(v, 95)),
                     "MeanDays": float(np.nanmean(v)),
                     "Npixels": int(v.size)})
    return rows

# ======================
# CDF & DISTRIBUTION PLOTS
# ======================
def plot_cdf(metric, pair_label, abs_vals):
    import seaborn as sns
    sns.set_style("whitegrid")
    v = abs_vals[abs_vals <= MAX_X]
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    if v.size:
        sns.ecdfplot(v, ax=ax, lw=2)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.4")
    ax.set_xlim(0, MAX_X); ax.set_ylim(0, 1)
    ax.set_xlabel("Absolute timing difference |Δ| (days)")
    ax.set_ylabel("Cumulative Fraction of Pixels")
    ax.set_title(f"CDF of |Δ| • {metric} • {pair_label} • k={K_FIXED}")
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.82")
    fig.tight_layout()
    fn = Path(OUT_ROOT) / "cdf" / f"cdf_absdiff_{metric}_{pair_label}.png"
    fn.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fn, dpi=DPI); plt.close(fig); rclone_copy(fn)

def plot_distribution(metric, pair_label, signed_vals):
    bins = np.arange(-30, 31, 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.hist(signed_vals, bins=bins, density=True, alpha=0.9)
    med = np.median(signed_vals); q25, q75 = np.percentile(signed_vals, [25, 75])
    pct5 = 100.0 * np.mean(np.abs(signed_vals) > 5)
    ax.axvline(med, color="k", lw=1)
    ax.axvline(q25, color="k", lw=1, ls=":")
    ax.axvline(q75, color="k", lw=1, ls=":")
    ax.set_title(f"{metric} Δ({pair_label})  •  median {med:+.1f} d | IQR {q25:+.1f}–{q75:+.1f} d | %|Δ|>5: {pct5:.1f}%")
    ax.set_xlabel("Δ timing (days)"); ax.set_ylabel("Density"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fn = Path(OUT_ROOT) / "distributions" / f"dist_signed_delta_{metric}_{pair_label}.png"
    fn.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fn, dpi=DPI); plt.close(fig); rclone_copy(fn)

# ======================
# TEMPORAL TRENDS & JOINT HISTOGRAM
# ======================
def compute_year_series_for_pair(metric, thrA, thrB, cano, active_mask=None):
    """Returns DataFrame: year, Metric, Pair, Region, MeanAbsDiff (active pixels only if mask supplied)."""
    dA = open_phase_dict(metric, thrA, kdays=K_FIXED)
    dB = open_phase_dict(metric, thrB, kdays=K_FIXED)
    years = align_years([dA, dB])

    A = stack_years(dA, years, f"{metric}_thr{thrA}")
    B = stack_years(dB, years, f"{metric}_thr{thrB}")

    valid = (~np.isnan(A)) & (~np.isnan(B))
    if active_mask is not None:
        valid = valid & active_mask.astype(bool)

    D = xr.apply_ufunc(wrapped_abs_diff, A, B, PERIOD, dask="allowed")
    D = D.where(valid)

    vo  = cano["valid_ocean"].astype(bool)
    sid = cano["sector_id"].astype(np.int16)

    rows = []
    for yi, yr in enumerate(D["year"].values):
        # circumpolar
        v = D.isel(year=yi).where(vo).values
        v = v[np.isfinite(v)]
        if v.size:
            rows.append({"year": int(yr), "Metric": metric, "Pair": f"thr{thrA}_vs_thr{thrB}",
                         "Region": "Circumpolar", "MeanAbsDiff": float(np.nanmean(v))})
        # per sector
        for k, name in SECTOR_ID_TO_NAME.items():
            mask_k = ((sid == k) & vo).values
            a = D.isel(year=yi).values[mask_k]
            a = a[np.isfinite(a)]
            if a.size:
                rows.append({"year": int(yr), "Metric": metric, "Pair": f"thr{thrA}_vs_thr{thrB}",
                             "Region": name, "MeanAbsDiff": float(np.nanmean(a))})
    return pd.DataFrame(rows)

def plot_trend_lines(df, out_png, title, per_sector=False):
    years = sorted(df["year"].unique())
    if not per_sector:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for pair in sorted(df["Pair"].unique()):
            sub = df[(df["Region"]=="Circumpolar") & (df["Pair"]==pair)]
            sub = sub.groupby("year")["MeanAbsDiff"].mean().reindex(years)
            ax.plot(years, sub.values, lw=2, label=pair)
        ax.set_xlabel("Year"); ax.set_ylabel("mean(|Δ|) (days)")
        ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(title="Pair")
        plt.tight_layout(); plt.savefig(out_png, dpi=DPI); plt.close(); rclone_copy(out_png)
        print(f"[OK] wrote {out_png}")
        return

    sectors = list(SECTOR_ID_TO_NAME.values())
    n = len(sectors); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*2.6), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, name in enumerate(sectors):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        for pair in sorted(df["Pair"].unique()):
            sub = df[(df["Region"]==name) & (df["Pair"]==pair)]
            sub = sub.groupby("year")["MeanAbsDiff"].mean().reindex(years)
            ax.plot(years, sub.values, lw=1.8, label=pair)
        ax.set_title(name, fontsize=9); ax.grid(True, alpha=0.3)
    for j in range(n, nrows*ncols):
        r, c = divmod(j, ncols); axes[r, c].set_visible(False)
    fig.supxlabel("Year"); fig.supylabel("mean(|Δ|) (days)")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=True)
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(out_png, dpi=DPI); plt.close(); rclone_copy(out_png)
    print(f"[OK] wrote {out_png}")

def plot_joint_hist(da_meanLM, da_meanMH, cano, metric, out_png):
    vo = cano["valid_ocean"].astype(bool).values
    x = da_meanLM.where(vo).values.ravel()
    y = da_meanMH.where(vo).values.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size:
        slope, intercept = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0,1]
    else:
        slope = intercept = r = np.nan

    fig, ax = plt.subplots(figsize=(5.1, 5.0))
    if x.size:
        hb = ax.hexbin(x, y, gridsize=60, norm=LogNorm(), mincnt=1)
        cb = plt.colorbar(hb, ax=ax); cb.set_label("pixel count (log)")
    lim = max(6, float(np.nanmax([x.max() if x.size else 0, y.max() if y.size else 0])))
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="1:1")
    xx = np.linspace(0, lim, 100)
    if np.isfinite(slope):
        ax.plot(xx, slope*xx + intercept, color="C1", lw=1.6, label=f"fit: y={slope:.2f}x+{intercept:.2f}")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("mean(|Δ10–15|) (days)"); ax.set_ylabel("mean(|Δ15–30|) (days)")
    ax.set_title(f"Joint histogram • {metric}\n$r = {r:.2f}$")
    ax.legend(loc="lower right", frameon=True); ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(out_png, dpi=DPI); plt.close(); rclone_copy(out_png)
    print(f"[OK] wrote {out_png}")

# ======================
# MAIN
# ======================
def main():
    out_dir = Path(OUT_ROOT)
    maps_dir = out_dir / "maps"
    cdf_dir  = out_dir / "cdf"
    dist_dir = out_dir / "distributions"
    trends_dir = out_dir / "trends"
    joint_dir  = out_dir / "joint"
    for d in [out_dir, maps_dir, cdf_dir, dist_dir, trends_dir, joint_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cano = load_canonical()

    # Build active masks once from baseline thr=15,k=5
    active_masks = {}
    for metric in ["MS", "FS"]:
        amask, _ = compute_active_mask(metric, cano)
        active_masks[metric] = amask

    # Collect rows for summary CSV
    all_rows = []

    # Pairs to compare
    thr = sorted(THRESHOLDS)
    PAIRS = [(thr[0], thr[1]), (thr[1], thr[2]), (thr[0], thr[2])]  # 10–15, 15–30, 10–30

    # MAPS + STATS + CDF/DISTRIBUTION
    for metric in ["MS", "FS"]:
        for (A, B) in PAIRS:
            pair_label = f"thr{A}_vs_thr{B}"

            # Per-pixel maps
            maps = compute_maps_for_metric_pair(metric, A, B, cano, active_mask=active_masks[metric])

            # Plot maps
            plot_map_cartopy(maps["mean"], cano,
                             title=f"mean(|Δ|) days • {metric} • {pair_label}",
                             out_png=maps_dir / f"mean_absdiff_{metric}_{pair_label}.png",
                             vmin=0, vmax=MEAN_VMAX, cmap="viridis", white_under=True)

            plot_map_cartopy(maps["std"], cano,
                             title=f"std(|Δ|) days • {metric} • {pair_label}",
                             out_png=maps_dir / f"std_absdiff_{metric}_{pair_label}.png",
                             vmin=0, vmax=STD_VMAX, cmap="magma", white_under=True)

            plot_map_cartopy(maps["q95"], cano,
                             title=f"q95(|Δ|) days • {metric} • {pair_label}",
                             out_png=maps_dir / f"q95_absdiff_{metric}_{pair_label}.png",
                             vmin=0, vmax=Q95_VMAX, cmap="plasma", white_under=True)

            # stats to CSV
            all_rows += summarize_to_rows(maps["mean"], cano, metric, pair_label, "mean_absdiff")
            all_rows += summarize_to_rows(maps["std"],  cano, metric, pair_label, "std_absdiff")
            all_rows += summarize_to_rows(maps["q95"],  cano, metric, pair_label, "q95_absdiff")

            # Flat deltas → CDF + distribution (active pixels only)
            d_signed, d_abs = compute_flat_deltas_pair(metric, A, B, cano, active_mask=active_masks[metric])
            plot_cdf(metric, pair_label, d_abs)
            plot_distribution(metric, pair_label, d_signed)

    # Write map summary CSV
    df = pd.DataFrame(all_rows,
                      columns=["Metric","Pair","Sector","Stat","MedianDays","P95Days","MeanDays","Npixels"])
    csv_path = out_dir / "summary_stats.csv"; df.to_csv(csv_path, index=False)
    print(f"[OK] wrote {csv_path}"); rclone_copy(csv_path)

    # TEMPORAL TRENDS + JOINT HISTOGRAMS (active pixels)
    trend_rows = []
    for metric in ["MS", "FS"]:
        # trend lines for each pair
        df_pairs = []
        for (A,B) in PAIRS:
            df_year = compute_year_series_for_pair(metric, A, B, cano, active_mask=active_masks[metric])
            df_pairs.append(df_year)
        df_year_all = pd.concat(df_pairs, ignore_index=True)
        trend_rows.append(df_year_all)

        # circumpolar trend (all pairs)
        plot_trend_lines(df_year_all,
                         out_png=trends_dir / f"trend_circ_{metric}.png",
                         title=f"Circumpolar mean(|Δ|) per year • {metric}",
                         per_sector=False)

        # sector panels
        plot_trend_lines(df_year_all,
                         out_png=trends_dir / f"trend_sectors_{metric}.png",
                         title=f"Sectoral mean(|Δ|) per year • {metric}",
                         per_sector=True)

        # Joint histogram compares per-pixel mean(|Δ10–15|) vs mean(|Δ15–30|)
        mean_lm = compute_maps_for_metric_pair(metric, thr[0], thr[1], cano, active_mask=active_masks[metric])["mean"]
        mean_mh = compute_maps_for_metric_pair(metric, thr[1], thr[2], cano, active_mask=active_masks[metric])["mean"]
        plot_joint_hist(mean_lm, mean_mh, cano, metric, out_png=Path(joint_dir) / f"joint_hist_mean_absdiff_{metric}.png")

    df_trends = pd.concat(trend_rows, ignore_index=True)
    df_trends.to_csv(out_dir / "trend_timeseries.csv", index=False)
    print(f"[OK] wrote {out_dir / 'trend_timeseries.csv'}"); rclone_copy(out_dir / "trend_timeseries.csv")

    # Provenance: year coverage
    meta = {"thresholds": THRESHOLDS, "k": K_FIXED, "years_MS": None, "years_FS": None}
    try:
        meta["years_MS"] = compute_maps_for_metric_pair("MS", thr[0], thr[1], cano)["years"]
        meta["years_FS"] = compute_maps_for_metric_pair("FS", thr[0], thr[1], cano)["years"]
    except Exception:
        pass
    with open(out_dir / "summary_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    rclone_copy(out_dir / "summary_meta.json")

if __name__ == "__main__":
    main()
