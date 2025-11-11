#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Threshold sensitivity for static (slope+H) phase detection.
Compares FS/MS dates between thresholds at fixed k, builds maps, CDFs, boxplots,
and a joint hexbin, and pushes outputs to Google Drive via rclone.

Outputs (local OUT_ROOT mirrors Drive REMOTE_DIR):
  maps/<metric>_<pair>_{mean|std|q95}.png
  cdf/CDF_<metric>_<pair>.png
  box/BOX_<metric>_<pair>.png
  joint/JOINT_<metric>_thr10-thr15_vs_thr15-thr30_k5.png
  summary_stats.csv
"""

import os, re, glob, json, subprocess
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import seaborn as sns

# ============== CONFIG ==============
VERSION_TAG  = "static_v2_slopeH"
SENSOR       = "SMMR"
K_FIXED      = 5
THRESHOLDS   = [10, 15, 30]               # percent, matches folder names
PERIOD       = 366                        # wrapped DOY
MAX_X        = 30                         # plot x-limit for |Δ| (days)

# Inputs: your FS/MS per-year NetCDFs from the static slope+H detector
INPUT_ROOT   = f"/user/geog/falejandraperez/sea-ice-phase/results/{SENSOR}_phase"
CANONICAL    = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"

# Outputs (local) + Drive
OUT_ROOT     = f"/user/geog/falejandraperez/sea-ice-phase/results/{VERSION_TAG}/threshold_sensitivity/{SENSOR}_k{K_FIXED}"
REMOTE_DIR   = f"sea-ice-phase/Results/Static2/slope_threshold_sensitivity/{VERSION_TAG}/{SENSOR}_k{K_FIXED}"
RCLONE       = {
    "enabled": True,                 # set False if you want to keep local only
    "remote": "gdrive",              # your rclone remote name
    "dst_dir": REMOTE_DIR,
    "extra_flags": ["--transfers=8", "--checkers=8", "--fast-list"],
    "dry_run": False
}

# plotting
DPI          = 200
MEAN_VMAX    = 10
STD_VMAX     = 8
Q95_VMAX     = 10

sns.set_style("whitegrid")
sns.set_context("talk")

# ============== helpers ==============
year_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    m = year_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def rclone_copy(local_path, cfg=RCLONE):
    if not cfg.get("enabled", False):
        return
    dst = f"{cfg['remote']}:{cfg['dst_dir']}"
    cmd = ["rclone", "copy", str(local_path), dst] + cfg.get("extra_flags", [])
    if cfg.get("dry_run"):
        cmd.insert(1, "--dry-run")
    print("[rclone]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!! rclone failed: {e}")

def load_thr_dict(metric, thr_pct, k=K_FIXED):
    folder = os.path.join(INPUT_ROOT, f"{metric}_thr{thr_pct}_k{k}")
    files  = sorted(glob.glob(os.path.join(folder, f"{metric}_*.nc")))
    d = {}
    for f in files:
        y = parse_year(f)
        if y is None:
            continue
        with xr.open_dataset(f) as ds:
            d[y] = ds[metric].load()
    if not d:
        raise FileNotFoundError(f"No files for {metric} thr{thr_pct} k{k} in {folder}")
    return d

def align_years(dicts):
    common = set.intersection(*[set(d.keys()) for d in dicts])
    years = sorted(common)
    if not years:
        raise ValueError("No overlapping years across thresholds.")
    return years

def stack_years(d, years, name):
    arrs = [d[y].expand_dims(year=[y]) for y in years]
    return xr.concat(arrs, dim="year").rename(name)

def wrapped_abs_diff(a, b, period=PERIOD):
    return np.abs(((a - b + period//2) % period) - (period//2))

def get_canonical():
    cano = xr.open_dataset(CANONICAL).load()
    vo   = cano["valid_ocean"].astype(bool)
    lon  = cano["lon"]; lat = cano["lat"]
    sid  = cano["sector_id"].astype(np.int16)
    sector_map = {
        1: "Amundsen–Bellingshausen",
        2: "Weddell",
        3: "King Haakon VII",
        4: "East Antarctic",
        5: "Ross–Amundsen",
    }
    return cano, vo, lon, lat, sid, sector_map

CANO, VO, LON, LAT, SID, SECTOR_MAP = get_canonical()

# ============== compute diffs + summary rows ==============
def diffs_for_metric_pair(metric, thrA, thrB):
    dA = load_thr_dict(metric, thrA)
    dB = load_thr_dict(metric, thrB)
    years = align_years([dA, dB])
    A = stack_years(dA, years, f"{metric}_thr{thrA}")  # (year,y,x)
    B = stack_years(dB, years, f"{metric}_thr{thrB}")
    valid = (~np.isnan(A)) & (~np.isnan(B)) & VO
    D = xr.apply_ufunc(wrapped_abs_diff, A, B, PERIOD, dask="allowed")
    return years, D.where(valid)

def summarize_regionwise(da, metric, pair, statname):
    rows = []
    vo = VO.values
    sid = SID.values
    vals = da.where(VO).values
    vals = vals[np.isfinite(vals)]
    if vals.size:
        rows.append({
            "Metric": metric, "Pair": pair, "Region": "Circumpolar", "Stat": statname,
            "MedianDays": float(np.nanmedian(vals)),
            "P95Days": float(np.nanpercentile(vals, 95)),
            "MeanDays": float(np.nanmean(vals)),
            "Npixels": int(vals.size)
        })
    for k, name in SECTOR_MAP.items():
        v = da.where((SID==k) & VO).values
        v = v[np.isfinite(v)]
        if v.size:
            rows.append({
                "Metric": metric, "Pair": pair, "Region": name, "Stat": statname,
                "MedianDays": float(np.nanmedian(v)),
                "P95Days": float(np.nanpercentile(v, 95)),
                "MeanDays": float(np.nanmean(v)),
                "Npixels": int(v.size)
            })
    return rows

# ============== plotting ==============
def _round_polar_axes(ax):
    theta = np.linspace(0, 2*np.pi, 200)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

SECTOR_BOUNDARIES_E = [71, 162, 250, 290, 346]
def draw_sector_meridians(ax):
    lats = np.linspace(-90, -45, 256)
    for degE in SECTOR_BOUNDARIES_E:
        lon = degE if degE <= 180 else degE - 360
        ax.plot(np.full_like(lats, lon), lats, transform=ccrs.Geodetic(),
                color="k", lw=0.6, alpha=0.7, zorder=6)

def plot_map(da, title, vmin, vmax, out_png, cmap="viridis"):
    proj = ccrs.SouthPolarStereo()
    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw={"projection": proj})
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="0.2", lw=0.4, zorder=2)
    ax.coastlines("110m", color="0.2", linewidth=0.5, zorder=3)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_under("white")
    pc = ax.pcolormesh(LON, LAT, da, transform=ccrs.PlateCarree(),
                       vmin=vmin + (1e-6 if vmin==0 else 0), vmax=vmax, cmap=cmap_obj, zorder=4)
    cb = plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.02, shrink=0.85)
    cb.set_label(title.split("•")[0].strip(), fontsize=9)
    draw_sector_meridians(ax)
    ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
    _round_polar_axes(ax)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=DPI); plt.close()
    print("[OK] wrote", out_png)
    rclone_copy(out_png)

def ecdf(vals):
    x = np.asarray(vals)
    x = x[np.isfinite(x)]
    x.sort()
    y = np.linspace(0, 1, x.size, endpoint=False)
    return x, y

def plot_cdf(metric, pair_label, D, out_png):
    vals = D.values[np.isfinite(D.values)]
    vals = vals[vals <= MAX_X]
    x, y = ecdf(vals)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(x, y, lw=2)
    ax.set_xlim(0, MAX_X); ax.set_ylim(0,1)
    ax.set_xlabel("|Δ| (days)"); ax.set_ylabel("Cumulative Fraction of Pixels")
    ax.set_title(f"CDF • {metric} • {pair_label} • k={K_FIXED}")
    ax.grid(True, ls=":", lw=0.7)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)
    print("[OK] wrote", out_png)
    rclone_copy(out_png)

def plot_box(metric, pair_label, D, out_png):
    data = []
    v = D.values[np.isfinite(D.values)]
    if v.size:
        data.append(("Circumpolar", v))
    for k, name in SECTOR_MAP.items():
        m = (SID.values == k) & VO.values
        vv = D.values[:, m]
        vv = vv[np.isfinite(vv)]
        if vv.size:
            data.append((name, vv))
    labels = [d[0] for d in data]
    series = [d[1] for d in data]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.boxplot(series, labels=labels, showfliers=False)
    ax.set_ylabel("|Δ| (days)")
    ax.set_title(f"Box • {metric} • {pair_label} • k={K_FIXED}")
    ax.set_ylim(0, MAX_X)
    ax.grid(True, axis="y", ls=":", lw=0.7)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)
    print("[OK] wrote", out_png)
    rclone_copy(out_png)

def plot_joint_hist(metric, pair_low_mid, pair_mid_high, D_lm, D_mh, out_png):
    X = D_lm.mean("year", skipna=True).values
    Y = D_mh.mean("year", skipna=True).values
    m = np.isfinite(X) & np.isfinite(Y)
    x = X[m].ravel(); y = Y[m].ravel()

    if x.size >= 2:
        r = np.corrcoef(x, y)[0,1]
        slope, intercept = np.polyfit(x, y, 1)
    else:
        r = np.nan; slope = intercept = np.nan

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    hb = ax.hexbin(x, y, gridsize=80, norm=LogNorm(), mincnt=1, cmap="magma")
    cb = fig.colorbar(hb, ax=ax, label="pixel count (log)")
    ax.plot([0, MAX_X], [0, MAX_X], ls="--", lw=1, c="0.6", label="1:1")
    if np.isfinite(slope):
        xx = np.array([0, MAX_X])
        ax.plot(xx, slope*xx + intercept, lw=2, label=f"fit: y={slope:.2f}x+{intercept:.2f}")
    ax.set_xlim(0, MAX_X); ax.set_ylim(0, MAX_X)
    ax.set_xlabel(f"mean |Δ| ({pair_low_mid}) (days)")
    ax.set_ylabel(f"mean |Δ| ({pair_mid_high}) (days)")
    ax.set_title(f"Joint histogram • {metric} • r={r:.2f}")
    ax.legend(frameon=True, loc="lower right")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)
    print("[OK] wrote", out_png)
    rclone_copy(out_png)

# ============== main ==============
def main():
    out_root = Path(OUT_ROOT)
    for sub in ["maps", "cdf", "box", "joint"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    pairs = [(THRESHOLDS[i], THRESHOLDS[j])
             for i in range(len(THRESHOLDS))
             for j in range(i+1, len(THRESHOLDS))]
    all_rows = []

    for metric in ["MS", "FS"]:
        # compute per-pair |Δ| datasets
        pair_D = {}
        for a, b in pairs:
            years, D = diffs_for_metric_pair(metric, a, b)  # (year,y,x)
            pair_label = f"thr{a}-thr{b}"

            # reduce across years for maps
            mean_da = D.mean("year", skipna=True)
            std_da  = D.std("year",  skipna=True)
            q95_da  = D.quantile(0.95, dim="year", skipna=True)

            # maps
            plot_map(mean_da, f"mean(|Δ|) days • {metric} • {pair_label}",
                     0, MEAN_VMAX, out_root / "maps" / f"mean_absdiff_{metric}_{pair_label}.png", cmap="viridis")
            plot_map(std_da,  f"std(|Δ|) days • {metric} • {pair_label}",
                     0, STD_VMAX,  out_root / "maps" / f"std_absdiff_{metric}_{pair_label}.png", cmap="magma")
            plot_map(q95_da,  f"q95(|Δ|) days • {metric} • {pair_label}",
                     0, Q95_VMAX,  out_root / "maps" / f"q95_absdiff_{metric}_{pair_label}.png", cmap="plasma")

            # summary rows
            all_rows += summarize_regionwise(mean_da, metric, pair_label, "mean_absdiff")
            all_rows += summarize_regionwise(std_da,  metric, pair_label, "std_absdiff")
            all_rows += summarize_regionwise(q95_da,  metric, pair_label, "q95_absdiff")

            # distributions
            plot_cdf(metric, pair_label, D, out_root / "cdf" / f"CDF_{metric}_{pair_label}_k{K_FIXED}.png")
            plot_box(metric, pair_label, D, out_root / "box" / f"BOX_{metric}_{pair_label}_k{K_FIXED}.png")

            pair_D[(a,b)] = D

        # joint hist for (low–mid) vs (mid–high) if we have 3 thresholds
        thr_sorted = sorted(THRESHOLDS)
        if len(thr_sorted) >= 3:
            lm = (thr_sorted[0], thr_sorted[1])
            mh = (thr_sorted[1], thr_sorted[2])
            if lm in pair_D and mh in pair_D:
                plot_joint_hist(metric,
                                f"thr{lm[0]}-thr{lm[1]}",
                                f"thr{mh[0]}-thr{mh[1]}",
                                pair_D[lm], pair_D[mh],
                                out_root / "joint" / f"JOINT_{metric}_thr{lm[0]}-thr{lm[1]}_vs_thr{mh[0]}-thr{mh[1]}_k{K_FIXED}.png")

    # write summary CSV
    df = pd.DataFrame(all_rows, columns=["Metric","Pair","Region","Stat","MedianDays","P95Days","MeanDays","Npixels"])
    csv_path = out_root / "summary_stats.csv"
    df.to_csv(csv_path, index=False)
    print("[OK] wrote", csv_path); rclone_copy(csv_path)

    # simple provenance
    meta = {"thresholds": THRESHOLDS, "k_fixed": K_FIXED, "version": VERSION_TAG}
    with open(out_root / "summary_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    rclone_copy(out_root / "summary_meta.json")

if __name__ == "__main__":
    main()
