#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chapter 2 figure set: FS/MS only (classic vs dynamic).

Assumptions (matching your existing code):

Classic static outputs:
  CLASSIC_DIR/FS_thr15_k5/FS_YYYY.nc  (dims: y,x)
  CLASSIC_DIR/MS_thr15_k5/MS_YYYY.nc

Dynamic outputs (quantile p=0.70, k=5):
  DYN_ROOT/quantile_k5/FS/p0.7/FS_YYYY.nc
  DYN_ROOT/quantile_k5/MS/p0.7/MS_YYYY.nc

Sector mask:
  CANONICAL (NetCDF) with a sector field on the same grid (y,x or lat,lon).

Figures created (all FS + MS only):
  - Mean FS/MS maps for classic and dynamic plus Δ (dyn-classic)
  - σ(FS/MS) maps (interannual variability)
  - ECDF of |ΔFS| and |ΔMS|
  - Violins of |ΔFS| and |ΔMS| by canonical sector
  - Joint histograms: classic vs dynamic FS and MS
  - Trend maps for FS/MS (classic + dynamic)
  - Circumpolar FS/MS anomaly time series

All figures saved at ~6.5–7 inch width, 300 dpi → Word-ready.
"""

import os
import glob
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
import seaborn as sns

sns.set_context("paper", font_scale=1.0)

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

OUT_DIR = Path("/user/geog/falejandraperez/sea-ice-phase/results/Ch2_Figures/fig_set_final")

# Classic (static) phase outputs
CLASSIC_DIR = Path("/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase")
CLASSIC_T   = 0.15   # threshold
CLASSIC_K   = 5      # smoothing window

# Dynamic (percentile) phase outputs
DYN_ROOT   = Path("/user/geog/falejandraperez/sea-ice-phase/results/static_v2_slopeH/dynamic")
DYN_SCHEME = "quantile"
DYN_K      = 5
DYN_TAG    = "p0.7"   # directory under FS/MS; change if your tag differs

# Canonical sector mask
CANONICAL = Path("/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc")

# Rclone configuration (your style)
RCLONE = dict(
    enabled=True,
    remote="gdrive",
    dst_dir="sea-ice-phase/results/Ch2_Figures/fig_set_final",
    extra_flags=["--transfers=8", "--checkers=8", "--fast-list", "--copy-links"]
)
RCLONE_REMOTE = f"{RCLONE['remote']}:{RCLONE['dst_dir']}"

# Trend baseline years
BASELINE_YEARS = (1979, 2016)

# Word-friendly sizes
FIG_W_SINGLE = 6.5
FIG_W_MULTI  = 7.0
DPI          = 300

PHASE_VARS = ["FS", "MS"]

# -------------------------------------------------------
# UTILS
# -------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def rclone_upload(local_path: Path):
    if not RCLONE["enabled"]:
        return
    rel = local_path.relative_to(OUT_DIR)
    dest_dir = os.path.join(RCLONE_REMOTE, str(rel.parent))
    cmd = ["rclone", "copy", str(local_path), dest_dir] + RCLONE["extra_flags"]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"WARNING: rclone upload failed for {local_path}: {e}")

def save_figure(fig, subdir: str, fname: str, tight=True):
    out_subdir = OUT_DIR / subdir
    ensure_dir(out_subdir)
    fpath = out_subdir / fname
    if tight:
        fig.tight_layout()
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    rclone_upload(fpath)
    print(f"Saved {fpath}")
    return fpath

# -------------------------------------------------------
# DATA LOADERS
# -------------------------------------------------------

def _load_year_stack(base_dir: Path, varname: str, pattern: str):
    """
    base_dir: directory containing files like FS_YYYY.nc
    pattern:  e.g. 'FS_*.nc'
    Returns DataArray (year, lat, lon).
    """
    files = sorted(glob.glob(str(base_dir / pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {base_dir}/{pattern}")
    da_list = []
    years = []
    for f in files:
        y = int(Path(f).stem.split("_")[-1])
        ds = xr.open_dataset(f)
        da = ds[varname]
        # unify dims: rename y,x -> lat,lon
        if "y" in da.dims:
            da = da.rename({"y": "lat"})
        if "x" in da.dims:
            da = da.rename({"x": "lon"})
        da_list.append(da.expand_dims(year=[y]))
        years.append(y)
    combined = xr.concat(da_list, dim="year").sortby("year")
    return combined


def load_classic_phase():
    fs_dir = CLASSIC_DIR / f"FS_thr{int(CLASSIC_T*100):02d}_k{CLASSIC_K}"
    ms_dir = CLASSIC_DIR / f"MS_thr{int(CLASSIC_T*100):02d}_k{CLASSIC_K}"
    FS = _load_year_stack(fs_dir, "FS", "FS_*.nc")
    MS = _load_year_stack(ms_dir, "MS", "MS_*.nc")
    ds = xr.Dataset({"FS": FS, "MS": MS})
    return ds


def load_dynamic_phase():
    fs_dir = DYN_ROOT / f"{DYN_SCHEME}_k{DYN_K}" / "FS" / DYN_TAG
    ms_dir = DYN_ROOT / f"{DYN_SCHEME}_k{DYN_K}" / "MS" / DYN_TAG
    FS = _load_year_stack(fs_dir, "FS", "FS_*.nc")
    MS = _load_year_stack(ms_dir, "MS", "MS_*.nc")
    ds = xr.Dataset({"FS": FS, "MS": MS})
    return ds


def load_sector_mask():
    ds = xr.open_dataset(CANONICAL)
    # try common variable names
    if "sector" in ds:
        sec = ds["sector"]
    else:
        # fall back to first data variable
        vname = list(ds.data_vars)[0]
        sec = ds[vname]
    if "y" in sec.dims:
        sec = sec.rename({"y": "lat"})
    if "x" in sec.dims:
        sec = sec.rename({"x": "lon"})
    return sec

# -------------------------------------------------------
# COMPUTATIONS
# -------------------------------------------------------

def compute_mean_phase(ds):
    return ds[PHASE_VARS].mean("year", skipna=True)

def compute_phase_diff(c_mean, d_mean):
    ds = xr.Dataset()
    for v in PHASE_VARS:
        ds[v] = d_mean[v] - c_mean[v]
    return ds

def compute_phase_sigma(ds):
    sig = xr.Dataset()
    for v in PHASE_VARS:
        sig[v] = ds[v].std("year", skipna=True)
    return sig

def compute_theil_sen_trend(da, years):
    # simple centered least squares; close enough
    t = years.astype(float)
    t = t - t.mean()
    num = (da * t[:, None, None]).sum("year")
    den = (t ** 2).sum()
    return num / den

def compute_phase_trends(classic_ds, dynamic_ds, year_range=BASELINE_YEARS):
    y0, y1 = year_range
    c = classic_ds.sel(year=slice(y0, y1))
    d = dynamic_ds.sel(year=slice(y0, y1))
    years = c.year.values

    out = {}
    for label, ds in [("classic", c), ("dynamic", d)]:
        tds = xr.Dataset()
        for v in PHASE_VARS:
            tds[v] = compute_theil_sen_trend(ds[v], years)
        out[label] = tds
    return out

def compute_circumpolar_ts(classic_ds, dynamic_ds, baseline_range=BASELINE_YEARS):
    y0, y1 = baseline_range
    sub_c = classic_ds.sel(year=slice(y0, y1))
    sub_d = dynamic_ds.sel(year=slice(y0, y1))

    lat = sub_c.lat
    w = np.cos(np.deg2rad(lat))
    w2d = w / w.mean()

    def area_mean(da):
        return (da * w2d).mean(dim=("lat", "lon"), skipna=True)

    rows = []
    for method, ds in [("classic", sub_c), ("dynamic", sub_d)]:
        for v in PHASE_VARS:
            am = area_mean(ds[v])
            for year, val in zip(am.year.values, am.values):
                rows.append({"year": int(year), "method": method, "var": v, "value": float(val)})
    df = pd.DataFrame(rows)

    base_mask = (df["year"] >= y0) & (df["year"] <= y1)
    basemean = df[base_mask].groupby(["method", "var"])["value"].mean()
    df["anom"] = df.apply(lambda r: r["value"] - basemean.loc[(r["method"], r["var"])], axis=1)
    return df

# -------------------------------------------------------
# PLOTTING
# -------------------------------------------------------

def plot_mean_and_diff_maps(c_mean, d_mean, diff):
    for v in PHASE_VARS:
        fig, axes = plt.subplots(1, 3, figsize=(FIG_W_MULTI, 3), sharex=True, sharey=True)
        for ax, (label, ds) in zip(
            axes,
            [("Classic", c_mean), ("Dynamic", d_mean), ("Δ (dyn-classic)", diff)]
        ):
            im = ax.pcolormesh(ds["lon"], ds["lat"], ds[v], shading="auto")
            ax.set_title(label)
        fig.suptitle(f"Mean {v} timing")
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.08, pad=0.1)
        cbar.set_label("Day of year")
        save_figure(fig, "maps", f"fig_mean_{v}.png")


def plot_sigma_maps(sig_c, sig_d):
    for v in PHASE_VARS:
        fig, axes = plt.subplots(1, 2, figsize=(FIG_W_MULTI, 3), sharex=True, sharey=True)
        vmax = max(sig_c[v].max().item(), sig_d[v].max().item())
        for ax, (label, ds) in zip(axes, [("Classic", sig_c), ("Dynamic", sig_d)]):
            im = ax.pcolormesh(ds["lon"], ds["lat"], ds[v], shading="auto", vmin=0, vmax=vmax)
            ax.set_title(f"{label} σ({v})")
        fig.suptitle(f"Interannual σ of {v}")
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.08, pad=0.1)
        cbar.set_label("Std dev (days)")
        save_figure(fig, "maps", f"fig_sigma_{v}.png")


def plot_ecdf_and_violins(diff, sectors_da):
    records = []
    for v in PHASE_VARS:
        arr = diff[v]
        for sec in np.unique(sectors_da.values):
            mask = (sectors_da == sec)
            vals = np.abs(arr.where(mask).values.ravel())
            vals = vals[~np.isnan(vals)]
            for val in vals:
                records.append({"var": v, "sector": str(sec), "abs_diff": float(val)})
    df = pd.DataFrame(records)

    # ECDF (circumpolar)
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 3.5))
    for v in PHASE_VARS:
        vals = df[df["var"] == v]["abs_diff"].values
        vals = np.sort(vals)
        y = np.linspace(0, 1, len(vals), endpoint=False)
        ax.step(vals, y, where="post", label=v)
    ax.set_xlabel("|Δ DOY| (dynamic - classic)")
    ax.set_ylabel("ECDF")
    ax.legend()
    save_figure(fig, "ecdf", "fig_ecdf_absdiff.png")

    # Violin by sector
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 4.0))
    sns.violinplot(
        data=df,
        x="sector",
        y="abs_diff",
        hue="var",
        inner="quartile",
        cut=0,
        ax=ax
    )
    ax.set_ylabel("|Δ DOY|")
    ax.set_xlabel("Sector")
    ax.legend(title="Phase")
    save_figure(fig, "violin", "fig_violin_absdiff_by_sector.png")


def plot_joint_hist(c_mean, d_mean):
    for v in PHASE_VARS:
        cvals = c_mean[v].values.ravel()
        dvals = d_mean[v].values.ravel()
        mask = np.isfinite(cvals) & np.isfinite(dvals)
        cvals = cvals[mask]
        dvals = dvals[mask]

        fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 4))
        h = ax.hist2d(cvals, dvals, bins=80, norm=LogNorm())
        ax.plot([0, 366], [0, 366], "k--", linewidth=1)
        ax.set_xlabel(f"{v} classic DOY")
        ax.set_ylabel(f"{v} dynamic DOY")
        fig.colorbar(h[3], ax=ax, label="Count")
        fig.suptitle(f"Joint distribution of {v} DOY: classic vs dynamic")
        save_figure(fig, "joint", f"fig_joint_{v}.png")


def plot_trend_maps(trends):
    for v in PHASE_VARS:
        fig, axes = plt.subplots(1, 2, figsize=(FIG_W_MULTI, 3), sharex=True, sharey=True)
        all_vals = [trends["classic"][v].values, trends["dynamic"][v].values]
        vmax = np.nanmax(np.abs(all_vals))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

        for ax, label in zip(axes, ["classic", "dynamic"]):
            da = trends[label][v]
            im = ax.pcolormesh(da["lon"], da["lat"], da, shading="auto", norm=norm, cmap="coolwarm")
            ax.set_title(f"{label.capitalize()} trend {v}")
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.08, pad=0.1)
        cbar.set_label("Trend (days/year)")
        fig.suptitle(f"Trend in {v} timing")
        save_figure(fig, "trends", f"fig_trend_{v}.png")


def plot_circumpolar_timeseries(df):
    vars_order = PHASE_VARS
    labels = {"FS": "Freeze-up (FS)", "MS": "Maximum (MS)"}

    nrows = len(vars_order)
    fig, axes = plt.subplots(nrows, 1, figsize=(FIG_W_SINGLE, 1.8 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, var in zip(axes, vars_order):
        sub = df[df["var"] == var]
        for method, color in [("classic", "C0"), ("dynamic", "C1")]:
            mdf = sub[sub["method"] == method]
            ax.plot(mdf["year"], mdf["anom"], label=method, alpha=0.8)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel("Anom (days)")
        ax.set_title(labels[var])

    axes[-1].set_xlabel("Year")
    axes[0].legend(loc="upper left", ncol=2)
    save_figure(fig, "timeseries", "fig_ts_circumpolar_fs_ms.png")

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    ensure_dir(OUT_DIR)

    print("Loading classic phase...")
    classic_ds = load_classic_phase()

    print("Loading dynamic phase...")
    dynamic_ds = load_dynamic_phase()

    # Align years and space
    years = np.intersect1d(classic_ds.year.values, dynamic_ds.year.values)
    classic_ds = classic_ds.sel(year=years)
    dynamic_ds = dynamic_ds.sel(year=years)

    # Means & differences
    c_mean = compute_mean_phase(classic_ds)
    d_mean = compute_mean_phase(dynamic_ds)
    diff   = compute_phase_diff(c_mean, d_mean)

    print("Plotting mean & diff maps...")
    plot_mean_and_diff_maps(c_mean, d_mean, diff)

    # σ maps
    print("Computing σ maps...")
    sig_c = compute_phase_sigma(classic_ds)
    sig_d = compute_phase_sigma(dynamic_ds)
    plot_sigma_maps(sig_c, sig_d)

    # Sector mask
    print("Loading sector mask...")
    sectors_da = load_sector_mask()
    # align to phase grid if needed
    sectors_da = sectors_da.interp(lat=c_mean.lat, lon=c_mean.lon, method="nearest")

    print("Plotting ECDF & violins...")
    plot_ecdf_and_violins(diff, sectors_da)

    print("Plotting joint histograms...")
    plot_joint_hist(c_mean, d_mean)

    print("Computing trends...")
    trends = compute_phase_trends(classic_ds, dynamic_ds, BASELINE_YEARS)
    plot_trend_maps(trends)

    print("Computing circumpolar time series...")
    df_ts = compute_circumpolar_ts(classic_ds, dynamic_ds, BASELINE_YEARS)
    plot_circumpolar_timeseries(df_ts)

    print("All figures done.")

if __name__ == "__main__":
    main()
