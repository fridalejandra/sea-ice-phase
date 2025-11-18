#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window / Threshold Sensitivity – Static Method (Slope + Persistence Version)
Frida A. Perez — updated for version tagging and rclone sync
"""

import os, re, glob, subprocess
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil

# NEW: cartopy + polar helpers for map style
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.path import Path as MplPath

# ----------------- RUN CONTEXT -----------------
VERSION_TAG   = "static_v2_slopeH"           # identifies slope + persistence version
SENSOR        = "SMMR"                       # "SMMR" or "AMSRE"
THRESH_PCT    = 15                           # e.g., 10/15/20
PERIOD        = 366                          # DOY wrap (for diffs)
MAX_X         = 30                           # x-limit in days for |Δ|

# --- paths ---
INPUT_ROOT    = f"/user/geog/falejandraperez/sea-ice-phase/results/{SENSOR}_phase"
OUTDIR_FIGS   = f"figures/{VERSION_TAG}/{SENSOR}_thr{THRESH_PCT}"
OUTDIR_RESULTS= f"results/{VERSION_TAG}/{SENSOR}_thr{THRESH_PCT}"
os.makedirs(OUTDIR_FIGS, exist_ok=True)
os.makedirs(OUTDIR_RESULTS, exist_ok=True)

# --- canonical sector file ---
CANONICAL     = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"

# --- Google Drive (rclone) ---
RCLONE_REMOTE = "gdrive"                     # your rclone remote name
RCLONE_PATH   = f"sea-ice-phase/results/{VERSION_TAG}/{SENSOR}_thr{THRESH_PCT}"
RCLONE_DEST   = f"{RCLONE_REMOTE}:{RCLONE_PATH}"

# --- aesthetics ---
sns.set_context("talk")
sns.set_style("whitegrid")

# ===========================================================
# HELPERS
# ===========================================================
year_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    m = year_re.search(os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))

def load_window_dict(metric, kdays):
    """Return {year: DataArray} for a given metric ('MS'/'FS') and window length (3/5/7)."""
    subdir = f"{metric}_thr{THRESH_PCT}_k{kdays}"
    folder = os.path.join(INPUT_ROOT, subdir)
    files = sorted(glob.glob(os.path.join(folder, f"{metric}_*.nc")))
    d = {}
    for f in files:
        yr = parse_year(f)
        if yr is None:
            print(f"Skip (no year): {f}")
            continue
        try:
            with xr.open_dataset(f) as ds:
                da = ds[metric].load()
            d[yr] = da
        except Exception as e:
            print(f"Skip {f}: {e}")
    if not d:
        raise FileNotFoundError(f"No {metric} files in {folder}")
    return d

def align_years(dicts):
    common = set.intersection(*[set(d.keys()) for d in dicts])
    years = sorted(common)
    if not years:
        raise ValueError("No overlapping years across windows.")
    return years

def stack_years(d, years, name):
    arrs = [d[y].expand_dims(year=[y]) for y in years]
    out = xr.concat(arrs, dim="year")
    out.name = name
    return out

def wrapped_diff_np(a_minus_b, period=PERIOD):
    # keep diffs symmetric around 0
    return ((a_minus_b + period//2) % period) - (period//2)

def ecdf_data(abs_diff, max_x=MAX_X):
    abs_diff = abs_diff[~np.isnan(abs_diff)]
    return abs_diff[abs_diff <= max_x], abs_diff

# ===========================================================
# SECTOR MASKS (from canonical file)
# ===========================================================
SECTOR_ID_TO_NAME = {
    1: "Amundsen–Bellingshausen",
    2: "Weddell",
    3: "King Haakon VII",
    4: "East Antarctic",
    5: "Ross–Amundsen",
}

def load_sector_masks(canonical_path=CANONICAL):
    ds = xr.open_dataset(canonical_path).load()
    sid = ds["sector_id"].astype(np.int16)
    valid_ocean = ds["valid_ocean"].astype(bool) if "valid_ocean" in ds else xr.ones_like(sid, dtype=bool)
    masks = {}
    for k, name in SECTOR_ID_TO_NAME.items():
        m = (sid == k) & valid_ocean
        masks[name] = m.values.astype(bool)
    ds.close()
    return masks

def rclone_copy(local_path, remote_dir=None):
    if remote_dir is None:
        remote_dir = f"{RCLONE_REMOTE}:{RCLONE_PATH}/figures/"
    cmd = f"rclone copy '{local_path}' '{remote_dir}' --transfers=8 --checkers=8 --fast-list"
    rc = os.system(cmd)
    if rc != 0:
        print(f"!! rclone failed (code {rc}) for {local_path}")

# ===========================================================
# DIFFS (whole domain + sector)
# ===========================================================
def diffs_for_metric(metric):
    """Whole Antarctica |Δ(3-5)| and |Δ(7-5)| flattened."""
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    years = align_years([d3, d5, d7])
    A3 = stack_years(d3, years, f"{metric}_k3")
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")
    valid = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7))
    d35 = wrapped_diff_np((A3 - A5).values)
    d75 = wrapped_diff_np((A7 - A5).values)
    v35 = np.abs(d35[valid.values]).ravel()
    v75 = np.abs(d75[valid.values]).ravel()
    return v35, v75

def diffs_for_metric_sector_mask(metric, sector_mask_2d):
    """Sectoral |Δ(3-5)| and |Δ(7-5)| flattened (mask broadcast over years)."""
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    years = align_years([d3, d5, d7])
    A3 = stack_years(d3, years, f"{metric}_k3")  # (year,y,x)
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")
    valid = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7))
    sm = xr.DataArray(np.asarray(sector_mask_2d, dtype=bool), dims=("y", "x"))
    if sm.shape != valid.isel(year=0).shape:
        raise ValueError(f"Sector mask shape {sm.shape} does not match grid {valid.isel(year=0).shape}")

    sector_mask = sm.broadcast_like(valid)
    valid = valid & sector_mask
    d35 = wrapped_diff_np((A3 - A5).values)
    d75 = wrapped_diff_np((A7 - A5).values)
    v35 = np.abs(d35[valid.values]).ravel()
    v75 = np.abs(d75[valid.values]).ravel()
    return v35, v75

# ===========================================================
# DIAGNOSTIC
# ===========================================================
def debug_years(metric):
    """Prints year coverage and overlap for k3/k5/k7 for the metric."""
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    s3, s5, s7 = set(d3.keys()), set(d5.keys()), set(d7.keys())
    inter = s3 & s5 & s7
    def rng(s):
        return f"{min(s)}–{max(s)}" if s else "—"
    print(f"\n[{metric}] years per window:")
    print(f"  k3: {rng(s3)}  (n={len(s3)})")
    print(f"  k5: {rng(s5)}  (n={len(s5)})")
    print(f"  k7: {rng(s7)}  (n={len(s7)})")
    print(f"  overlap: n={len(inter)}  {sorted(list(inter))[:8]}{' …' if len(inter)>8 else ''}")
    if not inter:
        print("!! No overlapping years across k3/k5/k7 — check file availability / THRESH_PCT / paths.")

# ===========================================================
# POLAR MAP HELPERS (match climatology style)
# ===========================================================
def make_polar_axes(fig, position):
    """South polar stereographic axes with circular boundary, grey land, black ocean."""
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(position, projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    theta = np.linspace(0, 2*np.pi, 200)
    center = np.array([0.5, 0.5])
    radius = 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center
    circle = MplPath(verts)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(cfeature.OCEAN, facecolor="black", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="0.7", edgecolor="0.7", zorder=1)
    ax.coastlines(linewidth=0.3, zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="0.5",
                 alpha=0.5, linestyle="--")
    return ax

def compute_window_diff_means(metric):
    """
    Mean timing difference (k=3-5 and k=7-5) over all overlapping years,
    returned as two DataArrays with (y,x) + coords.
    """
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    years = align_years([d3, d5, d7])

    A3 = stack_years(d3, years, f"{metric}_k3")   # (year,y,x)
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")

    d35 = wrapped_diff_np((A3 - A5).values)
    d75 = wrapped_diff_np((A7 - A5).values)

    d35_da = xr.DataArray(d35, coords=A3.coords, dims=A3.dims).mean("year", skipna=True)
    d75_da = xr.DataArray(d75, coords=A3.coords, dims=A3.dims).mean("year", skipna=True)

    return d35_da, d75_da

def plot_window_diff_maps(dpi=300, vmax=15):
    """
    2x2 panel: FS and MS window sensitivity maps (k=3-5 and 7-5),
    static method, using same polar style as climatology.
    """
    print("\nComputing mean window differences for maps...")
    fs_d35, fs_d75 = compute_window_diff_means("FS")
    ms_d35, ms_d75 = compute_window_diff_means("MS")

    cmap = plt.cm.RdBu_r  # diverging, symmetric about 0

    fig = plt.figure(figsize=(8.2, 6.0))
    panels = [
        ("FS", fs_d35, "k = 3 − 5"),
        ("FS", fs_d75, "k = 7 − 5"),
        ("MS", ms_d35, "k = 3 − 5"),
        ("MS", ms_d75, "k = 7 − 5"),
    ]

    mappable = None
    for idx, (metric, da, label) in enumerate(panels, start=1):
        ax = make_polar_axes(fig, 220 + idx)  # 2 rows, 2 cols, position idx
        x = da["x"]
        y = da["y"]

        im = ax.pcolormesh(
            x, y, da,
            transform=ccrs.SouthPolarStereo(),
            cmap=cmap,
            vmin=-vmax, vmax=vmax
        )
        mappable = im  # last one is fine

        if metric == "FS":
            phase_name = "Freeze start (FS)"
        else:
            phase_name = "Melt start (MS)"

        ax.set_title(f"{phase_name}, {label}", fontsize=9)

    # Shared colorbar
    cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label("Timing difference relative to 5-day window (days)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    fig.suptitle(
        f"Window sensitivity of phase timing (static, thr={THRESH_PCT}%, {SENSOR})",
        fontsize=11
    )

    fig.tight_layout(rect=[0, 0.14, 1, 0.94])
    fname = f"{VERSION_TAG}_maps_window_FS_MS_{SENSOR}_thr{THRESH_PCT}.png"
    local_path = os.path.join(OUTDIR_FIGS, fname)
    fig.savefig(local_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # upload to Drive
    rclone_copy(local_path)
    print(f"✓ Saved and uploaded window-sensitivity maps: {local_path}")

# ===========================================================
# PLOTTING – CDFs (unchanged)
# ===========================================================
def plot_cdf_sectors_with_masks(metric, masks_dict, ncols=3, dpi=300,
                                panel_size=(2.4, 2.0)):
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=0.9)

    names = list(masks_dict.keys())
    n = len(names); nrows = ceil(n / ncols)

    fig_w = ncols * panel_size[0]
    fig_h = nrows * panel_size[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    legend_handles = None
    legend_labels  = ["3 vs 5-day window", "7 vs 5-day window"]

    for i, name in enumerate(names):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        v35, v75 = diffs_for_metric_sector_mask(metric, masks_dict[name])
        v35_clip, _ = ecdf_data(v35, MAX_X)
        v75_clip, _ = ecdf_data(v75, MAX_X)

        if v35_clip.size == 0 and v75_clip.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="0.4", fontsize=7)
        else:
            sns.ecdfplot(v35_clip, lw=1.4, ax=ax)
            sns.ecdfplot(v75_clip, lw=1.4, ax=ax)
            if legend_handles is None:
                legend_handles = [ax.lines[-2], ax.lines[-1]]

        ax.text(0.02, 0.96, name, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color="0.25")
        ax.grid(True, linestyle=":", linewidth=0.6, color="0.82")
        ax.tick_params(labelsize=7)

    # hide empties
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    for ax in axes.ravel():
        if ax.get_visible():
            ax.set_xlim(0, MAX_X)
            ax.set_ylim(0, 1.0)

    fig.supxlabel("Absolute timing difference (days)", fontsize=9, color="0.3")
    fig.supylabel("Cumulative fraction of pixels", fontsize=9, color="0.3")

    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels,
                   loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=2, frameon=True, fontsize=8,
                   title="Window comparison", title_fontsize=8)

    # no suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fname = f"{VERSION_TAG}_CDF_sectors_{metric}_{SENSOR}_thr{THRESH_PCT}.png"
    local_path = os.path.join(OUTDIR_FIGS, fname)
    fig.savefig(local_path, dpi=dpi)
    plt.close(fig)

    remote_dest = f"{RCLONE_REMOTE}:{RCLONE_PATH}/figures/"
    os.system(f"rclone copy '{local_path}' '{remote_dest}' --progress")
    print(f"✓ Saved and uploaded: {fname}")



def plot_cdf_single_sector(metric, name, mask_2d,
                           figsize=(5.0, 3.5), dpi=300, save=True):
    """Bigger, single-sector panel (no annotations)."""
    v35, v75 = diffs_for_metric_sector_mask(metric, mask_2d)
    v35_clip, _ = ecdf_data(v35, MAX_X)
    v75_clip, _ = ecdf_data(v75, MAX_X)
    fig, ax = plt.subplots(figsize=figsize)
    if v35_clip.size == 0 and v75_clip.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.4", fontsize=10)
    else:
        sns.ecdfplot(v35_clip, label="3 vs 5-day window", lw=2, ax=ax)
        sns.ecdfplot(v75_clip, label="7 vs 5-day window", lw=2, ax=ax)
        handles = [ax.lines[-2], ax.lines[-1]]
        ax.legend(handles=handles, labels=["3 vs 5-day window", "7 vs 5-day window"],
                  title="Window comparison", frameon=True, loc="lower right")
    ax.set_xlim(0, MAX_X); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Absolute timing difference (days)", fontsize=11, fontweight="bold", color="0.3")
    ax.set_ylabel("Cumulative Fraction of Pixels",    fontsize=11, fontweight="bold", color="0.3")
    ax.set_title(f"{name} • {metric}", fontsize=11, color="0.25", pad=3)
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.82")
    fig.tight_layout()
    if save:
        fname = f"{VERSION_TAG}_CDF_sector_{name.replace(' ', '_')}_{metric}_{SENSOR}_thr{THRESH_PCT}.png"
        local_path = os.path.join(OUTDIR_FIGS, fname)
        fig.savefig(local_path, dpi=dpi)
        plt.close(fig)

        remote_dest = f"{RCLONE_REMOTE}:{RCLONE_PATH}/figures/"
        os.system(f"rclone copy '{local_path}' '{remote_dest}' --progress")
        print(f"✓ Saved and uploaded: {fname}")
    else:
        plt.show()

# ===========================================================
# RUN
# ===========================================================
if __name__ == "__main__":
    # Build sector masks from canonical file
    SECTOR_MASKS = load_sector_masks(CANONICAL)

    # Individual sector CDF plots
    for metric in ["MS", "FS"]:
        print(f"\n=== Individual sector plots: {metric} ===")
        for name, mask in SECTOR_MASKS.items():
            plot_cdf_single_sector(metric, name, mask, figsize=(5.5, 4.0), dpi=300, save=True)

    # Diagnostics
    debug_years("MS")
    debug_years("FS")

    # Faceted CDF figures
    plot_cdf_sectors_with_masks("MS", SECTOR_MASKS, ncols=3)
    plot_cdf_sectors_with_masks("FS", SECTOR_MASKS, ncols=3)

    # NEW: 2x2 map panel for window sensitivity (FS & MS)
    plot_window_diff_maps(dpi=300, vmax=15)
