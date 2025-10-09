# cdf_sectoral_window_sensitivity_MS_FS_diagnostics.py
# ---------------------------------------------------
# Purpose
#   - Reproduces your ECDF figures for MS/FS across Antarctic sectors
#   - Adds hard diagnostics to verify masks, coverage, DOY wrapping, and distribution differences
#
# Usage
#   - Edit USER CONFIG below as needed
#   - Run: python cdf_sectoral_window_sensitivity_MS_FS_diagnostics.py
#   - Toggle which diagnostics/plots run at the bottom (__main__)
#
# Notes
#   - Uses dask-backed xarray (no .load()); computes after masking
#   - All new helpers are clearly commented; you can prune later
# ---------------------------------------------------

import os, re, glob, warnings
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil

# Optional (for KS test). If not installed, code will skip KS diagnostics gracefully.
try:
    from scipy.stats import ks_2samp
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ----------------- USER CONFIG -----------------
SENSOR       = "SMMR"      # "SMMR" or "AMSRE"
THRESH_PCT   = 15          # e.g., 10/15/20
INPUT_ROOT   = f"/user/geog/falejandraperez/sea-ice-phase/results/{SENSOR}_phase"
RCLONE_DEST  = f"gdrive:sea-ice-phase/results/figures/cdf_ms_fs/{SENSOR}_thr{THRESH_PCT}"

PERIOD       = 366         # DOY wrap (0..365)
MAX_X        = 30          # x-limit in days for |Δ|

NSIDC_SAMPLE = "/user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/raw/NSIDC0079_SEAICE_PS_S25km_20241228_v4.0.nc"
LATLON_NPZ   = "/user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/raw/latlon_grid_25km_epsg3412.npz"

# 5 sectors (lat, lon in degrees). Longitudes may wrap.
SECTORS = {
    "East Antarctic":          dict(lats=(-90, -50), lons=( 70, 170)),
    "King Hakon VII":          dict(lats=(-90, -50), lons=(-10,  70)),
    "Ross–Amundsen":           dict(lats=(-90, -50), lons=(165, 250)),
    "Amundsen–Bellingshausen": dict(lats=(-90, -50), lons=(250, 290)),
    "Weddell":                 dict(lats=(-90, -50), lons=(290, 349)),
}

# Aesthetics
sns.set_context("talk")
sns.set_style("whitegrid")

# ----------------- HELPERS: file/years -----------------
year_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    """Extract 4-digit year from filename suffix ..._YYYY.nc; return int or None."""
    m = year_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def load_window_dict(metric, kdays):
    """
    Load a dict of {year: DataArray} for a given metric ('MS'/'FS') and window (3/5/7).
    - Dask-backed (no .load()) for memory efficiency.
    - Validates variable presence.
    """
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
            # Chunk moderately; adjust to your machine
            ds = xr.open_dataset(f, chunks={"y": 400, "x": 400})
            if metric not in ds:
                print(f"Skip {f}: '{metric}' variable missing")
                continue
            d[yr] = ds[metric]
        except Exception as e:
            print(f"Skip {f}: {e}")
    if not d:
        raise FileNotFoundError(f"No {metric} files in {folder}")
    return d

def align_years(dicts):
    """Return sorted list of overlapping years across multiple {year: DataArray} dicts."""
    common = set.intersection(*[set(d.keys()) for d in dicts])
    years = sorted(common)
    if not years:
        raise ValueError("No overlapping years across windows.")
    return years

def stack_years(d, years, name):
    """Stack {year: DataArray(y,x)} into DataArray(year,y,x) with a name."""
    arrs = [d[y].expand_dims(year=[y]) for y in years]
    out = xr.concat(arrs, dim="year")
    out.name = name
    return out

# ----------------- CORE MATH -----------------
def wrapped_diff_np(a_minus_b, period=PERIOD):
    """
    Wrap differences on a circular DOY axis.
    Returns values in [-period/2, +period/2) so abs diff <= period/2.
    """
    return ((a_minus_b + period//2) % period) - (period//2)

def ecdf_data(abs_diff, max_x=MAX_X):
    """Drop NaN and clip to <= max_x for plotting and comparable summaries."""
    abs_diff = abs_diff[~np.isnan(abs_diff)]
    return abs_diff[abs_diff <= max_x], abs_diff

# ----------------- LAT/LON GRID -----------------
def ensure_latlon_npz(nsidc_path=NSIDC_SAMPLE, out_npz=LATLON_NPZ):
    """
    Create a lat/lon cache from an EPSG:3412 grid file if missing.
    - Avoids reprojecting for every run.
    """
    if os.path.exists(out_npz):
        return out_npz
    try:
        import pyproj
    except ImportError:
        raise RuntimeError("pyproj is required. Install: pip install --user pyproj")
    ds = xr.open_dataset(nsidc_path)
    x = ds["x"].values  # (nx,)
    y = ds["y"].values  # (ny,)
    X, Y = np.meshgrid(x, y)  # (ny,nx) == (y,x)
    tfm = pyproj.Transformer.from_crs("EPSG:3412", "EPSG:4326", always_xy=True)
    lon, lat = tfm.transform(X, Y)  # (y,x)
    np.savez(out_npz, lat=lat, lon=lon)
    print(f"✓ wrote lat/lon grids to {out_npz}  shape={lat.shape}")
    return out_npz

def build_sector_masks_from_npz(latlon_npz_path, sectors=SECTORS):
    """
    Build boolean sector masks {name -> (y,x)} from cached lat/lon.
    Handles longitude wrap-around robustly by using 0–360.
    """
    g = np.load(latlon_npz_path)
    lat = g["lat"]; lon = g["lon"]
    lon360 = (lon + 360.0) % 360.0

    def lonmask(lo, hi):
        lo = lo % 360.0; hi = hi % 360.0
        if lo <= hi:
            return (lon360 >= lo) & (lon360 <= hi)
        else:
            return (lon360 >= lo) | (lon360 <= hi)

    masks = {}
    for name, spec in sectors.items():
        lat_lo, lat_hi = spec["lats"]; lon_lo, lon_hi = spec["lons"]
        m = (lat >= min(lat_lo, lat_hi)) & (lat <= max(lat_lo, lat_hi))
        m &= lonmask(lon_lo, lon_hi)
        masks[name] = m.astype(bool)
    return masks

# ----------------- DIFFS (whole domain + sector) -----------------
def diffs_for_metric(metric):
    """
    Whole Antarctica |Δ(3-5)| and |Δ(7-5)| flattened (valid where all three windows are finite).
    """
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    years = align_years([d3, d5, d7])
    A3 = stack_years(d3, years, f"{metric}_k3")  # (year,y,x)
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")
    valid = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7))

    d35 = wrapped_diff_np((A3 - A5))
    d75 = wrapped_diff_np((A7 - A5))

    # compute only once at the end
    v35 = np.abs(d35.where(valid).values).ravel()
    v75 = np.abs(d75.where(valid).values).ravel()
    v35 = v35[~np.isnan(v35)]
    v75 = v75[~np.isnan(v75)]
    return v35, v75

def diffs_for_metric_sector_mask(metric, sector_mask_2d):
    """
    Sectoral |Δ(3-5)| and |Δ(7-5)| flattened (mask broadcast over years).
    """
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    years = align_years([d3, d5, d7])
    A3 = stack_years(d3, years, f"{metric}_k3")  # (year,y,x)
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")

    sm = xr.DataArray(np.asarray(sector_mask_2d, dtype=bool), dims=("y","x"))
    if sm.shape != A5.isel(year=0).shape:
        raise ValueError(f"Sector mask shape {sm.shape} does not match grid {A5.isel(year=0).shape}")

    valid = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7)) & sm

    d35 = wrapped_diff_np((A3 - A5))
    d75 = wrapped_diff_np((A7 - A5))

    v35 = np.abs(d35.where(valid).values).ravel()
    v75 = np.abs(d75.where(valid).values).ravel()
    v35 = v35[~np.isnan(v35)]
    v75 = v75[~np.isnan(v75)]
    return v35, v75

# ----------------- DIAGNOSTICS -----------------
def diagnose_masks(latlon_npz_path=LATLON_NPZ, sectors=SECTORS):
    """
    1) Print grid extents
    2) Pixel counts per sector mask
    3) Overlap diagnostics (none or tiny is expected)
    4) Quick hashes so we can see masks are truly different
    """
    g = np.load(latlon_npz_path)
    lat = g["lat"]; lon = g["lon"]
    H, W = lat.shape
    print(f"grid shape: {H}x{W}, lat[{lat.min():.2f},{lat.max():.2f}], lon[{lon.min():.2f},{lon.max():.2f}]")

    masks = build_sector_masks_from_npz(latlon_npz_path, sectors)
    counts = {k: int(m.sum()) for k, m in masks.items()}
    print("\nSector pixel counts (mask only):")
    for k, v in counts.items():
        print(f"  {k:28s} {v:,}")

    M = np.zeros((H, W), dtype=int)
    for _, m in masks.items():
        M += m.astype(int)
    print(f"\nPixels in no sector: {(M==0).sum():,}")
    print(f"Pixels in >1 sector: {(M>1).sum():,}  (near-zero is expected)")

    for k, m in masks.items():
        print(f"  {k:28s} hash={hash(m.tobytes())}")
    return masks

def sector_valid_coverage(metric, masks_dict):
    """
    For k=5 (reference), report how many pixel-years are valid per sector.
    If these are near-identical or tiny, that explains similar ECDFs.
    """
    d5 = load_window_dict(metric, 5)
    years = sorted(d5.keys())
    A5 = stack_years(d5, years, f"{metric}_k5")
    vmap = ~np.isnan(A5)  # (year,y,x)

    print(f"\n[{metric}] valid coverage (k5):")
    for name, m in masks_dict.items():
        sm = xr.DataArray(np.asarray(m, bool), dims=("y","x"))
        n_all = int(sm.sum().values)
        n_valid = int((vmap & sm).sum().values)
        frac = n_valid / (n_all * vmap.sizes["year"]) if n_all>0 else np.nan
        print(f"  {name:28s} valid px-yrs={n_valid:,}  (mask px={n_all:,}, years={vmap.sizes['year']})  frac={frac:.3f}")

def test_wrapped_diff():
    """
    Quick sanity for DOY wrap. Expect small diffs near year boundary.
    """
    arr = np.array([0, 1, 365, 364, 10, 350])   # pretend actual DOYs
    ref = np.array([365, 365,   0,   0, 12, 349])  # reference DOYs
    dd  = wrapped_diff_np(arr - ref, period=PERIOD)
    print("\nwrapped_diff test (arr, ref, diff, |diff|):")
    print(list(zip(arr, ref, dd, np.abs(dd))))

def sector_summary(metric, masks_dict, max_x=MAX_X):
    """
    Descriptive stats for |Δ| pooled (Δ3–5 + Δ7–5) within each sector.
    If these match closely, the similarity is real (or upstream).
    """
    print(f"\n[{metric}] |Δ| summary (clipped at ≤{max_x} days):")
    hdr = f"{'sector':28s} {'n':>9s} {'p50':>7s} {'p90':>7s} {'p95':>7s} {'p99':>7s} {'mean':>7s}"
    print(hdr); print("-"*len(hdr))
    for name, m in masks_dict.items():
        v35, v75 = diffs_for_metric_sector_mask(metric, m)
        v = np.concatenate([v35, v75])
        v = v[(~np.isnan(v)) & (v <= max_x)]
        if v.size == 0:
            print(f"{name:28s} {'0':>9} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7}")
            continue
        p = np.percentile(v, [50, 90, 95, 99])
        print(f"{name:28s} {v.size:9d} {p[0]:7.2f} {p[1]:7.2f} {p[2]:7.2f} {p[3]:7.2f} {v.mean():7.2f}")

def ks_matrix(metric, masks_dict, which="35"):
    """
    Pairwise Kolmogorov–Smirnov distances between sector ECDFs for |Δ(3–5)| or |Δ(7–5)|.
    Small KS + large p-values -> distributions indistinguishable.
    """
    if not HAVE_SCIPY:
        print("KS diagnostics skipped (scipy not installed).")
        return
    names = list(masks_dict.keys())
    vals = {}
    for name, m in masks_dict.items():
        v35, v75 = diffs_for_metric_sector_mask(metric, m)
        v = v35 if which == "35" else v75
        v = v[(~np.isnan(v)) & (v <= MAX_X)]
        vals[name] = v
    print(f"\n[{metric}] KS distances for |Δ({which[0]}–{which[1]})|:")
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if vals[a].size and vals[b].size:
                stat, p = ks_2samp(vals[a], vals[b])
                print(f"  {a:28s} vs {b:28s}  KS={stat:.3f}  p={p:.3g}")

def peek_file(fpath, var):
    """
    Peek at one NetCDF to confirm variable range and dtype look like DOY (0..365).
    """
    try:
        with xr.open_dataset(fpath) as ds:
            if var not in ds:
                print(f"{os.path.basename(fpath)}: '{var}' missing")
                return
            da = ds[var]
            vmin = float(da.min().values)
            vmax = float(da.max().values)
            print(f"peek {os.path.basename(fpath)}  {var}: dtype={da.dtype} shape={tuple(da.shape)} range=[{vmin:.1f},{vmax:.1f}]")
    except Exception as e:
        print(f"peek failed {fpath}: {e}")

# ----------------- PLOTTING -----------------
def plot_cdf_sectors_with_masks(metric, masks_dict, ncols=3, dpi=300,
                                panel_size=(3.2, 2.6)):
    """
    Faceted plot per sector (shared axes; figure-level legend).
    """
    names = list(masks_dict.keys())
    n = len(names); nrows = ceil(n / ncols)

    fig_w = ncols * panel_size[0]
    fig_h = nrows * panel_size[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    print(f"\n[{metric}] sector pixel counts (valid & in mask):")
    print("sector".ljust(28), "n_pixels")

    legend_handles = None
    legend_labels  = ["3 vs 5-day window", "7 vs 5-day window"]

    for i, name in enumerate(names):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        v35, v75 = diffs_for_metric_sector_mask(metric, masks_dict[name])
        print(name.ljust(28), f"{(v35.size + v75.size)//2:,}")

        v35_clip, _ = ecdf_data(v35, MAX_X)
        v75_clip, _ = ecdf_data(v75, MAX_X)

        if v35_clip.size == 0 and v75_clip.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="0.4", fontsize=9)
        else:
            sns.ecdfplot(v35_clip, label=legend_labels[0], lw=2, ax=ax)
            sns.ecdfplot(v75_clip, label=legend_labels[1], lw=2, ax=ax)
            if legend_handles is None:
                legend_handles = [ax.lines[-2], ax.lines[-1]]

        ax.set_title(name, fontsize=10, pad=3, color="0.25")
        ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.82")

    total = nrows * ncols
    for j in range(n, total):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    for ax in axes.ravel():
        if ax.get_visible():
            ax.set_xlim(0, MAX_X)
            ax.set_ylim(0, 1.0)
            ax.tick_params(labelsize=9)

    fig.supxlabel("Absolute timing difference (days)", fontsize=11, fontweight="bold", color="0.3")
    fig.supylabel("Cumulative Fraction of Pixels",    fontsize=11, fontweight="bold", color="0.3")

    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, title="Window comparison",
                   loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=2, frameon=True)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    fname = f"CDF_sectors_{metric}_{SENSOR}_thr{THRESH_PCT}.png"
    local_path = f"/tmp/{fname}"
    fig.savefig(local_path, dpi=dpi)
    plt.close(fig)
    os.system(f"rclone copy '{local_path}' '{RCLONE_DEST}'")
    print(f"✓ Uploaded: {fname}")

def plot_cdf_single_sector(metric, name, mask_2d,
                           figsize=(5.5, 4.0), dpi=300, save=True):
    """
    Larger single-sector ECDF panel.
    """
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
        fname = f"CDF_sector_{name.replace(' ','_')}_{metric}_{SENSOR}_thr{THRESH_PCT}.png"
        local_path = f"/tmp/{fname}"
        fig.savefig(local_path, dpi=dpi)
        plt.close(fig)
        os.system(f"rclone copy '{local_path}' '{RCLONE_DEST}'")
        print(f"✓ Uploaded: {fname}")
    else:
        plt.show()

# ----------------- QUICK DIAGNOSTIC: years -----------------
def debug_years(metric):
    """
    Print year coverage and overlap for k3/k5/k7 for the metric.
    Helps catch missing or mismatched year sets across windows.
    """
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    s3, s5, s7 = set(d3.keys()), set(d5.keys()), set(d7.keys())
    inter = s3 & s5 & s7
    def rng(s): return f"{min(s)}–{max(s)}" if s else "—"
    print(f"\n[{metric}] years per window:")
    print(f"  k3: {rng(s3)}  (n={len(s3)})")
    print(f"  k5: {rng(s5)}  (n={len(s5)})")
    print(f"  k7: {rng(s7)}  (n={len(s7)})")
    print(f"  overlap: n={len(inter)}  {sorted(list(inter))[:8]}{' …' if len(inter)>8 else ''}")
    if not inter:
        print("!! No overlapping years across k3/k5/k7 — check file availability / THRESH_PCT / paths.")

# ----------------- RUN -----------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", ".*converting a masked element to nan.*")

    # 0) Optional: spot-check a file's range looks like DOY
    # Example: peek a k5 FS file (edit path if needed)
    # peek_file(os.path.join(INPUT_ROOT, f"FS_thr{THRESH_PCT}_k5", "FS_1980.nc"), "FS")

    # 1) Ensure lat/lon cache exists
    ensure_latlon_npz(NSIDC_SAMPLE, LATLON_NPZ)

    # 2) Build masks + diagnose them (distinctness, overlaps)
    SECTOR_MASKS = diagnose_masks(LATLON_NPZ)

    # 3) Quick diagnostics on year availability
    debug_years("MS")
    debug_years("FS")

    # 4) Validate data footprint per sector (are we sampling different pixels?)
    sector_valid_coverage("MS", SECTOR_MASKS)
    sector_valid_coverage("FS", SECTOR_MASKS)

    # 5) DOY wrap sanity
    test_wrapped_diff()

    # 6) Summaries (are sectors numerically different at all?)
    sector_summary("MS", SECTOR_MASKS, max_x=MAX_X)
    sector_summary("FS", SECTOR_MASKS, max_x=MAX_X)

    # 7) KS distances (optional but useful)
    ks_matrix("MS", SECTOR_MASKS, which="35")
    ks_matrix("MS", SECTOR_MASKS, which="75")
    ks_matrix("FS", SECTOR_MASKS, which="35")
    ks_matrix("FS", SECTOR_MASKS, which="75")

    # 8) Produce single-sector panels (same as your originals)
    for metric in ["MS", "FS"]:
        print(f"\n=== Individual sector plots: {metric} ===")
        for name, mask in SECTOR_MASKS.items():
            plot_cdf_single_sector(metric, name, mask, figsize=(5.5, 4.0), dpi=300, save=True)

    # 9) Faceted figures (multi-panel)
    plot_cdf_sectors_with_masks("MS", SECTOR_MASKS, ncols=3)
    plot_cdf_sectors_with_masks("FS", SECTOR_MASKS, ncols=3)

    # 10) Optional: zoom into a single sector for detail
    # plot_cdf_single_sector("MS", "Weddell", SECTOR_MASKS["Weddell"])
    # plot_cdf_single_sector("FS", "East Antarctic", SECTOR_MASKS["East Antarctic"])

    # 11) Optional: per-year ECDF comparison across sectors (uncomment & pick a year)
    # ax = ecdf_one_year("FS", SECTOR_MASKS, year=1980); plt.show()
