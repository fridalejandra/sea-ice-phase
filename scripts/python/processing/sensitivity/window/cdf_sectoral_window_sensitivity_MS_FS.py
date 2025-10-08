# cdf_window_sensitivity_MS_FS_SECTORS.py
import os, re, glob
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil

# ----------------- USER CONFIG -----------------
SENSOR       = "SMMR"      # "SMMR" or "AMSRE"
THRESH_PCT   = 15          # e.g., 10/15/20
INPUT_ROOT   = f"/user/geog/falejandraperez/sea-ice-phase/results/{SENSOR}_phase"
RCLONE_DEST  = f"gdrive:sea-ice-phase/results/figures/cdf_ms_fs/{SENSOR}_thr{THRESH_PCT}"
PERIOD       = 366         # DOY wrap
MAX_X        = 30          # plot x-limit in days for |Δ|
MARKS        = [2, 5, 10]  # (unused in sector figs since we skip annotations here)
sns.set_context("talk"); sns.set_style("whitegrid")

# NSIDC daily file (ANY one on the same 25 km SH grid) used to derive lat/lon once.
NSIDC_SAMPLE = "/user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/raw/NSIDC0079_SEAICE_PS_S25km_20241228_v4.0.nc"
LATLON_NPZ   = "/user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/raw/latlon_grid_25km_epsg3412.npz"  # will be created if not present

# ----------------- SECTORS (lat, lon in degrees) -----------------
SECTORS = {
    "East Antarctic":          dict(lats=(-90, -50), lons=( 70, 170)),
    "King Hakon VII":          dict(lats=(-90, -50), lons=(-10,  70)),
    "Ross–Amundsen":           dict(lats=(-90, -50), lons=(165, 250)),
    "Amundsen–Bellingshausen": dict(lats=(-90, -50), lons=(250, 290)),
    "Weddell":                 dict(lats=(-90, -50), lons=(290, 349)),
}

# ----------------- HELPERS (your originals) -----------------
year_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    m = year_re.search(os.path.basename(path))
    if not m: return None
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
    return ((a_minus_b + period//2) % period) - (period//2)

def ecdf_data(abs_diff, max_x=MAX_X):
    abs_diff = abs_diff[~np.isnan(abs_diff)]
    return abs_diff[abs_diff <= max_x], abs_diff

def frac_within(arr, m):
    if arr.size == 0: return np.nan
    return (arr <= m).mean() * 100.0

# ----------------- BASELINE: whole-Antarctica CDF (kept for reference) -----------------
def diffs_for_metric(metric):
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    years = align_years([d3, d5, d7])
    A3 = stack_years(d3, years, f"{metric}_k3")  # (year,y,x)
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")

    mask = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7))
    d35 = wrapped_diff_np((A3 - A5).values)
    d75 = wrapped_diff_np((A7 - A5).values)
    v35 = np.abs(d35[mask.values]).ravel()
    v75 = np.abs(d75[mask.values]).ravel()
    return v35, v75

# ----------------- NEW: build lat/lon once from NSIDC grid -----------------
def ensure_latlon_npz(nsidc_path=NSIDC_SAMPLE, out_npz=LATLON_NPZ):
    if os.path.exists(out_npz):
        return out_npz
    # Derive lat/lon from EPSG:3412 using pyproj
    try:
        import pyproj
    except ImportError:
        raise RuntimeError("pyproj is required to derive lat/lon. Install with: pip install pyproj")

    ds = xr.open_dataset(nsidc_path)
    x = ds["x"].values  # (316,)
    y = ds["y"].values  # (332,)
    X, Y = np.meshgrid(x, y)  # (332,316)

    tfm = pyproj.Transformer.from_crs("EPSG:3412", "EPSG:4326", always_xy=True)
    lon, lat = tfm.transform(X, Y)  # each (332,316)

    np.savez(out_npz, lat=lat, lon=lon)
    print(f"✓ wrote lat/lon grids to {out_npz}  shape={lat.shape}")
    return out_npz

# ----------------- NEW: build sector masks from lat/lon -----------------
def build_sector_masks_from_npz(latlon_npz_path, sectors=SECTORS):
    g = np.load(latlon_npz_path)
    lat = g["lat"]        # (y,x)
    lon = g["lon"]        # (y,x) in [-180,180]
    lon360 = (lon + 360.0) % 360.0

    def lonrange_mask(lo, hi):
        lo = lo % 360.0; hi = hi % 360.0
        if lo <= hi:
            return (lon360 >= lo) & (lon360 <= hi)
        else:
            return (lon360 >= lo) | (lon360 <= hi)

    masks = {}
    for name, spec in sectors.items():
        lat_lo, lat_hi = spec["lats"]; lon_lo, lon_hi = spec["lons"]
        m = (lat >= min(lat_lo, lat_hi)) & (lat <= max(lat_lo, lat_hi))
        m &= lonrange_mask(lon_lo, lon_hi)
        masks[name] = m.astype(bool)
    return masks

# ----------------- NEW: sector diffs + faceted plot (NO annotations) -----------------
def diffs_for_metric_sector_mask(metric, sector_mask_2d):
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    years = align_years([d3, d5, d7])
    A3 = stack_years(d3, years, f"{metric}_k3")  # (year,y,x)
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")

    valid = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7))  # xr (year,y,x)
    sm = xr.DataArray(sector_mask_2d, dims=("y","x"))
    if sm.shape != valid.isel(year=0).shape:
        raise ValueError(f"Sector mask shape {sm.shape} does not match grid {valid.isel(year=0).shape}")
    valid &= sm

    d35 = wrapped_diff_np((A3 - A5).values)
    d75 = wrapped_diff_np((A7 - A5).values)
    v35 = np.abs(d35[valid.values]).ravel()
    v75 = np.abs(d75[valid.values]).ravel()
    return v35, v75

def plot_cdf_sectors_with_masks(metric, masks_dict, ncols=3, dpi=300,
                                panel_size=(3.2, 2.6), grey_bold_axes=True):
    names = list(masks_dict.keys())
    n = len(names); nrows = ceil(n / ncols)
    fig_w = ncols * panel_size[0]; fig_h = nrows * panel_size[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    lines_for_legend = None
    for i, name in enumerate(names):
        r, c = divmod(i, ncols); ax = axes[r, c]
        v35, v75 = diffs_for_metric_sector_mask(metric, masks_dict[name])
        v35_clip, _ = ecdf_data(v35, MAX_X)
        v75_clip, _ = ecdf_data(v75, MAX_X)

        if v35_clip.size == 0 and v75_clip.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.4", fontsize=9)
            ax.set_xlim(0, MAX_X); ax.set_ylim(0, 1.0)
            ax.set_title(name, fontsize=10, pad=3, color="0.25")
            continue

        l1 = sns.ecdfplot(v35_clip, label="3 vs 5-day window", lw=2, ax=ax)
        l2 = sns.ecdfplot(v75_clip, label="7 vs 5-day window", lw=2, ax=ax)
        if lines_for_legend is None:
            lines_for_legend = (l1, l2)

        ax.set_xlim(0, MAX_X); ax.set_ylim(0, 1.0)
        if grey_bold_axes:
            ax.set_xlabel("Absolute timing difference (days)", fontsize=10, fontweight="bold", color="0.3")
            ax.set_ylabel("Cumulative Fraction of Pixels",    fontsize=10, fontweight="bold", color="0.3")
        else:
            ax.set_xlabel("Absolute timing difference (days)")
            ax.set_ylabel("Cumulative Fraction of Pixels")
        ax.set_title(name, fontsize=10, pad=3, color="0.25")
        ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.82")

    # hide empties (if any)
    total = nrows * ncols
    for j in range(n, total):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    # one figure-level legend
    if lines_for_legend is not None:
        fig.legend(lines_for_legend, ["3 vs 5-day window", "7 vs 5-day window"],
                   title="Window comparison", loc="upper right",
                   bbox_to_anchor=(0.98, 0.98), frameon=True)

    fig.tight_layout()
    fname = f"CDF_sectors_{metric}_{SENSOR}_thr{THRESH_PCT}.png"
    local_path = f"/tmp/{fname}"
    fig.savefig(local_path, dpi=dpi)
    plt.close(fig)
    os.system(f"rclone copy '{local_path}' '{RCLONE_DEST}'")
    print(f"✓ Uploaded: {fname}")

# ----------------- RUN -----------------
if __name__ == "__main__":
    # 1) Ensure we have lat/lon cached (once)
    ensure_latlon_npz(NSIDC_SAMPLE, LATLON_NPZ)

    # 2) Build sector masks from lat/lon
    SECTOR_MASKS = build_sector_masks_from_npz(LATLON_NPZ)

    # 3) Make the faceted figures (no annotations)
    plot_cdf_sectors_with_masks("MS", SECTOR_MASKS, ncols=3)  # me_
