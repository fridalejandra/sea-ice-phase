#!/usr/bin/env python3
# sector_mask_diagnostics.py
# ------------------------------------------------------------
# Purpose:
#   Prove that sector masks are correct and actually applied.
#   - Build masks from a cached lat/lon grid
#   - Check sizes, overlaps, and alignment with data files
#   - Print lon/lat stats and random sample coords per sector
#   - Compare valid pixel-year coverage per sector
#   - Save per-sector binary mask PNGs (quick eyeball check)
#
# Dependencies:
#   numpy, xarray, matplotlib, (optional) pyproj for building the npz
# ------------------------------------------------------------

import os, re, glob, warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- USER CONFIG -----------------
SENSOR       = "SMMR"      # "SMMR" or "AMSRE"
THRESH_PCT   = 15
INPUT_ROOT   = f"/user/geog/falejandraperez/sea-ice-phase/results/{SENSOR}_phase"

NSIDC_SAMPLE = "/user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/raw/NSIDC0079_SEAICE_PS_S25km_20241228_v4.0.nc"
LATLON_NPZ   = "/user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/raw/latlon_grid_25km_epsg3412.npz"

OUT_DIR      = Path(f"/tmp/sector_mask_diagnostics/{SENSOR}_thr{THRESH_PCT}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sectors (lat, lon in degrees). Longitudes may wrap.
SECTORS = {
    "East Antarctic":          dict(lats=(-90, -50), lons=( 70, 170)),
    "King Hakon VII":          dict(lats=(-90, -50), lons=(-10,  70)),
    "Ross–Amundsen":           dict(lats=(-90, -50), lons=(165, 250)),
    "Amundsen–Bellingshausen": dict(lats=(-90, -50), lons=(250, 290)),
    "Weddell":                 dict(lats=(-90, -50), lons=(290, 349)),
}

# ----------------- UTILS -----------------
year_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    m = year_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def ensure_latlon_npz(nsidc_path=NSIDC_SAMPLE, out_npz=LATLON_NPZ):
    """Build a lat/lon cache (npz) from an EPSG:3412 reference file if missing."""
    if os.path.exists(out_npz):
        return out_npz
    try:
        import pyproj
    except ImportError:
        raise RuntimeError("pyproj is required to build the cache. Install: pip install --user pyproj")
    ds = xr.open_dataset(nsidc_path)
    x = ds["x"].values
    y = ds["y"].values
    X, Y = np.meshgrid(x, y)  # (y,x)
    tfm = pyproj.Transformer.from_crs("EPSG:3412", "EPSG:4326", always_xy=True)
    lon, lat = tfm.transform(X, Y)
    np.savez(out_npz, lat=lat, lon=lon)
    print(f"✓ wrote lat/lon grids to {out_npz}  shape={lat.shape}")
    return out_npz

def build_sector_masks_from_npz(latlon_npz_path, sectors=SECTORS):
    """Return dict[name -> bool (y,x)] sector masks built from cached lat/lon, handling lon wrap."""
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
    return masks, lat, lon

def open_any_data(metric="FS", kdays=5):
    """
    Open one k=5 file (reference) for shape/NaN checks.
    We just need shape + valid mask; pick the first available year.
    """
    folder = os.path.join(INPUT_ROOT, f"{metric}_thr{THRESH_PCT}_k{kdays}")
    files = sorted(glob.glob(os.path.join(folder, f"{metric}_*.nc")))
    if not files:
        raise FileNotFoundError(f"No files in {folder}")
    f = files[0]
    with xr.open_dataset(f) as ds:
        if metric not in ds:
            raise KeyError(f"{metric} not in {f}")
        da = ds[metric]
        shape = tuple(da.shape)
        vmin = float(da.min().values) if np.isfinite(da.min()) else np.nan
        vmax = float(da.max().values) if np.isfinite(da.max()) else np.nan
        print(f"sample file: {os.path.basename(f)}  var={metric}  shape={shape}  range=[{vmin:.1f},{vmax:.1f}]")
        valid = ~np.isnan(da)
    return shape, valid

def plot_mask(mask, title, out_png, alpha=0.9):
    """Save a binary mask as a PNG for quick eyeballing (imshow in grid coordinates)."""
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.imshow(mask, origin="lower", interpolation="nearest", alpha=alpha)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def rand_coords(mask, lat, lon, n=6, seed=42):
    """Return up to n random (lat,lon) from True pixels of mask (for spot checks)."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return []
    rng = np.random.default_rng(seed)
    sel = idx[rng.choice(idx.shape[0], size=min(n, idx.shape[0]), replace=False)]
    out = []
    for y, x in sel:
        out.append((float(lat[y, x]), float(lon[y, x])))
    return out

# ----------------- MAIN DIAGNOSTIC -----------------
def main():
    warnings.filterwarnings("ignore", ".*converting a masked element to nan.*")

    # 1) Ensure we have lat/lon (or build it once)
    ensure_latlon_npz(NSIDC_SAMPLE, LATLON_NPZ)

    # 2) Build masks
    masks, lat, lon = build_sector_masks_from_npz(LATLON_NPZ, SECTORS)
    H, W = lat.shape
    print(f"grid shape (lat/lon): {H}x{W}")
    print(f"lat range: [{lat.min():.2f}, {lat.max():.2f}]  lon range: [{lon.min():.2f}, {lon.max():.2f}]")

    # 3) Save each mask as quick-look PNG + print counts and lon/lat stats
    total_union = np.zeros((H, W), dtype=int)
    for name, m in masks.items():
        total_union += m.astype(int)

        count = int(m.sum())
        if count == 0:
            print(f"[{name}] mask has ZERO pixels — check sector lon/lat bounds")
        lat_vals = lat[m]; lon_vals = lon[m]
        lon360 = (lon_vals + 360.0) % 360.0

        print(f"\n{name}")
        print(f"  pixels in mask: {count:,}")
        print(f"  lat  median={np.nanmedian(lat_vals):6.2f}  q5={np.nanpercentile(lat_vals,5):6.2f}  q95={np.nanpercentile(lat_vals,95):6.2f}")
        print(f"  lon°E median={np.nanmedian(lon360):6.2f}  q5={np.nanpercentile(lon360,5):6.2f}  q95={np.nanpercentile(lon360,95):6.2f}")

        # random coordinate samples for eyeballing
        samples = rand_coords(m, lat, lon, n=6)
        for j, (la, lo) in enumerate(samples, 1):
            lo360 = (lo + 360) % 360
            print(f"    sample{j}: lat={la:7.3f}, lon={lo:8.3f}  (lon°E={lo360:7.3f})")

        # write mask image
        out_png = OUT_DIR / f"mask_{name.replace(' ','_')}.png"
        plot_mask(m, f"{name} (mask)", out_png)
        print(f"  ✓ wrote {out_png}")

    # 4) Overlap & coverage diagnostics (union/overlap counts)
    n_unassigned = int((total_union == 0).sum())
    n_overlap    = int((total_union > 1).sum())
    print(f"\nPixels in NO sector: {n_unassigned:,}")
    print(f"Pixels in >1 sector: {n_overlap:,} (should be ~0; small boundary overlaps are acceptable)")

    # 5) Check alignment with data grid: open one FS k=5 file
    shape, valid = open_any_data(metric="FS", kdays=5)
    if shape[-2:] != (H, W):
        raise ValueError(f"Data grid shape {shape[-2:]} != lat/lon grid shape {(H,W)}. Check the reference file & grid.")

    # 6) For each sector: how many valid (non-NaN) pixels do we actually have in k=5?
    valid = valid.compute() if hasattr(valid.data, "compute") else valid  # dask -> numpy if needed
    print("\n[FS k=5] valid pixel counts per sector (single sample year):")
    for name, m in masks.items():
        n_mask   = int(m.sum())
        n_valid  = int((valid.values & m).sum())
        frac     = n_valid / n_mask if n_mask else np.nan
        print(f"  {name:28s} mask px={n_mask:7d}  valid px={n_valid:7d}  frac={frac:5.2f}")

    # 7) Optional: write a composite figure with all masks tiled
    cols = 3
    names = list(masks.keys())
    rows = int(np.ceil(len(names)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4.2, rows*3.6), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(masks[name], origin="lower", interpolation="nearest")
        ax.set_title(name, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    # blank any unused
    for j in range(len(names), rows*cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")
    fig.suptitle(f"Sector masks • {SENSOR} thr{THRESH_PCT}", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    composite = OUT_DIR / "masks_composite.png"
    fig.savefig(composite, dpi=200)
    plt.close(fig)
    print(f"✓ wrote {composite}")

    print("\nDone. If counts, lon/lat stats, and valid fractions look sensible—and the PNGs look distinct—you are truly slicing sectors.")

# ----------------- ENTRY -----------------
if __name__ == "__main__":
    main()
