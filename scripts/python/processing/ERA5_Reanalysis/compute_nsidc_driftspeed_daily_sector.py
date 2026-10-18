# =====================================================
# NSIDC-0116 drift speed → daily sector-mean CSV
# Mirrors compute_ERA5_winds_daily_sector.py structure.
#
# Source grid: NSIDC EASE-Grid South 25 km (EPSG:3409)
# Target grid: NSIDC Polar Stereographic South 25 km (EPSG:3412)
# =====================================================

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import glob
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from tqdm import tqdm
from pathlib import Path
from pyproj import CRS, Transformer

# =====================================================
# PATHS
# =====================================================
DRIFT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/drift_nsidc"

SIE_GRID_FILE = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/"
    "merged_bootstrap_SH_latest.nc"
)

SECTOR_MASK_FILE = (
    "/user/geog/falejandraperez/sea-ice-phase/data/"
    "canonical_sectors.nc"
)

OUT_DIR  = "/user/geog/falejandraperez/sea-ice-phase/results/NSIDC0116"
OUT_FILE = "NSIDC0116_driftSpeed_daily_sector.csv"
OUT_NC   = "NSIDC0116_driftSpeed_period_season_climatology.nc"
WEIGHTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "nsidc_drift_to_sie_bilinear_weights.nc",
)

START_YEAR = 1979
END_YEAR   = 2024
PERIOD_SPLIT = 2016  # years >= this are "post"

# NSIDC-0116 u/v are cm/s. Convert to m/s to match wind convention.
UNIT_SCALE = 0.01

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD SIE GRID + SECTOR MASK (ONCE)
# =====================================================
sie = xr.open_dataset(SIE_GRID_FILE)
sector_mask = xr.open_dataset(SECTOR_MASK_FILE)["sector_id"]

assert "x" in sie.dims and "y" in sie.dims

# =====================================================
# TARGET lat/lon: SIE grid is NSIDC Polar Stereographic South (EPSG:3412)
# =====================================================
sie_crs = CRS.from_epsg(3412)
to_wgs = Transformer.from_crs(sie_crs, CRS.from_epsg(4326), always_xy=True)
sxx, syy = np.meshgrid(sie.x.values, sie.y.values)
sie_lon, sie_lat = to_wgs.transform(sxx, syy)
sie = sie.assign_coords(
    lon=(("y", "x"), sie_lon),
    lat=(("y", "x"), sie_lat),
)

# =====================================================
# SOURCE lat/lon: NSIDC drift is EASE-Grid South (EPSG:3409)
# =====================================================
sample_file = sorted(glob.glob(f"{DRIFT_DIR}/icemotion_daily_sh_25km_*.nc"))[0]
sample = xr.open_dataset(sample_file)

ease_crs = CRS.from_proj4(
    "+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 "
    "+a=6371228 +b=6371228 +units=m +no_defs"
)
ease_to_wgs = Transformer.from_crs(ease_crs, CRS.from_epsg(4326), always_xy=True)
dxx, dyy = np.meshgrid(sample.x.values, sample.y.values)
drift_lon, drift_lat = ease_to_wgs.transform(dxx, dyy)

sample = sample.assign_coords(
    lon=(("y", "x"), drift_lon),
    lat=(("y", "x"), drift_lat),
)

# =====================================================
# BUILD REGRIDDER (ONCE)
# =====================================================
regridder = xe.Regridder(
    sample,
    sie,
    method="bilinear",
    reuse_weights=os.path.exists(WEIGHTS_FILE),
    filename=WEIGHTS_FILE,
)
sample.close()
print(f"Regridder ready (weights: {WEIGHTS_FILE})")

# Precompute unique sector IDs
sid_values = np.unique(sector_mask.values)
sid_values = sid_values[~np.isnan(sid_values)] if sector_mask.dtype.kind == "f" else sid_values
sid_values = [int(s) for s in sid_values if s != 0]  # drop background 0 if present
print(f"Sector ids: {sid_values}")

# =====================================================
# Running sums for seasonal × period climatologies
# =====================================================
# Keys like ("pre", "DJF"); values: (sum_2d, count_2d)
clim_sums = {}
clim_counts = {}

def season_of(month):
    if month in (12, 1, 2):   return "DJF"
    if month in (3, 4, 5):    return "MAM"
    if month in (6, 7, 8):    return "JJA"
    return "SON"

def period_of(year):
    return "post" if year >= PERIOD_SPLIT else "pre"

# =====================================================
# MAIN LOOP (year-by-year, memory safe)
# =====================================================
rows = []

for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Years"):
    matches = glob.glob(f"{DRIFT_DIR}/icemotion_daily_sh_25km_{year}*.nc")
    if not matches:
        print(f"  [skip] no file for {year}")
        continue
    ds = xr.open_dataset(matches[0])

    # Mask fill, compute speed in m/s
    u = ds.u.where(ds.u != -9999) * UNIT_SCALE
    v = ds.v.where(ds.v != -9999) * UNIT_SCALE
    speed = np.sqrt(u**2 + v**2)  # (time, y, x)

    # Attach lat/lon so regridder recognizes it
    speed = speed.assign_coords(
        lon=(("y", "x"), drift_lon),
        lat=(("y", "x"), drift_lat),
    )

    # Regrid all days at once
    speed_rg = regridder(speed)  # (time, y, x) on SIE grid

    # Sector-daily means + accumulate climatology sums
    for i, t_val in enumerate(ds.time.values):
        day = speed_rg.isel(time=i)
        ts = pd.Timestamp(t_val.strftime("%Y-%m-%d"))

        # climatology accumulation (period × season)
        key = (period_of(ts.year), season_of(ts.month))
        day_vals = day.values
        valid = np.isfinite(day_vals)
        if key not in clim_sums:
            clim_sums[key]   = np.zeros_like(day_vals, dtype=np.float64)
            clim_counts[key] = np.zeros_like(day_vals, dtype=np.int64)
        clim_sums[key]   += np.where(valid, day_vals, 0.0)
        clim_counts[key] += valid.astype(np.int64)

        # sector means (CSV)
        for sid in sid_values:
            mask = sector_mask == sid
            mean_val = float(day.where(mask).mean(skipna=True).values)
            rows.append({
                "date": ts,
                "sector": sid,
                "drift_speed_mps": mean_val,
            })

    ds.close()

df = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, OUT_FILE)
df.to_csv(out_path, index=False)
print(f"\nSaved NSIDC-0116 drift speed to:\n{out_path}")
print(df.head())
print(f"Rows: {len(df):,}")

# =====================================================
# Build + save period × season climatology NetCDF
# =====================================================
periods = ["pre", "post"]
seasons = ["DJF", "MAM", "JJA", "SON"]
ny, nx = sie.sizes["y"], sie.sizes["x"]
clim_arr = np.full((len(periods), len(seasons), ny, nx), np.nan, dtype=np.float32)

for ip, p in enumerate(periods):
    for isn, s in enumerate(seasons):
        key = (p, s)
        if key in clim_sums:
            with np.errstate(invalid="ignore", divide="ignore"):
                clim_arr[ip, isn] = np.where(
                    clim_counts[key] > 0,
                    clim_sums[key] / np.maximum(clim_counts[key], 1),
                    np.nan,
                )

clim_ds = xr.Dataset(
    {
        "drift_speed_mps": (
            ("period", "season", "y", "x"),
            clim_arr,
            {"long_name": "Mean sea-ice drift speed", "units": "m s-1"},
        ),
    },
    coords={
        "period": periods,
        "season": seasons,
        "y": sie.y,
        "x": sie.x,
        "lat": (("y", "x"), sie_lat),
        "lon": (("y", "x"), sie_lon),
    },
    attrs={
        "source": "NSIDC-0116 v4.1",
        "regridded_to": "NSIDC Polar Stereographic South 25 km (EPSG:3412)",
        "period_split_year": PERIOD_SPLIT,
        "note": f"'pre' = years < {PERIOD_SPLIT}; 'post' = years >= {PERIOD_SPLIT}",
    },
)
nc_path = os.path.join(OUT_DIR, OUT_NC)
clim_ds.to_netcdf(nc_path)
print(f"\nSaved climatology NetCDF to:\n{nc_path}")
print(clim_ds)
