# =====================================================
# ERA5 wind speed → daily sector-mean CSV + climatology NC
# Parallel to compute_NSIDC_driftSpeed_daily_sector.py
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
ERA5_BASE = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/winds"

SIE_GRID_FILE = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/"
    "merged_bootstrap_SH_latest.nc"
)
SECTOR_MASK_FILE = (
    "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"
)

OUT_DIR  = "/user/geog/falejandraperez/sea-ice-phase/results/ERA5"
OUT_FILE = "ERA5_windSpeed_daily_sector_v2.csv"
OUT_NC   = "ERA5_windSpeed_period_season_climatology.nc"
WEIGHTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "era5_to_sie_bilinear_weights_v2.nc",
)

START_YEAR = 1979
END_YEAR   = 2024
PERIOD_SPLIT = 2016

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD SIE GRID + SECTOR MASK, attach lat/lon via EPSG:3412
# =====================================================
sie = xr.open_dataset(SIE_GRID_FILE)
sector_mask = xr.open_dataset(SECTOR_MASK_FILE)["sector_id"]

sie_crs = CRS.from_epsg(3412)
to_wgs = Transformer.from_crs(sie_crs, CRS.from_epsg(4326), always_xy=True)
sxx, syy = np.meshgrid(sie.x.values, sie.y.values)
sie_lon, sie_lat = to_wgs.transform(sxx, syy)
sie = sie.assign_coords(
    lon=(("y", "x"), sie_lon),
    lat=(("y", "x"), sie_lat),
)

# =====================================================
# BUILD REGRIDDER FROM ERA5 (lat/lon) → SIE GRID
# =====================================================
sample_file = sorted(glob.glob(f"{ERA5_BASE}/{START_YEAR}/*.nc"))[0]
sample = xr.open_dataset(sample_file)
regridder = xe.Regridder(
    sample, sie, method="bilinear",
    reuse_weights=os.path.exists(WEIGHTS_FILE),
    filename=WEIGHTS_FILE,
)
sample.close()
print(f"Regridder ready: {WEIGHTS_FILE}")

# =====================================================
# Sector IDs (integer, drop 0/NaN background)
# =====================================================
sm_vals = sector_mask.values
unique = np.unique(sm_vals[np.isfinite(sm_vals)]) if sm_vals.dtype.kind == "f" else np.unique(sm_vals)
sid_values = [int(s) for s in unique if int(s) != 0]
print(f"Sector ids: {sid_values}")

# =====================================================
# Running sums for season × period climatology
# =====================================================
clim_sums = {}
clim_counts = {}

def season_of(month):
    if month in (12, 1, 2): return "DJF"
    if month in (3, 4, 5):  return "MAM"
    if month in (6, 7, 8):  return "JJA"
    return "SON"

def period_of(year):
    return "post" if year >= PERIOD_SPLIT else "pre"

# =====================================================
# MAIN LOOP
# =====================================================
rows = []

for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Years"):
    files = sorted(glob.glob(f"{ERA5_BASE}/{year}/*.nc"))
    if not files:
        print(f"  [skip] no files for {year}")
        continue

    for f in tqdm(files, desc=str(year), leave=False):
        try:
            ds = xr.open_dataset(f, engine="netcdf4")
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})

        ts = pd.Timestamp(ds.time.values.item())

        # ---- compute speed, GUARANTEED non-negative ----
        wind_speed = np.sqrt(ds.u10.values**2 + ds.v10.values**2).squeeze()
        # wrap back into DataArray with ERA5 lat/lon so regridder recognizes it
        wind_da = xr.DataArray(
            wind_speed,
            dims=("latitude", "longitude"),
            coords={"latitude": ds.latitude, "longitude": ds.longitude},
            name="wind_speed",
        )

        # regrid onto SIE
        wind_rg = regridder(wind_da)  # (y, x)

        # climatology accumulate
        key = (period_of(ts.year), season_of(ts.month))
        day_vals = wind_rg.values
        valid = np.isfinite(day_vals)
        if key not in clim_sums:
            clim_sums[key]   = np.zeros_like(day_vals, dtype=np.float64)
            clim_counts[key] = np.zeros_like(day_vals, dtype=np.int64)
        clim_sums[key]   += np.where(valid, day_vals, 0.0)
        clim_counts[key] += valid.astype(np.int64)

        # sector means — plain np.nanmean, no weighted ambiguity
        for sid in sid_values:
            mask = (sector_mask.values == sid)
            vals = day_vals[mask]
            mean_val = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
            rows.append({
                "date": ts,
                "sector": sid,
                "wind_speed_mps": mean_val,
            })

        ds.close()

df = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, OUT_FILE)
df.to_csv(out_path, index=False)
print(f"\nSaved sector CSV: {out_path}")
print(df.head())
print(f"Rows: {len(df):,}")
print(f"Negative wind values: {(df['wind_speed_mps'] < 0).sum()}  (should be 0)")

# =====================================================
# Period × season climatology NetCDF
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
        "wind_speed_mps": (
            ("period", "season", "y", "x"), clim_arr,
            {"long_name": "Mean 10 m wind speed", "units": "m s-1"},
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
        "source": "ERA5 10m u/v (daily 12UTC)",
        "regridded_to": "NSIDC Polar Stereographic South 25 km (EPSG:3412)",
        "period_split_year": PERIOD_SPLIT,
        "note": f"'pre' = years < {PERIOD_SPLIT}; 'post' = years >= {PERIOD_SPLIT}",
    },
)
nc_path = os.path.join(OUT_DIR, OUT_NC)
clim_ds.to_netcdf(nc_path)
print(f"Saved climatology NC: {nc_path}")
