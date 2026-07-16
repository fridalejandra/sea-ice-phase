"""
build_forcing_sector_table.py  (WIND STRESS ONLY - DLWR removed)

Regrids ERA5 wind stress (tau_x, tau_y) from its native regular lat/lon grid
onto your polar-stereographic sector grid, computes daily sector-mean wind
stress magnitude, applies the accumulation-period unit fix, and merges
everything with your existing per-sector SIA record into one analysis-ready
daily table.
"""

import glob
import os
import numpy as np
import pandas as pd
import xarray as xr

# ---------------- CONFIG ----------------
WIND_STRESS_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/wind_stress"
SECTORS_FILE    = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"
SIA_CSV         = "/user/geog/falejandraperez/sea-ice-phase/data/merged/sia_by_sector_daily.csv"
OUT_CSV         = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_table_daily.csv"

START_YEAR = 1979
END_YEAR = 2024   # adjust to match actual wind stress coverage

ACCUM_SECONDS = 86400  # confirmed 24hr accumulation for wind stress

SECTOR_NAMES = {
    1: "Amundsen-Bellingshausen",
    2: "Weddell",
    3: "King Haakon VII",
    4: "East Antarctica",
    5: "Ross-Amundsen",
}
# ------------------------------------------

print("Loading sector definitions...")
sectors_ds = xr.open_dataset(SECTORS_FILE)
sector_id = sectors_ds["sector_id"]     # (y, x) on polar stereographic grid
valid_ocean = sectors_ds["valid_ocean"]
lat_ps = sectors_ds["lat"]
lon_ps = sectors_ds["lonE"]             # 0-360 convention, matches ERA5 longitude convention


def load_yearly_var(directory, fname_pattern, varname, year):
    f = os.path.join(directory, fname_pattern.format(year=year))
    if not os.path.exists(f):
        return None
    ds = xr.open_dataset(f)
    return ds[varname]


def regrid_to_sector_grid(field_latlon, lat_ps, lon_ps):
    """Nearest-neighbor interpolation from lat/lon field onto polar-stereo grid."""
    lat_da = xr.DataArray(lat_ps.values, dims=["y", "x"])
    lon_da = xr.DataArray(lon_ps.values, dims=["y", "x"])
    regridded = field_latlon.interp(
        latitude=lat_da, longitude=lon_da, method="nearest"
    )
    return regridded


def sector_daily_mean(field_ps, sector_mask):
    return field_ps.where(sector_mask).mean(dim=["x", "y"], skipna=True)


# ---------------- Process wind stress ----------------
print("\nProcessing wind stress...")
tau_x_years, tau_y_years = [], []

for year in range(START_YEAR, END_YEAR + 1):
    tx = load_yearly_var(WIND_STRESS_DIR + f"/{year}", "era5_windstress_tau_x_{year}.nc", "ewss", year)
    ty = load_yearly_var(WIND_STRESS_DIR + f"/{year}", "era5_windstress_tau_y_{year}.nc", "nsss", year)
    if tx is None or ty is None:
        print(f"  {year}: missing wind stress file(s), skipping")
        continue
    tau_x_years.append(tx / ACCUM_SECONDS)  # accumulated -> true stress
    tau_y_years.append(ty / ACCUM_SECONDS)

tau_x_all = xr.concat(tau_x_years, dim="valid_time").rename({"valid_time": "time"})
tau_y_all = xr.concat(tau_y_years, dim="valid_time").rename({"valid_time": "time"})
tau_magnitude = np.sqrt(tau_x_all**2 + tau_y_all**2)

print("Regridding wind stress onto sector grid...")
tau_ps = regrid_to_sector_grid(tau_magnitude, lat_ps, lon_ps)

# ---------------- Compute sector daily means ----------------
print("\nComputing sector-mean wind stress...")
records = []
for sid, name in SECTOR_NAMES.items():
    mask = (sector_id == sid) & valid_ocean
    tau_sector = sector_daily_mean(tau_ps, mask)

    df = pd.DataFrame({
        "date": pd.to_datetime(tau_sector.time.values),
        "sector": name,
        "wind_stress": tau_sector.values,
    })
    records.append(df)

forcing_df = pd.concat(records, ignore_index=True)

# ---------------- Merge with SIA ----------------
print("\nMerging with SIA record...")
sia_df = pd.read_csv(SIA_CSV, index_col="date", parse_dates=True)
sia_long = sia_df[list(SECTOR_NAMES.values())].reset_index().melt(
    id_vars="date", var_name="sector", value_name="SIA"
)

merged = forcing_df.merge(sia_long, on=["date", "sector"], how="inner")
merged = merged.sort_values(["sector", "date"])

# ---------------- Compute daily change (delta X) per sector ----------------
merged["delta_SIA"] = merged.groupby("sector")["SIA"].diff()

# ---------------- Save ----------------
merged.to_csv(OUT_CSV, index=False)
print(f"\nSaved analysis-ready table to:\n  {OUT_CSV}")
print(f"Shape: {merged.shape}")
print(merged.head())
print(merged.describe())
