"""
compute_ZW3_raphael.py
Computes the Raphael (2004) ZW3 index from ERA5 z500 monthly files.
Outputs: ZW3_raphael_monthly.csv and ZW3_raphael_annual.csv
Reference: Raphael (2004), GRL, 31, L23212.
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
from scipy.fft import fft, ifft

# =============================================================================
# SETTINGS
# =============================================================================

Z500_DIR  = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/z500_daily_geopotential_height/daily_nc/"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/indices/"

# Raphael (2004) fixed locations: (lat, lon)
RAPHAEL_LOCS = [
    (-50,  50),   # Indian Ocean ridge
    (-50, 170),   # Pacific ridge
    (-50, 290),   # Atlantic ridge (70°W)
]

# =============================================================================
# 1. LOAD AND MONTHLY-MEAN ALL Z500 FILES
# =============================================================================

files = sorted(glob.glob(os.path.join(Z500_DIR, "*.nc")))
print(f"Found {len(files)} files")

monthly_records = []

for f in files:
    # Extract YYYYMM from filename
    basename = os.path.basename(f)
    yyyymm   = basename.split("_")[-1].replace(".nc", "")
    year     = int(yyyymm[:4])
    month    = int(yyyymm[4:])

    ds = xr.open_dataset(f)

    # Monthly mean (averaging over daily time steps)
    z_mean = ds["zg"].mean(dim="valid_time")

    monthly_records.append({
        "year":  year,
        "month": month,
        "z":     z_mean  # DataArray (lat x lon)
    })

    ds.close()

print(f"Loaded {len(monthly_records)} months")

# =============================================================================
# 2. BUILD CLIMATOLOGICAL SEASONAL CYCLE (1979-2016 baseline)
# =============================================================================

# Stack into one DataArray
z_list = [r["z"] for r in monthly_records]
times  = pd.to_datetime([f"{r['year']}-{r['month']:02d}-01"
                          for r in monthly_records])
z_all  = xr.concat(z_list, dim=pd.Index(times, name="time"))

# Climatological mean per calendar month (baseline 1979-2016)
baseline = z_all.sel(time=slice("1979", "2016"))
clim     = baseline.groupby("time.month").mean("time")

# Anomalies
z_anom = z_all.groupby("time.month") - clim

print("Anomalies computed")

# =============================================================================
# 3. ZONAL WAVE 3 FOURIER FILTER (keep wavenumber 3 only)
# =============================================================================

def filter_zw3(z_field):
    """
    Fourier filter a 2D field (lat x lon) to retain only zonal wavenumber 3.
    Returns filtered field same shape as input.
    """
    Z_fft    = fft(z_field.values, axis=-1)
    Z_filt   = np.zeros_like(Z_fft)
    # Keep only wavenumber 3 (positive and negative frequencies)
    Z_filt[:, 3]  = Z_fft[:, 3]
    Z_filt[:, -3] = Z_fft[:, -3]
    return ifft(Z_filt, axis=-1).real

# =============================================================================
# 4. EXTRACT INDEX AT RAPHAEL LOCATIONS
# =============================================================================

results = []

for i, t in enumerate(z_anom.time.values):
    z_field = z_anom.sel(time=t)

    # Fourier filter to ZW3
    z_zw3 = filter_zw3(z_field)

    # Wrap into DataArray for easy lat/lon selection
    z_zw3_da = xr.DataArray(
        z_zw3,
        coords={"latitude":  z_field.latitude,
                "longitude": z_field.longitude},
        dims=["latitude", "longitude"]
    )

    # Extract and average at three Raphael locations
    vals = []
    for lat, lon in RAPHAEL_LOCS:
        val = float(z_zw3_da.sel(
            latitude=lat,  method="nearest"
        ).sel(
            longitude=lon, method="nearest"
        ).values)
        vals.append(val)

    zw3_index = np.mean(vals)
    t_pd      = pd.Timestamp(t)

    results.append({
        "year":      t_pd.year,
        "month":     t_pd.month,
        "ZW3_index": zw3_index
    })

    if i % 60 == 0:
        print(f"  Processed {t_pd.year}-{t_pd.month:02d}")

# =============================================================================
# 5. SAVE OUTPUTS
# =============================================================================

monthly_df = pd.DataFrame(results)

# Annual mean
annual_df = monthly_df.groupby("year")["ZW3_index"].mean().reset_index()
annual_df.columns = ["year", "ZW3_raphael_annual"]

# Seasonal means (useful for later correlation by season)
monthly_df["season"] = pd.cut(
    monthly_df["month"],
    bins=[0, 2, 5, 8, 11, 12],
    labels=["DJF", "MAM", "JJA", "SON", "DJF2"]
)
# Fix DJF wrapping
monthly_df.loc[monthly_df["month"].isin([12, 1, 2]), "season"] = "DJF"
monthly_df.loc[monthly_df["month"].isin([3, 4, 5]),  "season"] = "MAM"
monthly_df.loc[monthly_df["month"].isin([6, 7, 8]),  "season"] = "JJA"
monthly_df.loc[monthly_df["month"].isin([9, 10, 11]),"season"] = "SON"

os.makedirs(OUTPUT_DIR, exist_ok=True)
monthly_df.to_csv(os.path.join(OUTPUT_DIR, "ZW3_raphael_monthly.csv"), index=False)
annual_df.to_csv( os.path.join(OUTPUT_DIR, "ZW3_raphael_annual.csv"),  index=False)

print("\n=== Done ===")
print(f"Monthly: {len(monthly_df)} rows")
print(f"Annual:  {len(annual_df)} rows")
print(monthly_df.head(10))
print(annual_df.head(10))