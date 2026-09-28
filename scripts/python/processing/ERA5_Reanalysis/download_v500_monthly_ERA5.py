"""
download_v500_monthly_ERA5.py
Downloads ERA5 monthly mean 500-hPa meridional wind (v-component)
for the Southern Hemisphere, 1979-2023.
Output: one .nc file per year in v500_monthly/
"""

import cdsapi
import os

OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/v500_monthly/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

c = cdsapi.Client()

for year in range(1979, 2024):
    outfile = os.path.join(OUTPUT_DIR, f"era5_v500_monthly_{year}.nc")

    if os.path.exists(outfile):
        print(f"{year} already exists, skipping")
        continue

    print(f"Downloading {year}...")
    c.retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable":     "v_component_of_wind",
            "pressure_level": "500",
            "year":  str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time":  "00:00",
            "area":  [-40, 0, -70, 360],   # N, W, S, E — 40S to 70S global
            "format": "netcdf"
        },
        outfile
    )
    print(f"  Saved: {outfile}")

print("\n=== Download complete ===")