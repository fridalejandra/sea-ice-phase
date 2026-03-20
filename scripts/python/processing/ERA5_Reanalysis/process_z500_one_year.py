# download_z500_daily_mean.py

import os
import calendar
import cdsapi

# =========================
# User settings
# =========================
BASE = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/z500_daily"
YEARS = range(1979, 1980)   # test one year first
AREA = [-20, -180, -90, 180]  # North, West, South, East

os.makedirs(BASE, exist_ok=True)
c = cdsapi.Client()

for yr in YEARS:
    outdir = os.path.join(BASE, f"{yr}")
    os.makedirs(outdir, exist_ok=True)

    for mo in range(1, 13):
        month_str = f"{mo:02d}"
        ndays = calendar.monthrange(yr, mo)[1]
        days = [f"{d:02d}" for d in range(1, ndays + 1)]

        out = os.path.join(outdir, f"era5_z500_daily_mean_{yr}{month_str}.zip")
        if os.path.exists(out):
            print(f"Skipping existing file: {out}")
            continue

        print(f"Downloading daily mean z500 for {yr}-{month_str}")

        c.retrieve(
            "derived-era5-pressure-levels-daily-statistics",
            {
                "product_type": "reanalysis",
                "variable": "geopotential",
                "pressure_level": "500",
                "year": str(yr),
                "month": month_str,
                "day": days,
                "daily_statistic": "daily_mean",
                "frequency": "1_hourly",
                "time_zone": "utc+00:00",
                "area": AREA,
                "data_format": "netcdf",
                "download_format": "zip",
            },
            out,
        )

print("Done.")