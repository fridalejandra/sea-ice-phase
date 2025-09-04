# download_z700_daily_12utc.py
import os
from datetime import date, timedelta
import cdsapi

BASE = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/z700"
os.makedirs(BASE, exist_ok=True)

YEARS = range(1979, 2024)  # adjust
AREA = [-20, -180, -90, 180]  # N,W,S,E

c = cdsapi.Client()

for yr in YEARS:
    outdir = os.path.join(BASE, f"{yr}")
    os.makedirs(outdir, exist_ok=True)

    d = date(yr, 1, 1)
    d_end = date(yr, 12, 31)
    while d <= d_end:
        ymd = d.strftime("%Y%m%d")
        out = os.path.join(outdir, f"era5_z700_{ymd}_12UTC.nc")
        if os.path.exists(out):
            d += timedelta(days=1)
            continue

        # ERA5 pressure levels @ 12 UTC snapshot
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "pressure_level": ["700"],
                "variable": ["geopotential"],
                "year": f"{d.year}",
                "month": f"{d.month:02d}",
                "day": f"{d.day:02d}",
                "time": ["12:00"],
                "area": AREA,  # N,W,S,E
            },
            out,
        )
        d += timedelta(days=1)
why