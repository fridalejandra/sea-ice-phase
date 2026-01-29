#!/usr/bin/env python
# =====================================================
# ERA5 wind SPEED → SIE (x,y) grid → daily sector means
# POSTER-READY, SAFE, CLUSTER-READY
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
    "/user/geog/falejandraperez/sea-ice-phase/data/"
    "canonical_sectors.nc"
)

OUT_DIR  = "/user/geog/falejandraperez/sea-ice-phase/results/ERA5"
OUT_FILE = "ERA5_windSpeed_daily_sector.csv"

START_YEAR = 1979
END_YEAR   = 2024

SECTORS = {
    1: "Weddell",
    2: "Amundsen_Bellingshausen",
    3: "Ross",
    4: "East_Antarctica",
    5: "King_Haakon"
}

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD SIE GRID + SECTOR MASK (ONCE)
# =====================================================
sie = xr.open_dataset(SIE_GRID_FILE)
sector_mask = xr.open_dataset(SECTOR_MASK_FILE)["sector_id"]

assert "x" in sie.dims and "y" in sie.dims

# =====================================================
# EXPLICIT CRS: NSIDC Polar Stereographic South
# =====================================================
crs = CRS.from_epsg(3412)

transformer = Transformer.from_crs(
    crs,
    CRS.from_epsg(4326),  # WGS84 lon/lat
    always_xy=True
)

xx, yy = np.meshgrid(sie.x.values, sie.y.values)
lon, lat = transformer.transform(xx, yy)

sie = sie.assign_coords(
    lon=(("y", "x"), lon),
    lat=(("y", "x"), lat)
)

# =====================================================
# BUILD REGRIDDER (ONCE)
# =====================================================
sample_file = sorted(
    glob.glob(f"{ERA5_BASE}/{START_YEAR}/*.nc")
)[0]

era5_sample = xr.open_dataset(sample_file)

regridder = xe.Regridder(
    era5_sample,
    sie,
    method="bilinear",
    reuse_weights=False,
    filename="era5_to_sie_bilinear_weights.nc"
)

era5_sample.close()

# =====================================================
# WEIGHTS ON SIE GRID
# =====================================================
weights = xr.ones_like(sector_mask)

# =====================================================
# MAIN LOOP (FILE-BY-FILE = OOM SAFE)
# =====================================================
rows = []

for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Years"):

    files = sorted(glob.glob(f"{ERA5_BASE}/{year}/*.nc"))
    if not files:
        continue

    for f in tqdm(files, desc=str(year), leave=False):

        ds = xr.open_dataset(f)

        ds = ds.rename({"valid_time": "time"})

        # ---- FIXED: scalar timestamp ----
        time_val = pd.Timestamp(ds.time.values.item())

        # ---- compute wind speed only ----
        wind_speed = np.sqrt(ds.u10**2 + ds.v10**2)

        # regrid ERA5 → SIE grid
        wind_rg = regridder(wind_speed)

        row = {"time": time_val}

        # -------------------------
        # Circumpolar mean
        # -------------------------
        row["wind_circumpolar"] = float(
            wind_rg
            .where(sector_mask.notnull())
            .weighted(weights)
            .mean(dim=("y", "x"))
        )

        # -------------------------
        # Sector means
        # -------------------------
        for code, name in SECTORS.items():
            row[f"wind_{name}"] = float(
                wind_rg
                .where(sector_mask == code)
                .weighted(weights)
                .mean(dim=("y", "x"))
            )

        rows.append(row)

        ds.close()
        wind_rg.close()

# =====================================================
# WRITE OUTPUT
# =====================================================
df = pd.DataFrame(rows).sort_values("time")
out_path = f"{OUT_DIR}/{OUT_FILE}"
df.to_csv(out_path, index=False)

print(f"\nSaved ERA5 wind speed to:\n{out_path}")
