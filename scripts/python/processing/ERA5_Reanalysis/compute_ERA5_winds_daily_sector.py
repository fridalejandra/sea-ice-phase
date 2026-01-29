#!/usr/bin/env python
# =====================================================
# ERA5 winds → SIE (x,y) grid → daily sector means
# SAFE, CORRECT, CLUSTER-READY
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
OUT_FILE = "ERA5_winds_daily_sector.csv"

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

# Ensure consistent naming
# (SIE grid already uses x/y)
assert "x" in sie.dims and "y" in sie.dims

# =====================================================
# BUILD REGRIDDER (ONCE)
# =====================================================
sample_file = sorted(glob.glob(f"{ERA5_BASE}/{START_YEAR}/*.nc"))[0]
era5_sample = xr.open_dataset(sample_file)

regridder = xe.Regridder(
    era5_sample,
    sie,
    method="bilinear",
    reuse_weights=True
)

era5_sample.close()

# =====================================================
# WEIGHTS (PROJECTED GRID)
# =====================================================
# Option A (BEST): grid-cell area if available
# area = xr.open_dataset("grid_cell_area.nc")["area"]
# weights = area

# Option B (ACCEPTABLE): uniform weights
weights = xr.ones_like(sector_mask)

# =====================================================
# MAIN LOOP (FILE BY FILE = OOM SAFE)
# =====================================================
rows = []

for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Years"):

    files = sorted(glob.glob(f"{ERA5_BASE}/{year}/*.nc"))
    if not files:
        continue

    for f in tqdm(files, desc=str(year), leave=False):

        ds = xr.open_dataset(f)

        # time handling
        ds = ds.rename({"valid_time": "time"})
        time_val = pd.to_datetime(ds.time.values)

        # derive wind speed
        ds["wind_speed"] = np.sqrt(ds.u10**2 + ds.v10**2)

        # regrid ERA5 → SIE grid
        ds_rg = regridder(ds[["u10", "v10", "wind_speed"]])

        row = {"time": time_val}

        # ---------------------------------
        # circumpolar mean
        # ---------------------------------
        circ = (
            ds_rg
            .where(sector_mask.notnull())
            .weighted(weights)
            .mean(dim=("y", "x"))
        )

        row["u10_circumpolar"]  = float(circ.u10)
        row["v10_circumpolar"]  = float(circ.v10)
        row["wind_circumpolar"] = float(circ.wind_speed)

        # ---------------------------------
        # sector means
        # ---------------------------------
        for code, name in SECTORS.items():
            sec = (
                ds_rg
                .where(sector_mask == code)
                .weighted(weights)
                .mean(dim=("y", "x"))
            )

            row[f"u10_{name}"]  = float(sec.u10)
            row[f"v10_{name}"]  = float(sec.v10)
            row[f"wind_{name}"] = float(sec.wind_speed)

        rows.append(row)

        ds.close()
        ds_rg.close()

# =====================================================
# WRITE OUTPUT
# =====================================================
df = pd.DataFrame(rows).sort_values("time")
out_path = f"{OUT_DIR}/{OUT_FILE}"
df.to_csv(out_path, index=False)

print(f"\nSaved ERA5 winds to:\n{out_path}")
