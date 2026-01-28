#!/usr/bin/env python

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import glob
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from pathlib import Path

# =====================================================
# paths
# =====================================================
ERA5_BASE = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/winds"
MASK_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"
OUT_DIR   = "/user/geog/falejandraperez/sea-ice-phase/results/ERA5"
OUT_FILE  = "ERA5_winds_daily_sector.csv"

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
# load mask ONCE
# =====================================================
mask = xr.open_dataset(MASK_FILE)["sector_id"]

results = []

# =====================================================
# main loop (FILE BY FILE = OOM SAFE)
# =====================================================
for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Years"):

    files = sorted(glob.glob(f"{ERA5_BASE}/{year}/*.nc"))
    if not files:
        continue

    for f in tqdm(files, desc=f"{year}", leave=False):

        ds = xr.open_dataset(f)

        ds = ds.rename({"valid_time": "time"})
        ds["wind_speed"] = np.sqrt(ds.u10**2 + ds.v10**2)

        weights = np.cos(np.deg2rad(ds.latitude))

        row = {
            "time": pd.to_datetime(ds.time.values)
        }

        # circumpolar
        circ = (
            ds[["u10", "v10", "wind_speed"]]
            .where(mask.notnull())
            .weighted(weights)
            .mean(dim=("latitude", "longitude"))
        )

        row["u10_circumpolar"]   = float(circ.u10)
        row["v10_circumpolar"]   = float(circ.v10)
        row["wind_circumpolar"]  = float(circ.wind_speed)

        # sectors
        for code, name in SECTORS.items():
            sec = (
                ds[["u10", "v10", "wind_speed"]]
                .where(mask == code)
                .weighted(weights)
                .mean(dim=("latitude", "longitude"))
            )

            row[f"u10_{name}"]  = float(sec.u10)
            row[f"v10_{name}"]  = float(sec.v10)
            row[f"wind_{name}"] = float(sec.wind_speed)

        results.append(row)

        ds.close()

# =====================================================
# write output
# =====================================================
df = pd.DataFrame(results)
df = df.sort_values("time")
df.to_csv(f"{OUT_DIR}/{OUT_FILE}", index=False)

print(f"\nSaved:\n{OUT_DIR}/{OUT_FILE}")
