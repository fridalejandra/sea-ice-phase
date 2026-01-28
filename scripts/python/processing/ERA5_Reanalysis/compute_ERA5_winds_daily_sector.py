#!/usr/bin/env python

import os
# ---- hard safety limits for HPC ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import glob
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from pathlib import Path

# =====================================================
# paths (EDIT ONLY IF NEEDED)
# =====================================================
ERA5_BASE = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/winds"
MASK_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"
OUT_DIR   = "/user/geog/falejandraperez/sea-ice-phase/results/ERA5"

OUT_FILE  = "ERA5_winds_daily_sector.csv"

START_YEAR = 1979
END_YEAR   = 2024

# =====================================================
# sector definition (must match SIE)
# =====================================================
SECTORS = {
    1: "Weddell",
    2: "Amundsen_Bellingshausen",
    3: "Ross",
    4: "East_Antarctica",
    5: "King_Haakon"
}

# =====================================================
# setup output
# =====================================================
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# =====================================================
# load sector mask ONCE
# =====================================================
mask_ds = xr.open_dataset(MASK_FILE)
sector_mask = mask_ds["sector_id"]

# =====================================================
# container for results
# =====================================================
all_years = []

# =====================================================
# main loop (YEARLY = SAFE)
# =====================================================
for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Processing years"):

    files = sorted(
        glob.glob(f"{ERA5_BASE}/{year}/*.nc")
    )

    if len(files) == 0:
        continue

    # ---- open year (NO parallelism) ----
    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="valid_time",
        decode_times=True,
        parallel=False,
        chunks={}
    )

    ds = ds.rename({"valid_time": "time"})

    # ---- derived wind speed ----
    ds["wind_speed"] = np.sqrt(ds.u10**2 + ds.v10**2)

    # ---- area weights ----
    weights = np.cos(np.deg2rad(ds.latitude))
    weights.name = "weights"

    # =================================================
    # circumpolar mean (same philosophy as SIE)
    # =================================================
    circ = (
        ds[["u10", "v10", "wind_speed"]]
        .where(sector_mask.notnull())
        .weighted(weights)
        .mean(dim=("latitude", "longitude"))
    )

    circ = circ.rename({
        "u10": "u10_circumpolar",
        "v10": "v10_circumpolar",
        "wind_speed": "wind_circumpolar"
    })

    outputs = [circ]

    # =================================================
    # sector means
    # =================================================
    for code, name in SECTORS.items():
        sec = (
            ds[["u10", "v10", "wind_speed"]]
            .where(sector_mask == code)
            .weighted(weights)
            .mean(dim=("latitude", "longitude"))
        )

        sec = sec.rename({
            "u10": f"u10_{name}",
            "v10": f"v10_{name}",
            "wind_speed": f"wind_{name}"
        })

        outputs.append(sec)

    # =================================================
    # merge + daily aggregation
    # =================================================
    out_ds = xr.merge(outputs)

    out_ds = out_ds.resample(time="1D").mean()

    df_year = out_ds.to_dataframe().reset_index()
    all_years.append(df_year)

    # ---- CRITICAL CLEANUP ----
    ds.close()
    out_ds.close()

# =====================================================
# write final CSV
# =====================================================
df = pd.concat(all_years, ignore_index=True)

out_path = f"{OUT_DIR}/{OUT_FILE}"
df.to_csv(out_path, index=False)

print(f"\nSaved ERA5 winds to:\n{out_path}")
