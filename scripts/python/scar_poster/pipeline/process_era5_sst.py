"""
process_era5_sst_sector.py

Converts the raw ERA5 SST file (from fetch_era5_sst.py) into sector-mean daily
SST and SST anomaly, matching the (date, sector) grain expected by
wind_divergence_coupling_test.py's OCEAN_STATE_PATH.

ERA5 SPECIFICS TO VERIFY
  - Variable name is typically 'sst', units Kelvin. Confirm both via the
    printed header before trusting the conversion below.
  - SST is undefined (masked / fill) over land and, importantly, ALSO where
    the ERA5 model diagnoses sea ice cover -- i.e. cells with the coupled sea
    ice model active may not carry a valid open-water SST. Sector means will
    therefore silently under-sample or exclude ice-covered areas, particularly
    in winter -- worth checking the count of valid cells per sector per season
    doesn't collapse near zero in JJA, where much of the domain is ice-covered.
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

# ---------------- CONFIG ----------------
RAW_DIR = "era5_sst_raw"
RAW_GLOB = "era5_sst_*.nc"

SST_VAR = "sst"
LAT_COORD = "latitude"     # ERA5 uses 'latitude'/'longitude', not 'lat'/'lon'
LON_COORD = "longitude"
TIME_COORD = "valid_time"  # newer CDS downloads use 'valid_time'; older use 'time'

SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1987, 1991, 1995]

SECTORS = {
    "Amundsen-Bellingshausen": (230.0, 300.0),
    "Weddell":                 (300.0, 20.0),
    "King Haakon VII":         (20.0, 90.0),
    "East Antarctica":         (90.0, 160.0),
    "Ross-Amundsen":           (160.0, 230.0),
}

OUTPUT_PATH = "sst_anomaly_by_sector_daily.csv"
# -----------------------------------------


def resolve_time_coord(ds):
    for cand in (TIME_COORD, "time", "valid_time"):
        if cand in ds.coords or cand in ds.dims:
            return cand
    raise KeyError(f"No time-like coordinate found among {list(ds.coords)}")


def sector_mask(lon):
    lon360 = lon % 360.0
    labels = xr.full_like(lon360, fill_value="", dtype=object)
    for name, (lo, hi) in SECTORS.items():
        lo360, hi360 = lo % 360.0, hi % 360.0
        mask = ((lon360 >= lo360) & (lon360 < hi360) if lo360 <= hi360
                else (lon360 >= lo360) | (lon360 < hi360))
        labels = xr.where(mask, name, labels)
    return labels


def load_and_check(path):
    ds = xr.open_dataset(path)
    if SST_VAR not in ds:
        raise KeyError(f"'{SST_VAR}' not found. Available: {list(ds.data_vars)}")

    units = ds[SST_VAR].attrs.get("units", "")
    print(f"SST units attribute: {units!r}")
    sst = ds[SST_VAR]
    if "k" in units.lower() and "celsius" not in units.lower():
        sst = sst - 273.15
        print("Converted K -> degC")

    tcoord = resolve_time_coord(ds)
    return ds, sst, tcoord


def sector_daily_means(sst, tcoord):
    lon = sst[LON_COORD]
    sec = sector_mask(lon)

    rows = []
    for name in SECTORS:
        m = sst.where(sec == name)
        n_valid = m.notnull().sum(dim=[LAT_COORD, LON_COORD])
        daily_mean = m.mean(dim=[LAT_COORD, LON_COORD], skipna=True)
        rows.append(pd.DataFrame({
            "date": pd.to_datetime(sst[tcoord].values),
            "sector": name,
            "sst": daily_mean.values,
            "n_valid_cells": n_valid.values,
        }))
    return pd.concat(rows, ignore_index=True)


def add_period_climatology_anomaly(df):
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    df["post"] = (df["year"] >= SPLIT_YEAR).astype(int)

    df["sst_anom"] = np.nan
    for sector in df["sector"].unique():
        for post in (0, 1):
            m = (df["sector"] == sector) & (df["post"] == post)
            clim = df.loc[m].groupby("doy")["sst"].transform("mean")
            df.loc[m, "sst_anom"] = df.loc[m, "sst"] - clim
    return df


def run():
    files = sorted(glob.glob(os.path.join(RAW_DIR, RAW_GLOB)))
    if not files:
        raise FileNotFoundError(f"No files matched {RAW_GLOB} in {RAW_DIR}")

    all_parts = []
    for path in files:
        print(f"Processing {os.path.basename(path)}")
        ds, sst, tcoord = load_and_check(path)
        part = sector_daily_means(sst, tcoord)
        all_parts.append(part)
        ds.close()

    df = pd.concat(all_parts, ignore_index=True).sort_values(["sector", "date"])
    df = df[~df["date"].dt.year.isin(EXCLUDE_YEARS)]
    df = add_period_climatology_anomaly(df)

    # flag sectors/seasons where valid-cell count collapses -- likely winter
    # sea-ice masking of open-water SST
    df["month"] = df["date"].dt.month
    low_cov = df.groupby(["sector", "month"])["n_valid_cells"].mean()
    thin = low_cov[low_cov < low_cov.max() * 0.1]
    if len(thin):
        print("\n[warn] sector-months with <10% of max valid-cell coverage "
              "(likely winter sea-ice masking of open-water SST):")
        print(thin)

    df[["date", "sector", "sst", "sst_anom", "n_valid_cells"]].to_csv(
        OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df):,} rows -> {OUTPUT_PATH}")
    print(df.groupby("sector")["sst"].describe()[["mean", "std", "min", "max"]])

    return df


if __name__ == "__main__":
    run()