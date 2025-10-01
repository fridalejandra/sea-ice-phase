#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import xarray as xr

RAW_DIR = os.environ["RAW_DIR"]                    # e.g., /user/.../bootstrap_smmr/raw
EXISTING = os.environ["EXISTING_MERGED"]            # baseline (ends 2024-06-30)
OUT_MERGED = os.environ["OUT_MERGED"]                 # new dated output
START_YEAR = int(os.environ.get("START_YEAR", "2024"))

TARGET = "N07_ICECON"

def preprocess(ds: xr.Dataset) -> xr.Dataset:
    """Keep only SIC and standardize its name to TARGET."""
    # Decode CF times early if needed
    try:
        ds = xr.decode_cf(ds)
    except Exception:
        pass

    if TARGET in ds.data_vars:
        keep = ds[[TARGET]]
    else:
        # Try common aliases
        alias = None
        for cand in ["ice_conc", "ICECON", "sic", "SIC", "N07_SIC"]:
            if cand in ds.data_vars:
                alias = cand
                break
        if alias is None:
            # If nothing matches, keep nothing (gets dropped)
            return xr.Dataset()
        keep = ds[[alias]].rename({alias: TARGET})

    return keep

def find_new_files(raw_dir: str, start_year: int):
    files = []
    for y in range(start_year, 2100):
        ydir = os.path.join(raw_dir, f"{y}")
        if os.path.isdir(ydir):
            files.extend(glob.glob(os.path.join(ydir, "*.nc")))
    return sorted(files)

def main():
    files = find_new_files(RAW_DIR, START_YEAR)
    if not files:
        print("No new raw files found; nothing to merge.")
        return

    # Open new stack lazily (parallel if Dask is installed)
    new_stack = xr.open_mfdataset(
        files,
        combine="by_coords",
        preprocess=preprocess,
        parallel=True
    )

    if TARGET not in new_stack.data_vars:
        raise SystemExit(f"No '{TARGET}' variable found in new raw files after preprocessing.")

    # Open existing baseline
    base = xr.open_dataset(EXISTING, decode_times=True)

    # Merge → sort → drop duplicate time stamps
    merged = xr.concat([base[TARGET], new_stack[TARGET]], dim="time", combine_attrs="drop_conflicts")
    merged = merged.sortby("time")
    tvals = merged["time"].values
    _, uniq_idx = np.unique(tvals, return_index=True)
    merged = merged.isel(time=np.sort(uniq_idx))

    # Write atomically with light compression
    out_tmp = OUT_MERGED + ".tmp"
    enc = {TARGET: {"zlib": True, "complevel": 2}}
    merged.to_dataset(name=TARGET).to_netcdf(out_tmp, mode="w", format="NETCDF4", encoding=enc)
    os.replace(out_tmp, OUT_MERGED)
    print(f"✅ wrote {OUT_MERGED}")

if __name__ == "__main__":
    main()
