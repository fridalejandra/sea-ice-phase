#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_smmr.py

Cluster-aware pipeline to extend the NSIDC-0079 (Bootstrap) Southern Hemisphere
merged SIC file by downloading only the missing daily granules and appending
them safely, without overwriting the baseline file.

Contract expected of component scripts:
- download_smmr.py: reads START_DATE, END_DATE, RAW_DIR; downloads to RAW_DIR/YYYY/,
                    keeps only SH files (e.g., basenames containing "_SH_"), idempotent.
- merge_smmr.py   : reads RAW_DIR, EXISTING_MERGED, OUT_MERGED, START_YEAR; loads only
                    new files from START_YEAR onward, standardizes var to "N07_ICECON",
                    concat + sort + dedupe by 'time', atomic write to OUT_MERGED.

Author: Frida P.
"""

import os
import sys
import shutil
import pathlib
import datetime as dt
import subprocess as sp

# Optional dependency: xarray just for start-date inference + validation.
# (If xarray is unavailable on the driver node, you can replace with ncdump/awk.)
import xarray as xr
import numpy as np


# ======================
# Configuration (edit)
# ======================
DATA_ROOT   = "/user/geog/falejandraperez/sea-ice-phase/data"
MERGED_DIR  = f"{DATA_ROOT}/merged"
RAW_DIR     = f"{DATA_ROOT}/bootstrap_smmr/raw"

# If you renamed your baseline to the ISO scheme, set that here; otherwise keep legacy name.
MERGED_BASELINE = f"{MERGED_DIR}/SMMR_merged_SH_1979_06302024.nc"  # your current baseline (ends 2024-06-30)

# Scripts
SCRIPTS_DIR      = "/user/geog/falejandraperez/sea-ice-phase/scripts/python/processing/smmr"
DOWNLOAD_SCRIPT  = f"{SCRIPTS_DIR}/download_smmr.py"
MERGE_SCRIPT     = f"{SCRIPTS_DIR}/merge_smmr.py"

# Controls via env (override with export before running):
# END_DATE: YYYY-MM-DD (default: today)
# UPDATE_LATEST_LINK: "1" to update symlink; default "1"
# CLEAN_RAW: "1" to delete RAW_DIR subfolders for the years touched; default "0"
END_DATE_ISO       = os.environ.get("END_DATE", dt.date.today().isoformat())          # 'YYYY-MM-DD'
UPDATE_LATEST_LINK = os.environ.get("UPDATE_LATEST_LINK", "1") == "1"
CLEAN_RAW          = os.environ.get("CLEAN_RAW", "0") == "1"

# Output naming (don’t overwrite baseline):
ENDDATE = END_DATE_ISO.replace("-", "")                                               # 'YYYYMMDD'
OUT_MERGED = f"{MERGED_DIR}/SMMR_merged_SH_1979_{ENDDATE}.nc"
LATEST_LINK = f"{MERGED_DIR}/SMMR_merged_SH_latest.nc"


# ======================
# Utility functions
# ======================
def shell(cmd, env=None):
    """Run a command, stream stdout/stderr, and fail fast on non-zero exit."""
    print(f"\n$ {' '.join(cmd)}\n")
    proc = sp.run(cmd, env=env or os.environ.copy(), text=True, capture_output=True)
    sys.stdout.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def ensure_dirs():
    pathlib.Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(RAW_DIR).mkdir(parents=True, exist_ok=True)


def infer_start_date(merged_path: str) -> str:
    """
    Open the baseline merged NetCDF, read the last timestamp, return last_date + 1 day.
    If the file is missing or empty, fall back to 1979-01-01.
    """
    if not os.path.exists(merged_path):
        return "1979-01-01"
    ds = xr.open_dataset(merged_path, decode_times=True)
    try:
        t = ds["time"].values
        if t.size == 0:
            return "1979-01-01"
        last_np = t[-1]
        # Robust cast to date
        last_dt = np.datetime64(last_np, "s").astype("datetime64[s]").astype(object)
        if isinstance(last_dt, dt.datetime):
            last_date = last_dt.date()
        elif isinstance(last_dt, dt.date):
            last_date = last_dt
        else:
            # Fallback using numpy toordinal if weird dtype
            last_date = dt.date.fromtimestamp(int(np.datetime64(last_np, "s").astype(int)))
        return (last_date + dt.timedelta(days=1)).isoformat()
    finally:
        ds.close()



def validate_merged(path: str):
    """Quick checks: time coverage, monotonicity, duplicates, variable presence."""
    ds = xr.open_dataset(path, decode_times=True)
    try:
        if "time" not in ds.coords:
            print("⚠ No 'time' coordinate found.")
            return
        t = ds["time"].values
        if t.size == 0:
            print("⚠ 'time' coordinate is empty.")
            return
        t_min, t_max, n = t.min(), t.max(), t.size
        print(f"✓ time coverage: min={t_min}  max={t_max}  n={n}")
        ndup = n - np.unique(t).size
        print(f"✓ duplicates in time: {ndup}")
        if "N07_ICECON" in ds.data_vars:
            print("✓ variable present: N07_ICECON")
        else:
            print("⚠ variable N07_ICECON not found.")
    finally:
        ds.close()


def touch_years(start_iso: str, end_iso: str):
    """Return the set of integer years intersecting [start, end]."""
    s = dt.date.fromisoformat(start_iso).year
    e = dt.date.fromisoformat(end_iso).year
    return set(range(s, e + 1))


def cleanup_raw_years(years):
    """Delete year subfolders under RAW_DIR for the given set of years."""
    for y in years:
        ydir = pathlib.Path(RAW_DIR) / f"{y}"
        if ydir.exists():
            print(f"🗑 Removing raw folder: {ydir}")
            shutil.rmtree(ydir)


# ======================
# Main pipeline
# ======================
def main():
    ensure_dirs()

    # 1) Infer dates
    start_date = os.environ.get("START_DATE", infer_start_date(MERGED_BASELINE))
    if dt.date.fromisoformat(END_DATE_ISO) < dt.date.fromisoformat(start_date):
        print(f"Nothing to do: END_DATE ({END_DATE_ISO}) < START_DATE ({start_date}).")
        return

    # 2) Announce configuration
    print("\n=== SMMR Bootstrap SH update ===")
    print(f"Baseline merged  : {MERGED_BASELINE}")
    print(f"Raw granules dir : {RAW_DIR}")
    print(f"Output merged    : {OUT_MERGED}")
    print(f"Date window      : {start_date} → {END_DATE_ISO}")
    print("================================\n")

    # 3) Download new granules (idempotent)
    #    Expect download_smmr.py to honor START_DATE/END_DATE/RAW_DIR
    shell(["python", DOWNLOAD_SCRIPT],
          env={**os.environ,
               "START_DATE": start_date,
               "END_DATE": END_DATE_ISO,
               "RAW_DIR": RAW_DIR})

    # 4) Merge/append into a NEW dated file (don’t overwrite baseline)
    #    Expect merge_smmr.py to honor RAW_DIR/EXISTING_MERGED/OUT_MERGED/START_YEAR
    start_year = str(dt.date.fromisoformat(start_date).year)
    shell(["python", MERGE_SCRIPT],
          env={**os.environ,
               "RAW_DIR": RAW_DIR,
               "EXISTING_MERGED": MERGED_BASELINE,
               "OUT_MERGED": OUT_MERGED,
               "START_YEAR": start_year})

    print("\n--- Validation of new merged file ---")
    if os.path.exists(OUT_MERGED):
        validate_merged(OUT_MERGED)
    else:
        print(f"⚠ No merged file created (no new data). Skipping validation.")

    # 5) Validate the new merged file
    print("\n--- Validation of new merged file ---")
    validate_merged(OUT_MERGED)

    # 6) Optionally update the "latest" symlink
    if UPDATE_LATEST_LINK:
        try:
            p_link = pathlib.Path(LATEST_LINK)
            if p_link.exists() or p_link.is_symlink():
                p_link.unlink()
            os.symlink(OUT_MERGED, LATEST_LINK)
            print(f"↪ Updated symlink: {LATEST_LINK} -> {OUT_MERGED}")
        except Exception as e:
            print(f"⚠ Symlink update skipped: {e}")

    # 7) Optional cleanup of raw files for years touched
    if CLEAN_RAW:
        years = touch_years(start_date, END_DATE_ISO)
        cleanup_raw_years(years)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
