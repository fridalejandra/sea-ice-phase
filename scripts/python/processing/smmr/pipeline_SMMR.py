#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_SMMR.py
Merge daily SMMR Bootstrap files into a single, time-sorted NetCDF per variable.

Environment (must be set by the job script):
  RAW_DIR         : root with RAW_DIR/YYYY/*.nc  (downloaded daily files)
  OUT_MERGED      : path to write the merged file (NetCDF)
  EXISTING_MERGED : optional baseline merged file (read-only); will be extended
  START_YEAR      : optional; only consider RAW_DIR/START_YEAR and after (default=1979)

This script:
  - finds candidate daily files since START_YEAR
  - opens each, extracts a single sea-ice concentration variable, renaming to TARGET
  - constructs/normalizes a daily time coordinate
  - concatenates with (optional) existing merged dataset, de-duplicates on time
  - writes atomically to OUT_MERGED.tmp then mv to OUT_MERGED
"""

from __future__ import annotations
import os
import re
import sys
import glob
import shutil
import datetime as dt
from pathlib import Path
import numpy as np
import xarray as xr

# --------- ENV (cluster-aware) ----------
RAW_DIR         = os.environ.get("RAW_DIR")
OUT_MERGED      = os.environ.get("OUT_MERGED")
EXISTING_MERGED = os.environ.get("EXISTING_MERGED")  # optional
START_YEAR      = int(os.environ.get("START_YEAR", "1979"))

if not RAW_DIR or not OUT_MERGED:
    sys.stderr.write("ERROR: RAW_DIR and OUT_MERGED must be set in the environment.\n")
    sys.exit(2)

# --------- SIC var discovery ----------
TARGET = "N07_ICECON"  # standardized name in outputs
VAR_CANDIDATES = [
    "N07_ICECON",
    "ice_conc", "ice_concentration", "sea_ice_conc",
    "Sea_Ice_Concentration", "SI_ICE_CONC", "seaice_conc_cdr",
    "sic", "N07_SIC"  # be generous; rename to TARGET below
]

def preprocess(ds: xr.Dataset) -> xr.Dataset:
    """Return a dataset with a single variable named TARGET."""
    try:
        ds = xr.decode_cf(ds)
    except Exception:
        pass

    # Prefer already-standardized name
    if TARGET in ds.data_vars:
        keep = ds[TARGET]
        return keep.to_dataset(name=TARGET)

    # Search aliases
    alias = None
    for cand in VAR_CANDIDATES:
        if cand in ds.data_vars:
            alias = cand
            break

    if alias is None:
        # Nothing we recognize → empty dataset
        return xr.Dataset()

    keep = ds[alias]
    # Normalize scale if needed (0..100 → 0..1)
    try:
        if np.nanmax(keep.values) > 1.5:
            keep = keep / 100.0
    except Exception:
        pass

    return keep.to_dataset(name=TARGET)

# --------- Date parsing helpers ----------
def _parse_date_from_attrs(ds: xr.Dataset) -> dt.date | None:
    """Try common CF-ish attrs to pull a file's date."""
    g = ds.attrs
    for k in ("time_coverage_start", "RangeBeginningDate", "RANGEBEGINNINGDATE"):
        v = g.get(k)
        if v:
            try:
                return dt.date.fromisoformat(str(v)[:10])
            except Exception:
                pass
    return None

def _parse_date_from_filename(path: str) -> dt.date | None:
    """Accept YYYYMMDD or YYYY-MM-DD anywhere in basename."""
    base = os.path.basename(path)
    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", base)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return dt.date(y, mo, d)
        except Exception:
            return None
    return None

def open_and_prepare(path: str) -> xr.Dataset:
    """Open a file, extract SIC as TARGET, and attach/normalize a 1-day time coord."""
    try:
        raw = xr.open_dataset(path, decode_times=False)  # robust to odd units
    except Exception as e:
        sys.stderr.write(f"WARN: cannot open {path}: {e}\n")
        return xr.Dataset()

    ds = preprocess(raw)
    raw.close()

    if not ds.data_vars:
        sys.stderr.write(f"INFO: skip {path} (no known SIC var)\n")
        return xr.Dataset()

    # Try to get the date
    date = _parse_date_from_attrs(ds) or _parse_date_from_filename(path)
    if date is None:
        sys.stderr.write(f"INFO: skip {path} (no date)\n")
        return xr.Dataset()

    # Attach a time coordinate (single day)
    t = np.array([np.datetime64(date)])
    ds = ds.expand_dims(time=t)  # (time=1, y, x)

    # ensure dims ordering
    if "y" in ds.dims and "x" in ds.dims:
        ds = ds.transpose("time", ..., missing_dims="ignore")
    else:
        # If grid uses lat/lon, rename once to y/x for consistency
        rename = {}
        if "lat" in ds.dims: rename["lat"] = "y"
        if "latitude" in ds.dims: rename["latitude"] = "y"
        if "lon" in ds.dims: rename["lon"] = "x"
        if "longitude" in ds.dims: rename["longitude"] = "x"
        if rename:
            ds = ds.rename(rename)
        ds = ds.transpose("time", ..., missing_dims="ignore")

    return ds

# --------- File discovery ----------
def find_new_files(raw_dir: str, start_year: int) -> list[str]:
    """Collect candidate daily files from RAW_DIR/YYYY directories."""
    files: list[str] = []
    for y in range(start_year, 2101):
        ydir = os.path.join(raw_dir, f"{y}")
        if not os.path.isdir(ydir):
            continue
        files.extend(sorted(glob.glob(os.path.join(ydir, "*.nc"))))
    return files

# --------- Merge logic ----------
def merge_datasets(existing: xr.Dataset | None, parts: list[xr.Dataset]) -> xr.Dataset:
    """Concat along time, drop duplicate times (keep last), sort by time."""
    pieces = []
    if existing is not None and existing.data_vars:
        pieces.append(existing)

    for ds in parts:
        if ds is not None and ds.data_vars:
            pieces.append(ds)

    if not pieces:
        return xr.Dataset()

    merged = xr.concat(pieces, dim="time", join="outer", combine_attrs="override")

    # drop duplicate times (keep last → newest write wins)
    _, idx = np.unique(merged["time"].values, return_index=True)
    keep_mask = np.zeros(merged.time.size, dtype=bool)
    keep_mask[idx] = True
    # invert to find duplicates, then re-keep last by stable argsort
    if keep_mask.sum() != merged.time.size:
        # keep last occurrences
        order = np.argsort(merged.time.values)
        seen = set()
        keep_mask = np.zeros(merged.time.size, dtype=bool)
        for k in order[::-1]:
            t = merged.time.values[k]
            if t not in seen:
                keep_mask[k] = True
                seen.add(t)
        merged = merged.isel(time=keep_mask)

    merged = merged.sortby("time")
    merged.attrs["history"] = (
        merged.attrs.get("history", "") +
        f"\nmerged on {dt.datetime.utcnow().isoformat()}Z"
    )
    return merged

# --------- I/O helpers ----------
def safe_write_netcdf(ds: xr.Dataset, out_path: str):
    """Atomic write: write to .tmp then move."""
    tmp = out_path + ".tmp"
    encoding = {TARGET: {"zlib": True, "complevel": 4, "dtype": "float32"}}
    ds.to_netcdf(tmp, encoding=encoding)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    os.replace(tmp, out_path)

# --------- Main ----------
def main():
    print(f"[INFO] RAW_DIR={RAW_DIR}")
    print(f"[INFO] OUT_MERGED={OUT_MERGED}")
    print(f"[INFO] EXISTING_MERGED={EXISTING_MERGED or '(none)'}")
    print(f"[INFO] START_YEAR={START_YEAR}")

    # Optional baseline
    existing = None
    if EXISTING_MERGED and os.path.isfile(EXISTING_MERGED):
        try:
            existing = xr.open_dataset(EXISTING_MERGED)
            # ensure we only carry TARGET
            if TARGET not in existing.data_vars:
                existing = existing[[v for v in existing.data_vars][0]].rename(TARGET).to_dataset()
        except Exception as e:
            sys.stderr.write(f"WARN: cannot open EXISTING_MERGED: {e}\n")
            existing = None

    # Discover and open new daily files
    files = find_new_files(RAW_DIR, START_YEAR)
    if not files and existing is None:
        sys.stderr.write("ERROR: no files found and no existing merged dataset.\n")
        sys.exit(1)

    parts: list[xr.Dataset] = []
    for p in files:
        ds = open_and_prepare(p)
        if ds.data_vars:
            parts.append(ds)

    merged = merge_datasets(existing, parts)
    if not merged.data_vars:
        sys.stderr.write("WARN: nothing to write (no valid inputs). Exiting.\n")
        sys.exit(0)

    # Write atomically
    safe_write_netcdf(merged, OUT_MERGED)
    print(f"[OK] Wrote merged dataset: {OUT_MERGED}")
    # Close anything we opened via xarray (avoid file locks on some FS)
    try:
        merged.close()
        if existing is not None:
            existing.close()
        for ds in parts:
            ds.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
