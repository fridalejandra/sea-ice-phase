#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_smmr.py  — robust merge for Bootstrap daily files lacking a time dimension.

Env (set by pipeline):
  RAW_DIR          : root with RAW_DIR/YYYY/*.nc
  EXISTING_MERGED  : baseline merged file (read-only; optional)
  OUT_MERGED       : path to write the new merged file
  START_YEAR       : only consider RAW_DIR/START_YEAR and after
"""

import os, re, sys, glob, shutil, datetime as dt
import numpy as np
import xarray as xr

RAW_DIR         = os.environ["RAW_DIR"]
EXISTING_MERGED = os.environ["EXISTING_MERGED"]
OUT_MERGED      = os.environ["OUT_MERGED"]
START_YEAR      = int(os.environ.get("START_YEAR", "1979"))

# Common alias names for concentration in Bootstrap-ish files
VAR_CANDIDATES = [
    "N07_ICECON", "ice_conc", "ice_concentration", "sea_ice_conc",
    "Sea_Ice_Concentration", "SI_ICE_CONC", "seaice_conc_cdr"
]

def _parse_date_from_attrs(ds):
    """Try CF-ish globals commonly seen in NASA products."""
    g = ds.attrs
    for k in ("time_coverage_start", "RangeBeginningDate", "RANGEBEGINNINGDATE"):
        if k in g and g[k]:
            try:
                # keep only YYYY-MM-DD part if timestamp present
                s = str(g[k])[:10]
                return dt.date.fromisoformat(s)
            except Exception:
                pass
    return None

def _parse_date_from_filename(path):
    """Accept YYYYMMDD or YYYYDDD tokens anywhere in basename."""
    name = os.path.basename(path)
    m8 = re.search(r"(20\d{2})(\d{2})(\d{2})", name)   # YYYYMMDD
    if m8:
        y, mo, da = map(int, m8.groups())
        return dt.date(y, mo, da)
    m7 = re.search(r"(20\d{2})(\d{3})", name)          # YYYYDDD
    if m7:
        y, jjj = map(int, m7.groups())
        return (dt.date(y, 1, 1) + dt.timedelta(days=jjj - 1))
    return None

def _find_var(ds):
    for v in VAR_CANDIDATES:
        if v in ds.data_vars:
            return v
    raise ValueError(f"No known SIC var in {list(ds.data_vars)}")

def _preprocess_one(fp: str) -> xr.Dataset:
    """Open raw file → pick/rename SIC var → add time coord → expand_dims('time')."""
    ds = xr.open_dataset(fp, decode_times=False)

    # pick the SIC variable and rename to canonical
    v = _find_var(ds)
    if v != "N07_ICECON":
        ds = ds.rename({v: "N07_ICECON"})

    # derive date
    d = _parse_date_from_filename(fp) or _parse_date_from_attrs(ds)
    if d is None:
        # last ditch: file mtime
        d = dt.date.fromtimestamp(os.path.getmtime(fp))
    t = np.datetime64(d.isoformat())

    # ensure minimal payload; add time dim
    ds = ds[["N07_ICECON"]].expand_dims(time=[t])
    return ds

def _find_candidates(raw_dir: str, start_year: int):
    paths = []
    for y in range(start_year, 2100):
        ydir = os.path.join(raw_dir, f"{y}")
        if os.path.isdir(ydir):
            paths.extend(glob.glob(os.path.join(ydir, "*.nc")))
    return sorted(paths)

def _concat_stack(files):
    if not files:
        return None
    out = []
    for i, fp in enumerate(files):
        try:
            out.append(_preprocess_one(fp))
        except Exception as e:
            print(f"⚠ Skipping {os.path.basename(fp)}: {e}")
    if not out:
        return None
    ds = xr.concat(out, dim="time", data_vars="minimal", coords="minimal",
                   compat="override", join="override").sortby("time")
    return ds

def main():
    print(f"RAW_DIR={RAW_DIR}")
    print(f"EXISTING_MERGED={EXISTING_MERGED}")
    print(f"OUT_MERGED={OUT_MERGED}")
    print(f"START_YEAR={START_YEAR}")

    files = _find_candidates(RAW_DIR, START_YEAR)
    print(f"Found {len(files)} candidate files at/after {START_YEAR}")

    new = _concat_stack(files)
    if new is None:
        print("No usable new files. Exiting.")
        return

    # Append to baseline if present
    if os.path.exists(EXISTING_MERGED):
        base = xr.open_dataset(EXISTING_MERGED, decode_times=True)
        if "N07_ICECON" not in base.data_vars:
            raise RuntimeError("Baseline missing 'N07_ICECON'")
        bt = np.asarray(base["time"].values)
        nt = np.asarray(new["time"].values)
        mask = [t not in set(bt) for t in nt]
        if not any(mask):
            print("All candidate days already in baseline; nothing to append.")
            base.close()
            return
        new = new.isel(time=np.nonzero(mask)[0])
        merged = xr.concat([base, new], dim="time", data_vars="minimal",
                           coords="minimal", compat="override", join="override").sortby("time")
        base.close()
    else:
        print("⚠ Baseline not found; creating merged file from new stack only.")
        merged = new

    # Drop residual duplicate times (just in case)
    _, idx = np.unique(merged["time"].values, return_index=True)
    merged = merged.isel(time=np.sort(idx))

    # Atomic write
    tmp = OUT_MERGED + ".tmp"
    encoding = {"N07_ICECON": {"zlib": True, "complevel": 1, "_FillValue": None}}
    merged.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
    shutil.move(tmp, OUT_MERGED)
    print(f"✓ Wrote {OUT_MERGED}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
