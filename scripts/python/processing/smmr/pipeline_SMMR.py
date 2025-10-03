#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMMR/Bootstrap SH pipeline — download missing daily granules and append safely.

Fixed paths:
- Scripts: /user/geog/falejandraperez/sea-ice-phase/scripts/python/processing/smmr
- RAW_DIR (daily): /user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/raw/YYYY/*.nc
- TMP_DIR (staging): /user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/tmp
- MERGED_DIR (outputs): /user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr
- Baseline (read-only): /user/geog/falejandraperez/sea-ice-phase/data/merged/SMMR_merged_1979_06302024.nc
"""

import os, sys, pathlib, shutil
import datetime as dt
import subprocess as sp
import xarray as xr
import numpy as np

# ---------- CONFIG ----------
REPO_ROOT   = "/user/geog/falejandraperez/sea-ice-phase"
SCRIPTS_DIR = f"{REPO_ROOT}/scripts/python/processing/smmr"

DATA_ROOT   = f"{REPO_ROOT}/data/bootstrap_smmr"
RAW_DIR     = f"{DATA_ROOT}/raw"      # expects RAW_DIR/YYYY/*.nc
TMP_DIR     = f"{DATA_ROOT}/tmp"
MERGED_DIR  = DATA_ROOT

# Your baseline (ends 2024-06-30). If this path doesn’t exist, create a symlink to the real file.
MERGED_BASELINE = "/user/geog/falejandraperez/sea-ice-phase/data/merged/SMMR_merged_1979_06302024.nc"

# Controls via env (override when running): END_DATE, UPDATE_LATEST_LINK, CLEAN_RAW
END_DATE_ISO       = os.environ.get("END_DATE", dt.date.today().isoformat())     # YYYY-MM-DD
UPDATE_LATEST_LINK = os.environ.get("UPDATE_LATEST_LINK", "1") == "1"
CLEAN_RAW          = os.environ.get("CLEAN_RAW", "0") == "1"

# Output file for this run; never overwrite baseline
ENDDATE    = END_DATE_ISO.replace("-", "")                                       # YYYYMMDD
OUT_MERGED = f"{MERGED_DIR}/merged_bootstrap_SH_1979_{ENDDATE}.nc"
LATEST_LINK = f"{MERGED_DIR}/merged_bootstrap_SH_latest.nc"

DOWNLOAD_SCRIPT = f"{SCRIPTS_DIR}/download_smmr.py"   # expects START_DATE, END_DATE, RAW_DIR, TMP_DIR
MERGE_SCRIPT    = f"{SCRIPTS_DIR}/merge_smmr.py"      # expects RAW_DIR, EXISTING_MERGED, OUT_MERGED, START_YEAR

# ---------- HELPERS ----------
def shell(cmd, env=None):
    print(f"\n$ {' '.join(cmd)}\n")
    p = sp.run(cmd, env=env or os.environ.copy(), text=True, capture_output=True)
    sys.stdout.write(p.stdout or "")
    sys.stderr.write(p.stderr or "")
    if p.returncode != 0:
        raise SystemExit(f"Command failed ({p.returncode}): {' '.join(cmd)}")

def ensure_dirs():
    for d in (RAW_DIR, TMP_DIR, MERGED_DIR):
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)

def organize_flat_raw_into_years():
    """Move any RAW_DIR/*.nc into RAW_DIR/YYYY/ (one-time hygiene)."""
    root = pathlib.Path(RAW_DIR)
    for f in root.glob("*.nc"):
        name = f.name
        year = None
        for i in range(len(name) - 7):
            tok = name[i:i+8]
            if tok.isdigit() and tok.startswith("20"):
                year = int(tok[:4]); break
        if year is None:
            year = dt.date.fromtimestamp(f.stat().st_mtime).year
        ydir = root / f"{year}"; ydir.mkdir(exist_ok=True)
        dest = ydir / name
        if not dest.exists(): f.rename(dest)

def infer_start_date(merged_path: str) -> str:
    """Read last 'time' from baseline; return last_date + 1 day. Fallback 1979-01-01."""
    if not os.path.exists(merged_path):
        return "1979-01-01"
    ds = xr.open_dataset(merged_path, decode_times=True)
    try:
        if "time" not in ds or ds["time"].size == 0:
            return "1979-01-01"
        last_np = ds["time"].values[-1]
        last_dt = np.datetime64(last_np, "s").astype("datetime64[s]").astype(object)
        last_date = last_dt.date() if hasattr(last_dt, "date") else dt.date.fromtimestamp(int(np.datetime64(last_np, "s").astype(int)))
        return (last_date + dt.timedelta(days=1)).isoformat()
    finally:
        ds.close()

def touch_years(start_iso: str, end_iso: str):
    s = dt.date.fromisoformat(start_iso).year
    e = dt.date.fromisoformat(end_iso).year
    return set(range(s, e + 1))

def cleanup_raw_years(years):
    for y in years:
        ydir = pathlib.Path(RAW_DIR) / f"{y}"
        if ydir.exists():
            print(f"🗑 Removing {ydir}")
            shutil.rmtree(ydir)

# ---------- MAIN ----------
def main():
    ensure_dirs()
    organize_flat_raw_into_years()

    start_date = os.environ.get("START_DATE", infer_start_date(MERGED_BASELINE))

    if dt.date.fromisoformat(END_DATE_ISO) < dt.date.fromisoformat(start_date):
        print(f"Nothing to do: END_DATE ({END_DATE_ISO}) < START_DATE ({start_date}).")
        return

    print("\n=== SMMR Bootstrap SH update ===")
    print(f"Baseline merged  : {MERGED_BASELINE}")
    print(f"Raw granules dir : {RAW_DIR}")
    print(f"Staging tmp dir  : {TMP_DIR}")
    print(f"Output merged    : {OUT_MERGED}")
    print(f"Date window      : {start_date} → {END_DATE_ISO}")
    print("================================\n")

    # Step 1 — download (always use dates; never rely on an “empty folder” check)
    shell(
        ["python", DOWNLOAD_SCRIPT],
        env={**os.environ,
             "START_DATE": start_date,
             "END_DATE": END_DATE_ISO,
             "RAW_DIR": RAW_DIR,
             "TMP_DIR": TMP_DIR}
    )

    # Step 2 — merge/append to a new dated file
    start_year = str(dt.date.fromisoformat(start_date).year)
    shell(
        ["python", MERGE_SCRIPT],
        env={**os.environ,
             "RAW_DIR": RAW_DIR,
             "EXISTING_MERGED": MERGED_BASELINE,
             "OUT_MERGED": OUT_MERGED,
             "START_YEAR": start_year}
    )

    # Step 3 — validate (guarded)
    print("\n--- Validation of new merged file ---")
    if os.path.exists(OUT_MERGED):
        ds = xr.open_dataset(OUT_MERGED, decode_times=True)
        try:
            t = ds["time"].values if "time" in ds.coords else []
            print(f"✓ new file time span: n={getattr(t, 'size', 0)}")
        finally:
            ds.close()
    else:
        print("⚠ No merged file created (no new data). Skipping validation.")

    # Step 4 — update latest symlink
    if UPDATE_LATEST_LINK and os.path.exists(OUT_MERGED):
        link = pathlib.Path(LATEST_LINK)
        try:
            if link.exists() or link.is_symlink(): link.unlink()
            os.symlink(OUT_MERGED, LATEST_LINK)
            print(f"↪ latest → {OUT_MERGED}")
        except Exception as e:
            print(f"⚠ Symlink update skipped: {e}")

    # Step 5 — optional cleanup
    if CLEAN_RAW:
        years = touch_years(start_date, END_DATE_ISO)
        cleanup_raw_years(years)

    print("\n✓ Done.")

if __name__ == "__main__":
    main()
