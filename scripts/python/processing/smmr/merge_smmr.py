# merge_smmr.py
# Merges all downloaded NSIDC-0079 granules into a single NetCDF
# using CDO mergetime, then updates the latest symlink.

import os
import subprocess
import glob
from datetime import datetime

# === CONFIG ===

GRANULE_DIR  = "/user/geog/falejandraperez/sea-ice-phase/data/smmr/raw"
MERGED_DIR   = "/user/geog/falejandraperez/sea-ice-phase/data/merged"
today        = datetime.today().strftime("%m%d%Y")
MERGED_FILE  = f"{MERGED_DIR}/merged_bootstrap_SH_{today}.nc"
LATEST_LINK  = f"{MERGED_DIR}/merged_bootstrap_SH_latest.nc"

os.makedirs(MERGED_DIR, exist_ok=True)

# === FIND GRANULES ===

files = sorted(glob.glob(os.path.join(GRANULE_DIR, "*.nc")))
print(f"Found {len(files)} granule files in {GRANULE_DIR}")

if not files:
    print("No files found — run download_smmr.py first.")
    exit(1)

# === MERGE WITH CDO ===

print(f"Merging with CDO mergetime...")
print(f"Output: {MERGED_FILE}")

cmd = ["cdo", "mergetime"] + files + [MERGED_FILE]
result = subprocess.run(cmd, text=True, capture_output=True)

if result.stdout:
    print(result.stdout)
if result.stderr:
    print(result.stderr)

if result.returncode != 0:
    print("CDO mergetime failed.")
    exit(1)

# === UPDATE SYMLINK ===

if os.path.islink(LATEST_LINK) or os.path.exists(LATEST_LINK):
    os.remove(LATEST_LINK)
os.symlink(MERGED_FILE, LATEST_LINK)
print(f"Symlink updated: merged_bootstrap_SH_latest.nc → {os.path.basename(MERGED_FILE)}")

print(f"\nDone. Merged file: {MERGED_FILE}")