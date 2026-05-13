"""
download_smmr.py  –  Step 1
Download NSIDC-0079 Bootstrap SIC granules for the Southern Hemisphere.
Only keeps PS_S25km (Southern Hemisphere) files; NH files are deleted.
"""

import os
import sys
import earthaccess
from config import GRANULE_DIR, DOWNLOAD_START, TODAY

# ---- SETUP ---- #
GRANULE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading NSIDC-0079 granules: {DOWNLOAD_START} → {TODAY}")
print(f"   Saving to: {GRANULE_DIR}\n")

# ---- LOGIN ---- #
earthaccess.login()

# ---- SEARCH ---- #
results = earthaccess.search_data(
    short_name="NSIDC-0079",
    temporal=(DOWNLOAD_START, TODAY),
    bounding_box=(-180, -90, 180, -50),
)

if not results:
    print("No granules found. Check your date range or earthaccess credentials.")
    sys.exit(1)

print(f"Found {len(results)} granules. Starting download...")

# ---- DOWNLOAD ---- #
downloaded_files = earthaccess.download(results, str(GRANULE_DIR))

# ---- FILTER: keep SH only ---- #
deleted = 0
for f in downloaded_files:
    if "PS_N25km" in os.path.basename(f):
        print(f"  🗑  Removing NH file: {os.path.basename(f)}")
        os.remove(f)
        deleted += 1

kept = len(downloaded_files) - deleted
print(f"\n Download complete. {kept} SH files kept, {deleted} NH files removed.")
print(f"   Files in: {GRANULE_DIR}")