"""
download_smmr.py  –  for future incremental updates only
Downloads NSIDC-0079 granules from BASE_NC_END_DATE to TODAY.
Before running, update BASE_NC_END_DATE in config.py to the day
after your last downloaded granule to avoid re-downloading.
"""

import sys
import earthaccess
from config import GRANULE_DIR, BASE_NC_END_DATE, TODAY

GRANULE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading NSIDC-0079 granules: {BASE_NC_END_DATE} → {TODAY}")
print(f"   Saving to: {GRANULE_DIR}\n")

earthaccess.login()

results = earthaccess.search_data(
    short_name="NSIDC-0079",
    temporal=(BASE_NC_END_DATE, TODAY),
    bounding_box=(-180, -90, 180, -50),
)

if not results:
    print(" No granules found. Check date range or earthaccess credentials.")
    sys.exit(1)

print(f"Found {len(results)} granules. Starting download...")
downloaded_files = earthaccess.download(results, str(GRANULE_DIR))

print(f"\nDownload complete. {len(downloaded_files)} files saved.")
print(f"Update BASE_NC_END_DATE in config.py for next time!")