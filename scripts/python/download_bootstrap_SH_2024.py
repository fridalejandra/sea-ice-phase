import os
from datetime import datetime
import earthaccess

# ---- CONFIG ---- #
TEMP_DOWNLOAD_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/data/bootstrap_smmr/test_downloads/"
start_date = "2024-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# ---- LOGIN ---- #
earthaccess.login()

# ---- SEARCH & DOWNLOAD ---- #
results = earthaccess.search_data(
    short_name="NSIDC-0079",
    temporal=(start_date, end_date),
    bounding_box=(-180, -90, 180, -50),
)

print(f"Found {len(results)} granules.")
downloaded_files = earthaccess.download(results, TEMP_DOWNLOAD_DIR)

# ---- FILTER & DELETE NON-SH ---- #
for f in downloaded_files:
    if "PS_N25km" in os.path.basename(f):  # Northern Hemisphere
        print("❌ Deleting NH granule:", os.path.basename(f))
        os.remove(f)
