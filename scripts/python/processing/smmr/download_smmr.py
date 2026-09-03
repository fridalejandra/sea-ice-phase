# download_smmr.py

import earthaccess
from datetime import datetime

# === AUTHENTICATION ===
earthaccess.login()

# === CONFIG ===

# Date range
# For a full redownload from scratch use 1978-10-01
# For incremental updates change start_date to day after last granule
start_date = "1978-10-01"
end_date   = datetime.today().strftime("%Y-%m-%d")

# Output directory
output_dir = "/user/geog/falejandraperez/sea-ice-phase/data/smmr/raw/"

# === SEARCH ===

print(f"Searching for NSIDC-0079 granules from {start_date} to {end_date}...")
results = earthaccess.search_data(
    short_name="NSIDC-0079",
    temporal=(start_date, end_date),
    bounding_box=(-180, -90, 180, -50)  # Southern Hemisphere
)

print(f"Found {len(results)} granules.")

# === DOWNLOAD ===

import os
os.makedirs(output_dir, exist_ok=True)

downloaded = earthaccess.download(results, output_dir)
print(f"Downloaded {len(downloaded)} granules to: {output_dir}")