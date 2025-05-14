# download_amsre.py

import earthaccess
from datetime import datetime

# === AUTHENTICATION ===
earthaccess.login()

# === CONFIG ===

# Date range (feel free to edit)
start_date = "2024-07-09"
end_date = datetime.today().strftime("%Y-%m-%d")

# Output directory
# LOCAL:
output_dir = "/data/processed/amsre/raw/"

# CLUSTER substitute:
# output_dir = "/user/geog/falejandraperez/sea-ice-phase/data/amsre/raw/"

# === SEARCH ===

print(f"Searching for AU_SI12 granules from {start_date} to {end_date}...")
results = earthaccess.search_data(
    short_name="AU_SI12",
    temporal=(start_date, end_date),
    bounding_box=(-180, -90, 180, -50)  # Southern Hemisphere
)

print(f"Found {len(results)} granules.")

# === DOWNLOAD ===

downloaded = earthaccess.download(results, output_dir)
print(f"✅ Downloaded {len(downloaded)} granules to: {output_dir}")
