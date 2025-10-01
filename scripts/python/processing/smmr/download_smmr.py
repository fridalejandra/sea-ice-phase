import os
from datetime import datetime
import shutil
import re
import earthaccess

# ---- CONFIG (honor pipeline env) ----
RAW_DIR    = os.environ.get("RAW_DIR", "/user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/raw")
START_DATE = os.environ.get("START_DATE", "2024-06-30")
END_DATE   = os.environ.get("END_DATE",   datetime.today().strftime("%Y-%m-%d"))

os.makedirs(RAW_DIR, exist_ok=True)

# ---- LOGIN ----
earthaccess.login()

# ---- SEARCH ----
results = earthaccess.search_data(
    short_name="NSIDC-0079",
    temporal=(START_DATE, END_DATE),
    bounding_box=(-180, -90, 180, -50),  # SH box
)

print(f"Found {len(results)} granules for {START_DATE} → {END_DATE}")

# ---- DOWNLOAD to a temp staging folder ----
staging = os.path.join(RAW_DIR, "_staging")
os.makedirs(staging, exist_ok=True)
downloaded = earthaccess.download(results, staging)

# ---- Keep SH only; move into year subfolders RAW_DIR/YYYY/*.nc ----
moved = 0
for f in downloaded:
    base = os.path.basename(f)

    # Drop NH
    if "PS_N25km" in base:
        print("❌ Deleting NH granule:", base)
        os.remove(f)
        continue

    # Extract date token from filename (e.g., ..._YYYYDDD or YYYYMMDD variants)
    # Bootstrap V4 daily often encodes date as YYYYDDD; also accept YYYYMMDD.
    m = re.search(r"(20\d{2})(\d{3}|\d{4})", base)
    year = None
    if m:
        y = int(m.group(1))
        # crude sanity check
        if 1979 <= y <= 2100:
            year = y

    # Fallback: use file mtime
    if year is None:
        year = datetime.fromtimestamp(os.path.getmtime(f)).year

    ydir = os.path.join(RAW_DIR, str(year))
    os.makedirs(ydir, exist_ok=True)
    dest = os.path.join(ydir, base)

    if os.path.exists(dest):
        # already have it; delete duplicate download
        print("↪ Already have:", os.path.basename(dest))
        os.remove(f)
        continue

    shutil.move(f, dest)
    print("✓ Moved:", base, "→", dest)
    moved += 1

# clean staging if empty
try:
    if not os.listdir(staging):
        os.rmdir(staging)
except Exception:
    pass

print(f"Done. New files staged into {RAW_DIR} by year. Moved: {moved}")
