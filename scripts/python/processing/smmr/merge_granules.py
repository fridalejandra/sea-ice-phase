"""
merge_granules.py  –  Step 1
Concatenates new daily granule .nc files in GRANULE_DIR into a single
time-sorted file (MERGED_NEW) using CDO mergetime.
Only processes files from BASE_NC_END_DATE onward.
"""

import sys
import subprocess
from config import GRANULE_DIR, MERGED_NEW, BASE_NC_END_DATE

# ---- FIND FILES ---- #
all_files = sorted(GRANULE_DIR.glob("*.nc"))

def file_date(f):
    # filename: NSIDC0079_SEAICE_PS_S25km_YYYYMMDD_v4.0.nc
    try:
        return f.stem.split("_")[4]
    except IndexError:
        return ""

cutoff = BASE_NC_END_DATE.replace("-", "")
new_files = [f for f in all_files if file_date(f) >= cutoff]

if not new_files:
    print(f"No granule files found in {GRANULE_DIR} at or after {BASE_NC_END_DATE}")
    sys.exit(1)

print(f"Found {len(new_files)} new granule files (>= {BASE_NC_END_DATE})")
print(f"   First : {new_files[0].name}")
print(f"   Last  : {new_files[-1].name}")

# ---- CDO MERGETIME ---- #
MERGED_NEW.parent.mkdir(parents=True, exist_ok=True)
print(f"\nMerging with CDO mergetime...")
print(f"   Output: {MERGED_NEW}")

cmd = ["cdo", "mergetime"] + [str(f) for f in new_files] + [str(MERGED_NEW)]
result = subprocess.run(cmd, text=True, capture_output=True)

print(result.stdout)
if result.stderr:
    print(result.stderr)

if result.returncode != 0:
    print("CDO mergetime failed.")
    sys.exit(1)

print("Granule merge complete.")