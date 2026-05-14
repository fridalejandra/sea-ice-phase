"""
merge_smmr.py  –  Step 2
Merges the historical base record (BASE_NC) with the newly merged
granules (MERGED_NEW) into a single time-sorted file (FINAL_MERGED)
using CDO mergetime, then updates the LATEST_MERGED symlink.
"""

import sys
import subprocess
import xarray as xr
from config import BASE_NC, MERGED_NEW, FINAL_MERGED, LATEST_MERGED

# ---- CHECKS ---- #
for path, label in [(BASE_NC, "BASE_NC"), (MERGED_NEW, "MERGED_NEW")]:
    if not path.exists():
        print(f"{label} not found: {path}")
        sys.exit(1)

# ---- PREVIEW ---- #
ds_base = xr.open_dataset(BASE_NC)
ds_new  = xr.open_dataset(MERGED_NEW)
print(f"Base : {BASE_NC.name}")
print(f"Range: {str(ds_base.time.values[0])[:10]} → {str(ds_base.time.values[-1])[:10]}  ({ds_base.sizes['time']} steps)")
print(f"New  : {MERGED_NEW.name}")
print(f"   Range: {str(ds_new.time.values[0])[:10]} → {str(ds_new.time.values[-1])[:10]}  ({ds_new.sizes['time']} steps)")
ds_base.close()
ds_new.close()

# ---- CDO MERGETIME ---- #
FINAL_MERGED.parent.mkdir(parents=True, exist_ok=True)
print(f"\nMerging with CDO mergetime...")
print(f"   Output: {FINAL_MERGED}")

cmd = ["cdo", "mergetime", str(BASE_NC), str(MERGED_NEW), str(FINAL_MERGED)]
result = subprocess.run(cmd, text=True, capture_output=True)

print(result.stdout)
if result.stderr:
    print(result.stderr)

if result.returncode != 0:
    print("CDO mergetime failed.")
    sys.exit(1)

# ---- VERIFY ---- #
ds_final = xr.open_dataset(FINAL_MERGED)
print(f"Final range : {str(ds_final.time.values[0])[:10]} → {str(ds_final.time.values[-1])[:10]}")
print(f"   Total steps : {ds_final.sizes['time']}")
ds_final.close()

# ---- UPDATE SYMLINK ---- #
if LATEST_MERGED.is_symlink() or LATEST_MERGED.exists():
    LATEST_MERGED.unlink()
LATEST_MERGED.symlink_to(FINAL_MERGED)
print(f"🔗 Symlink updated: {LATEST_MERGED.name} → {FINAL_MERGED.name}")

print(f"\nMerge complete.")