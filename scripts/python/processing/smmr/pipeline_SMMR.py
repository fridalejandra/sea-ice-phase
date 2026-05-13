
"""
pipeline_SMMR.py  –  Main orchestrator
Runs the full pipeline to update the Bootstrap SIE CSV to today's date.

Steps:
  1. download_smmr.py   – fetch NSIDC-0079 SH granules via earthaccess
  2. merge_smmr.py      – merge new data with Stammerjohn 2008 base file
  3. compute_SIE_csv.py – compute daily SIE and write CSV

Usage:
  python pipeline_SMMR.py

To force re-running a step even if its output already exists, delete the
corresponding output file or pass --force on the command line.
"""

import subprocess
import sys
import shutil
from pathlib import Path
from config import (
    GRANULE_DIR, MERGED_2024, FINAL_MERGED, LATEST_MERGED,
    SIE_CSV, TODAY,
)

SCRIPT_DIR = Path(__file__).parent
FORCE = "--force" in sys.argv

# ---- UTILITY ---- #
def run(script_name):
    script = SCRIPT_DIR / script_name
    print(f"\n{'='*60}")
    print(f"🔧  Running {script_name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script)],
        text=True,
    )
    if result.returncode != 0:
        print(f"\n{script_name} failed (exit code {result.returncode}). Aborting.")
        sys.exit(result.returncode)

def skip(label, path):
    print(f"{label} already exists — skipping.")
    print(f"   {path}")

# ---- STEP 1: DOWNLOAD ---- #
already_downloaded = GRANULE_DIR.exists() and any(GRANULE_DIR.iterdir())
if not FORCE and already_downloaded:
    skip("Downloaded granules", GRANULE_DIR)
else:
    run("download_smmr.py")

# ---- STEP 2: MERGE ---- #
if not FORCE and FINAL_MERGED.exists():
    skip(f"Final merged file (until {TODAY})", FINAL_MERGED)
    # Still ensure the symlink is current
    if LATEST_MERGED.is_symlink() or LATEST_MERGED.exists():
        LATEST_MERGED.unlink()
    LATEST_MERGED.symlink_to(FINAL_MERGED)
    print(f"🔗  Symlink refreshed: {LATEST_MERGED.name} → {FINAL_MERGED.name}")
else:
    run("merge_smmr.py")

# ---- STEP 3: COMPUTE SIE CSV ---- #
if not FORCE and SIE_CSV.exists():
    skip("SIE CSV", SIE_CSV)
else:
    run("compute_SIE_csv.py")

# ---- CLEANUP ---- #
print(f"\n{'='*60}")
print("Cleaning up temporary files...")

if MERGED_2024.exists():
    MERGED_2024.unlink()
    print(f"   🗑  Deleted intermediate: {MERGED_2024.name}")

if GRANULE_DIR.exists():
    shutil.rmtree(GRANULE_DIR)
    print(f"   🗑  Deleted granule directory: {GRANULE_DIR}")

# ---- DONE ---- #
print(f"\n{'='*60}")
print("Pipeline complete!")
print(f"Merged NetCDF : {FINAL_MERGED}")
print(f"SIE CSV       : {SIE_CSV}")
print(f"{'='*60}\n")
