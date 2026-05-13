"""
pipeline_SMMR.py  –  Main orchestrator

Steps:
  1. merge_granules.py  – concatenate the 567 new daily .nc files into one
  2. merge_smmr.py      – append new data to historical base record
  3. compute_SIE_csv.py – compute daily SIE and write CSV

Usage:
  python pipeline_SMMR.py           # skips steps whose output already exists
  python pipeline_SMMR.py --force   # re-runs all steps

To fetch new granules in future runs, update BASE_NC_END_DATE in config.py
and run download_smmr.py before this pipeline.
"""

import subprocess
import sys
import shutil
from pathlib import Path
from config import GRANULE_DIR, MERGED_NEW, FINAL_MERGED, LATEST_MERGED, SIE_CSV, TODAY

SCRIPT_DIR = Path(__file__).parent
FORCE = "--force" in sys.argv

# ---- UTILITY ---- #
def run(script_name):
    script = SCRIPT_DIR / script_name
    print(f"\n{'='*60}")
    print(f"🔧  Running {script_name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(script)], text=True)
    if result.returncode != 0:
        print(f"\n{script_name} failed (exit code {result.returncode}). Aborting.")
        sys.exit(result.returncode)

def skip(label, path):
    print(f"{label} already exists — skipping.")
    print(f"   {path}")

# ---- STEP 1: MERGE NEW GRANULES ---- #
if not FORCE and MERGED_NEW.exists():
    skip("Merged new granules", MERGED_NEW)
else:
    run("merge_granules.py")

# ---- STEP 2: MERGE WITH HISTORICAL BASE ---- #
if not FORCE and FINAL_MERGED.exists():
    skip(f"Final merged file (until {TODAY})", FINAL_MERGED)
    if LATEST_MERGED.is_symlink() or LATEST_MERGED.exists():
        LATEST_MERGED.unlink()
    LATEST_MERGED.symlink_to(FINAL_MERGED)
    print(f"🔗  Symlink refreshed → {FINAL_MERGED.name}")
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
if MERGED_NEW.exists():
    MERGED_NEW.unlink()
    print(f"   🗑  Deleted: {MERGED_NEW.name}")

# ---- DONE ---- #
print(f"\n{'='*60}")
print("Pipeline complete!")
print(f"Merged NetCDF : {FINAL_MERGED}")
print(f"SIE CSV       : {SIE_CSV}")
print(f"{'='*60}\n")