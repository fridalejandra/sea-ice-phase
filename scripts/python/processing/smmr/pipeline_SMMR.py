import subprocess
import os
import shutil
from datetime import datetime

# ---- CONFIG ---- #
SCRIPT_DIR = "/scripts/python/"
DATA_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/data/bootstrap_smmr/"
PHASE_DIR = "/Users/fridaperez/Developer/repos/phase_project/Stammerjohn_2008/"

END_DATE = datetime.today().strftime("%Y-%m-%d")

# File paths
granule_dir = os.path.join(DATA_DIR, "test_downloads")
merged_2024 = os.path.join(DATA_DIR, f"merged_bootstrap_SH_2024_until_{END_DATE}.nc")
final_merged = os.path.join(DATA_DIR, f"merged_bootstrap_extended_SH_until_{END_DATE}.nc")

# ---- UTILITY ---- #
def run_script(script_path, env_vars=None):
    print(f"\n🔧 Running {os.path.basename(script_path)}...\n")
    result = subprocess.run(
        ["python", script_path],
        capture_output=True,
        text=True,
        env={**os.environ, **(env_vars or {})}
    )
    print(result.stdout)
    if result.stderr:
        print("⚠️ STDERR:")
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"❌ {script_path} failed.")

# ---- STEP 1: DOWNLOAD ---- #
download_script = os.path.join(SCRIPT_DIR, "download_SH_2024.py")
if os.path.exists(granule_dir) and len(os.listdir(granule_dir)) > 0:
    print("✅ SH granules already downloaded. Skipping Step 1.")
else:
    run_script(download_script, env_vars={"END_DATE": END_DATE})

# ---- STEP 2: MERGE 2024 GRANULES ---- #
merge_2024_script = os.path.join(SCRIPT_DIR, "merge_SH_2024.py")
if os.path.exists(merged_2024):
    print(f"✅ 2024 merged file exists: {merged_2024}. Skipping Step 2.")
else:
    run_script(merge_2024_script, env_vars={"END_DATE": END_DATE})

# ---- STEP 3: MERGE INTO FINAL FILE ---- #
merge_final_script = os.path.join(SCRIPT_DIR, "merge_smmr.py")
if os.path.exists(final_merged):
    print(f"✅ Final merged file already exists: {final_merged}. Skipping Step 3.")
else:
    run_script(merge_final_script, env_vars={
        "END_DATE": END_DATE,
        "PHASE_DIR": PHASE_DIR
    })

# ---- CLEANUP ---- #
print("\n🧹 Cleaning up temporary files...")

if os.path.exists(merged_2024):
    os.remove(merged_2024)
    print(f"🗑 Deleted: {merged_2024}")

if os.path.exists(granule_dir):
    shutil.rmtree(granule_dir)
    print(f"🗑 Deleted granule directory: {granule_dir}")

print("\n✅ All done.")
print(f"📦 Final merged file located at: {final_merged}")
