"""
config.py  –  shared paths and settings for the SMMR Bootstrap SIE pipeline.

Edit ONLY this file when paths change (new computer, new data location, etc.)
All other scripts import from here.
"""

from pathlib import Path
from datetime import datetime

# ============================================================
# ROOT DIRECTORIES  ← update these when you change machines
# ============================================================
REPO_DIR        = Path("/user/geog/falejandraperez/sea-ice-phase")

# ============================================================
# DATA DIRECTORIES  (derived from root — usually no need to edit)
# ============================================================
DATA_DIR        = REPO_DIR / "data"
SMMR_DIR        = DATA_DIR / "bootstrap_smmr"
MERGED_DIR      = DATA_DIR / "merged"
RESULTS_DIR     = REPO_DIR / "results" / "SIE"

# ============================================================
# STATIC INPUT FILES
# ============================================================
BASE_NC         = MERGED_DIR / "SMMR_merged_1979_06302024.nc"
AREA_FILE       = DATA_DIR / "NSIDC0771_CellArea_PS_S25km_v1.0.nc"
MASK_FILE       = DATA_DIR / "canonical_sectors.nc"

# ============================================================
# DYNAMIC PATHS  (dated filenames)
# ============================================================
TODAY           = datetime.today().strftime("%Y-%m-%d")

GRANULE_DIR     = SMMR_DIR / "downloads"
MERGED_2024     = SMMR_DIR / f"merged_bootstrap_SH_2024_until_{TODAY}.nc"
FINAL_MERGED    = MERGED_DIR / f"merged_bootstrap_SH_until_{TODAY}.nc"

# Stable symlink / alias always pointing at the latest merged file
# (compute_SIE_csv.py reads this so it never needs a dated name)
LATEST_MERGED   = MERGED_DIR / "merged_bootstrap_SH_latest.nc"

# ============================================================
# OUTPUT CSV
# ============================================================
SIE_CSV         = RESULTS_DIR / "SIE_daily_sector_and_circumpolar_million_km2.csv"

# ============================================================
# SCIENCE SETTINGS
# ============================================================
DOWNLOAD_START  = "1979-01-01"   # earthaccess temporal start
SIC_THRESHOLD   = 0.15           # 15 % extent threshold
SIC_VAR         = "N07_ICECON"   # expected variable name (fallback: any *_ICECON)
AREA_VAR        = "cell_area"
MASK_VAR        = "sector_id"

SECTORS = {
    1: "Weddell",
    2: "Amundsen_Bellingshausen",
    3: "Ross",
    4: "East_Antarctica",
    5: "King_Haakon",
}