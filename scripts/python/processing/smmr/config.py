"""
config.py  –  shared paths and settings for the SMMR Bootstrap SIE pipeline.
Edit ONLY this file when paths change. All other scripts import from here.
"""

from pathlib import Path
from datetime import datetime

# ============================================================
# ROOT DIRECTORIES  ← update these when you change machines
# ============================================================
REPO_DIR         = Path("/user/geog/falejandraperez/sea-ice-phase")

# ============================================================
# DATA DIRECTORIES
# ============================================================
DATA_DIR         = REPO_DIR / "data"
SMMR_DIR         = DATA_DIR / "bootstrap_smmr"
MERGED_DIR       = DATA_DIR / "merged"
RESULTS_DIR      = REPO_DIR / "results" / "SIE"

# ============================================================
# STATIC INPUT FILES
# ============================================================
BASE_NC          = MERGED_DIR / "SMMR_merged_1979_06302024.nc"  # 1978-11-01 → 2024-06-30
AREA_FILE        = DATA_DIR / "NSIDC0771_CellArea_PS_S25km_v1.0.nc"
MASK_FILE        = DATA_DIR / "canonical_sectors.nc"

# ============================================================
# DOWNLOAD DATE RANGE
# Only fetch granules after the base file ends.
# Update BASE_NC_END_DATE if you ever replace the base file.
# ============================================================
BASE_NC_END_DATE = "2024-07-01"   # day after SMMR_merged_1979_06302024.nc ends
TODAY            = datetime.today().strftime("%Y-%m-%d")

# ============================================================
# DYNAMIC PATHS
# ============================================================
GRANULE_DIR      = SMMR_DIR / "downloads"
MERGED_NEW       = SMMR_DIR / f"merged_new_until_{TODAY}.nc"
FINAL_MERGED     = MERGED_DIR / f"merged_bootstrap_SH_until_{TODAY}.nc"
LATEST_MERGED    = MERGED_DIR / "merged_bootstrap_SH_latest.nc"

# ============================================================
# OUTPUT CSV
# ============================================================
SIE_CSV          = RESULTS_DIR / "SIE_daily_sector_and_circumpolar_million_km2.csv"

# ============================================================
# SCIENCE SETTINGS
# ============================================================
SIC_THRESHOLD    = 0.15
SIC_VAR          = "N07_ICECON"   # fallback: any *_ICECON variable
AREA_VAR         = "cell_area"
MASK_VAR         = "sector_id"

SECTORS = {
    1: "Weddell",
    2: "Amundsen_Bellingshausen",
    3: "Ross",
    4: "East_Antarctica",
    5: "King_Haakon",
}