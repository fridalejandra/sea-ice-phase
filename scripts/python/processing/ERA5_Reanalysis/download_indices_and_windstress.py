"""
Climate Indices & ERA5 Wind Stress Download Script
===================================================
Downloads all pre-computed climate indices from Table 1 of Eabry et al. (2025)
and the ERA5 wind stress components needed for wind stress curl (Ch4).

Indices downloaded:
  1. Marshall SAM index (monthly)     — BAS website
  2. Goyal ZW3 magnitude (monthly)    — Mendeley
  3. Raphael ZW3 index (daily)        — Mendeley
  4. TPI index (monthly)              — NOAA PSL
  5. Daily AAO/SAM index (daily)      — NOAA CPC

ERA5 variables downloaded:
  6. Eastward wind stress  (tau_x)    — CDS API, daily, 1979–2023
  7. Northward wind stress (tau_y)    — CDS API, daily, 1979–2023

References:
  Marshall (2003)       https://legacy.bas.ac.uk/met/gjma/sam.html
  Goyal et al. (2022)   https://doi.org/10.17632/382gmc8937.1
  Raphael (2004, 2007)  https://doi.org/10.17632/382gmc8937.1
  Henley et al. (2015)  https://psl.noaa.gov/data/timeseries/IPOTPI/
  Mo (2000)             https://www.cpc.ncep.noaa.gov/products/precip/CWlink/
                        daily_ao_index/aao/aao.shtml

Usage (on cluster, in a screen session):
    screen -S indices
    cd /user/geog/falejandraperez/sea-ice-phase/scripts/python/processing/ERA5_Reanalysis
    python download_indices_and_windstress.py

Check progress:
    tail -f /user/geog/falejandraperez/sea-ice-phase/logs/indices_download.log
"""

import cdsapi
import logging
import urllib.request
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATHS — matched to your existing cluster structure
# =============================================================================

BASE_DATA   = Path("/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5")
INDEX_DIR   = Path("/user/geog/falejandraperez/sea-ice-phase/data/indices")
LOG_DIR     = Path("/user/geog/falejandraperez/sea-ice-phase/logs")

# Wind stress saves alongside your existing ERA5 data
WINDSTRESS_DIR = BASE_DATA / "wind_stress"

# =============================================================================
# SETUP
# =============================================================================

INDEX_DIR.mkdir(parents=True, exist_ok=True)
WINDSTRESS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "indices_download.log"),
    ]
)
log = logging.getLogger(__name__)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def already_exists(filepath: Path, min_size_bytes: int = 500) -> bool:
    """Return True if file exists and is non-empty."""
    if filepath.exists() and filepath.stat().st_size > min_size_bytes:
        log.info(f"  Already exists, skipping: {filepath.name}")
        return True
    return False


def download_url(url: str, dest: Path, description: str = "") -> bool:
    """
    Download a file from a URL using urllib.
    Returns True on success, False on failure.
    """
    if already_exists(dest):
        return True
    log.info(f"Downloading {description}")
    log.info(f"  URL : {url}")
    log.info(f"  Dest: {dest}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response:
            with open(dest, "wb") as f:
                shutil.copyfileobj(response, f)
        size_kb = dest.stat().st_size / 1000
        log.info(f"  Saved: {dest.name} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        log.error(f"  FAILED: {description} | {e}")
        return False


# =============================================================================
# 1. MARSHALL SAM INDEX (monthly, station-based)
#    SAM = P_40S - P_65S, normalised zonal-mean SLP
#    Marshall (2003), updated regularly
# =============================================================================

def download_marshall_sam():
    log.info("\n--- 1. Marshall SAM Index (monthly) ---")
    dest = INDEX_DIR / "marshall_sam_monthly.txt"
    url  = "https://legacy.bas.ac.uk/met/gjma/sam.html"

    # The SAM data is embedded in the HTML page as a text table
    # We download the raw data file directly
    data_url = "https://legacy.bas.ac.uk/met/gjma/newsam.1957.2007.txt"
    success = download_url(data_url, dest,
                           "Marshall SAM index (monthly, 1957–present)")
    if not success:
        log.warning("  Try downloading manually from: "
                    "https://legacy.bas.ac.uk/met/gjma/sam.html")
        log.warning("  Save as: " + str(dest))


# =============================================================================
# 2 & 3. GOYAL ZW3 MAGNITUDE + RAPHAEL ZW3 INDEX
#    Both available from the same Mendeley dataset
#    Goyal et al. (2022): https://doi.org/10.17632/382gmc8937.1
#    Monthly Goyal ZW3mag = sqrt(PC1^2 + PC2^2)
#    Daily Raphael ZW3: normalised deviations at 49S/50E, 49S/166E, 49S/76W
# =============================================================================

def download_zw3_indices():
    log.info("\n--- 2 & 3. ZW3 Indices (Goyal monthly + Raphael daily) ---")

    # Mendeley direct download URL for the dataset zip
    # Dataset: "Southern Hemisphere ZW3 indices"
    # DOI: 10.17632/382gmc8937.1
    mendeley_url = ("https://data.mendeley.com/public-files/datasets/"
                    "382gmc8937/files/")

    # The Mendeley dataset contains multiple files — we note what to expect
    # after downloading and unzipping
    dest_zip = INDEX_DIR / "zw3_indices_mendeley.zip"
    dest_dir = INDEX_DIR / "zw3_indices"
    dest_dir.mkdir(exist_ok=True)

    log.info("  Mendeley datasets require manual download via browser.")
    log.info("  Please download from:")
    log.info("  https://data.mendeley.com/datasets/382gmc8937/1")
    log.info("  Click 'Download All' and save the zip to:")
    log.info(f"  {dest_zip}")
    log.info("  Then re-run this script — it will unzip automatically.")

    # If the zip was manually downloaded, unzip it
    if dest_zip.exists() and dest_zip.stat().st_size > 10_000:
        log.info("  Found zip file — extracting...")
        try:
            with zipfile.ZipFile(dest_zip, 'r') as z:
                z.extractall(dest_dir)
            log.info(f"  Extracted to: {dest_dir}")
            log.info("  Contents:")
            for f in sorted(dest_dir.rglob("*")):
                if f.is_file():
                    log.info(f"    {f.name}")
        except Exception as e:
            log.error(f"  Failed to extract zip: {e}")
    else:
        log.warning("  Zip not found — manual download required (see above).")


# =============================================================================
# 4. TPI INDEX (monthly, Tripole Index for IPO)
#    Based on Pacific SST anomalies
#    Henley et al. (2015)
#    NOAA PSL: https://psl.noaa.gov/data/timeseries/IPOTPI/
# =============================================================================

def download_tpi():
    log.info("\n--- 4. TPI Index (monthly, Henley et al. 2015) ---")
    dest = INDEX_DIR / "tpi_monthly.txt"

    # NOAA PSL provides the unfiltered TPI as a plain text file
    url = "https://psl.noaa.gov/data/correlation/tpi.data"
    success = download_url(url, dest, "TPI index (monthly, unfiltered)")
    if not success:
        log.warning("  Try downloading manually from: "
                    "https://psl.noaa.gov/data/timeseries/IPOTPI/")
        log.warning("  Save as: " + str(dest))


# =============================================================================
# 5. DAILY AAO/SAM INDEX
#    Computed from daily 700 hPa height fields projected onto AO EOF
#    Mo (2000); NOAA CPC
# =============================================================================

def download_daily_aao():
    log.info("\n--- 5. Daily AAO/SAM Index (NOAA CPC) ---")
    dest = INDEX_DIR / "daily_aao_sam.txt"

    # NOAA CPC provides the daily AAO index as a plain text file
    url = ("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/"
           "daily_ao_index/aao/monthly.aao.index.b79.current.ascii.table")
    success = download_url(url, dest, "Daily AAO/SAM index (NOAA CPC)")

    if not success:
        # Try alternative URL format
        alt_url = ("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/"
                   "daily_ao_index/aao/aao.shtml")
        log.warning("  Primary URL failed. Visit manually:")
        log.warning(f"  {alt_url}")
        log.warning("  Download the daily index file and save as:")
        log.warning(f"  {dest}")


# =============================================================================
# 6 & 7. ERA5 WIND STRESS (daily, Ch4)
#    Eastward  turbulent surface stress (tau_x)
#    Northward turbulent surface stress (tau_y)
#    Wind stress curl = d(tau_y)/dx - d(tau_x)/dy (computed in post-processing)
#    Saved as one file per year, matching your existing winds/ convention
# =============================================================================

def download_wind_stress_year(client: cdsapi.Client,
                               variable_name: str,
                               cds_name: str,
                               year: int) -> None:
    """Download daily wind stress for one year, one file per year."""
    year_dir = WINDSTRESS_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    fname = year_dir / f"era5_windstress_{variable_name}_{year}.nc"

    if already_exists(fname, min_size_bytes=100_000):
        return

    log.info(f"  Downloading {variable_name} | {year}")
    try:
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type"  : "reanalysis",
                "variable"      : cds_name,
                "year"          : str(year),
                "month"         : [f"{m:02d}" for m in range(1, 13)],
                "day"           : [f"{d:02d}" for d in range(1, 32)],
                "time"          : ["12:00"],   # 12 UTC, matches your mslp/winds
                "area"          : [-40, -180, -90, 180],  # 40S–90S
                "format"        : "netcdf",
                "grid"          : "1.0/1.0",
            },
            str(fname),
        )
        size_mb = fname.stat().st_size / 1e6
        log.info(f"    Saved: {fname.name} ({size_mb:.1f} MB)")
    except Exception as e:
        log.error(f"    FAILED: {variable_name} {year} | {e}")


def download_wind_stress():
    log.info("\n--- 6 & 7. ERA5 Wind Stress (daily, 1979–2023) ---")
    log.info("  tau_x: eastward_turbulent_surface_stress")
    log.info("  tau_y: northward_turbulent_surface_stress")
    log.info(f"  Saving to: {WINDSTRESS_DIR}")

    client = cdsapi.Client()

    wind_stress_vars = [
        ("tau_x", "eastward_turbulent_surface_stress"),
        ("tau_y", "northward_turbulent_surface_stress"),
    ]

    years = list(range(1979, 2024))

    for var_name, cds_name in wind_stress_vars:
        log.info(f"\n  Variable: {var_name.upper()}")
        for year in years:
            download_wind_stress_year(client, var_name, cds_name, year)
        log.info(f"  {var_name.upper()} complete.")


# =============================================================================
# SUMMARY — print what was downloaded and what needs manual steps
# =============================================================================

def print_summary():
    log.info("\n" + "=" * 60)
    log.info("DOWNLOAD SUMMARY")
    log.info("=" * 60)

    files = {
        "Marshall SAM (monthly)"    : INDEX_DIR / "marshall_sam_monthly.txt",
        "TPI index (monthly)"       : INDEX_DIR / "tpi_monthly.txt",
        "Daily AAO/SAM"             : INDEX_DIR / "daily_aao_sam.txt",
        "ZW3 zip (manual download)" : INDEX_DIR / "zw3_indices_mendeley.zip",
    }

    for name, path in files.items():
        status = "✓ Downloaded" if (path.exists() and
                                     path.stat().st_size > 500) \
                 else "✗ Missing"
        log.info(f"  {status} | {name}")
        log.info(f"            {path}")

    # Wind stress
    tau_x_count = len(list(WINDSTRESS_DIR.rglob("*tau_x*.nc")))
    tau_y_count = len(list(WINDSTRESS_DIR.rglob("*tau_y*.nc")))
    log.info(f"\n  Wind stress tau_x: {tau_x_count} yearly files")
    log.info(f"  Wind stress tau_y: {tau_y_count} yearly files")

    log.info("\nMANUAL STEPS STILL NEEDED:")
    log.info("  1. ZW3 indices: download zip from Mendeley and re-run")
    log.info("     https://data.mendeley.com/datasets/382gmc8937/1")
    log.info("  2. If Marshall SAM failed: visit")
    log.info("     https://legacy.bas.ac.uk/met/gjma/sam.html")
    log.info("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Climate Indices + ERA5 Wind Stress Download")
    log.info(f"Index directory    : {INDEX_DIR}")
    log.info(f"Wind stress dir    : {WINDSTRESS_DIR}")
    log.info(f"Started            : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info("=" * 60)

    # --- Pre-computed indices (fast, no CDS needed) ---
    download_marshall_sam()
    download_tpi()
    download_daily_aao()
    download_zw3_indices()   # prints manual instructions + unzips if present

    # --- ERA5 wind stress (slow, uses CDS API) ---
    download_wind_stress()

    # --- Summary ---
    print_summary()

    log.info(f"\nFinished: {datetime.now():%Y-%m-%d %H:%M:%S}")