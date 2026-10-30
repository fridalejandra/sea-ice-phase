from pathlib import Path
import xarray as xr
ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
LATEST_MERGED = ROOT / "data/merged/merged_bootstrap_SH_latest.nc"
AREA_FILE     = ROOT / "data/NSIDC0771_CellArea_PS_S25km_v1.1.nc"
MASK_FILE     = ROOT / "data/canonical_sectors.nc"
SIE_CSV       = ROOT / "results/SIE/SIE_daily_sector_and_circumpolar_million_km2.csv"
SIC_VAR       = "N07_ICECON"
AREA_VAR      = "cell_area"
MASK_VAR      = "sector_id"
SIC_THRESHOLD = 0.15
# Sector names are READ FROM THE MASK FILE -- never hand-typed (see 2026-10-29 audit).
_m = xr.open_dataset(MASK_FILE)
SECTORS = {k: ("Ross" if str(_m[f"sector_{k}_name"].values).startswith("Ross") else
              str(_m[f"sector_{k}_name"].values).replace("\u2013", "_").replace(" ", "_").replace("_VII", ""))
           for k in range(1, 6)}
_m.close()
