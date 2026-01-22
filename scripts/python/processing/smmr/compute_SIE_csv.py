# calc_SIE_pan_and_sector.py
import xarray as xr
import numpy as np
from pathlib import Path

# =====================================================
# User settings
# =====================================================
sic_file   = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
area_file  = "/user/geog/falejandraperez/sea-ice-phase/data/NSIDC0771_CellArea_PS_S25km_v1.0.nc"
mask_file  = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"

out_dir = Path("/user/geog/falejandraperez/sea-ice-phase/scripts/R/Sea_Ice_Sheets")
out_file = out_dir / "SIE_daily_pan_and_sector.csv"

sic_var  = "N07_ICECON"
area_var = "cell_area"    # km^2
mask_var = "sector"

sic_threshold = 0.15

sectors = {
    1: "Weddell",
    2: "Amundsen_Bellingshausen",
    3: "Ross",
    4: "East_Antarctica",
    5: "King_Haakon"
}

# =====================================================
# Load data (lazy, cluster-safe)
# =====================================================
ds   = xr.open_dataset(sic_file, chunks={"time": 365})
area = xr.open_dataset(area_file)[area_var]
mask = xr.open_dataset(mask_file)[mask_var]

sic = ds[sic_var]

# =====================================================
# Binary ice mask
# =====================================================
ice = (sic >= sic_threshold).astype("int8")

# =====================================================
# Pan-Antarctic SIE
# =====================================================
sie_pan = (ice * area).sum(dim=("lat", "lon"))
sie_pan = sie_pan.rename("SIE_panAntarctic")

# =====================================================
# Sector-wise SIE
# =====================================================
sie_vars = [sie_pan]

for code, name in sectors.items():
    area_sector = area.where(mask == code)
    sie_sector = (ice * area_sector).sum(dim=("lat", "lon"))
    sie_sector = sie_sector.rename(f"SIE_{name}")
    sie_vars.append(sie_sector)

# =====================================================
# Combine → DataFrame → CSV
# =====================================================
sie_ds = xr.merge(sie_vars)

df = sie_ds.to_dataframe().reset_index()

# optional: cleaner column order
cols = ["time", "SIE_panAntarctic"] + [f"SIE_{v}" for v in sectors.values()]
df = df[cols]

df.to_csv(out_file, index=False)

print(f"Saved CSV: {out_file}")