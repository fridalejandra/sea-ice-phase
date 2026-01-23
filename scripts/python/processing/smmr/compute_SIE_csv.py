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
mask_var = "sector_id"

thr = 0.15

sectors = {
    1: "Weddell",
    2: "Amundsen_Bellingshausen",
    3: "Ross",
    4: "East_Antarctica",
    5: "King_Haakon"
}

# =====================================================
# load data
# =====================================================
ds   = xr.open_dataset(sic_file, chunks={"time": 365})
sic  = ds[sic_var]
area = xr.open_dataset(area_file)[area_var]
mask = xr.open_dataset(mask_file)[mask_var]

# =====================================================
# detect spatial dimensions (x/y safe)
# =====================================================
spatial_dims = [d for d in sic.dims if d != "time"]
if len(spatial_dims) != 2:
    raise ValueError(f"Expected 2 spatial dims, found {spatial_dims}")

print("Using spatial dimensions:", spatial_dims)

# =====================================================
# UNIT CONVERSION
# m^2 → million km^2
# =====================================================
area = area / 1e12
area.attrs["units"] = "million km^2"

# =====================================================
# Antarctic ocean mask (domain definition)
# =====================================================
antarctic_ocean = area.notnull()

# =====================================================
# binary ice mask
# =====================================================
ice = (sic >= thr)

# =====================================================
# PAN-ANTARCTIC SIE (sector-independent)
# =====================================================
sie_pan = (ice * area.where(antarctic_ocean)).sum(dim=spatial_dims)
sie_pan = sie_pan.rename("SIE_panAntarctic")

# =====================================================
# SECTOR SIE (diagnostic decomposition)
# =====================================================
sie_vars = [sie_pan]

for code, name in sectors.items():
    area_sector = area.where(mask == code)
    sie_sector = (ice * area_sector).sum(dim=spatial_dims)
    sie_sector = sie_sector.rename(f"SIE_{name}")
    sie_vars.append(sie_sector)

# =====================================================
# to CSV
# =====================================================
sie_ds = xr.merge(sie_vars)
df = sie_ds.to_dataframe().reset_index()

cols = ["time", "SIE_panAntarctic"] + [f"SIE_{v}" for v in sectors.values()]
df = df[cols]

df.to_csv(out_file, index=False)
print(f"Saved: {out_file}")

# =====================================================
# HARD SANITY CHECKS (do not remove)
# =====================================================
assert df["SIE_panAntarctic"].max() < 25, "Pan-Antarctic SIE exceeds physical limit"
assert df["SIE_panAntarctic"].min() > 0,  "Negative or zero SIE detected"