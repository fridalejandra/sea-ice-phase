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

sic_threshold = 0.15

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
area = xr.open_dataset(area_file)[area_var]
mask = xr.open_dataset(mask_file)[mask_var]

sic = ds[sic_var]

# =====================================================
# detect spatial dimensions (x/y safe)
# =====================================================
spatial_dims = [d for d in sic.dims if d != "time"]
if len(spatial_dims) != 2:
    raise ValueError(f"Expected 2 spatial dims, found {spatial_dims}")

print("Using spatial dimensions:", spatial_dims)

# =====================================================
# UNIT CONVERSION (DO THIS ONCE)
# m^2 → million km^2
# =====================================================
area = area / 1e12
area.attrs["units"] = "million km^2"

# =====================================================
# binary ice mask
# =====================================================
ice = (sic >= thr).astype("int8")

# =====================================================
# pan-Antarctic SIE
# =====================================================
sie_pan = (ice * area).sum(dim=spatial_dims)
sie_pan = sie_pan.rename("SIE_panAntarctic")

# =====================================================
# sector-wise SIE
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