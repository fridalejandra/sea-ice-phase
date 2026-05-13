"""
compute_SIE_csv.py  –  Step 3
Reads the latest merged Bootstrap SIC NetCDF and computes daily Sea Ice
Extent (SIE) for circumpolar Antarctic and 5 standard sectors.

Output: CSV [time, SIE_circumpolar, SIE_<sector>, ...]  in million km²
"""

import sys
import xarray as xr
from config import (
    LATEST_MERGED, AREA_FILE, MASK_FILE, SIE_CSV,
    SIC_VAR, AREA_VAR, MASK_VAR, SIC_THRESHOLD, SECTORS,
)

# ---- CHECKS ---- #
for path, label in [
    (LATEST_MERGED, "LATEST_MERGED"),
    (AREA_FILE,     "AREA_FILE"),
    (MASK_FILE,     "MASK_FILE"),
]:
    if not path.exists():
        print(f"{label} not found:\n   {path}")
        sys.exit(1)

# ---- LOAD ---- #
print(f"📂 SIC  : {LATEST_MERGED.name}")
ds   = xr.open_dataset(LATEST_MERGED, chunks={"time": 365})
area = xr.open_dataset(AREA_FILE)[AREA_VAR]
mask = xr.open_dataset(MASK_FILE)[MASK_VAR]

# ---- IDENTIFY SIC VARIABLE ---- #
if SIC_VAR in ds.data_vars:
    sic_var = SIC_VAR
else:
    candidates = [v for v in ds.data_vars if v.endswith("_ICECON")]
    if not candidates:
        raise ValueError(f"No *_ICECON variable found. Available: {list(ds.data_vars)}")
    sic_var = candidates[0]
    print(f"SIC_VAR '{SIC_VAR}' not found; using '{sic_var}' instead.")

sic = ds[sic_var]

# ---- SPATIAL DIMS ---- #
spatial_dims = [d for d in sic.dims if d != "time"]
if len(spatial_dims) != 2:
    raise ValueError(f"Expected 2 spatial dims, found: {spatial_dims}")
print(f"   Variable     : {sic_var}")
print(f"   Spatial dims : {spatial_dims}")
print(f"   Time range   : {str(sic.time.values[0])[:10]} → {str(sic.time.values[-1])[:10]}")
print(f"   Time steps   : {sic.sizes['time']}")

# ---- UNIT CONVERSION: m² → million km² ---- #
area = area / 1e12
area.attrs["units"] = "million km²"

# ---- CLEAN SIC ---- #
sic = sic.where((sic >= 0) & (sic <= 1))

# ---- BINARY ICE MASK ---- #
ice = sic >= SIC_THRESHOLD

# ---- CIRCUMPOLAR SIE ---- #
antarctic_ocean = mask.notnull()
sie_circ = (ice * area.where(antarctic_ocean)).sum(dim=spatial_dims)
sie_circ = sie_circ.rename("SIE_circumpolar")

# ---- SECTOR SIE ---- #
sie_vars = [sie_circ]
for code, name in SECTORS.items():
    sie_sector = (ice * area.where(mask == code)).sum(dim=spatial_dims)
    sie_sector = sie_sector.rename(f"SIE_{name}")
    sie_vars.append(sie_sector)

# ---- TO CSV ---- #
SIE_CSV.parent.mkdir(parents=True, exist_ok=True)
sie_ds = xr.merge(sie_vars)
df = sie_ds.to_dataframe().reset_index()
cols = ["time", "SIE_circumpolar"] + [f"SIE_{v}" for v in SECTORS.values()]
df = df[cols]
df.to_csv(SIE_CSV, index=False)

print(f"\nCSV saved: {SIE_CSV}")
print(f"   Rows: {len(df)}  |  Columns: {list(df.columns)}")

# ---- SANITY CHECKS ---- #
assert df["SIE_circumpolar"].max() < 25, "FAIL: Circumpolar SIE exceeds physical limit (25 M km²)"
assert df["SIE_circumpolar"].min() >  0, "FAIL: Non-positive SIE detected"
print("Sanity checks passed.")