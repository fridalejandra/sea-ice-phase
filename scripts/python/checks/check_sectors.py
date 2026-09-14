"""
check_sectors.py  --  cut the Raphael & Hobbs (2014) sectors from one day of the
NSIDC-0079 v4 concentration grid and compare with the sector CSV.

Written 2026-09-14 to diagnose the sector labels in
data/raw/SIE_daily_sector_and_circumpolar_million_km2.csv.

Run on the cluster (where the granules live). Output for 2000-09-15:

    NSIDC0079_SEAICE_PS_S25km_20000915_v4.0.nc  F13_ICECON
    circumpolar   19.97
    King_Haakon                  5.89
    East_Antarctica              2.89
    Ross                         5.53
    Amundsen_Bellingshausen      1.40
    Weddell                      4.26

The CSV for calendar 2000 had max 1.49 under "SIE_Weddell" and max 4.31 under
"SIE_Amundsen_Bellingshausen": those two column names were exchanged. The
mask file (data/canonical_sectors.nc) is correct; see check_mask.py. The two
headers were swapped in the CSV on 2026-09-14 (git history shows the one-line
change). Nominal 625 km^2 cells are used here, so numbers differ from the
CSV by a few percent; the point is which column is near 4 and which near 1.4.

Bounds (degrees east) read off Fig. 1 of Raphael & Hobbs (2014), GRL 41:
King Hakon VII 346-71, East Antarctica 71-162, Ross-Amundsen 162-250,
Amundsen-Bellingshausen 250-290, Weddell 290-346.
"""
import glob
import numpy as np
import xarray as xr
from pyproj import Transformer

DATE = "20000915"
RAW = "/user/geog/falejandraperez/sea-ice-phase/data/smmr/raw/"

f = sorted(glob.glob(f"{RAW}*{DATE}*.nc"))[0]
ds = xr.open_dataset(f)
var = [v for v in ds.data_vars if v.upper().endswith("ICECON")][0]
c = ds[var].squeeze().values.astype(float)
if np.nanmax(c) > 1.5:
    c = c / 100.0
x, y = np.meshgrid(ds["x"].values, ds["y"].values)
lon, lat = Transformer.from_crs("EPSG:3412", "EPSG:4326", always_xy=True).transform(x, y)
lon = lon % 360
ice = (c >= 0.15) & (c <= 1.0)
cell = 25.0 * 25.0 / 1e6  # 10^6 km^2 per cell, nominal

sectors = {"King_Haakon": (346, 71), "East_Antarctica": (71, 162), "Ross": (162, 250),
           "Amundsen_Bellingshausen": (250, 290), "Weddell": (290, 346)}
print(f, var)
print(f"circumpolar  {ice.sum() * cell:6.2f}")
for name, (a, b) in sectors.items():
    m = ((lon >= a) & (lon < b)) if a < b else ((lon >= a) | (lon < b))
    print(f"{name:26s} {(ice & m).sum() * cell:6.2f}")
