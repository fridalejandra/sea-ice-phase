"""
check_mask.py  --  list each sector id in data/canonical_sectors.nc with the
name stored in the file and the longitudes its cells actually cover.

Written 2026-09-14. Output on the cluster:

    AB:[250,290), WE:[290,346), KH:[346,360)U[0,71), EA:[71,162), RA:[162,250)
    sector_id 1: Amundsen-Bellingshausen    lon  250.0E to  290.0E
    sector_id 2: Weddell                    lon  290.0E to  345.9E
    sector_id 3: King Haakon VII            lon  346.0E to   71.0E
    sector_id 4: East Antarctica            lon   71.0E to  162.0E
    sector_id 5: Ross-Amundsen              lon  162.0E to  250.0E

So the mask is correct and consistent with Raphael & Hobbs (2014). The
exchange of the Weddell and Amundsen-Bellingshausen columns in the sector
CSV came from the id-to-name dictionary (SECTORS in the config imported by
compute_SIE_csv.py), which must have had ids 1 and 2 the wrong way round.
Any future regeneration of the CSV must use
    SECTORS = {1: "Amundsen_Bellingshausen", 2: "Weddell", 3: "King_Haakon",
               4: "East_Antarctica", 5: "Ross"}
and should be checked against this script and check_sectors.py.
"""
import numpy as np
import xarray as xr

m = xr.open_dataset("canonical_sectors.nc")
print(m.attrs["sector_bounds_degE"])
sid = m["sector_id"].values
lonE = m["lonE"].values
lat = m["lat"].values
for code in np.unique(sid[sid > 0]):
    name = str(m[f"sector_{code}_name"].values) if f"sector_{code}_name" in m else "?"
    sel = (sid == code) & (lat < -55) & (lat > -75)
    lons = np.sort(lonE[sel])
    gaps = np.diff(np.concatenate([lons, [lons[0] + 360]]))
    i = np.argmax(gaps)
    print(f"sector_id {code}: {name:26s} lon {lons[(i + 1) % len(lons)]:6.1f}E to {lons[i]:6.1f}E")
