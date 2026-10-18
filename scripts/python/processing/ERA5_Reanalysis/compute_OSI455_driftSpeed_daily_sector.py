# =====================================================
# OSI-455 (via C3S CDS "satellite-sea-ice-drift") -> daily sector-mean CSV.
# One CDS request per year, process, delete. Checkpoints after every year.
# Output matches NSIDC0116_driftSpeed_daily_sector.csv in shape and units.
# =====================================================

import os
os.environ["OMP_NUM_THREADS"] = "1"

import glob, shutil, zipfile, tarfile
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import cdsapi
from pathlib import Path
from pyproj import CRS, Transformer

SCRATCH = os.path.expanduser("~/osi455_tmp")
SIE_GRID_FILE    = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
SECTOR_MASK_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"
OUT_DIR  = "/user/geog/falejandraperez/sea-ice-phase/results/OSI455"
OUT_FILE = "OSI455_driftSpeed_daily_sector.csv"
WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "osi455_to_sie_bilinear_weights.nc")

START_YEAR, END_YEAR = 1991, 2020
MONTHS = ["03", "04", "05", "06", "07", "08", "09", "10"]   # drop 03-05 for JJA/SON only
USE_STATUS_FLAG = True        # keep nominal-quality vectors only (status_flag == 30)
KM_PER_DAY_TO_MPS = 1000.0 / 86400.0

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(SCRATCH).mkdir(parents=True, exist_ok=True)

# ---- target grid + sectors (identical to the other scripts)
sie = xr.open_dataset(SIE_GRID_FILE)
sector_mask = xr.open_dataset(SECTOR_MASK_FILE)["sector_id"]
to_wgs = Transformer.from_crs(CRS.from_epsg(3412), CRS.from_epsg(4326), always_xy=True)
sxx, syy = np.meshgrid(sie.x.values, sie.y.values)
sie_lon, sie_lat = to_wgs.transform(sxx, syy)
sie = sie.assign_coords(lon=(("y", "x"), sie_lon), lat=(("y", "x"), sie_lat))
sid_values = [int(s) for s in np.unique(sector_mask.values[np.isfinite(sector_mask.values)]) if int(s) != 0]

client = cdsapi.Client()

def fetch_year(year):
    """Download one year (selected months) from CDS, extract, return list of SH .nc files."""
    ydir = os.path.join(SCRATCH, str(year))
    Path(ydir).mkdir(parents=True, exist_ok=True)
    target = os.path.join(ydir, f"osi455_{year}.zip")
    if not os.path.exists(target):
        client.retrieve("satellite-sea-ice-drift", {
            "variable": "all",
            "region": "southern_hemisphere",
            "year": [str(year)],
            "month": MONTHS,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "version": "1_0",
        }).download(target)
    # CDS hands back a zip (sometimes tar); handle both
    if zipfile.is_zipfile(target):
        with zipfile.ZipFile(target) as z: z.extractall(ydir)
    elif tarfile.is_tarfile(target):
        with tarfile.open(target) as t: t.extractall(ydir)
    else:
        # occasionally a single .nc is returned directly
        os.rename(target, target.replace(".zip", ".nc"))
    return sorted(glob.glob(os.path.join(ydir, "**", "*.nc"), recursive=True))

out_path = os.path.join(OUT_DIR, OUT_FILE)
rows, done = [], set()
if os.path.exists(out_path):
    prev = pd.read_csv(out_path, parse_dates=["date"])
    rows, done = prev.to_dict("records"), set(prev["date"].dt.year)
    print(f"Resuming: years done = {sorted(done)}")

regridder = None
for year in range(START_YEAR, END_YEAR + 1):
    if year in done:
        continue
    files = fetch_year(year)
    if not files:
        print(f"  [skip] {year}: no files"); continue

    ds = xr.concat([xr.open_dataset(f) for f in files], dim="time")
    # make 2-D coordinates visible to xesmf under the names it expects
    ren = {}
    for k in ("latitude", "Latitude"):  ren[k] = "lat" if k in ds else None
    for k in ("longitude", "Longitude"): ren[k] = "lon" if k in ds else None
    ds = ds.rename({k: v for k, v in ren.items() if v})
    if "lat" in ds and "lat" not in ds.coords:
        ds = ds.set_coords(["lat", "lon"])

    dX, dY = ds["dX"], ds["dY"]                      # km per 24 h
    if USE_STATUS_FLAG and "status_flag" in ds:
        ok = ds["status_flag"] == 30
        dX, dY = dX.where(ok), dY.where(ok)
    speed = (np.sqrt(dX**2 + dY**2) * KM_PER_DAY_TO_MPS).squeeze()   # m/s

    if regridder is None:
        regridder = xe.Regridder(ds, sie, method="bilinear",
                                 reuse_weights=os.path.exists(WEIGHTS_FILE), filename=WEIGHTS_FILE)
        print(f"Regridder ready: {WEIGHTS_FILE}")
        print(f"First file vars: {list(ds.data_vars)}")

    speed_rg = regridder(speed)                       # (time, y, x)
    for i, t in enumerate(pd.to_datetime(ds["time"].values)):
        day = speed_rg.isel(time=i).values
        for sid in sid_values:
            vals = day[sector_mask.values == sid]
            rows.append({"date": t.normalize(), "sector": sid,
                         "drift_speed_mps": float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan})
    ds.close()
    shutil.rmtree(os.path.join(SCRATCH, str(year)), ignore_errors=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  {year}: {len(files)} days -> checkpointed")

shutil.rmtree(SCRATCH, ignore_errors=True)
df = pd.DataFrame(rows)
print(f"\nSaved: {out_path}\nRows: {len(df):,}\nNegative: {(df.drift_speed_mps < 0).sum()}")
print(df.groupby(df['date'].dt.year)['drift_speed_mps'].mean().round(4).to_string())
