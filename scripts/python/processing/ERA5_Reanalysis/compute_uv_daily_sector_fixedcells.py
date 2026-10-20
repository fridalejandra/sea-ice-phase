#!/usr/bin/env python
# =====================================================
# Daily sector-mean VECTORS of NSIDC-0116 ice drift and ERA5 10 m wind,
# both in an east/north frame, averaged over the FIXED cells used in
# make_ch4_figures.py (pre-period mean drift > 0.03 m/s, >= 15 valid pre
# years, >= 3 valid post years; one mask per season).
#
# Output: results/ch4/tables/uv_daily_sector_fixedcells_<Y0>-<Y1>.csv
#   date, sector, season, ui_e, ui_n, uw_e, uw_n, n_cells
#
# Feeds windfactor_analysis.py (daily complex regression ice = A*wind + C).
# Run in a screen session; the ERA5 loop is one file per day.
# =====================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import glob
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from tqdm import tqdm
from pyproj import CRS, Transformer

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
DRIFT_DIR = f"{ROOT}/data/drift_nsidc"
ERA5_BASE = f"{ROOT}/data/Reanalysis_ERA5/winds"
SIE_GRID_FILE = f"{ROOT}/data/merged/merged_bootstrap_SH_latest.nc"
SECTOR_MASK_FILE = f"{ROOT}/data/canonical_sectors.nc"
DRIFT_SEASON_NC = f"{ROOT}/results/NSIDC0116/NSIDC0116_uv_season_year.nc"
WEIGHTS_DIR = f"{ROOT}/scripts/python/processing/ERA5_Reanalysis"
W_NSIDC = f"{WEIGHTS_DIR}/nsidc_drift_to_sie_bilinear_weights.nc"
W_ERA5 = f"{WEIGHTS_DIR}/era5_to_sie_bilinear_weights_v2.nc"
OUT_DIR = f"{ROOT}/results/ch4/tables"

Y0, Y1 = 1992, 2024
BASE0, BASE1, POST0, POST1 = 1992, 2015, 2016, 2020
UNIT_SCALE = 0.01          # NSIDC cm/s -> m/s
os.makedirs(OUT_DIR, exist_ok=True)
OUT = f"{OUT_DIR}/uv_daily_sector_fixedcells_{Y0}-{Y1}.csv"


def season_of(month):
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA"}.get(month, "SON")


# ---- target grid + sector mask
sie = xr.open_dataset(SIE_GRID_FILE)
sector = xr.open_dataset(SECTOR_MASK_FILE)["sector_id"].values
to_wgs = Transformer.from_crs(CRS.from_epsg(3412), CRS.from_epsg(4326), always_xy=True)
sxx, syy = np.meshgrid(sie.x.values, sie.y.values)
sie_lon, sie_lat = to_wgs.transform(sxx, syy)
sie = sie.assign_coords(lon=(("y", "x"), sie_lon), lat=(("y", "x"), sie_lat))
sids = [int(s) for s in np.unique(sector[np.isfinite(sector)]) if int(s) != 0]

# ---- fixed cell masks, one per season (same definition as make_ch4_figures.fixed_mask_series)
d = xr.open_dataset(DRIFT_SEASON_NC)["speed"]
dpre, dpost = d.sel(year=slice(BASE0, BASE1)), d.sel(year=slice(POST0, POST1))
fixed = ((dpre.mean("year") > 0.03) & (np.isfinite(dpre).sum("year") >= 15)
         & (np.isfinite(dpost).sum("year") >= 3))
FIXED = {s: fixed.sel(season=s).values for s in ["DJF", "MAM", "JJA", "SON"]}
print("fixed cells per season:", {k: int(v.sum()) for k, v in FIXED.items()})

# ---- NSIDC source grid (EASE south, laea) + local east/north unit vectors in grid frame
sample = xr.open_dataset(sorted(glob.glob(f"{DRIFT_DIR}/icemotion_daily_sh_25km_*.nc"))[0])
ease = CRS.from_proj4("+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 +a=6371228 +b=6371228 +units=m +no_defs")
ease_to_wgs = Transformer.from_crs(ease, CRS.from_epsg(4326), always_xy=True)
wgs_to_ease = Transformer.from_crs(CRS.from_epsg(4326), ease, always_xy=True)
dxx, dyy = np.meshgrid(sample.x.values, sample.y.values)
dlon, dlat = ease_to_wgs.transform(dxx, dyy)
eps = 1e-3
x0, y0 = wgs_to_ease.transform(dlon, dlat)
xe_, ye_ = wgs_to_ease.transform(dlon + eps, dlat)            # step east
xn_, yn_ = wgs_to_ease.transform(dlon, np.minimum(dlat + eps, -0.001))  # step north
ex, ey = xe_ - x0, ye_ - y0
nx, ny = xn_ - x0, yn_ - y0
en = np.hypot(ex, ey); nn = np.hypot(nx, ny)
ex, ey, nx, ny = ex / en, ey / en, nx / nn, ny / nn
polar = dlat < -89.9
ex[polar] = ey[polar] = nx[polar] = ny[polar] = np.nan
sample = sample.assign_coords(lon=(("y", "x"), dlon), lat=(("y", "x"), dlat))
rg_nsidc = xe.Regridder(sample, sie, method="bilinear", reuse_weights=os.path.exists(W_NSIDC), filename=W_NSIDC)
sample.close()

# ---- ERA5 regridder
sample = xr.open_dataset(sorted(glob.glob(f"{ERA5_BASE}/{Y0}/*.nc"))[0])
rg_era5 = xe.Regridder(sample, sie, method="bilinear", reuse_weights=os.path.exists(W_ERA5), filename=W_ERA5)
sample.close()
print("regridders ready")


def sector_means(field_e, field_n, season):
    """Mean east/north components over fixed(season) & sector, for each sector id."""
    out = {}
    fm = FIXED[season]
    for sid in sids:
        m = fm & (sector == sid) & np.isfinite(field_e) & np.isfinite(field_n)
        n = int(m.sum())
        out[sid] = (float(field_e[m].mean()) if n else np.nan, float(field_n[m].mean()) if n else np.nan, n)
    return out


# ---- ICE: one file per year, all days regridded at once
ice_rows = {}
for year in tqdm(range(Y0, Y1 + 1), desc="NSIDC"):
    files = glob.glob(f"{DRIFT_DIR}/icemotion_daily_sh_25km_{year}*.nc")
    if not files:
        print(f"  [skip] no NSIDC file for {year}"); continue
    ds = xr.open_dataset(files[0])
    u = ds.u.where(ds.u != -9999) * UNIT_SCALE
    v = ds.v.where(ds.v != -9999) * UNIT_SCALE
    ue = (u * ex + v * ey).assign_coords(lon=(("y", "x"), dlon), lat=(("y", "x"), dlat))
    un = (u * nx + v * ny).assign_coords(lon=(("y", "x"), dlon), lat=(("y", "x"), dlat))
    ue_rg, un_rg = rg_nsidc(ue), rg_nsidc(un)
    for i, t in enumerate(ds.time.values):
        ts = pd.Timestamp(t.strftime("%Y-%m-%d")) if hasattr(t, "strftime") else pd.Timestamp(t).normalize()
        sm = sector_means(ue_rg.isel(time=i).values, un_rg.isel(time=i).values, season_of(ts.month))
        for sid, (e, n, k) in sm.items():
            ice_rows[(ts, sid)] = (e, n, k)
    ds.close()

# ---- WIND: one file per day
wind_rows = {}
for year in tqdm(range(Y0, Y1 + 1), desc="ERA5"):
    files = sorted(glob.glob(f"{ERA5_BASE}/{year}/*.nc"))
    if not files:
        print(f"  [skip] no ERA5 files for {year}"); continue
    for f in tqdm(files, desc=str(year), leave=False):
        ds = xr.open_dataset(f)
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        ts = pd.Timestamp(ds.time.values.item()).normalize()
        ue = xr.DataArray(ds.u10.values.squeeze(), dims=("latitude", "longitude"),
                          coords={"latitude": ds.latitude, "longitude": ds.longitude})
        un = xr.DataArray(ds.v10.values.squeeze(), dims=("latitude", "longitude"),
                          coords={"latitude": ds.latitude, "longitude": ds.longitude})
        ue_rg, un_rg = rg_era5(ue).values, rg_era5(un).values
        sm = sector_means(ue_rg, un_rg, season_of(ts.month))
        for sid, (e, n, k) in sm.items():
            wind_rows[(ts, sid)] = (e, n)
        ds.close()

# ---- join and write
rows = []
for (ts, sid), (ie, inn, k) in ice_rows.items():
    w = wind_rows.get((ts, sid))
    if w is None:
        continue
    rows.append(dict(date=ts, sector=sid, season=season_of(ts.month),
                     ui_e=ie, ui_n=inn, uw_e=w[0], uw_n=w[1], n_cells=k))
df = pd.DataFrame(rows).sort_values(["date", "sector"])
df.to_csv(OUT, index=False)
print(f"\nSaved {OUT}\n{len(df):,} rows, {df.date.min().date()} to {df.date.max().date()}")
print(df.groupby('season').n_cells.mean().round(0))
