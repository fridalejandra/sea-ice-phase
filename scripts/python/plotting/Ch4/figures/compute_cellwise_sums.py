#!/usr/bin/env python
# Per-cell, per-season, per-year SUFFICIENT STATISTICS for  U_ice = A*U_wind + C
# on the SIE 25 km grid:  n, sum Uw, sum Ui, sum |Uw|^2, sum conj(Uw)Ui, sum |Ui|^2.
# A, C, R^2 for any set of years follow in closed form; the year-block bootstrap
# becomes a resample of ~30 small arrays.  cellwise_residual_map.py does the rest.
# Output: results/ch4/cellwise/sums_<syear>.nc  (December -> next year's DJF).
# Restartable: finished season-years are never overwritten.  Run in screen.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import glob, sys
import numpy as np, pandas as pd, xarray as xr, xesmf as xe
from tqdm import tqdm
from pyproj import CRS, Transformer

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
DRIFT_DIR = f"{ROOT}/data/drift_nsidc"
ERA5_BASE = f"{ROOT}/data/Reanalysis_ERA5/winds"
SIE_GRID_FILE = f"{ROOT}/data/merged/merged_bootstrap_SH_latest.nc"
WEIGHTS_DIR = f"{ROOT}/scripts/python/processing/ERA5_Reanalysis"
W_NSIDC = f"{WEIGHTS_DIR}/nsidc_drift_to_sie_bilinear_weights.nc"
W_ERA5 = f"{WEIGHTS_DIR}/era5_to_sie_bilinear_weights_v2.nc"
OUT_DIR = f"{ROOT}/results/ch4/cellwise"
os.makedirs(OUT_DIR, exist_ok=True)
Y0, Y1 = 1992, 2024
if len(sys.argv) == 3:
    Y0, Y1 = int(sys.argv[1]), int(sys.argv[2])
UNIT_SCALE = 0.01
SEASONS = ["DJF", "MAM", "JJA", "SON"]

def season_of(month):
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA"}.get(month, "SON")

sie = xr.open_dataset(SIE_GRID_FILE)
to_wgs = Transformer.from_crs(CRS.from_epsg(3412), CRS.from_epsg(4326), always_xy=True)
sxx, syy = np.meshgrid(sie.x.values, sie.y.values)
sie_lon, sie_lat = to_wgs.transform(sxx, syy)
sie = sie.assign_coords(lon=(("y", "x"), sie_lon), lat=(("y", "x"), sie_lat))
NY, NX = len(sie.y), len(sie.x)

sample = xr.open_dataset(sorted(glob.glob(f"{DRIFT_DIR}/icemotion_daily_sh_25km_*.nc"))[0])
ease = CRS.from_proj4("+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 +a=6371228 +b=6371228 +units=m +no_defs")
ease_to_wgs = Transformer.from_crs(ease, CRS.from_epsg(4326), always_xy=True)
wgs_to_ease = Transformer.from_crs(CRS.from_epsg(4326), ease, always_xy=True)
dxx, dyy = np.meshgrid(sample.x.values, sample.y.values)
dlon, dlat = ease_to_wgs.transform(dxx, dyy)
eps = 1e-3
x0, y0 = wgs_to_ease.transform(dlon, dlat)
xe_, ye_ = wgs_to_ease.transform(dlon + eps, dlat)
xn_, yn_ = wgs_to_ease.transform(dlon, np.minimum(dlat + eps, -0.001))
ex, ey = xe_ - x0, ye_ - y0
nx, ny = xn_ - x0, yn_ - y0
en = np.hypot(ex, ey); nn = np.hypot(nx, ny)
ex, ey, nx, ny = ex / en, ey / en, nx / nn, ny / nn
polar = dlat < -89.9
ex[polar] = ey[polar] = nx[polar] = ny[polar] = np.nan
sample = sample.assign_coords(lon=(("y", "x"), dlon), lat=(("y", "x"), dlat))
rg_nsidc = xe.Regridder(sample, sie, method="bilinear", reuse_weights=os.path.exists(W_NSIDC), filename=W_NSIDC)
sample.close()
sample = xr.open_dataset(sorted(glob.glob(f"{ERA5_BASE}/{Y0}/*.nc"))[0])
rg_era5 = xe.Regridder(sample, sie, method="bilinear", reuse_weights=os.path.exists(W_ERA5), filename=W_ERA5)
sample.close()
print("regridders ready; grid", NY, "x", NX)

def new_acc():
    return dict(n=np.zeros((NY, NX), np.int16),
                sw=np.zeros((NY, NX), np.complex64), si=np.zeros((NY, NX), np.complex64),
                sww=np.zeros((NY, NX), np.float32), swi=np.zeros((NY, NX), np.complex64),
                sii=np.zeros((NY, NX), np.float32))

def write(syear, acc):
    out = f"{OUT_DIR}/sums_{syear}.nc"
    if os.path.exists(out):
        print(f"  keep existing {out}"); return
    ds = xr.Dataset({k: (("season", "y", "x"), np.stack([acc[s][k] for s in SEASONS]))
                     for k in ("n", "sw", "si", "sww", "swi", "sii")},
                    coords=dict(season=SEASONS, y=sie.y.values, x=sie.x.values, syear=syear))
    for k in ("sw", "si", "swi"):
        ds[k + "_re"], ds[k + "_im"] = ds[k].real.astype(np.float32), ds[k].imag.astype(np.float32)
        ds = ds.drop_vars(k)
    ds.attrs.update(model="U_ice = A*U_wind + C, daily, east/north frame, SIE grid",
                    units="m s-1", december="counted in following season-year DJF")
    ds.to_netcdf(out)
    print(f"  wrote {out}")

acc = {}
def get(syear, season):
    if syear not in acc:
        acc[syear] = {s: new_acc() for s in SEASONS}
    return acc[syear][season]

for year in range(Y0, Y1 + 1):
    if os.path.exists(f"{OUT_DIR}/sums_{year}.nc") and os.path.exists(f"{OUT_DIR}/sums_{year + 1}.nc"):
        print(f"{year}: already done"); continue
    files = glob.glob(f"{DRIFT_DIR}/icemotion_daily_sh_25km_{year}*.nc")
    if not files:
        print(f"  [skip] no NSIDC file for {year}"); continue
    ds = xr.open_dataset(files[0])
    u = ds.u.where(ds.u != -9999) * UNIT_SCALE
    v = ds.v.where(ds.v != -9999) * UNIT_SCALE
    ue = (u * ex + v * ey).assign_coords(lon=(("y", "x"), dlon), lat=(("y", "x"), dlat))
    un = (u * nx + v * ny).assign_coords(lon=(("y", "x"), dlon), lat=(("y", "x"), dlat))
    ice_e, ice_n = rg_nsidc(ue).values.astype(np.float32), rg_nsidc(un).values.astype(np.float32)
    ice_dates = {pd.Timestamp(str(t)[:10]).normalize(): i for i, t in enumerate(ds.time.values)}
    ds.close()
    for f in tqdm(sorted(glob.glob(f"{ERA5_BASE}/{year}/*.nc")), desc=str(year)):
        w = xr.open_dataset(f)
        if "valid_time" in w.dims:
            w = w.rename({"valid_time": "time"})
        ts = pd.Timestamp(w.time.values.item()).normalize()
        i = ice_dates.get(ts)
        if i is None:
            w.close(); continue
        we = xr.DataArray(w.u10.values.squeeze(), dims=("latitude", "longitude"),
                          coords={"latitude": w.latitude, "longitude": w.longitude})
        wn = xr.DataArray(w.v10.values.squeeze(), dims=("latitude", "longitude"),
                          coords={"latitude": w.latitude, "longitude": w.longitude})
        uw = (rg_era5(we).values + 1j * rg_era5(wn).values).astype(np.complex64)
        w.close()
        ui = (ice_e[i] + 1j * ice_n[i]).astype(np.complex64)
        ok = np.isfinite(ui) & np.isfinite(uw)
        ui, uw = np.where(ok, ui, 0), np.where(ok, uw, 0)
        a = get(ts.year + (ts.month == 12), season_of(ts.month))
        a["n"] += ok
        a["sw"] += uw
        a["si"] += ui
        a["sww"] += (np.abs(uw) ** 2).astype(np.float32)
        a["swi"] += np.conj(uw) * ui
        a["sii"] += (np.abs(ui) ** 2).astype(np.float32)
    if year in acc:
        write(year, acc.pop(year))
for syear in sorted(acc):
    write(syear, acc.pop(syear))
print("done")
