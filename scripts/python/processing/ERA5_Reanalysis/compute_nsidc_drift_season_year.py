import os
os.environ["OMP_NUM_THREADS"] = "1"
import glob
import numpy as np, pandas as pd, xarray as xr, xesmf as xe
from pyproj import CRS, Transformer

DRIFT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/drift_nsidc"
SIE_GRID_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
OUT_NC = "/user/geog/falejandraperez/sea-ice-phase/results/NSIDC0116/NSIDC0116_driftSpeed_season_year.nc"
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nsidc_drift_to_sie_bilinear_weights.nc")
START_YEAR, END_YEAR = 1979, 2024

sie = xr.open_dataset(SIE_GRID_FILE)
to_wgs = Transformer.from_crs(CRS.from_epsg(3412), CRS.from_epsg(4326), always_xy=True)
sxx, syy = np.meshgrid(sie.x.values, sie.y.values)
sie_lon, sie_lat = to_wgs.transform(sxx, syy)
sie = sie.assign_coords(lon=(("y","x"), sie_lon), lat=(("y","x"), sie_lat))

sample = xr.open_dataset(sorted(glob.glob(f"{DRIFT_DIR}/icemotion_daily_sh_25km_*.nc"))[0])
ease = CRS.from_proj4("+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 +a=6371228 +b=6371228 +units=m +no_defs")
t = Transformer.from_crs(ease, CRS.from_epsg(4326), always_xy=True)
dxx, dyy = np.meshgrid(sample.x.values, sample.y.values)
dlon, dlat = t.transform(dxx, dyy)
sample = sample.assign_coords(lon=(("y","x"), dlon), lat=(("y","x"), dlat))
rg = xe.Regridder(sample, sie, method="bilinear", reuse_weights=os.path.exists(WEIGHTS), filename=WEIGHTS)
sample.close()

seasons = ["DJF","MAM","JJA","SON"]
def season_of(m): return "DJF" if m in (12,1,2) else "MAM" if m in (3,4,5) else "JJA" if m in (6,7,8) else "SON"
years = list(range(START_YEAR, END_YEAR+1))
ny, nx = sie.sizes["y"], sie.sizes["x"]
out = np.full((len(years), len(seasons), ny, nx), np.nan, dtype=np.float32)

for iy, year in enumerate(years):
    files = glob.glob(f"{DRIFT_DIR}/icemotion_daily_sh_25km_{year}*.nc")
    if not files: continue
    ds = xr.open_dataset(files[0])
    u = ds.u.where(ds.u != -9999) * 0.01; v = ds.v.where(ds.v != -9999) * 0.01
    sp = np.sqrt(u**2 + v**2).assign_coords(lon=(("y","x"), dlon), lat=(("y","x"), dlat))
    sp_rg = rg(sp).values
    months = np.array([pd.Timestamp(x.strftime("%Y-%m-%d")).month for x in ds.time.values])
    for isn, s in enumerate(seasons):
        sel = np.array([season_of(m) == s for m in months])
        if sel.any():
            with np.errstate(invalid="ignore"):
                out[iy, isn] = np.nanmean(sp_rg[sel], axis=0)
    ds.close(); print(year, end=" ", flush=True)

xr.Dataset({"drift_speed_mps": (("year","season","y","x"), out)},
           coords={"year": years, "season": seasons, "y": sie.y, "x": sie.x,
                   "lat": (("y","x"), sie_lat), "lon": (("y","x"), sie_lon)},
           attrs={"source": "NSIDC-0116 v4.1, regridded to NSIDC PS South 25 km", "units": "m s-1"}
          ).to_netcdf(OUT_NC)
print(f"\nSaved {OUT_NC}")
