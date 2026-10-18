# Seasonal-mean u, v, speed per year on the SIE grid, for NSIDC drift and ERA5 wind.
# Vectors are in the polar-stereographic x/y frame so they can be drawn directly.
import os, sys, glob
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, xarray as xr, xesmf as xe
from pyproj import CRS, Transformer

WHICH = sys.argv[1] if len(sys.argv) > 1 else "drift"     # "drift" or "wind"
SIE_GRID_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
DRIFT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/drift_nsidc"
ERA5_BASE = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/winds"
OUT = {"drift": "/user/geog/falejandraperez/sea-ice-phase/results/NSIDC0116/NSIDC0116_uv_season_year.nc",
       "wind":  "/user/geog/falejandraperez/sea-ice-phase/results/ERA5/ERA5_uv_season_year.nc"}[WHICH]
HERE = os.path.dirname(os.path.abspath(__file__))
START_YEAR, END_YEAR = 1979, 2024
seasons = ["DJF","MAM","JJA","SON"]
def season_of(m): return "DJF" if m in (12,1,2) else "MAM" if m in (3,4,5) else "JJA" if m in (6,7,8) else "SON"

sie = xr.open_dataset(SIE_GRID_FILE)
to_wgs = Transformer.from_crs(CRS.from_epsg(3412), CRS.from_epsg(4326), always_xy=True)
sxx, syy = np.meshgrid(sie.x.values, sie.y.values)
sie_lon, sie_lat = to_wgs.transform(sxx, syy)
sie = sie.assign_coords(lon=(("y","x"), sie_lon), lat=(("y","x"), sie_lat))
ny, nx = sie.sizes["y"], sie.sizes["x"]
years = list(range(START_YEAR, END_YEAR+1))
out = np.full((len(years), len(seasons), 3, ny, nx), np.nan, dtype=np.float32)   # u, v, speed

if WHICH == "drift":
    sample = xr.open_dataset(sorted(glob.glob(f"{DRIFT_DIR}/icemotion_daily_sh_25km_*.nc"))[0])
    ease = CRS.from_proj4("+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 +a=6371228 +b=6371228 +units=m +no_defs")
    t = Transformer.from_crs(ease, CRS.from_epsg(4326), always_xy=True)
    dxx, dyy = np.meshgrid(sample.x.values, sample.y.values); dlon, dlat = t.transform(dxx, dyy)
    sample = sample.assign_coords(lon=(("y","x"), dlon), lat=(("y","x"), dlat))
    rg = xe.Regridder(sample, sie, "bilinear", reuse_weights=True, filename=f"{HERE}/nsidc_drift_to_sie_bilinear_weights.nc")
    sample.close()
    for iy, year in enumerate(years):
        files = glob.glob(f"{DRIFT_DIR}/icemotion_daily_sh_25km_{year}*.nc")
        if not files: continue
        ds = xr.open_dataset(files[0])
        u = (ds.u.where(ds.u != -9999) * 0.01).assign_coords(lon=(("y","x"), dlon), lat=(("y","x"), dlat))
        v = (ds.v.where(ds.v != -9999) * 0.01).assign_coords(lon=(("y","x"), dlon), lat=(("y","x"), dlat))
        # EASE-South and PS-South (both lon_0=0, azimuthal about the pole) share axis directions
        U, V = rg(u).values, rg(v).values; S = np.sqrt(U**2 + V**2)
        months = np.array([pd.Timestamp(x.strftime("%Y-%m-%d")).month for x in ds.time.values])
        for isn, s in enumerate(seasons):
            sel = np.array([season_of(m) == s for m in months])
            if sel.any():
                with np.errstate(invalid="ignore"):
                    out[iy, isn, 0] = np.nanmean(U[sel], 0); out[iy, isn, 1] = np.nanmean(V[sel], 0); out[iy, isn, 2] = np.nanmean(S[sel], 0)
        ds.close(); print(year, end=" ", flush=True)
    attrs = {"source": "NSIDC-0116 v4.1", "units": "m s-1", "frame": "NSIDC PS South x/y"}

else:
    sample = xr.open_dataset(sorted(glob.glob(f"{ERA5_BASE}/{START_YEAR}/*.nc"))[0])
    rg = xe.Regridder(sample, sie, "bilinear", reuse_weights=True, filename=f"{HERE}/era5_to_sie_bilinear_weights_v2.nc")
    lon2d = np.broadcast_to(np.deg2rad(sample.longitude.values)[None, :], (sample.sizes["latitude"], sample.sizes["longitude"]))
    sample.close()
    cosl, sinl = np.cos(lon2d), np.sin(lon2d)
    for iy, year in enumerate(years):
        files = sorted(glob.glob(f"{ERA5_BASE}/{year}/*.nc"))
        if not files: continue
        sums = np.zeros((len(seasons), 3, ny, nx), np.float64); cnt = np.zeros((len(seasons), 1, ny, nx), np.int32)
        for f in files:
            ds = xr.open_dataset(f, engine="netcdf4")
            if "valid_time" in ds.dims: ds = ds.rename({"valid_time": "time"})
            m = pd.Timestamp(ds.time.values.item()).month
            ue, vn = ds.u10.values.squeeze(), ds.v10.values.squeeze()
            # rotate east/north -> PS-South x/y (lon_0 = 0: +y toward 0E, +x toward 90E)
            ux = ue * cosl + vn * sinl; uy = -ue * sinl + vn * cosl
            mk = lambda a: xr.DataArray(a, dims=("latitude","longitude"), coords={"latitude": ds.latitude, "longitude": ds.longitude})
            U, V = rg(mk(ux)).values, rg(mk(uy)).values; S = np.sqrt(U**2 + V**2)
            isn = seasons.index(season_of(m))
            for k, A in enumerate((U, V, S)):
                ok = np.isfinite(A); sums[isn, k][ok] += A[ok]
            cnt[isn, 0] += np.isfinite(U)
            ds.close()
        with np.errstate(invalid="ignore", divide="ignore"):
            out[iy] = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        print(year, end=" ", flush=True)
    attrs = {"source": "ERA5 10 m wind, daily 12 UTC", "units": "m s-1", "frame": "NSIDC PS South x/y"}

xr.Dataset({"u": (("year","season","y","x"), out[:, :, 0]), "v": (("year","season","y","x"), out[:, :, 1]),
            "speed": (("year","season","y","x"), out[:, :, 2])},
           coords={"year": years, "season": seasons, "y": sie.y, "x": sie.x,
                   "lat": (("y","x"), sie_lat), "lon": (("y","x"), sie_lon)}, attrs=attrs).to_netcdf(OUT)
print(f"\nSaved {OUT}")
