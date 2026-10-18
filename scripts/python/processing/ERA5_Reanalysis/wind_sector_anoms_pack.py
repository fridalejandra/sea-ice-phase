import numpy as np, xarray as xr
R = "/user/geog/falejandraperez/sea-ice-phase"
w = xr.open_dataset(f"{R}/results/ERA5/ERA5_uv_season_year.nc")
d = xr.open_dataset(f"{R}/results/NSIDC0116/NSIDC0116_uv_season_year.nc")
sec = xr.open_dataset(f"{R}/data/canonical_sectors.nc"); sid = sec["sector_id"]
names = {int(i): str(sec[f"sector_{i}_name"].values) for i in range(1, 6)}
pack = d["speed"].notnull()                      # ice present that season-year
wu, wv, ws = (w[k].where(pack) for k in ("u", "v", "speed"))
def an(x): return x.sel(year=slice(2016,2024)).mean("year") - x.sel(year=slice(1992,2015)).mean("year")
du, dv, ds = an(wu), an(wv), an(ws); base = ws.sel(year=slice(1992,2015)).mean("year")
print(f"{'sector':28s} {'season':6s} {'du':>7s} {'dv':>7s} {'griddir':>7s} {'dspeed%':>8s}")
for i, nm in names.items():
    m = sid == i
    for s in w.season.values:
        u = du.sel(season=s).where(m).mean().item(); v = dv.sel(season=s).where(m).mean().item()
        p = 100 * ds.sel(season=s).where(m).mean().item() / base.sel(season=s).where(m).mean().item()
        print(f"{nm:28s} {s:6s} {u:+7.3f} {v:+7.3f} {(np.degrees(np.arctan2(u,v))+360)%360:7.0f} {p:+7.1f}%")
