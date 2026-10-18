import numpy as np, xarray as xr
R = "/user/geog/falejandraperez/sea-ice-phase"
FILES = {"wind":  f"{R}/results/ERA5/ERA5_uv_season_year.nc",
         "drift": f"{R}/results/NSIDC0116/NSIDC0116_uv_season_year.nc"}
sec  = xr.open_dataset(f"{R}/data/canonical_sectors.nc")
sid  = sec["sector_id"]
names = {int(i): str(sec[f"sector_{i}_name"].values) for i in range(1, 6)}
land = xr.open_dataset(f"{R}/data/landmask_sic.nc")
land = land[[k for k in land.data_vars if land[k].ndim == 2][0]].values.astype(bool)

def anomaly(da, ydim):
    return (da.sel({ydim: slice(2016, 2024)}).mean(ydim)
            - da.sel({ydim: slice(1992, 2015)}).mean(ydim))

for prod, f in FILES.items():
    d = xr.open_dataset(f)
    print(f"\n#### {prod}: {f}\n{d}\n")
    ydim = [k for k in d.dims if "year" in k.lower()][0]
    sdim = [k for k in d.dims if "season" in k.lower()][0]
    uvar = [v for v in d.data_vars if v.lower().startswith("u")][0]
    vvar = [v for v in d.data_vars if v.lower().startswith("v")][0]
    du, dv = anomaly(d[uvar], ydim), anomaly(d[vvar], ydim)
    spd  = np.hypot(d[uvar], d[vvar])
    dspd = anomaly(spd, ydim)
    base = spd.sel({ydim: slice(1992, 2015)}).mean(ydim)
    print(f"{'sector':28s} {'season':6s} {'du':>7s} {'dv':>7s} {'|anom|':>7s} {'dir':>5s} {'dspeed':>8s} {'dspeed%':>8s}")
    for i, nm in names.items():
        m = (sid == i) & ~land
        for s in d[sdim].values:
            u = du.sel({sdim: s}).where(m).mean().item()
            v = dv.sel({sdim: s}).where(m).mean().item()
            ds = dspd.sel({sdim: s}).where(m).mean().item()
            b  = base.sel({sdim: s}).where(m).mean().item()
            ang = (np.degrees(np.arctan2(u, v)) + 360) % 360   # direction the anomaly blows TOWARD, deg from N
            print(f"{nm:28s} {str(s):6s} {u:+7.3f} {v:+7.3f} {np.hypot(u,v):7.3f} {ang:5.0f} {ds:+8.3f} {100*ds/b:+7.1f}%")
