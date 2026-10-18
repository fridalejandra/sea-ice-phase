#!/usr/bin/env python
"""Does a sector-scale level drop explain the post-2016 compression of the ice season?
Predicted dFS = -dL / advance rate, dMS = dL / |retreat rate|, against measured static shifts.
Split as in Chapter 2: 1979-2015 vs 2016-2024 (1986, 1987 excluded)."""
import numpy as np, pandas as pd, xarray as xr
ROOT = "/user/geog/falejandraperez/sea-ice-phase"
SIC = f"{ROOT}/data/merged/merged_bootstrap_SH_latest.nc"
SECTORS = {1: "ABS", 2: "WS", 3: "KHV", 4: "EA", 5: "RS"}
ANOM = f"{ROOT}/data/anomalies/SMMR"
def _clim(tag, names):
    with xr.open_dataset(f"{ANOM}/{tag}_climatology.nc", decode_times=False) as c:
        for n in names:
            if n in c: return c[n].values
FS_CLIM = _clim("FS_static_thr15_k5", ["FS_static_thr15_k5_clim"])
MS_CLIM = _clim("MS_static_thr15_k5", ["MS_static_thr15_k5_clim_dsa", "MS_static_thr15_k5_clim"])
MEASURED = {"ABS": (5.0, 0.7), "WS": (9.5, -12.5), "KHV": (8.9, -7.7), "EA": (4.9, -7.3), "RS": (7.1, -4.0)}
CELL = 625.0  # km^2 per 25 km pixel (approximate; polar stereographic grid)

with xr.open_dataset(f"{ROOT}/data/canonical_sectors.nc") as m:
    sid = m["sector_id"].values; ocean = m["valid_ocean"].astype(bool).values
ds = xr.open_dataset(SIC)
t = pd.to_datetime(ds.time.values)
rows = []
for y in range(1979, 2026):
    idx = np.where(t.year == y)[0]
    if idx.size == 0: continue
    c = ds["N07_ICECON"].isel(time=idx).values.astype("float32")
    c = np.where(c < 1050, c, np.nan); c = c / 1000.0 if np.nanmax(c) > 1.5 else c
    ice = (c >= 0.15) & ocean
    for s in SECTORS:
        ext = ice[:, sid == s].sum(1) * CELL / 1e6
        rows += [{"date": d, "sector": SECTORS[s], "ext": e} for d, e in zip(t[idx], ext)]
df = pd.DataFrame(rows)
df = df[~((df.date.dt.month == 2) & (df.date.dt.day == 29))]
df["doy"] = df.date.dt.dayofyear - ((df.date.dt.is_leap_year) & (df.date.dt.month > 2)).astype(int)
df["yr"] = df.date.dt.year

print(f"{'sector':6s} {'dL(1e6 km2)':>11s} {'adv rate':>9s} {'ret rate':>9s} {'pred dFS':>9s} {'meas dFS':>9s} {'pred dMS':>9s} {'meas dMS':>9s}")
for lab in SECTORS.values():
    d = df[df.sector == lab]
    pre = d[(d.yr <= 2015) & ~d.yr.isin([1986, 1987])]
    post = d[(d.yr >= 2016) & (d.yr <= 2024)]
    dL = post.groupby("yr").ext.mean().mean() - pre.groupby("yr").ext.mean().mean()
    clim = pre.groupby("doy").ext.mean().rolling(15, center=True, min_periods=1).mean()
    rate = np.gradient(clim.values)                       # 1e6 km2 per day
    sec = (sid == [k for k, v in SECTORS.items() if v == lab][0]) & ocean
    fs_med = int(round(np.nanmedian(FS_CLIM[sec])))
    ms_med = int(round(np.nanmedian(MS_CLIM[sec]) + 227 - 1) % 365 + 1)   # DSA -> day of year
    wrap = lambda c0: [((c0 + i - 1) % 365) + 1 for i in range(-20, 21)]
    adv = rate[clim.index.get_indexer(wrap(fs_med))].mean()
    ret = rate[clim.index.get_indexer(wrap(ms_med))].mean()
    print(f"  {lab}: median FS day {fs_med}, median MS day {ms_med}")
    mfs, mms = MEASURED[lab]
    print(f"{lab:6s} {dL:11.3f} {adv:9.4f} {ret:9.4f} {-dL/adv:9.1f} {mfs:9.1f} {dL/abs(ret):9.1f} {mms:9.1f}")
