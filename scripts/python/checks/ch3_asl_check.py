#!/usr/bin/env python
"""ch3_asl_check.py -- test the Raphael & Hobbs (2014) ASL->Ross prediction in forms closer to theirs.
Independent of the pipeline: daily sector SIE from raw SIC + canonical mask (identity by longitude)."""
import re, numpy as np, pandas as pd, xarray as xr
R = "/user/geog/falejandraperez/sea-ice-phase"
m = xr.open_dataset(f"{R}/data/canonical_sectors.nc"); sid = m["sector_id"].values
lon = (m["lonE"] if "lonE" in m else m["lon"]).values % 360
B = {"ABS": (250, 290), "WS": (290, 346), "KH": (346, 71), "EA": (71, 162), "RA": (162, 250)}
ident = {}
for k in range(1, 6):
    L = float(np.nanmedian(np.where(sid == k, lon, np.nan)))
    for nm, (a, b) in B.items():
        if (a < b and a <= L < b) or (a > b and (L >= a or L < b)): ident[k] = nm
print("ids by longitude:", ident)
area = xr.open_dataset(f"{R}/data/NSIDC0771_CellArea_PS_S25km_v1.1.nc")["cell_area"].values / 1e12
ds = xr.open_dataset(f"{R}/data/merged/merged_bootstrap_SH_latest.nc"); v = [x for x in ds.data_vars if x.endswith("_ICECON")][0]
rows = []
for y in range(1979, 2026):
    s = ds[v].sel(time=str(y)).values; ice = (s >= 0.15) & (s <= 1.0); t = pd.to_datetime(ds.time.sel(time=str(y)).values)
    for k in range(1, 6):
        rows.append(pd.DataFrame({"date": t, "sec": ident[k], "sie": (ice * (area * (sid == k))).sum(axis=(1, 2))}))
d = pd.concat(rows, ignore_index=True); d["year"] = d.date.dt.year; d["mon"] = d.date.dt.month
SEAS = {"annual": list(range(1, 13)), "advance": [3, 4, 5, 6, 7, 8], "retreat": [10, 11, 12, 1]}
def sy(df, months):
    wrap = [mm for mm in months if mm < months[0]]; return df.year - df.mon.isin(wrap).astype(int)
ice = {}
for sec, g in d.groupby("sec"):
    w = g[g.mon.between(6, 11)]; s = g[g.mon.between(1, 3)]
    out = pd.DataFrame({"amp": w.groupby("year").sie.max() - s.groupby("year").sie.min(),
                        "maxdoy": w.loc[w.groupby("year").sie.idxmax()].set_index("year").date.dt.dayofyear})
    for sn, ms in SEAS.items():
        gg = g[g.mon.isin(ms)].copy(); gg["sy"] = sy(gg, ms); out[f"sie_{sn}"] = gg.groupby("sy").sie.mean()
    ice[sec] = out
a = pd.read_csv(f"{R}/data/indices/asli_era5_v3-latest.csv", comment="#")
tc = [c for c in a if c.lower() in ("time", "date")][0]; tt = pd.to_datetime(a[tc])
col = lambda pat: [c for c in a if re.search(pat, c.lower().replace("_", ""))][0]
print("ASL columns used:", col(r"relcen"), col(r"^lon$"))
A = pd.DataFrame({"year": tt.dt.year, "mon": tt.dt.month, "depth": a[col(r"relcen")].astype(float),
                  "lonA": a[col(r"^lon$")].astype(float) % 360})
def idx(var, ms):
    z = A[A.mon.isin(ms)].copy(); z["sy"] = sy(z, ms); return z.groupby("sy")[var].mean()
def detr(s):
    s = s.dropna(); t = s.index.values.astype(float); return s - np.polyval(np.polyfit(t, s.values, 1), t)
def r(x, y, lo=1979, hi=2024):
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1).loc[lo:hi].dropna()
    return np.nan if len(j) < 8 else np.corrcoef(detr(j.x), detr(j.y))[0, 1]
secs = ["ABS", "WS", "KH", "EA", "RA"]
print("\nASL index (season) vs ice quantity, detrended r 1979-2024.  |r| > 0.29 ~ p<0.05.  Ross = RA")
print(f"  {'index':18s}{'ice quantity':16s}" + "".join(f"{s:>7s}" for s in secs))
for var, vlab in (("depth", "depth"), ("lonA", "longitude")):
    for sn, ms in SEAS.items():
        x = idx(var, ms)
        for q in ("amp", "maxdoy", f"sie_{sn}"):
            print(f"  {vlab+' '+sn:18s}{q:16s}" + "".join(f"{r(x, ice[s][q]):+7.2f}" for s in secs))
print("\nASL depth vs same-season mean extent, 1979-2000 -> 2001-2024")
for sn, ms in SEAS.items():
    x = idx("depth", ms)
    print(f"  {sn:8s}" + "".join(f"   {s} {r(x, ice[s][f'sie_{sn}'], 1979, 2000):+.2f}->{r(x, ice[s][f'sie_{sn}'], 2001, 2024):+.2f}" for s in secs))
