#!/usr/bin/env python
"""ch3_independent_check.py -- rebuild Ch3's key numbers from raw inputs, independent of the pipeline.
Reads ONLY gridded SIC, canonical_sectors.nc (sector identity by LONGITUDE), cell areas, and index files."""
import re, numpy as np, pandas as pd, xarray as xr
R = "/user/geog/falejandraperez/sea-ice-phase"
m = xr.open_dataset(f"{R}/data/canonical_sectors.nc")
sid = m["sector_id"]; lon = m["lonE"] if "lonE" in m else (m["lon"] % 360)
BOUNDS = {"ABS": (250, 290), "WS": (290, 346), "KH": (346, 71), "EA": (71, 162), "RA": (162, 250)}
ident = {}
for k in range(1, 6):
    L = float(np.nanmedian(np.where(sid.values == k, lon.values, np.nan) % 360))
    for nm, (a, b) in BOUNDS.items():
        if (a < b and a <= L < b) or (a > b and (L >= a or L < b)): ident[k] = nm
print("sector_id -> region by median longitude:", ident)
assert sorted(ident.values()) == sorted(BOUNDS), "longitude test did not identify all five sectors"
area = xr.open_dataset(f"{R}/data/NSIDC0771_CellArea_PS_S25km_v1.1.nc")["cell_area"].values / 1e12
ds = xr.open_dataset(f"{R}/data/merged/merged_bootstrap_SH_latest.nc")
v = [x for x in ds.data_vars if x.endswith("_ICECON")][0]
masks = {ident[k]: (sid.values == k) for k in range(1, 6)}
rows = []
for y in range(1979, 2026):
    s = ds[v].sel(time=str(y)).values; ice = (s >= 0.15) & (s <= 1.0)
    t = pd.to_datetime(ds.time.sel(time=str(y)).values)
    for nm, mk in masks.items():
        rows.append(pd.DataFrame({"date": t, "sec": nm, "sie": (ice * (area * mk)).sum(axis=(1, 2))}))
d = pd.concat(rows); d["year"] = d.date.dt.year; d["mon"] = d.date.dt.month
def per_year(g):
    w = g[g.mon.between(6, 11)]; s = g[g.mon.between(1, 3)]
    return pd.Series({"mean": g.sie.mean(), "amp": w.sie.max() - s.sie.min(), "maxdoy": w.loc[w.sie.idxmax(), "date"].dayofyear})
Y = d.groupby(["sec", "year"]).apply(per_year).reset_index()
P = pd.read_csv(f"{R}/data/ch3/annual_params.csv"); P = P[P.period == "FULL"]
NAME = {"ABS": "SIE_Amundsen_Bellingshausen", "WS": "SIE_Weddell", "KH": "SIE_King_Haakon", "EA": "SIE_East_Antarctica", "RA": "SIE_Ross"}
print("\n(1) independent vs pipeline (max abs difference over 1979-2025)")
for sec, col in NAME.items():
    a = Y[Y.sec == sec].set_index("year"); b = P[P.sector == col].set_index("Year")
    j = a.join(b[["sie_annual", "amplitude_raw_yr", "max_doy_raw"]], how="inner")
    print(f"  {sec:3s}  mean {abs(j['mean']-j.sie_annual).max():.3f}   amp {abs(j.amp-j.amplitude_raw_yr).max():.3f}"
          f"   maxdoy {abs(j.maxdoy-j.max_doy_raw).max():.0f} d   (typical mean {j['mean'].mean():.2f}, amp {j.amp.mean():.2f})")
def rd_table(p):
    rows = []
    for line in open(p):
        t = line.split()
        if len(t) >= 13 and re.fullmatch(r"\d{4}", t[0]): rows += [(int(t[0]), i + 1, float(t[i + 1])) for i in range(12)]
    x = pd.DataFrame(rows, columns=["year", "month", "v"]); x.loc[x.v <= -99, "v"] = np.nan; return x
def rd_csv(p, vpat):
    x = pd.read_csv(p, comment="#"); c = {k.lower(): k for k in x}
    if "year" in c and "month" in c: yy, mm = x[c["year"]], x[c["month"]]
    else: tt = pd.to_datetime(x[[k for k in x if k.lower() in ("time", "date")][0]]); yy, mm = tt.dt.year, tt.dt.month
    vc = [k for k in x if re.search(vpat, k.lower().replace("_", ""))][0]
    return pd.DataFrame({"year": yy.astype(int), "month": mm.astype(int), "v": x[vc].astype(float)})
I = {"sam": rd_table(f"{R}/data/indices/marshall_sam_monthly.txt"), "nino": rd_table(f"{R}/data/indices/nina34.data"),
     "asl": rd_csv(f"{R}/data/indices/asli_era5_v3-latest.csv", r"relcen"),
     "zw3": rd_csv(f"{R}/data/indices/ZW3_raphael_monthly.csv", r"zw3index|index")}
def seas(x, months):
    wrap = [mm for mm in months if mm < months[0]]; z = x.copy(); z["sy"] = z.year - z.month.isin(wrap).astype(int)
    return z[z.month.isin(months)].groupby("sy").v.mean()
def detr(s):
    s = s.dropna(); t = s.index.values.astype(float); return s - np.polyval(np.polyfit(t, s.values, 1), t)
def r(x, y, lo=1979, hi=2025):
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1).loc[lo:hi].dropna()
    return np.nan if len(j) < 8 else np.corrcoef(detr(j.x), detr(j.y))[0, 1]
REL = [("Nino3.4 SON", "nino", [9, 10, 11], "amp", "ABS"), ("Nino3.4 annual", "nino", list(range(1, 13)), "amp", "KH"),
       ("ASL annual", "asl", list(range(1, 13)), "amp", "RA"), ("SAM JJA", "sam", [6, 7, 8], "amp", "WS"),
       ("SAM Oct-Jan", "sam", [10, 11, 12, 1], "maxdoy", "EA"), ("ZW3 SON", "zw3", [9, 10, 11], "maxdoy", "KH"),
       ("ZW3 annual", "zw3", list(range(1, 13)), "amp", "ABS")]
secs = ["ABS", "WS", "KH", "EA", "RA"]
print("\n(2) each index-season vs the component in ALL five sectors (detrended r, 1979-2025); [ ] = predicted sector")
print(f"  {'relationship':28s}" + "".join(f"{s:>8s}" for s in secs))
for lab, k, mon, comp, pred in REL:
    x = seas(I[k], mon); out = ""
    for s in secs:
        cell = f"{r(x, Y[Y.sec == s].set_index('year')[comp]):+.2f}"
        out += f"{('['+cell+']') if s == pred else cell:>8s}"
    print(f"  {lab + ' -> ' + comp:28s}" + out)
print("\n(3) ASL annual -> amplitude, 1979-2000 vs 2001-2025, by sector")
x = seas(I["asl"], list(range(1, 13)))
for s in secs:
    y = Y[Y.sec == s].set_index("year")["amp"]; print(f"  {s:3s}  {r(x, y, 1979, 2000):+.2f} -> {r(x, y, 2001, 2025):+.2f}")
