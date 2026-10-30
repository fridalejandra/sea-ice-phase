#!/usr/bin/env python
# ch3_mapping_change.py -- has the way the atmosphere and the ocean map onto the cycle's
# components changed after 2016, and is any change independent of the annual mean?
#  (a) component-mean coupling: r(annual mean, amplitude / day of max / day of min)
#  (b) atmosphere -> component (Table 3.3): r and partial r controlling for annual mean
#  (c) ocean (HadISST 55-65S band) -> amplitude / day of max: r and partial r controlling for annual mean
# Each for 1979-2015 (pre), 2016-2024 (post), 1979-2024 (full), detrended within period,
# Fisher z p for pre vs post with df corrected for detrending.
import os, re, numpy as np, pandas as pd, xarray as xr
from scipy.stats import norm
ROOT = os.environ.get("SIP_ROOT", "/user/geog/falejandraperez/sea-ice-phase")
ANNUAL = f"{ROOT}/data/ch3/annual_params.csv"; IDX = f"{ROOT}/data/indices"
SST = f"{ROOT}/data/ocean/HadISST_sst.nc"; OUT = f"{ROOT}/results/ch3/tables"
os.makedirs(OUT, exist_ok=True)
PRE, POST, FULL = (1979, 2015), (2016, 2024), (1979, 2024)
SEC = {"WS": "SIE_Weddell", "KH": "SIE_King_Haakon", "EA": "SIE_East_Antarctica",
       "RA": "SIE_Ross", "ABS": "SIE_Amundsen_Bellingshausen", "Total": "SIE_circumpolar"}
LON = {"WS": (-70, -14), "KH": (-14, 71), "EA": (71, 162), "RA": (162, -110), "ABS": (-110, -70)}   # canonical_sectors.nc bounds
COMP = {"amplitude": "amplitude_raw_yr", "day of max": "max_doy_raw", "day of min": "min_doy_raw"}
REL = [("Nino3.4 SON", "nino34", [9, 10, 11], "ABS", "amplitude"),
       ("SAM JJA", "sam", [6, 7, 8], "WS", "amplitude"),
       ("SAM Oct-Jan", "sam", [10, 11, 12, 1], "EA", "day of max"),
       ("ZW3 annual", "zw3", list(range(1, 13)), "ABS", "amplitude"),
       ("[search] Nino3.4 annual", "nino34", list(range(1, 13)), "RA", "amplitude"),
       ("[search] ZW3 SON", "zw3", [9, 10, 11], "RA", "day of max")]
SST_SEAS = {"annual": list(range(1, 13)), "JJA": [6, 7, 8], "SON": [9, 10, 11]}

def rd_table(path, n=13):
    rows = []
    for line in open(path):
        t = line.split()
        if len(t) >= n and re.fullmatch(r"\d{4}", t[0]):
            rows += [(int(t[0]), m + 1, float(t[m + 1])) for m in range(12)]
    d = pd.DataFrame(rows, columns=["year", "month", "v"]); d.loc[d.v <= -99, "v"] = np.nan; return d
def rd_asl(path):
    d = pd.read_csv(path, comment="#"); tc = [c for c in d if c.lower() in ("time", "date")][0]
    vc = [c for c in d if "relcen" in c.lower().replace("_", "")][0]; t = pd.to_datetime(d[tc])
    return pd.DataFrame({"year": t.dt.year, "month": t.dt.month, "v": d[vc].astype(float)})
def rd_zw3(path):
    d = pd.read_csv(path, comment="#"); c = {k.lower(): k for k in d}
    if "year" in c and "month" in c: y, m = d[c["year"]], d[c["month"]]
    else: t = pd.to_datetime(d[[k for k in d if k.lower() in ("time", "date")][0]]); y, m = t.dt.year, t.dt.month
    vc = [k for k in d if any(s in k.lower() for s in ("amp", "mag", "index"))][0]
    return pd.DataFrame({"year": y.astype(int), "month": m.astype(int), "v": d[vc].astype(float)})
def seasonal(mon, months):
    d = mon.copy(); wrap = [m for m in months if m < months[0]]
    d["sy"] = d.year - d.month.isin(wrap).astype(int)
    return d[d.month.isin(months)].groupby("sy").v.mean()

def detr(s):
    s = s.dropna(); x = s.index.values.astype(float)
    return s - np.polyval(np.polyfit(x, s.values, 1), x) if len(s) > 3 else s * np.nan
def resid(y, z):
    b = np.polyfit(z, y, 1); return y - np.polyval(b, z)
def stat(x, y, z=None, per=FULL):
    df = pd.concat([x.rename("x"), y.rename("y")] + ([z.rename("z")] if z is not None else []), axis=1)
    df = df.loc[per[0]:per[1]].dropna()
    if len(df) < 6: return np.nan, len(df)
    df = df.apply(detr)
    if z is not None: df["x"] = resid(df.x, df.z); df["y"] = resid(df.y, df.z)
    return np.corrcoef(df.x, df.y)[0, 1], len(df)
def fz(r1, n1, r2, n2, k=0):
    if np.isnan(r1) or np.isnan(r2): return np.nan
    se = np.sqrt(1 / (n1 - 5 - k) + 1 / (n2 - 5 - k))   # -2 df for within-period detrend
    return 2 * (1 - norm.cdf(abs(np.arctanh(r1) - np.arctanh(r2)) / se))
def three(x, y, z=None, k=0):
    rf, nf = stat(x, y, z, FULL); r1, n1 = stat(x, y, z, PRE); r2, n2 = stat(x, y, z, POST)
    return dict(r_full=rf, r_pre=r1, r_post=r2, p_change=fz(r1, n1, r2, n2, k), n_pre=n1, n_post=n2)

def main():
    a = pd.read_csv(ANNUAL); a = a[a.period == "FULL"] if "period" in a else a
    g = {k: a[a.sector == v].set_index("Year").sort_index() for k, v in SEC.items()}
    pd.set_option("display.width", 200)
    rows = [dict(sector=s, component=c, **three(g[s]["sie_annual"], g[s][col])) for s in SEC for c, col in COMP.items()]
    A = pd.DataFrame(rows); A.to_csv(f"{OUT}/ch3_map_a_mean_coupling.csv", index=False)
    print("\n(a) r(annual mean, component), detrended within period")
    print(A.drop(columns=["n_pre", "n_post"]).round(2).to_string(index=False))
    idx = {"sam": rd_table(f"{IDX}/marshall_sam_monthly.txt"), "nino34": rd_table(f"{IDX}/nina34.data"),
           "asl": rd_asl(f"{IDX}/asli_era5_v3-latest.csv"), "zw3": rd_zw3(f"{IDX}/ZW3_raphael_monthly.csv")}
    rows = []
    for lab, k, mon, s, c in REL:
        x = seasonal(idx[k], mon); y = g[s][COMP[c]]; m = g[s]["sie_annual"]
        r0 = three(x, y); r1 = three(x, y, m, k=1)
        rows.append(dict(relation=f"{lab} -> {s} {c}", r_full=r0["r_full"], r_pre=r0["r_pre"], r_post=r0["r_post"],
                         p=r0["p_change"], pr_full=r1["r_full"], pr_pre=r1["r_pre"], pr_post=r1["r_post"], pp=r1["p_change"]))
    B = pd.DataFrame(rows); B.to_csv(f"{OUT}/ch3_map_b_atmos.csv", index=False)
    print("\n(b) atmosphere -> component: r, and partial r controlling for the annual mean (pr_)")
    print("    check r_full against corrected set (+0.42, -0.40, +0.38, -0.35, -0.42, +0.40)")
    print(B.round(2).to_string(index=False))
    ds = xr.open_dataset(SST); sst = ds["sst"].where(ds["sst"] > -2).sel(latitude=slice(-55, -65))
    rows = []
    for s, (lo, hi) in LON.items():
        box = sst.where(((ds.longitude >= lo) & (ds.longitude <= hi)) if lo < hi else ((ds.longitude >= lo) | (ds.longitude <= hi)), drop=True)
        ser = box.mean(("latitude", "longitude")).to_pandas(); ser.index = pd.to_datetime(ser.index)
        mon = pd.DataFrame({"year": ser.index.year, "month": ser.index.month, "v": ser.values})
        for sn, ms in SST_SEAS.items():
            x = seasonal(mon, ms); m = g[s]["sie_annual"]; rm = three(x, m)
            for c in ("amplitude", "day of max"):
                y = g[s][COMP[c]]; r0 = three(x, y); r1 = three(x, y, m, k=1)
                rows.append(dict(sector=s, sst=sn, component=c, r_mean_full=rm["r_full"],
                                 r_pre=r0["r_pre"], r_post=r0["r_post"], p=r0["p_change"],
                                 pr_pre=r1["r_pre"], pr_post=r1["r_post"], pp=r1["p_change"]))
    C = pd.DataFrame(rows); C.to_csv(f"{OUT}/ch3_map_c_ocean.csv", index=False)
    print("\n(c) SST -> amplitude / day of max: r, and partial r controlling for annual mean (pr_)")
    print("    r_mean_full = r(SST, annual mean) over 1979-2024 for reference")
    print(C.round(2).to_string(index=False))
    for name, T, col in (("(b) partial", B, "pp"), ("(c) raw", C, "p"), ("(c) partial", C, "pp")):
        k = int((T[col] < 0.05).sum()); n = int(T[col].notna().sum())
        print(f"{name}: {k} of {n} pre/post changes at p<0.05; about {0.05*n:.1f} expected by chance alone")
    print(f"\nwrote 3 tables to {OUT}")

if __name__ == "__main__":
    main()
