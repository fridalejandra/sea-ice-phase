#!/usr/bin/env python
"""
11_osisaf_validation.py -- Ch4 Approach 6: does an INDEPENDENT drift product show the
same rise in opening/closing intensity as NSIDC-0116?

Independent product: EUMETSAT OSI SAF Global Sea Ice Drift CDR v1 (OSI-455),
75 km EASE2 (LAEA) grid, 24 h drift, 1991-2020 (Lavergne & Down 2023, ESSD 15, 5807).
Unlike NSIDC-0116 it does NOT assimilate buoys or NCEP winds. Summer (Nov-Feb in the SH)
is gap-filled with an ERA5 free-drift model -> we keep ONLY status_flag == 30
(nominal satellite retrieval), so no wind-model vectors enter.

Usage (run steps in order, in `screen`):
  python 11_osisaf_validation.py --probe      # download ONE file, print its structure
  python 11_osisaf_validation.py --download   # stream JJA+SON 1991-2020, month by month;
                                              # raw files deleted after processing (no disk growth);
                                              # resumable: re-run to continue
  python 11_osisaf_validation.py --analyze    # trends 1991-2020, OSI SAF vs NSIDC-0116

Divergence: div = d(u)/dx + d(v)/dy on the native 75 km grid, u = dX/dt, v = dY/dt
(dX, dY along grid axes, km). np.gradient with the file's xc/yc coordinates handles sign.
Compare RELATIVE trends (% per decade), not magnitudes: 75 km vs 25 km resolution.
"""
import ftplib
import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
HOST = "osisaf.met.no"
BASE = "/reprocessed/ice/drift_lr/v1/merged"
YEARS = range(1991, 2021)
MONTHS = (6, 7, 8, 9, 10, 11)
SEASONS = {"JJA": (6, 7, 8), "SON": (9, 10, 11)}
TMP = f"{ROOT}/tmp_osisaf"
OUT_DAILY = f"{ROOT}/results/ch4/tables/osisaf_daily_sector_opening_closing.csv"
OUT_TAB = f"{ROOT}/results/ch4/tables/osisaf_vs_nsidc_trends_1991_2020.csv"
NSIDC_CSV = f"{ROOT}/results/ch4/tables/ice_divergence_by_sector_season.csv"
SAT_FLAG = 30
MIN_DAYS = {"JJA": 60, "SON": 40}   # SON: Nov is mostly model gap-filled in SH -> fewer sat days
SECTORS = {"WS": (300.0, 20.0), "KH": (20.0, 90.0), "EA": (90.0, 160.0),
           "RA": (160.0, 230.0), "ABS": (230.0, 300.0)}
SHORT = {"WED": "WS", "Weddell": "WS", "WS": "WS", "KHV": "KH", "King Haakon VII": "KH",
         "KH": "KH", "EA": "EA", "East Antarctica": "EA", "RA": "RA", "Ross-Amundsen": "RA",
         "ABS": "ABS", "Amundsen-Bellingshausen": "ABS"}


def sector_masks(lon):
    lon = lon % 360
    return {k: ((lon >= lo) | (lon < hi)) if lo > hi else ((lon >= lo) & (lon < hi))
            for k, (lo, hi) in SECTORS.items()}


def ftp_connect():
    f = ftplib.FTP(HOST, timeout=120)
    f.login()
    return f


def sh_files(ftp, y, m):
    try:
        ftp.cwd(f"{BASE}/{y}/{m:02d}")
    except ftplib.error_perm:
        return []
    return sorted(n for n in ftp.nlst() if n.startswith("ice_drift_sh") and n.endswith(".nc"))


def fetch(ftp, name, dest):
    with open(dest, "wb") as fh:
        ftp.retrbinary(f"RETR {name}", fh.write)


def coord_km(ds, name):
    c = ds[name].values.astype(float)
    units = ds[name].attrs.get("units", "km").lower()
    return c / 1000.0 if units in ("m", "meter", "meters", "metre", "metres") else c


def process_file(path):
    ds = xr.open_dataset(path)
    dX = ds["dX"].squeeze().values.astype(float)
    dY = ds["dY"].squeeze().values.astype(float)
    flag = ds["status_flag"].squeeze().values
    xc, yc = coord_km(ds, "xc"), coord_km(ds, "yc")
    if "t0" in ds and "t1" in ds:
        dt = (ds["t1"].squeeze().values - ds["t0"].squeeze().values)
        dt_days = dt.astype("timedelta64[s]").astype(float) / 86400.0 if np.issubdtype(dt.dtype, np.timedelta64) \
            else np.full(dX.shape, 1.0)
        dt_days = np.where(np.isfinite(dt_days) & (dt_days > 0.2), dt_days, np.nan)
    else:
        dt_days = np.ones_like(dX)
    sat = (flag == SAT_FLAG)
    u = np.where(sat, dX / dt_days, np.nan)            # km/day along grid x
    v = np.where(sat, dY / dt_days, np.nan)
    div = np.gradient(u, xc, axis=1) + np.gradient(v, yc, axis=0)   # 1/day
    div = div / 86400.0                                              # s^-1, same units as NSIDC
    date = pd.Timestamp(ds["time"].values.ravel()[0]).normalize()
    return date, div, ds["lon"].values, int(sat.sum())


def probe():
    os.makedirs(TMP, exist_ok=True)
    ftp = ftp_connect()
    names = sh_files(ftp, 2005, 7)
    print(f"{len(names)} SH files in 2005/07, e.g.:", names[:3])
    p = os.path.join(TMP, names[0])
    fetch(ftp, names[0], p)
    ftp.quit()
    ds = xr.open_dataset(p)
    print(ds)
    fl = ds["status_flag"].squeeze().values
    vals, cnt = np.unique(fl[np.isfinite(fl)] if np.issubdtype(fl.dtype, np.floating) else fl, return_counts=True)
    print("status_flag counts:", dict(zip(vals.tolist(), cnt.tolist())))
    date, div, lon, nsat = process_file(p)
    fin = div[np.isfinite(div)]
    print(f"{date.date()}: {nsat} satellite vectors; divergence finite on {fin.size} cells; "
          f"median |div| = {np.median(np.abs(fin)):.2e} s^-1 (NSIDC-0116 at 25 km is ~1e-7-1e-6)")
    os.remove(p)


def download():
    os.makedirs(TMP, exist_ok=True)
    done = set()
    if os.path.exists(OUT_DAILY):
        done = set(pd.read_csv(OUT_DAILY, usecols=["date"]).date.str[:7])
    smask = None
    for y in YEARS:
        for m in MONTHS:
            key = f"{y}-{m:02d}"
            if key in done:
                continue
            for attempt in range(3):
                try:
                    ftp = ftp_connect()
                    names = sh_files(ftp, y, m)
                    rows = []
                    for n in names:
                        p = os.path.join(TMP, n)
                        fetch(ftp, n, p)
                        date, div, lon, nsat = process_file(p)
                        os.remove(p)
                        if smask is None:
                            smask = sector_masks(lon)
                        for k, S in smask.items():
                            d = div[S]
                            rows.append(dict(date=date.date(), sector=k, n_sat=nsat,
                                             opening=np.nanmean(np.where(d > 0, d, np.nan)),
                                             closing=-np.nanmean(np.where(d < 0, d, np.nan)),
                                             n_div=int(np.isfinite(d).sum())))
                    ftp.quit()
                    break
                except Exception as ex:
                    print(f"  {key}: attempt {attempt + 1} failed ({ex}); retrying")
                    time.sleep(20)
            else:
                print(f"  {key}: FAILED, will retry on next run")
                continue
            if rows:
                pd.DataFrame(rows).to_csv(OUT_DAILY, mode="a", index=False,
                                          header=not os.path.exists(OUT_DAILY), float_format="%.5g")
            print(f"  {key}: {len(names)} files", flush=True)


def seasonal(df, val_cols):
    out = []
    for s, months in SEASONS.items():
        g = df[df.date.dt.month.isin(months)].copy()
        g["sy"] = g.date.dt.year
        n = g.groupby(["sector", "sy"])["opening"].count()
        y = g.groupby(["sector", "sy"])[val_cols].mean()
        y = y[n >= MIN_DAYS[s]].reset_index()
        y["season"] = s
        out.append(y)
    return pd.concat(out)


def trend(yrs, v):
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 10:
        return np.nan, np.nan, int(ok.sum())
    f = sm.OLS(np.log(v[ok]), sm.add_constant((yrs[ok] - yrs[ok].mean()) / 10.0)).fit()
    return (np.exp(f.params[1]) - 1) * 100, f.pvalues[1], int(ok.sum())


def analyze():
    osi = pd.read_csv(OUT_DAILY, parse_dates=["date"]).drop_duplicates(["date", "sector"])
    ns = pd.read_csv(NSIDC_CSV, parse_dates=["date"])
    ns["sector"] = ns.sector.map(SHORT)
    ns = ns.groupby(["date", "sector"])[["div_positive", "div_negative"]].mean().reset_index()
    ns["opening"], ns["closing"] = ns.div_positive, -ns.div_negative
    ns = ns[ns.date.dt.year.between(min(YEARS), max(YEARS))]
    # restrict NSIDC to the same days OSI SAF has satellite data (fair comparison)
    ns = ns.merge(osi[["date", "sector"]], on=["date", "sector"], how="inner")
    so, sn = seasonal(osi, ["opening", "closing"]), seasonal(ns, ["opening", "closing"])
    rows = []
    for (k, s), a in so.groupby(["sector", "season"]):
        b = sn[(sn.sector == k) & (sn.season == s)]
        both = a.merge(b, on="sy", suffixes=("_osi", "_nsidc"))
        for v in ("opening", "closing"):
            to, po, no = trend(a.sy.values.astype(float), a[v].values)
            tn, pn, nn = trend(b.sy.values.astype(float), b[v].values)
            r = np.corrcoef(both[f"{v}_osi"], both[f"{v}_nsidc"])[0, 1] if len(both) > 5 else np.nan
            rows.append(dict(sector=k, season=s, variable=v, n_years_osi=no,
                             osi_trend_pct_dec=to, osi_p=po, nsidc_trend_pct_dec=tn, nsidc_p=pn,
                             interannual_r=r))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_TAB, index=False, float_format="%.4g")
    print(f"wrote {OUT_TAB}\n")
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    {"--probe": probe, "--download": download, "--analyze": analyze}.get(
        arg, lambda: print(__doc__))()
