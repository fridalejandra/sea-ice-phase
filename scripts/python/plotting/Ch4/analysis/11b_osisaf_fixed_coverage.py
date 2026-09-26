#!/usr/bin/env python
"""
11b_osisaf_fixed_coverage.py -- Ch4 Approach 6, done fairly: OSI SAF (OSI-455) divergence and
convergence rate trends over a FIXED set of cells that have valid satellite-only divergence in
every season-year 1991-2020. Removes the coverage artifact found in 11_osisaf_validation.py
(valid-cell counts grew 7-13%/decade and anti-correlated with mean opening).

Usage (in order):
  python 11b_osisaf_fixed_coverage.py --download   # per-day divergence fields, one .npz per month
                                                   # (float16, ~1 MB/month, ~150 MB total); resumable
  python 11b_osisaf_fixed_coverage.py --analyze    # fixed-coverage trends vs NSIDC-0116

Divergence exactly as in 11_osisaf_validation.py (status_flag == 30 only, native 75 km grid).
"""
import ftplib
import glob
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
MONTHS = (6, 7, 8, 9, 10)          # JJA + Sep-Oct (Nov is model gap-filled in the SH)
SEASONS = {"JJA": (6, 7, 8), "SO": (9, 10)}
TMP = f"{ROOT}/tmp_osisaf"
FIELD_DIR = f"{ROOT}/data/osisaf_div_fields"
OUT_TAB = f"{ROOT}/results/ch4/tables/osisaf_fixed_coverage_trends.csv"
NSIDC_CSV = f"{ROOT}/results/ch4/tables/ice_divergence_by_sector_season.csv"
SAT_FLAG = 30
MIN_FRAC = 0.5      # a cell must have valid divergence on >= 50% of days in EVERY season-year
SECTORS = {"WS": (300.0, 20.0), "KH": (20.0, 90.0), "EA": (90.0, 160.0),
           "RA": (160.0, 230.0), "ABS": (230.0, 300.0)}
SHORT = {"WED": "WS", "Weddell": "WS", "WS": "WS", "KHV": "KH", "King Haakon VII": "KH",
         "KH": "KH", "EA": "EA", "East Antarctica": "EA", "RA": "RA", "Ross-Amundsen": "RA",
         "ABS": "ABS", "Amundsen-Bellingshausen": "ABS"}


def sector_masks(lon):
    lon = lon % 360
    return {k: ((lon >= lo) | (lon < hi)) if lo > hi else ((lon >= lo) & (lon < hi))
            for k, (lo, hi) in SECTORS.items()}


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
    dt = ds["t1"].squeeze().values - ds["t0"].squeeze().values
    dt_days = dt.astype("timedelta64[s]").astype(float) / 86400.0
    dt_days = np.where(np.isfinite(dt_days) & (dt_days > 0.2), dt_days, np.nan)
    sat = flag == SAT_FLAG
    u = np.where(sat, dX / dt_days, np.nan)
    v = np.where(sat, dY / dt_days, np.nan)
    div = (np.gradient(u, xc, axis=1) + np.gradient(v, yc, axis=0)) / 86400.0   # s^-1
    date = pd.Timestamp(ds["time"].values.ravel()[0]).normalize()
    return date, div, ds["lon"].values, ds["lat"].values


def download():
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(FIELD_DIR, exist_ok=True)
    for y in YEARS:
        for m in MONTHS:
            out = f"{FIELD_DIR}/div_{y}{m:02d}.npz"
            if os.path.exists(out):
                continue
            for attempt in range(3):
                try:
                    ftp = ftplib.FTP(HOST, timeout=120)
                    ftp.login()
                    try:
                        ftp.cwd(f"{BASE}/{y}/{m:02d}")
                        names = sorted(n for n in ftp.nlst() if n.startswith("ice_drift_sh") and n.endswith(".nc"))
                    except ftplib.error_perm:
                        names = []
                    dates, fields, lon, lat = [], [], None, None
                    for n in names:
                        p = os.path.join(TMP, n)
                        with open(p, "wb") as fh:
                            ftp.retrbinary(f"RETR {n}", fh.write)
                        d, div, lon, lat = process_file(p)
                        os.remove(p)
                        dates.append(str(d.date()))
                        fields.append((div * 1e7).astype("float16"))   # stored in 1e-7 s^-1
                    ftp.quit()
                    if fields:
                        np.savez_compressed(out, dates=np.array(dates), div=np.stack(fields))
                        if not os.path.exists(f"{FIELD_DIR}/grid.npz"):
                            np.savez_compressed(f"{FIELD_DIR}/grid.npz", lon=lon, lat=lat)
                    print(f"  {y}-{m:02d}: {len(fields)} days", flush=True)
                    break
                except Exception as ex:
                    print(f"  {y}-{m:02d}: attempt {attempt + 1} failed ({ex})", flush=True)
                    time.sleep(20)


def trend(yrs, v):
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 10:
        return np.nan, np.nan
    f = sm.OLS(np.log(v[ok]), sm.add_constant((yrs[ok] - yrs[ok].mean()) / 10.0)).fit()
    return (np.exp(f.params[1]) - 1) * 100, f.pvalues[1]


def analyze():
    g = np.load(f"{FIELD_DIR}/grid.npz")
    smask = sector_masks(g["lon"])
    dates, fields = [], []
    for f in sorted(glob.glob(f"{FIELD_DIR}/div_*.npz")):
        z = np.load(f)
        dates += list(z["dates"])
        fields.append(z["div"].astype("float32"))
    t = pd.DatetimeIndex(pd.to_datetime(dates))
    D = np.concatenate(fields)                      # (days, y, x), 1e-7 s^-1
    print(f"loaded {len(t)} days, {t.min().date()} -> {t.max().date()}")

    ns = pd.read_csv(NSIDC_CSV, parse_dates=["date"])
    ns["sector"] = ns.sector.map(SHORT)
    ns = ns.groupby(["date", "sector"])[["div_positive", "div_negative"]].mean().reset_index()

    rows = []
    for s, months in SEASONS.items():
        yrs = np.array([y for y in YEARS])
        frac = np.full((len(yrs),) + D.shape[1:], np.nan)
        for i, y in enumerate(yrs):
            idx = (t.year == y) & t.month.isin(months)
            if idx.sum() > 0:
                frac[i] = np.isfinite(D[idx]).mean(0)
        fixed = np.nanmin(frac, 0) >= MIN_FRAC
        for sec, M in smask.items():
            cells = fixed & M
            op, cl, op_all, n_all = [], [], [], []
            for y in yrs:
                idx = (t.year == y) & t.month.isin(months)
                X = D[idx][:, cells]
                op.append(np.nanmean(np.where(X > 0, X, np.nan)))
                cl.append(np.nanmean(np.where(X < 0, -X, np.nan)))
                Xa = D[idx][:, M]
                op_all.append(np.nanmean(np.where(Xa > 0, Xa, np.nan)))
                n_all.append(np.isfinite(Xa).sum() / max(idx.sum(), 1))
            op, cl, op_all = map(np.array, (op, cl, op_all))
            nsd = ns[(ns.sector == sec) & ns.date.dt.month.isin(months) & ns.date.dt.year.between(1991, 2020)]
            nsy = nsd.groupby(nsd.date.dt.year).div_positive.mean().reindex(yrs).values
            to, po = trend(yrs.astype(float), op)
            tc, pc = trend(yrs.astype(float), cl)
            ta, pa = trend(yrs.astype(float), op_all)
            tn, pn = trend(yrs.astype(float), nsy)
            ok = np.isfinite(op) & np.isfinite(nsy)
            rows.append(dict(season=s, sector=sec, n_fixed_cells=int(cells.sum()),
                             osi_fixed_opening_pct_dec=to, p_open=po,
                             osi_fixed_closing_pct_dec=tc, p_close=pc,
                             osi_allcells_opening_pct_dec=ta, p_allcells=pa,
                             nsidc_opening_pct_dec=tn, p_nsidc=pn,
                             r_fixed_vs_nsidc=np.corrcoef(op[ok], nsy[ok])[0, 1] if ok.sum() > 5 else np.nan,
                             r_count_vs_opening_all=np.corrcoef(n_all, op_all)[0, 1]))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_TAB, index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(f"wrote {OUT_TAB}\n")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    {"--download": download, "--analyze": analyze}.get(arg, lambda: print(__doc__))()
