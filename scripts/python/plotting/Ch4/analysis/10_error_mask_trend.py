#!/usr/bin/env python
"""
10_error_mask_trend.py -- Ch4 Approach 3: is the rise in opening/closing intensity
driven by poorly-constrained (noisy / wind-filled) drift vectors?

NSIDC-0116 v4 daily files carry `icemotion_error_estimate` (OI vector error):
  < 0      within 25 km of coast (possible false ice)       -> excluded in "clean"/"low"
  >= 1000  nearest input vector > 1250 km away (wind/interp) -> excluded in "clean"/"low"
Masks compared (JJA + SON, 1988-2024):
  all   : every cell with divergence (as in tests 08/09)
  clean : 0 <= err < 1000
  low   : clean AND err <= LOW_Q quantile of 1988-2001 clean ice-cell errors (fixed threshold)
If the trend in `low` matches `all`, the rise is not an artifact of poorly-observed cells.
Also reports the trend in the mean error itself and in the fraction of ice cells that are `low`.
"""
import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
DIV_NC = f"{ROOT}/results/ch4/derived_nc/ease_divergence_with_latlon.nc"
DRIFT_DIR = f"{ROOT}/data/drift_nsidc/"
DRIFT_GLOB = "icemotion_daily_sh_25km_*_v4.1.nc"
ERR_VAR = "icemotion_error_estimate"
OUT_DAILY = f"{ROOT}/results/ch4/tables/error_mask_daily_sector.csv"
OUT_TAB = f"{ROOT}/results/ch4/tables/error_mask_trend.csv"
START, END = 1988, 2024
MONTHS = (6, 7, 8, 9, 10, 11)
SEASONS = {"JJA": (6, 7, 8), "SON": (9, 10, 11)}
LOW_Q = 0.5
# From the NSIDC-0116 v4 user guide (NH dates; SH assumed the same -- verify).
# Value = first season-year with the new input configuration.
SENSOR_BREAKS = {"AVHRR ends": 2001, "AMSR-E starts": 2002, "SSMIS replaces SSM/I": 2007,
                 "AMSR-E ends": 2012, "buoys+NCEP end": 2021}
SECTORS = {"WS": (300.0, 20.0), "KH": (20.0, 90.0), "EA": (90.0, 160.0),
           "RA": (160.0, 230.0), "ABS": (230.0, 300.0)}


def sector_masks(lon):
    lon = lon % 360
    out = {}
    for k, (lo, hi) in SECTORS.items():
        out[k] = ((lon >= lo) | (lon < hi)) if lo > hi else ((lon >= lo) & (lon < hi))
    return out


def raw_file_for(year):
    hits = [f for f in glob.glob(os.path.join(DRIFT_DIR, DRIFT_GLOB))
            if re.search(rf"_{year}\d{{4}}", os.path.basename(f))]
    return sorted(hits)[0] if hits else None


def load_err(year):
    f = raw_file_for(year)
    if f is None:
        return None, None
    try:
        ds = xr.open_dataset(f)
        t = pd.DatetimeIndex(ds.time.values)
    except Exception:
        ds = xr.open_dataset(f, decode_times=False)
        t = pd.date_range(f"{year}-01-01", periods=ds.sizes["time"], freq="D")
    e = ds[ERR_VAR]
    e = e.transpose("time", *[d for d in e.dims if d != "time"])
    return e, t


def fit(yr, v):
    ok = np.isfinite(v) & (v > 0)
    yr, y = yr[ok], np.log(v[ok])
    x = (yr - yr.mean()) / 10.0
    one = np.ones_like(x)
    steps = np.column_stack([(yr >= b).astype(float) for b in sorted(set(SENSOR_BREAKS.values()))])
    steps = steps[:, steps.std(0) > 0]
    fT = sm.OLS(y, np.column_stack([one, x])).fit()
    fTS = sm.OLS(y, np.column_stack([one, x, steps])).fit()
    fS = sm.OLS(y, np.column_stack([one, steps])).fit()
    aic = {"T": fT.aic, "S": fS.aic, "TS": fTS.aic}
    return dict(n_years=len(y), trend_pct_dec=(np.exp(fT.params[1]) - 1) * 100, trend_p=fT.pvalues[1],
                trend_given_sensor_p=fTS.pvalues[1], best=min(aic, key=aic.get))


def build_daily():
    div = xr.open_dataset(DIV_NC)
    latn = "lat" if "lat" in div else "latitude"
    lonn = "lon" if "lon" in div else "longitude"
    smask = sector_masks(div[lonn].values)
    pos_da = div["div_positive"].transpose("time", ...)
    neg_da = div["div_negative"].transpose("time", ...)
    dtimes = pd.DatetimeIndex(div.time.values)
    grid_shape = pos_da.shape[1:]

    # threshold from 1988-2001 (every 5th JJA/SON day)
    samp = []
    for yr in range(START, 2002):
        e, t = load_err(yr)
        if e is None:
            continue
        idx = np.nonzero(np.isin(t.month, MONTHS))[0][::5]
        ev = e.isel(time=idx).values
        di = dtimes.get_indexer(t[idx])
        ok = di >= 0
        fin = np.isfinite(pos_da.isel(time=di[ok]).values) | np.isfinite(neg_da.isel(time=di[ok]).values)
        ev = ev[ok]
        samp.append(ev[fin & (ev >= 0) & (ev < 1000)])
    thr = float(np.quantile(np.concatenate(samp), LOW_Q))
    print(f"low-error threshold (q={LOW_Q} of 1988-2001 clean ice cells): {thr:.3f}")

    rows = []
    for yr in range(START, END + 1):
        e, t = load_err(yr)
        if e is None:
            print(f"  {yr}: no raw file, skipped")
            continue
        idx = np.nonzero(np.isin(t.month, MONTHS))[0]
        di = dtimes.get_indexer(t[idx])
        ok = di >= 0
        idx, di = idx[ok], di[ok]
        ev = e.isel(time=idx).values.astype("float32")
        if ev.shape[1:] != grid_shape:
            sys.exit(f"grid mismatch: raw {ev.shape[1:]} vs divergence {grid_shape}")
        pos = pos_da.isel(time=di).values
        neg = neg_da.isel(time=di).values
        ice = np.isfinite(pos) | np.isfinite(neg)
        if yr == START:
            overlap = np.isfinite(ev[ice]).mean()
            print(f"  sanity: error field finite on {overlap:.1%} of divergence cells (expect ~100%)")
        clean = (ev >= 0) & (ev < 1000)
        masks = {"all": np.ones_like(ice), "clean": clean, "low": clean & (ev <= thr)}
        for k, S in smask.items():
            Sb = S[None]
            n_ice = (ice & Sb).sum((1, 2))
            e_mean = np.nanmean(np.where(ice & clean & Sb, ev, np.nan), axis=(1, 2))
            frac_low = (ice & masks["low"] & Sb).sum((1, 2)) / np.maximum((ice & clean & Sb).sum((1, 2)), 1)
            out = {"err_mean": e_mean, "frac_low": frac_low, "n_ice": n_ice}
            for mname, M in masks.items():
                w = M & Sb
                out[f"opening_{mname}"] = np.nanmean(np.where(w, pos, np.nan), axis=(1, 2))
                out[f"closing_{mname}"] = -np.nanmean(np.where(w, neg, np.nan), axis=(1, 2))
            df = pd.DataFrame(out)
            df.insert(0, "date", t[idx])
            df.insert(1, "sector", k)
            rows.append(df)
        print(f"  {yr}: {len(idx)} days")
    daily = pd.concat(rows, ignore_index=True)
    daily.to_csv(OUT_DAILY, index=False, float_format="%.5g")
    print(f"wrote {OUT_DAILY}")
    return daily


def main():
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    daily = pd.read_csv(OUT_DAILY, parse_dates=["date"]) if (
        os.path.exists(OUT_DAILY) and "--rebuild" not in sys.argv) else build_daily()
    daily["sy"] = daily.date.dt.year
    cols = [c for c in daily.columns if c.startswith(("opening_", "closing_"))] + ["err_mean", "frac_low"]
    rows = []
    for (k, season), g in [((k, s), daily[(daily.sector == k) & daily.date.dt.month.isin(m)])
                           for k in SECTORS for s, m in SEASONS.items()]:
        y = g.groupby("sy")[cols].mean()
        n = g.groupby("sy").size()
        y = y[n >= 60]
        for c in cols:
            if c in ("err_mean", "frac_low"):
                v = y[c].values
                lr = sm.OLS(v, sm.add_constant((y.index.values - y.index.values.mean()) / 10.0)).fit()
                rows.append(dict(sector=k, season=season, series=c,
                                 trend_pct_dec=100 * lr.params[1] / np.nanmean(v), trend_p=lr.pvalues[1]))
            else:
                rows.append(dict(sector=k, season=season, series=c,
                                 **fit(y.index.values.astype(float), y[c].values)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_TAB, index=False, float_format="%.4g")
    print(f"wrote {OUT_TAB}\n")
    pd.set_option("display.width", 220)
    for v in ("opening", "closing"):
        sub = res[res.series.str.startswith(v)]
        print(f"== {v}: trend %/decade by mask (p in second table)")
        print(sub.pivot_table(index=["sector", "season"], columns="series", values="trend_pct_dec").round(1).to_string())
        print(sub.pivot_table(index=["sector", "season"], columns="series", values="trend_p").round(3).to_string(), "\n")
    sub = res[res.series.isin(["err_mean", "frac_low"])]
    print("== error diagnostics: trend %/decade (p)")
    print(sub.pivot_table(index=["sector", "season"], columns="series",
                          values=["trend_pct_dec", "trend_p"]).round(3).to_string())


if __name__ == "__main__":
    main()
