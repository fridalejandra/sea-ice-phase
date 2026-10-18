#!/usr/bin/env python3
"""
ch2_consolidation_time.py

Threshold-free, locally referenced consolidation diagnostic. For every Freeze
Start the static method detects (1988-2024, analysis footprint), record:

  cons_days : days from the static FS (15% crossing) to the first time the
              k-day-smoothed SIC holds >= the pixel's dynamic level for k days
              (after a k-day run below it). NO slope condition. NaN if never.
  slope05   : centred slope (per day) where smoothed SIC first reaches 0.5
              after the static FS (mid-ramp growth rate, independent of level)
  plateau   : median raw SIC 30-60 days after the static FS

Summaries by zone x band (coastal <= 100 km / interior) x sector,
1988-2015 vs 2016-2024, year-bootstrap 95% CIs:
  - absolute: yearly medians of cons_days, slope05, plateau
  - LOCAL: each value minus that pixel's own 1988-2015 median (days), and the
    share of freeze-ups slower than the pixel's own baseline 90th percentile
    (~10% in the baseline by construction)
  - never_pct: share of freeze-ups that never reach the level

Per pixel-year fields are saved to results/ch2_consolidation_time.nc.
Uses v3's load_sic, slice_season, precompute_clim_percentiles,
centered_slope_vectorized and DYN_* / FS_* constants (same folder).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_phase_dates_v3 as v3  # noqa: E402

SENSOR = "SMMR"
ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
ANOM = ROOT / "data" / "anomalies" / "SMMR"
SECT = ROOT / "data" / "canonical_sectors.nc"
LANDMASK = ROOT / "data" / "landmask_sic.nc"
OUT = ROOT / "results"

SECTORS = {1: "ABS", 2: "WS", 3: "KHV", 4: "EA", 5: "RAS"}
TAGS = {("FS", "static"): "FS_static_thr15_k5", ("FS", "dynamic"): "FS_dynamic_k5_q70",
        ("MS", "static"): "MS_static_thr15_k5", ("MS", "dynamic"): "MS_dynamic_k5_q70"}
PRE, POST = (1988, 2015), (2016, 2024)
BASE_FRAC, MONTH, AUG15_DOY = 0.5, 30.4, 227
LB, UB = 4.0, 8.5
COAST_KM, PIXEL_KM = 100, 25.0
MID_LEVEL = 0.5
PLATEAU_WIN = (30, 60)
K, SLOPE_HW, ABS_MIN = v3.DYN_K, v3.DYN_SLOPE_HW, v3.DYN_ABS_MIN
NBOOT = 2000
rng = np.random.default_rng(42)


def _var(path, names):
    with xr.open_dataset(path, decode_times=False) as ds:
        for n in names:
            if n in ds:
                return ds[n].load()
    raise KeyError(f"{names} not in {path}")


def load_dates(phase, method):
    t = TAGS[(phase, method)]
    cs = [f"{t}_clim_dsa", f"{t}_clim"] if phase == "MS" else [f"{t}_clim"]
    an = [f"{t}_anom_dsa", f"{t}_anom"] if phase == "MS" else [f"{t}_anom"]
    clim = _var(ANOM / f"{t}_climatology.nc", cs).values
    anom = _var(ANOM / f"{t}_anomalies.nc", an)
    return clim[None] + anom.values, anom["year"].values


def boot_change(pre, post):
    pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
    if len(pre) < 3 or len(post) < 3:
        return (np.nan,) * 5
    bs = (rng.choice(post, (NBOOT, len(post))).mean(1)
          - rng.choice(pre, (NBOOT, len(pre))).mean(1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pre.mean(), post.mean(), post.mean() - pre.mean(), lo, hi


def fmt(b, scale=1.0, nd=1):
    if not np.isfinite(b[2]):
        return "nan"
    star = "*" if (b[3] > 0 or b[4] < 0) else " "
    return f"{b[2]*scale:+.{nd}f} [{b[3]*scale:+.{nd}f},{b[4]*scale:+.{nd}f}]{star}"


def year_fields(ice365, clim_q, yr, fs_doy):
    """cons_days, slope05, plateau (y, x) for one year, given static FS DOY map."""
    ts = v3.slice_season(ice365, f"{yr}{v3.FS_START_MMDD}", f"{yr}{v3.FS_END_MMDD}")
    T = ts.time.size
    nan = np.full(fs_doy.shape, np.nan)
    if T < K:
        return nan, nan, nan
    ts_s = ts.rolling(time=K, center=True, min_periods=max(1, K // 2)).mean()
    slope = v3.centered_slope_vectorized(ts_s, SLOPE_HW).values
    doys = ts.time.dt.dayofyear.values
    thr_map = np.maximum(clim_q.sel(doy=doys, method="nearest").mean("doy").values, ABS_MIN)

    thr = xr.DataArray(np.broadcast_to(thr_map[None], ts_s.shape), dims=ts_s.dims, coords=ts_s.coords)
    sustained = (ts_s >= thr).rolling(time=K, min_periods=K).construct("window").all("window")
    start = sustained.shift(time=-(K - 1), fill_value=False)
    opp = (ts_s < thr).rolling(time=K, min_periods=K).construct("window").all("window")
    prior = opp.cumsum("time").shift(time=1, fill_value=0) > 0
    cand = (start & prior).values

    t15 = np.searchsorted(doys, fs_doy)
    ok = np.isfinite(fs_doy) & (t15 < T)
    t15 = np.where(ok, t15, T)
    tt = np.arange(T)[:, None, None]
    after = tt >= t15[None]

    hit = cand & after
    has = hit.any(0)
    cons = np.where(ok & has, hit.argmax(0) - t15, np.nan).astype(float)

    s = ts_s.values
    up = (s >= MID_LEVEL) & after
    has05 = up.any(0)
    i05 = up.argmax(0)
    slope05 = np.take_along_axis(slope, i05[None], 0)[0]
    slope05 = np.where(ok & has05, slope05, np.nan)

    lo, hi = PLATEAU_WIN
    win = (tt >= t15[None] + lo) & (tt <= t15[None] + hi)
    raw = np.where(win, ts.values, np.nan)
    enough = np.isfinite(raw).sum(0) >= 10
    plateau = np.where(ok & enough, np.nanmedian(raw, 0), np.nan)
    return cons, slope05, plateau


def main():
    with xr.open_dataset(SECT) as m:
        sid = m["sector_id"].values
    date = {}
    for key in TAGS:
        date[key], years = load_dates(*key)
    ipre = (years >= PRE[0]) & (years <= PRE[1])
    valid = {k: np.isfinite(v) for k, v in date.items()}
    foot = np.all([valid[k][ipre].mean(0) >= BASE_FRAC for k in valid], axis=0)
    dur = np.nanmean(((date[("MS", "static")] + AUG15_DOY) - date[("FS", "static")])[ipre], axis=0) / MONTH
    zones = {"short": foot & (dur < LB), "mid": foot & (dur >= LB) & (dur <= UB), "long": foot & (dur > UB)}
    land = xr.open_dataset(LANDMASK)["land"].values.astype(bool)
    coastal = distance_transform_edt(~land) * PIXEL_KM <= COAST_KM
    fs = np.where(foot[None], date[("FS", "static")], np.nan)

    print(f"Loading daily SIC ({SENSOR}) and precomputing Q{int(v3.DYN_Q_FS*100)} climatology ...", flush=True)
    ice365, _ = v3.load_sic(SENSOR)
    clim_q = v3.precompute_clim_percentiles(ice365, v3.DYN_Q_FS, v3.DYN_DOY_HW)

    keep = (years >= PRE[0]) & (years <= POST[1])
    yrs = years[keep]
    F = {k: np.full((len(yrs),) + foot.shape, np.nan) for k in ["cons_days", "slope05", "plateau"]}
    for n, (iy, yr) in enumerate(zip(np.where(keep)[0], yrs)):
        c, s5, p = year_fields(ice365, clim_q, int(yr), fs[iy])
        F["cons_days"][n], F["slope05"][n], F["plateau"][n] = c, s5, p
        print(f"  {yr}: {int(np.isfinite(fs[iy]).sum())} freeze-ups, "
              f"{int(np.isfinite(c).sum())} reach level", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    xr.Dataset({k: (("year", "y", "x"), v) for k, v in F.items()},
               coords={"year": yrs}).to_netcdf(OUT / "ch2_consolidation_time.nc")

    ip, iq = (yrs <= PRE[1]), (yrs >= POST[0])
    detected = np.isfinite(fs[keep])
    cons = F["cons_days"]
    base_med = np.nanmedian(np.where(ip[:, None, None], cons, np.nan), 0)
    base_p90 = np.nanpercentile(np.where(ip[:, None, None], cons, np.nan), 90, axis=0)
    local_anom = cons - base_med[None]
    slow = np.where(np.isfinite(cons) & np.isfinite(base_p90)[None], cons > base_p90[None], np.nan)

    rows = []
    bands = [("coastal", coastal), ("interior", ~coastal)]
    for zname, zmask in zones.items():
        for bname, bmask in bands:
            for s in list(SECTORS) + [0]:
                sel = zmask & bmask & ((sid == s) if s else True)
                if sel.sum() < 20:
                    continue

                def ymed(a):
                    return np.array([np.nanmedian(a[i][sel]) for i in range(len(yrs))])

                def ymean(a):
                    return np.array([np.nanmean(a[i][sel]) for i in range(len(yrs))])

                nd = detected[:, sel].sum(1).astype(float)
                never = np.where(nd > 0, (detected[:, sel] & ~np.isfinite(cons[:, sel])).sum(1) / np.maximum(nd, 1), np.nan)
                b_c = boot_change(ymed(cons)[ip], ymed(cons)[iq])
                b_l = boot_change(ymean(local_anom)[ip], ymean(local_anom)[iq])
                b_s = boot_change(ymean(slow)[ip], ymean(slow)[iq])
                b_5 = boot_change(ymed(F["slope05"])[ip], ymed(F["slope05"])[iq])
                b_p = boot_change(ymed(F["plateau"])[ip], ymed(F["plateau"])[iq])
                b_n = boot_change(never[ip], never[iq])
                rows.append(dict(
                    zone=zname, band=bname, sector=SECTORS.get(s, "ALL"), n_pix=int(sel.sum()),
                    cons_pre_d=round(b_c[0], 1), cons_chg_d=fmt(b_c),
                    local_anom_chg_d=fmt(b_l),
                    slow_pre_pct=round(100 * b_s[0], 1), slow_chg_pp=fmt(b_s, 100),
                    slope05_pre=round(b_5[0], 3), slope05_chg=fmt(b_5, 1, 4),
                    plateau_pre=round(b_p[0], 3), plateau_chg=fmt(b_p, 1, 3),
                    never_pre_pct=round(100 * b_n[0], 1), never_chg_pp=fmt(b_n, 100)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ch2_consolidation_time_summary.csv", index=False)
    pd.set_option("display.width", 250)
    for zname in ["long", "mid", "short"]:
        sub = df[df.zone == zname].drop(columns="zone")
        print(f"\n=== {zname}-season zone: consolidation (1988-2015 vs 2016-2024; change [95% CI], * excludes 0) ===")
        print(sub.to_string(index=False))
    print(f"\nWritten: {OUT/'ch2_consolidation_time.nc'}, {OUT/'ch2_consolidation_time_summary.csv'}")


if __name__ == "__main__":
    main()
