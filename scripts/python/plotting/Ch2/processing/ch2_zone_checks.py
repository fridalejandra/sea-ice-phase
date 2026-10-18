#!/usr/bin/env python
"""
ch2_zone_checks.py

Season-duration zones from the 1988-2015 baseline (static method, Stammerjohn-
style duration = Melt Start - Freeze Start):
    short-season  < 4 months          (winter-only fringe)
    mid-season    4 - UB months       (broad seasonal pack)
    long-season   > UB months         (ice lasting into summer)
UB = 8.5 months (main); 8.0 and 9.0 as sensitivity on the weaker boundary.

Per sector x zone, 1988-2015 vs 2016-2024, with 95% CIs from bootstrapping
years within each period:
  (a) non-detection rate, static and dynamic, and the dynamic-minus-static
      gap (does consolidation fail more often than presence?)
  (b) shoulder = dynamic - static (days), yearly zone means

Footprint: pixels valid in >= 50% of baseline years for FS and MS under both
methods (baseline-only, so no survivor bias). Also prints the baseline
duration histogram over this footprint to confirm the zone boundaries.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
ANOM = ROOT / "data" / "anomalies" / "SMMR"
SECT = ROOT / "data" / "canonical_sectors.nc"
OUT = ROOT / "results"

SECTORS = {1: "ABS", 2: "WS", 3: "KHV", 4: "EA", 5: "RAS"}
TAGS = {("FS", "static"): "FS_static_thr15_k5", ("FS", "dynamic"): "FS_dynamic_k5_q70",
        ("MS", "static"): "MS_static_thr15_k5", ("MS", "dynamic"): "MS_dynamic_k5_q70"}
PRE, POST = (1979, 2015), (2016, 2024)
BASE_FRAC = 0.5
MONTH = 30.4
AUG15_DOY = 227
LB, UB_MAIN, UB_SENS = 4.0, 8.5, (8.0, 9.0)
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


def boot_change(series, ipre, ipost):
    """post - pre mean of a yearly series, with bootstrap 95% CI over years."""
    pre, post = series[ipre], series[ipost]
    pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
    if len(pre) < 3 or len(post) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    d = post.mean() - pre.mean()
    bs = (rng.choice(post, (NBOOT, len(post))).mean(1)
          - rng.choice(pre, (NBOOT, len(pre))).mean(1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pre.mean(), post.mean(), d, lo, hi


def fmt(d, lo, hi, scale=1.0, nd=1):
    if not np.isfinite(d):
        return "nan"
    star = "*" if (lo > 0 or hi < 0) else " "
    return f"{d*scale:+.{nd}f} [{lo*scale:+.{nd}f},{hi*scale:+.{nd}f}]{star}"


def main():
    with xr.open_dataset(SECT) as m:
        ocean = m["valid_ocean"].astype(bool).values
        sid = m["sector_id"].values

    date, years = {}, None
    for key in TAGS:
        date[key], years = load_dates(*key)
    ipre = (years >= PRE[0]) & (years <= PRE[1])
    ipost = (years >= POST[0]) & (years <= POST[1])
    valid = {k: np.isfinite(v) for k, v in date.items()}

    foot = ocean & np.all([valid[k][ipre].mean(0) >= BASE_FRAC for k in valid], axis=0)

    dur_y = (date[("MS", "static")] + AUG15_DOY) - date[("FS", "static")]
    dur = np.nanmean(dur_y[ipre], axis=0) / MONTH

    d = dur[foot & np.isfinite(dur)]
    h, e = np.histogram(d, bins=np.arange(0, 12.5, 0.5))
    print(f"\n=== Baseline (1988-2015) season duration, analysis footprint (n={d.size}) ===")
    for n, lo in zip(h, e[:-1]):
        print(f"{lo:4.1f}-{lo+0.5:4.1f} mo | {'#' * int(50 * n / max(h.max(), 1))} {n}")

    def zones(ub):
        return {"short": foot & (dur < LB),
                "mid": foot & (dur >= LB) & (dur <= ub),
                "long": foot & (dur > ub)}

    rows_a, rows_b, rows_s = [], [], []
    for ub in (UB_MAIN,) + UB_SENS:
        Z = zones(ub)
        if ub == UB_MAIN:
            np.savez(OUT / "ch2_zone_masks.npz", short=Z["short"], mid=Z["mid"], long=Z["long"], dur=dur, foot=foot)
        for phase in ["FS", "MS"]:
            ks, kd = (phase, "static"), (phase, "dynamic")
            diff = date[kd] - date[ks]
            for s in list(SECTORS) + [0]:
                lab = SECTORS.get(s, "ALL")
                for zname, zmask in Z.items():
                    sel = zmask & ((sid == s) if s else True)
                    n = int(sel.sum())
                    if n == 0:
                        continue
                    r_sta = 1 - valid[ks][:, sel].mean(1)
                    r_dyn = 1 - valid[kd][:, sel].mean(1)
                    sh = np.nanmean(diff[:, sel], axis=1)
                    bs_sta = boot_change(r_sta, ipre, ipost)
                    bs_dyn = boot_change(r_dyn, ipre, ipost)
                    bs_gap = boot_change(r_dyn - r_sta, ipre, ipost)
                    bs_sh = boot_change(sh, ipre, ipost)

                    if ub == UB_MAIN:
                        rows_a.append(dict(
                            phase=phase, sector=lab, zone=zname, n_pix=n,
                            sta_pre=round(100 * bs_sta[0], 1), sta_post=round(100 * bs_sta[1], 1),
                            dyn_pre=round(100 * bs_dyn[0], 1), dyn_post=round(100 * bs_dyn[1], 1),
                            sta_change=fmt(*bs_sta[2:], scale=100),
                            dyn_change=fmt(*bs_dyn[2:], scale=100),
                            gap_change=fmt(*bs_gap[2:], scale=100)))
                        rows_b.append(dict(
                            phase=phase, sector=lab, zone=zname, n_pix=n,
                            pre=round(bs_sh[0], 1), post=round(bs_sh[1], 1),
                            change=fmt(*bs_sh[2:])))
                    if zname == "long":
                        rows_s.append(dict(
                            ub_months=ub, phase=phase, sector=lab, n_pix=n,
                            gap_change_pp=fmt(*bs_gap[2:], scale=100),
                            shoulder_change_d=fmt(*bs_sh[2:])))

    OUT.mkdir(parents=True, exist_ok=True)
    tables = [
        (f"(a) non-detection rate (%) by duration zone, {PRE[0]}-{PRE[1]} vs {POST[0]}-{POST[1]}; "
         "changes in percentage points [95% CI], * = CI excludes 0; "
         "gap = dynamic minus static", rows_a, "ch2_zone_nodetection"),
        ("(b) shoulder = dynamic - static (days), yearly zone means; change [95% CI]",
         rows_b, "ch2_zone_shoulders"),
        (f"(c) sensitivity of the long-season zone to the upper boundary {(UB_MAIN,) + UB_SENS}",
         rows_s, "ch2_zone_sensitivity"),
    ]
    for title, rows, fname in tables:
        df = pd.DataFrame(rows)
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))
        df.to_csv(OUT / f"{fname}.csv", index=False)
    print(f"\nCSVs written to {OUT}")


if __name__ == "__main__":
    main()
