#!/usr/bin/env python3
"""
ch2_slope_sensitivity.py

Does the long-season consolidation-failure increase depend on the 0.02/day
rate cutoff? Lowering the cutoff can only turn failures into detections, and
ch2_level_vs_rate_pixelyears.csv records each failure's best slope, so the
conditional failure rate at any cutoff c <= 0.02 is exact:

  failure at c  <=>  level failure, or rate failure with best_slope <= c
  rate(c) = failures at c / static FS detections   (per year, per region)

Reports long-season interior (> 100 km from land) and all long-season pixels,
by sector, 1988-2015 vs 2016-2024, with year-bootstrap 95% CIs.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
ANOM = ROOT / "data" / "anomalies" / "SMMR"
SECT = ROOT / "data" / "canonical_sectors.nc"
LANDMASK = ROOT / "data" / "landmask_sic.nc"
LVR_CSV = ROOT / "results" / "ch2_level_vs_rate_pixelyears.csv"
OUT = ROOT / "results"

SECTORS = {1: "ABS", 2: "WS", 3: "KHV", 4: "EA", 5: "RAS"}
TAGS = {("FS", "static"): "FS_static_thr15_k5", ("FS", "dynamic"): "FS_dynamic_k5_q70",
        ("MS", "static"): "MS_static_thr15_k5", ("MS", "dynamic"): "MS_dynamic_k5_q70"}
PRE, POST = (1988, 2015), (2016, 2024)
BASE_FRAC, MONTH, AUG15_DOY, UB = 0.5, 30.4, 227, 8.5
COAST_KM, PIXEL_KM = 100, 25.0
CUTOFFS = (0.02, 0.015, 0.01, 0.005)
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


def main():
    with xr.open_dataset(SECT) as m:
        sid = m["sector_id"].values
    date = {}
    for key in TAGS:
        date[key], years = load_dates(*key)
    ipre = (years >= PRE[0]) & (years <= PRE[1])
    ipost = (years >= POST[0]) & (years <= POST[1])
    valid = {k: np.isfinite(v) for k, v in date.items()}
    foot = np.all([valid[k][ipre].mean(0) >= BASE_FRAC for k in valid], axis=0)
    dur = np.nanmean(((date[("MS", "static")] + AUG15_DOY) - date[("FS", "static")])[ipre], axis=0) / MONTH
    long_zone = foot & (dur > UB)
    land = xr.open_dataset(LANDMASK)["land"].values.astype(bool)
    interior = distance_transform_edt(~land) * PIXEL_KM > COAST_KM

    det = valid[("FS", "static")]
    lvr = pd.read_csv(LVR_CSV)
    lvr = lvr[lvr["confirmed"]]
    yidx = {int(y): i for i, y in enumerate(years)}

    rows = []
    for c in CUTOFFS:
        still_fail = (lvr["kind"] == "level") | (lvr["best_slope"] <= c)
        f = np.zeros(det.shape, dtype=bool)
        sub = lvr[still_fail]
        f[[yidx[int(v)] for v in sub["year"]], sub["y"].values, sub["x"].values] = True
        for region, rmask in [("long, interior", long_zone & interior), ("long, all", long_zone)]:
            for s in list(SECTORS) + [0]:
                sel = rmask & ((sid == s) if s else True)
                if not sel.any():
                    continue
                nd = det[:, sel].sum(1).astype(float)
                nf = f[:, sel].sum(1).astype(float)
                rate_y = np.where(nd > 0, nf / np.maximum(nd, 1), np.nan)
                b = boot_change(rate_y[ipre], rate_y[ipost])
                star = "*" if (b[3] > 0 or b[4] < 0) else " "
                rows.append(dict(cutoff=c, region=region, sector=SECTORS.get(s, "ALL"),
                                 pre_pct=round(100 * b[0], 1), post_pct=round(100 * b[1], 1),
                                 change_pp=f"{100*b[2]:+.1f} [{100*b[3]:+.1f},{100*b[4]:+.1f}]{star}"))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ch2_slope_sensitivity.csv", index=False)
    for region in ["long, interior", "long, all"]:
        print(f"\n=== Conditional failure rate (%) vs slope cutoff: {region} ===")
        print(df[df.region == region].drop(columns="region").to_string(index=False))
    print(f"\nCSV written to {OUT}")


if __name__ == "__main__":
    main()
