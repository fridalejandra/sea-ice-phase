#!/usr/bin/env python3
"""
ch2_coastal_buffer.py

Is the long-season consolidation-failure increase a coastal effect?

Conditional failure rate  P(dynamic misses FS | static detects FS)
recomputed (1) excluding pixels within 0/50/100/150 km of land, and
(2) split explicitly into a coastal band (<= COAST_KM) and interior
(> COAST_KM) within the long-season zone. If the level/rate pixel-year CSV
(with y, x columns) exists, its level/rate shares are split the same way.

Land mask: pixels whose daily SIC is missing on every day (same definition as
the landmask in compute_phase_dates_v3.validate). Computed once from the SIC
record and cached to data/landmask_sic.nc. (canonical_sectors.nc
valid_ocean is True everywhere, so it is not a land mask.)
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

SENSOR = "SMMR"
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
BASE_FRAC, MONTH, AUG15_DOY = 0.5, 30.4, 227
LB, UB = 4.0, 8.5
PIXEL_KM = 25.0
BUFFERS_KM = (0, 50, 100, 150)
COAST_KM = 100
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


def get_landmask(shape):
    if LANDMASK.exists():
        return xr.open_dataset(LANDMASK)["land"].values.astype(bool)
    import compute_phase_dates_v3 as v3
    print(f"Building land mask from daily SIC ({SENSOR}); cached to {LANDMASK} ...")
    ice365, _ = v3.load_sic(SENSOR)
    land = ice365.isnull().all("time").values
    assert land.shape == shape, "SIC grid does not match the anomaly grid"
    xr.Dataset({"land": (("y", "x"), land.astype("int8"))}).to_netcdf(LANDMASK)
    return land


def boot_change(pre, post):
    pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
    if len(pre) < 3 or len(post) < 3:
        return (np.nan,) * 5
    bs = (rng.choice(post, (NBOOT, len(post))).mean(1)
          - rng.choice(pre, (NBOOT, len(pre))).mean(1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pre.mean(), post.mean(), post.mean() - pre.mean(), lo, hi


def cond_rate_row(det, fail, sel, per, **labels):
    nd = det[:, sel].sum(1).astype(float)
    nf = fail[:, sel].sum(1).astype(float)
    rate_y = np.where(nd > 0, nf / np.maximum(nd, 1), np.nan)
    b = boot_change(rate_y[per["pre"]], rate_y[per["post"]])
    star = "*" if (b[3] > 0 or b[4] < 0) else " "
    return dict(**labels, n_pix=int(sel.sum()),
                pre_pct=round(100 * b[0], 1), post_pct=round(100 * b[1], 1),
                change_pp=f"{100*b[2]:+.1f} [{100*b[3]:+.1f},{100*b[4]:+.1f}]{star}")


def main():
    with xr.open_dataset(SECT) as m:
        sid = m["sector_id"].values

    date = {}
    for key in TAGS:
        date[key], years = load_dates(*key)
    per = {"pre": (years >= PRE[0]) & (years <= PRE[1]),
           "post": (years >= POST[0]) & (years <= POST[1])}
    valid = {k: np.isfinite(v) for k, v in date.items()}
    foot = np.all([valid[k][per["pre"]].mean(0) >= BASE_FRAC for k in valid], axis=0)
    dur = np.nanmean(((date[("MS", "static")] + AUG15_DOY) - date[("FS", "static")])[per["pre"]], axis=0) / MONTH
    long_zone = foot & (dur > UB)

    land = get_landmask(sid.shape)
    dist_km = distance_transform_edt(~land) * PIXEL_KM
    print(f"Land pixels: {int(land.sum())}; long-season pixels within {COAST_KM} km of land: "
          f"{int((long_zone & (dist_km <= COAST_KM)).sum())} of {int(long_zone.sum())}")

    det = valid[("FS", "static")]
    fail = det & ~valid[("FS", "dynamic")]

    rows = []
    for b in BUFFERS_KM:
        keep = long_zone & (dist_km > b)
        for s in list(SECTORS) + [0]:
            sel = keep & ((sid == s) if s else True)
            if sel.any():
                rows.append(cond_rate_row(det, fail, sel, per, buffer_km=b, sector=SECTORS.get(s, "ALL")))
    df1 = pd.DataFrame(rows)
    print("\n=== (1) Long-season zone, conditional failure rate (%) excluding pixels within buffer of land ===")
    print(df1.to_string(index=False))
    df1.to_csv(OUT / "ch2_coastal_buffer_exclusion.csv", index=False)

    rows = []
    for band, bm in [(f"coastal (<= {COAST_KM} km)", dist_km <= COAST_KM),
                     (f"interior (> {COAST_KM} km)", dist_km > COAST_KM)]:
        for s in list(SECTORS) + [0]:
            sel = long_zone & bm & ((sid == s) if s else True)
            if sel.any():
                rows.append(cond_rate_row(det, fail, sel, per, band=band, sector=SECTORS.get(s, "ALL")))
    df2 = pd.DataFrame(rows)
    print("\n=== (2) Long-season zone, coastal band vs interior ===")
    print(df2.to_string(index=False))
    df2.to_csv(OUT / "ch2_coastal_vs_interior.csv", index=False)

    if LVR_CSV.exists():
        lvr = pd.read_csv(LVR_CSV)
        if {"y", "x"}.issubset(lvr.columns):
            lvr = lvr[lvr["confirmed"] & (lvr["zone"] == "long")].copy()
            lvr["band"] = np.where(dist_km[lvr["y"], lvr["x"]] <= COAST_KM, "coastal", "interior")
            tab = (lvr.groupby(["band", "period"])
                      .apply(lambda g: pd.Series(dict(
                          n=len(g),
                          pct_level=round(100 * (g.kind == "level").mean(), 1),
                          pct_rate=round(100 * (g.kind == "rate").mean(), 1),
                          level_med_shortfall=round(g.loc[g.kind == "level", "shortfall"].median(), 3),
                          rate_med_best_slope=round(g.loc[g.kind == "rate", "best_slope"].median(), 4))))
                      .reset_index())
            print("\n=== (3) Level vs rate failures, long-season zone, coastal vs interior ===")
            print(tab.to_string(index=False))
            tab.to_csv(OUT / "ch2_level_vs_rate_coastal_interior.csv", index=False)
        else:
            print("\n(3) skipped: level/rate CSV has no y, x columns (rerun ch2_level_vs_rate.py with the patch)")
    else:
        print("\n(3) skipped: level/rate CSV not found yet (rerun this script when that job finishes)")
    print(f"\nCSVs written to {OUT}")


if __name__ == "__main__":
    main()
