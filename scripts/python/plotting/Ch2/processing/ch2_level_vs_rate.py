#!/usr/bin/env python3
"""
ch2_level_vs_rate.py

For each consolidation-failure pixel-year (static detects FS, dynamic does
not), classify WHY the dynamic method missed it, using the exact v3 logic:

  LEVEL : the smoothed SIC never holds >= the pixel's dynamic threshold for
          k days after a k-day run below it (ice never reaches its
          consolidated level)
  RATE  : it does, but the centred slope never exceeds DYN_SLOPE_MIN at the
          start of such a run (ice consolidates, but slowly)

Also reports, for level failures, how far short the ice fell (threshold minus
the highest k-day-sustained SIC), and for rate failures, the fastest slope at
a qualifying run start. Fidelity check: re-applies the full dynamic test to
every failure pixel-year; the "confirmed" share should be ~100%.

Imports load_sic, slice_season, precompute_clim_percentiles,
centered_slope_vectorized and the DYN_* / FS_* constants from
compute_phase_dates_v3.py (same folder).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_phase_dates_v3 as v3  # noqa: E402

SENSOR = "SMMR"
ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
ANOM = ROOT / "data" / "anomalies" / "SMMR"
SECT = ROOT / "data" / "canonical_sectors.nc"
OUT = ROOT / "results"

SECTORS = {1: "ABS", 2: "WS", 3: "KHV", 4: "EA", 5: "RAS"}
TAGS = {("FS", "static"): "FS_static_thr15_k5", ("FS", "dynamic"): "FS_dynamic_k5_q70",
        ("MS", "static"): "MS_static_thr15_k5", ("MS", "dynamic"): "MS_dynamic_k5_q70"}
PRE, POST = (1988, 2015), (2016, 2024)
BASE_FRAC, MONTH, AUG15_DOY = 0.5, 30.4, 227
LB, UB = 4.0, 8.5

K, SLOPE_HW, SLOPE_MIN, ABS_MIN = v3.DYN_K, v3.DYN_SLOPE_HW, v3.DYN_SLOPE_MIN, v3.DYN_ABS_MIN


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


def run_masks(ts_s, thr_map):
    """Replicates first_run_with_slope_vectorized (FS branch) up to the slope test."""
    thr = xr.DataArray(np.broadcast_to(thr_map[None], ts_s.shape), dims=ts_s.dims, coords=ts_s.coords)
    sustained = (ts_s >= thr).rolling(time=K, min_periods=K).construct("window").all("window")
    sustained_shifted = sustained.shift(time=-(K - 1), fill_value=False)
    opp_run = (ts_s < thr).rolling(time=K, min_periods=K).construct("window").all("window")
    prior_opp = opp_run.cumsum("time").shift(time=1, fill_value=0) > 0
    return sustained_shifted & prior_opp, thr


def main():
    with xr.open_dataset(SECT) as m:
        ocean = m["valid_ocean"].astype(bool).values
        sid = m["sector_id"].values

    date = {}
    for key in TAGS:
        date[key], years = load_dates(*key)
    ipre = (years >= PRE[0]) & (years <= PRE[1])
    valid = {k: np.isfinite(v) for k, v in date.items()}
    foot = ocean & np.all([valid[k][ipre].mean(0) >= BASE_FRAC for k in valid], axis=0)
    dur = np.nanmean(((date[("MS", "static")] + AUG15_DOY) - date[("FS", "static")])[ipre], axis=0) / MONTH
    zone = np.full(dur.shape, "", dtype=object)
    zone[foot & (dur < LB)] = "short"
    zone[foot & (dur >= LB) & (dur <= UB)] = "mid"
    zone[foot & (dur > UB)] = "long"

    fail = valid[("FS", "static")] & ~valid[("FS", "dynamic")] & foot[None]

    print(f"Loading daily SIC ({SENSOR}) and precomputing Q{int(v3.DYN_Q_FS*100)} climatology ...")
    ice365, _ = v3.load_sic(SENSOR)
    assert ice365.shape[1:] == ocean.shape, "SIC grid does not match the anomaly/sector grid"
    clim_q_fs = v3.precompute_clim_percentiles(ice365, v3.DYN_Q_FS, v3.DYN_DOY_HW)

    recs = []
    for iy, yr in enumerate(years):
        if yr < PRE[0] or yr > POST[1] or not fail[iy].any():
            continue
        ts = v3.slice_season(ice365, f"{yr}{v3.FS_START_MMDD}", f"{yr}{v3.FS_END_MMDD}")
        if ts.time.size < K:
            continue
        ts_s = ts.rolling(time=K, center=True, min_periods=max(1, K // 2)).mean()
        slope = v3.centered_slope_vectorized(ts_s, SLOPE_HW)
        doys = ts.time.dt.dayofyear.values
        thr_map = np.maximum(clim_q_fs.sel(doy=doys, method="nearest").mean("doy").values, ABS_MIN)

        cand, thr = run_masks(ts_s, thr_map)
        slope_ok = slope > SLOPE_MIN
        detected = (cand & slope_ok).any("time").values
        level_reached = cand.any("time").values

        sust_min = ts_s.rolling(time=K, min_periods=K).min().shift(time=-(K - 1)).max("time").values
        best_slope = slope.where(cand).max("time").values

        yy, xx = np.where(fail[iy])
        for j, i in zip(yy, xx):
            recs.append(dict(
                y=int(j), x=int(i), year=int(yr), period="pre" if yr <= PRE[1] else "post",
                sector=SECTORS.get(int(sid[j, i]), "?") if np.isfinite(sid[j, i]) else "?",
                zone=zone[j, i], confirmed=not detected[j, i],
                kind="rate" if level_reached[j, i] else "level",
                threshold=float(thr_map[j, i]),
                shortfall=float(thr_map[j, i] - sust_min[j, i]),
                best_slope=float(best_slope[j, i])))
        print(f"  {yr}: {len(yy)} failure pixels")

    df = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ch2_level_vs_rate_pixelyears.csv", index=False)

    print(f"\nFidelity: {100*df['confirmed'].mean():.1f}% of failure pixel-years are also missed "
          f"by the re-implemented dynamic test (should be ~100%)")
    d = df[df["confirmed"]]

    def summarise(g):
        lv, rt = g[g.kind == "level"], g[g.kind == "rate"]
        return pd.Series(dict(
            n=len(g), pct_level=round(100 * len(lv) / len(g), 1), pct_rate=round(100 * len(rt) / len(g), 1),
            med_threshold=round(g["threshold"].median(), 2),
            level_med_shortfall=round(lv["shortfall"].median(), 3) if len(lv) else np.nan,
            rate_med_best_slope=round(rt["best_slope"].median(), 4) if len(rt) else np.nan))

    for keys, title in [(["zone", "period"], "by zone"),
                        (["sector", "period"], "long-season zone, by sector")]:
        sub = d if title == "by zone" else d[d.zone == "long"]
        tab = sub.groupby(keys).apply(summarise).reset_index()
        print(f"\n=== Level vs rate failures, {title} (slope threshold {SLOPE_MIN}/day) ===")
        print(tab.to_string(index=False))
        tab.to_csv(OUT / f"ch2_level_vs_rate_{'zone' if title == 'by zone' else 'long_by_sector'}.csv", index=False)
    print(f"\nCSVs written to {OUT}")


if __name__ == "__main__":
    main()
