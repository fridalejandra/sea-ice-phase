#!/usr/bin/env python3
"""
check_level_to_timing.py -- does the post-2016 change in LEVEL alone account for the shift in the
timing of the seasonal advance and retreat?

Idea (back-of-envelope, sector-total extent; not pixel dates):
  1. Mean daily climatology of extent, 1979-2015 ("pre") and 2016-END ("post"), smoothed.
  2. For a set of extent values X on the advance limb and the retreat limb, find the day each
     climatology crosses X.  Observed shift = post crossing - pre crossing (days; + = later).
  3. Take the PRE climatology, lower it by the change in annual-mean extent (the level change)
     and nothing else, and find the crossings again.  Predicted shift = level-only crossing - pre.
  4. If level explains the timing, predicted ~ observed at every X.  What is left over
     (observed - predicted) is what amplitude and phase would have to supply.

Also prints the rule-of-thumb: shift ~ level change / seasonal rate of change of extent.

Usage:
    python check_level_to_timing.py            # circumpolar + five sectors, post = 2016-2024
    python check_level_to_timing.py 2023       # post = 2016-2023 (leaves out the noisy 2024-25)
    python check_level_to_timing.py 2024 SIE_Weddell

Reads  data/ch3/daily_fitted.csv (FULL fit; only the Extent column is used, no fitted terms)
Writes results/ch3/level_to_timing.csv
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = "/Users/fridaperez/Research/repos/sea-ice-phase"
DAILY = os.path.join(ROOT, "data", "ch3", "daily_fitted.csv")
OUT = os.path.join(ROOT, "results", "ch3", "level_to_timing.csv")
PERIOD = "FULL"
PRE = (1979, 2015)
POST_START = 2016
POST_END = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
SECTORS = ([sys.argv[2]] if len(sys.argv) > 2 else
           ["SIE_circumpolar", "SIE_Weddell", "SIE_Amundsen_Bellingshausen", "SIE_Ross",
            "SIE_East_Antarctica", "SIE_King_Haakon"])
FRACTIONS = (0.2, 0.35, 0.5, 0.65, 0.8)   # X as a fraction of the pre-climatology range
SMOOTH = 15                               # days, circular running mean


def climatology(df, y0, y1):
    g = df[(df.Y >= y0) & (df.Y <= y1)]
    g = g[~((g.Date.dt.month == 2) & (g.Date.dt.day == 29))]       # drop 29 Feb
    doy = g.Date.dt.dayofyear.where(~g.Date.dt.is_leap_year, g.Date.dt.dayofyear - (g.Date.dt.month > 2))
    c = g.groupby(doy)["x"].mean().reindex(range(1, 366)).interpolate(limit_direction="both").values
    k = np.ones(SMOOTH) / SMOOTH                                   # circular smoothing
    ext = np.concatenate([c[-SMOOTH:], c, c[:SMOOTH]])
    return np.convolve(ext, k, mode="same")[SMOOTH:-SMOOTH]


def crossings(c, X, i_min, i_max):
    """Fractional day of the advance and of the retreat crossing of X.
    The year is started at i_min and split at i_max, both taken from the PRE climatology, so every
    curve is read on the same limbs and the dates are comparable."""
    n = len(c)
    roll = np.roll(c, -i_min)
    imax = (i_max - i_min) % n
    adv, ret = roll[: imax + 1], roll[imax:]
    ia, ir = np.where(adv >= X)[0], np.where(ret <= X)[0]
    if len(ia) == 0 or len(ir) == 0 or ia[0] == 0 or ir[0] == 0:
        return np.nan, np.nan
    a0, r0 = ia[0], ir[0]
    a = (a0 - 1) + (X - adv[a0 - 1]) / (adv[a0] - adv[a0 - 1])
    r = imax + (r0 - 1) + (ret[r0 - 1] - X) / (ret[r0 - 1] - ret[r0])
    return a + i_min, r + i_min


def main():
    if not os.path.exists(DAILY):
        sys.exit(f"Input not found: {DAILY}")
    d = pd.read_csv(DAILY, parse_dates=["Date"])
    if "period" in d.columns:
        d = d[d["period"] == PERIOD]
    col = next(c for c in ("Extent", "extent", "SIE", "sie") if c in d.columns)
    d["x"] = d[col] * (1e-6 if d[col].max() > 1000 else 1.0)       # 10^6 km^2
    d["Y"] = d.Date.dt.year
    rows = []
    for sec in SECTORS:
        g = d[d["sector"] == sec]
        if g.empty:
            print(f"(no rows for {sec})"); continue
        pre = climatology(g, *PRE)
        post = climatology(g, POST_START, POST_END)
        dL = (g[(g.Y >= POST_START) & (g.Y <= POST_END)].groupby("Y")["x"].mean().mean()
              - g[(g.Y >= PRE[0]) & (g.Y <= PRE[1])].groupby("Y")["x"].mean().mean())
        lo, hi = pre.min(), pre.max()
        i_min, i_max = int(np.argmin(pre)), int(np.argmax(pre))
        print(f"\n{sec}: level change {dL:+.2f} (10^6 km^2), post = {POST_START}-{POST_END}")
        print("   X      advance: obs  pred  left  |  retreat: obs  pred  left   (days; + = later)")
        for f in FRACTIONS:
            X = lo + f * (hi - lo)
            a0, r0 = crossings(pre, X, i_min, i_max)
            a1, r1 = crossings(post, X, i_min, i_max)
            a2, r2 = crossings(pre + dL, X, i_min, i_max)
            ao, ro, ap, rp = a1 - a0, r1 - r0, a2 - a0, r2 - r0
            rate_a = (np.interp(a0 + 3, np.arange(365), pre) - np.interp(a0 - 3, np.arange(365), pre)) / 6
            rows.append(dict(sector=sec, X=X, adv_obs=ao, adv_level_only=ap, ret_obs=ro, ret_level_only=rp,
                             level_change=dL, adv_rate=rate_a))
            print(f"  {X:5.1f}          {ao:+6.1f} {ap:+6.1f} {ao - ap:+6.1f}  |          {ro:+6.1f} {rp:+6.1f} {ro - rp:+6.1f}")
        t = pd.DataFrame([r for r in rows if r["sector"] == sec])
        for lab, o, p in (("advance", "adv_obs", "adv_level_only"), ("retreat", "ret_obs", "ret_level_only")):
            print(f"   mean {lab}: observed {t[o].mean():+.1f} d, level-only {t[p].mean():+.1f} d")
        obs_season = t["adv_obs"].mean() - t["ret_obs"].mean()
        pred_season = t["adv_level_only"].mean() - t["ret_level_only"].mean()
        print(f"   season shortened by: observed {obs_season:.1f} d, level-only {pred_season:.1f} d")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print("Reading it: level-only close to observed -> the timing shift is largely the level change.")
    print("This uses sector-total extent crossings, not pixel freeze/melt dates: it checks magnitude only.")


if __name__ == "__main__":
    main()
