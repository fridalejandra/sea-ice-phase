#!/usr/bin/env python
"""
21_build_clean_anomalies.py -- rebuild the daily analysis table with clean anomalies.

Why: the old *_periodclim table's pre-2016 day-of-year climatology was jagged (built over
1979-2015 incl. every-other-day SMMR years, unsmoothed). Subtracting it ADDED noise to the
pre-2016 anomalies and produced a spurious post-2016 "collapse" in day-to-day variability.

Here, from the raw SIA and wind_stress columns:
  * record starts 1988 (SSM/I era; every tendency a true one-day difference)
  * dSIA = SIA(t) - SIA(t-1), only for consecutive calendar days
  * climatologies are day-of-year means smoothed with a 31-day circular window
      *_anomaly            : period-specific (1988-2015 for pre, 2016-2023 for post), smoothed
      *_anomaly_commonclim : one 1988-2015 climatology for all years, smoothed
Output keeps the column names the existing scripts expect (delta_SIA_anomaly, wind_stress,
wind_stress_anomaly, SIA, sector, date), so they can be re-pointed with a one-line sed.
"""
import numpy as np
import pandas as pd

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
IN_CSV = f"{ROOT}/data/merged/analysis_table_daily_anomaly.csv"
OUT_CSV = f"{ROOT}/data/merged/analysis_table_daily_anomaly_clean.csv"
START, BREAK = 1988, 2016


def doy_clim(x, years):
    s = x[x.index.year.isin(years)]
    c = s.groupby(np.minimum(s.index.dayofyear.values, 365)).mean().reindex(range(1, 366))
    c = pd.concat([c.iloc[-15:], c, c.iloc[:15]]).rolling(31, center=True, min_periods=10).mean().iloc[15:-15]
    c.index = range(1, 366)
    return pd.Series(c.reindex(np.minimum(x.index.dayofyear.values, 365)).values, index=x.index)


def main():
    d = pd.read_csv(IN_CSV, parse_dates=["date"]).sort_values(["sector", "date"])
    out = []
    for sec, g in d.groupby("sector"):
        g = g.set_index("date")[["SIA", "wind_stress"]].asfreq("D")
        g["delta_SIA"] = g.SIA - g.SIA.shift(1)
        g = g[g.index.year >= START]
        last = g.index.year.max()
        pre, post = range(START, BREAK), range(BREAK, last + 1)
        for col, name in (("delta_SIA", "delta_SIA"), ("wind_stress", "wind_stress")):
            common = doy_clim(g[col], pre)
            per = pd.Series(np.where(g.index.year >= BREAK, doy_clim(g[col], post), common), index=g.index)
            g[f"{name}_anomaly"] = g[col] - per
            g[f"{name}_anomaly_commonclim"] = g[col] - common
        g["sector"] = sec
        out.append(g.reset_index())
    res = pd.concat(out).dropna(subset=["SIA"])
    res.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"wrote {OUT_CSV}: {len(res)} rows, {res.date.min().date()} -> {res.date.max().date()}")
    # roughness check: the climatology must be smooth in BOTH periods (the old table failed this)
    for sec, g in res.groupby("sector"):
        g = g[g.date.dt.month.isin([6, 7, 8])]
        clim = g.delta_SIA - g.delta_SIA_anomaly
        for lab, m in (("pre ", g.date.dt.year < BREAK), ("post", g.date.dt.year >= BREAK)):
            r = lambda s: np.nanstd(np.diff(s[m].values))
            print(f"  {sec[:14]:14s} {lab} roughness  dSIA={r(g.delta_SIA):7.0f}  clim={r(clim):6.0f}  anomaly={r(g.delta_SIA_anomaly):7.0f}")


if __name__ == "__main__":
    main()
