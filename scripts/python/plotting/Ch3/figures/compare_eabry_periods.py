#!/usr/bin/env python3
"""
compare_eabry_periods.py -- do Eabry et al. (2025)'s five 2016 decline periods
show up in the decomposition, and in which component?

Eabry et al. (J. Climate 38, 7105; Sect. 3b) identify five submonthly periods
of rapid decline in the SIE anomaly from a fixed 1979-2015 climatology:
    P1 29 Aug-2 Sep   dSIEa -0.46   P2 8-14 Sep   -0.68   P3 13-26 Oct  -0.49
    P4 4-20 Nov       -1.29         P5 2-14 Dec   -0.69   (10^6 km^2, circumpolar)

For each period and sector this prints
  d_anom    change in anomaly_from_iac (Extent - invariant cycle) from T1 to T2,
            the analogue of their dSIEa. The circumpolar values are the check:
            they should be close to theirs (differences: our IAC is a spline
            over 1979-2025, theirs a 1979-2015 day-of-year mean).
  d_trend, d_amp, d_phase, d_resid
            the same change split into the four components (they sum to d_anom).
  resid_pct percentile of d_resid among the same calendar window in every
            other year 1988-2023 (low = unusually large fall for the season).
  events    level/tendency events from find_residual_events.py overlapping
            the window (if the CSVs are present).

Question answered: of each episode Eabry et al. picked by hand, how much is a
departure from the adjusted cycle (residual) and how much is the cycle itself
shifting (phase, amplitude, trend)? And is the residual's change during their
windows unusual for the time of year?

Reads daily_fitted.csv (period == FULL) via ch3_data. Run from the figures dir.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ch3_data as D
from ch3_config import SECTORS, SECTOR_LABELS, TABLES_DIR

PERIODS = [("P1", "2016-08-29", "2016-09-02", -0.46),
           ("P2", "2016-09-08", "2016-09-14", -0.68),
           ("P3", "2016-10-13", "2016-10-26", -0.49),
           ("P4", "2016-11-04", "2016-11-20", -1.29),
           ("P5", "2016-12-02", "2016-12-14", -0.69)]
COMP = {"trend": "trend_component", "amp": "amplitude_component",
        "phase": "phase_component", "resid": "residual_apac"}
Y0, Y1 = 1988, 2023

d = D.load_daily(period="FULL")
d["Date"] = pd.to_datetime(d["Date"])
need = ["anomaly_from_iac"] + list(COMP.values())
miss = [c for c in need if c not in d.columns]
if miss:
    sys.exit(f"daily file lacks {miss}")
W = {s: d[d["sector"] == s].set_index("Date").sort_index() for s in SECTORS}


def value(s, date, col):
    """value on the date, or the nearest day within 2 (SMMR gaps)."""
    g = W[s]
    i = g.index.get_indexer([pd.Timestamp(date)], method="nearest")[0]
    if abs((g.index[i] - pd.Timestamp(date)).days) > 2:
        return np.nan
    return float(g[col].iloc[i])


def delta(s, t1, t2, col):
    return value(s, t2, col) - value(s, t1, col)


ev = {}
for kind in ("level", "tendency"):
    p = os.path.join(HERE, f"residual_events_{kind}.csv")
    if os.path.exists(p):
        ev[kind] = pd.read_csv(p, parse_dates=["start", "end", "peak_date"])

rows = []
for name, a, b, theirs in PERIODS:
    t1, t2 = pd.Timestamp(a), pd.Timestamp(b)
    print(f"\n== {name}  {a} to {b}   (Eabry et al. circumpolar dSIEa {theirs:+.2f})")
    print(f"   {'sector':26s} {'d_anom':>7s} {'d_trend':>8s} {'d_amp':>7s} {'d_phase':>8s} {'d_resid':>8s}"
          f"  {'resid %ile':>10s}   overlapping events")
    for s in SECTORS:
        dv = {k: delta(s, t1, t2, c) for k, c in COMP.items()}
        da = delta(s, t1, t2, "anomaly_from_iac")
        # same calendar window in every other year
        clim = []
        for y in range(Y0, Y1 + 1):
            if y == 2016:
                continue
            u1, u2 = t1.replace(year=y), t2.replace(year=y)
            clim.append(delta(s, u1, u2, "residual_apac"))
        clim = np.array([c for c in clim if np.isfinite(c)])
        pct = 100.0 * np.mean(clim <= dv["resid"]) if len(clim) and np.isfinite(dv["resid"]) else np.nan
        ov = []
        for kind, e in ev.items():
            q = e[(e["sector"] == s) & (e["start"] <= t2) & (e["end"] >= t1)]
            for _, r in q.iterrows():
                ov.append(f"{kind[0]}:{r['start']:%d %b}-{r['end']:%d %b} z{r['peak_z']:+.1f}")
        print(f"   {SECTOR_LABELS.get(s, s):26s} {da:+7.3f} {dv['trend']:+8.3f} {dv['amp']:+7.3f} "
              f"{dv['phase']:+8.3f} {dv['resid']:+8.3f}  {pct:9.0f}%   {'; '.join(ov) or '-'}")
        rows.append(dict(period=name, start=a, end=b, sector=SECTOR_LABELS.get(s, s),
                         d_anom=da, **{f"d_{k}": v for k, v in dv.items()},
                         resid_percentile=pct, n_years_clim=len(clim),
                         overlapping_events="; ".join(ov), eabry_circ_dSIEa=theirs))

t = pd.DataFrame(rows)
circ = t[t["sector"].str.contains("ircumpolar")]
print("\n== summary, circumpolar")
print(circ[["period", "d_anom", "eabry_circ_dSIEa", "d_trend", "d_amp", "d_phase", "d_resid",
            "resid_percentile"]].round(3).to_string(index=False))
share = (circ["d_resid"].abs() / (circ[["d_trend", "d_amp", "d_phase", "d_resid"]].abs().sum(axis=1)))
print("   residual's share of |component changes| per period:", ", ".join(f"{v:.0%}" for v in share))
os.makedirs(TABLES_DIR, exist_ok=True)
out = os.path.join(TABLES_DIR, "t35_eabry_periods.csv")
t.to_csv(out, index=False)
print(f"\nwrote {out}")
print("reading guide: d_anom (circumpolar) close to Eabry's value = the two anomalies agree. "
      "A period whose d_anom is carried by d_phase/d_amp is the cycle shifting, invisible to the residual "
      "by construction; one carried by d_resid, with a low percentile, is an episode the residual sees.")
