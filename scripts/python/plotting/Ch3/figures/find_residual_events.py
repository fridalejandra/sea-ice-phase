#!/usr/bin/env python3
"""
The raw anomaly as an episode detector: WHEN was the ice edge far from its
modulated cycle, and for how long.

Motivation
----------
Eabry et al. (2025) show that the 2016 decline happened in distinct submonthly
episodes, and they had to pick the start and end dates of those episodes by
hand because a monthly anomaly smears them together and a daily anomaly against
a FIXED climatology mixes them with the seasonal cycle. The raw APAC anomaly is
that diagnostic defined continuously: its e-folding time of 12-20 days is
exactly the submonthly band their episodes occupy.

This script turns the residual into a dated event list you can take to the maps.
It makes no claim about what caused anything. It says when, where, how big and
how long -- and whether the other sectors were doing the same thing at the time.

Two methodological points that matter
-------------------------------------
1. STANDARDISE BY DAY OF YEAR. The residual's standard deviation swings by more
   than a factor of ten through the year (smallest at the February minimum,
   largest during the retreat). A fixed threshold in 10^6 km^2 would select
   almost nothing but retreat-season events. Every series is divided by a
   smoothed day-of-year standard deviation first, so "large" means large FOR
   THE TIME OF YEAR.

2. LEVEL AND TENDENCY ARE DIFFERENT QUESTIONS. The residual LEVEL answers "the
   edge sat far from where the cycle says it should" -- a state. Its TENDENCY
   answers "the edge moved fast relative to the cycle" -- which is closer to
   what Eabry et al. picked out by hand. Both are detected and reported; they
   pick out overlapping but not identical dates.

Usage
-----
    python find_residual_events.py                 # paths from ch3_config
    python find_residual_events.py --demo FILE.csv # on a plain SIE file, to
                                                   # see the output shape
    python find_residual_events.py --year 2016     # only events peaking in 2016
    python find_residual_events.py --top 30
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

THRESH = float(os.environ.get("THRESH", 1.5))   # |z| to open an event
MIN_DAYS = int(os.environ.get("MIN_DAYS", 5))   # shortest event kept
SMOOTH_DOY = 15                                 # days, for the seasonal SD


def parse_dates(series):
    raw = series.astype(str)
    t = pd.to_datetime(raw.str.extract(r"(\d{4}-\d{2}-\d{2})")[0],
                       errors="coerce", format="%Y-%m-%d")
    if t.isna().mean() > 0.5:
        t = pd.to_datetime(raw, errors="coerce")
    return t


def seasonal_z(df, col):
    """Divide by a smoothed day-of-year standard deviation, so that 'large'
    means large for the time of year rather than large in absolute terms."""
    doy = df["Date"].dt.dayofyear
    sd = df.groupby(doy)[col].transform("std")
    # circular smoothing of the day-of-year SD
    per = df.groupby(doy)[col].std().reindex(range(1, 367))
    per = per.interpolate(limit_direction="both")
    sm = (pd.concat([per, per, per]).rolling(SMOOTH_DOY, center=True,
                                             min_periods=1).mean()
          .iloc[len(per):2 * len(per)])
    sm.index = range(1, 367)
    sd = doy.map(sm)
    return df[col] / sd.replace(0, np.nan)


def find_events(df, zcol, thresh=THRESH, min_days=MIN_DAYS):
    """Contiguous runs where |z| >= thresh, allowing single-day dropouts."""
    z = df[zcol].to_numpy()
    over = np.abs(z) >= thresh
    # bridge one-day gaps so a brief dip does not split one episode in two
    for i in range(1, len(over) - 1):
        if over[i - 1] and over[i + 1]:
            over[i] = True
    events, i, n = [], 0, len(over)
    while i < n:
        if not over[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and over[j + 1]:
            j += 1
        seg = z[i:j + 1]
        if (j - i + 1) >= min_days and np.isfinite(seg).any():
            k = i + int(np.nanargmax(np.abs(seg)))
            events.append({
                "start": df["Date"].iloc[i], "end": df["Date"].iloc[j],
                "peak_date": df["Date"].iloc[k], "days": j - i + 1,
                "peak_z": float(z[k]),
                "sign": "low" if z[k] < 0 else "high",
                "integrated_z": float(np.nansum(seg)),
            })
        i = j + 1
    return events


def build(daily, sectors, valcol):
    """z-score each sector's series, level and tendency."""
    out = {}
    for s in sectors:
        d = daily[daily["sector"] == s][["Date", valcol]].dropna().sort_values("Date")
        if len(d) < 400:
            continue
        full = pd.DataFrame({"Date": pd.date_range(d["Date"].min(), d["Date"].max(), freq="D")})
        d = full.merge(d, on="Date", how="left")
        d["lev"] = d[valcol]
        d["tend"] = d[valcol].diff()
        d.loc[d[valcol].shift(1).isna(), "tend"] = np.nan
        d["z_lev"] = seasonal_z(d, "lev")
        d["z_tend"] = seasonal_z(d, "tend")
        out[s] = d
    return out


def report(zs, which, top, year, label):
    allev = []
    for s, d in zs.items():
        for e in find_events(d, which):
            e["sector"] = s
            allev.append(e)
    if not allev:
        print("  (no events)")
        return pd.DataFrame()
    ev = pd.DataFrame(allev)
    if year:
        ev = ev[ev["peak_date"].dt.year == year]
        if ev.empty:
            print("  (no events peaking in %d)" % year)
            return ev
    ev = ev.reindex(ev["integrated_z"].abs().sort_values(ascending=False).index)

    print("\n%s  (|z| >= %.1f for >= %d days)" % (label, THRESH, MIN_DAYS))
    print("  %-26s %-11s %-11s %5s %7s %8s   %s"
          % ("sector", "start", "end", "days", "peak z", "sum z", "other sectors on the peak day"))
    for _, r in ev.head(top).iterrows():
        others = []
        for s2, d2 in zs.items():
            if s2 == r["sector"]:
                continue
            m = d2.loc[d2["Date"] == r["peak_date"], which]
            if len(m) and np.isfinite(m.iloc[0]) and abs(m.iloc[0]) >= 1.0:
                others.append("%s %+.1f" % (str(s2).replace("SIE_", "")[:11], m.iloc[0]))
        print("  %-26s %-11s %-11s %5d %+7.1f %+8.0f   %s"
              % (str(r["sector"]).replace("SIE_", ""), r["start"].date(), r["end"].date(),
                 r["days"], r["peak_z"], r["integrated_z"], ", ".join(others) or "-"))
    return ev


def main():
    args = sys.argv[1:]
    top = int(args[args.index("--top") + 1]) if "--top" in args else 20
    year = int(args[args.index("--year") + 1]) if "--year" in args else None

    if "--demo" in args:
        path = args[args.index("--demo") + 1]
        w = pd.read_csv(path)
        w["Date"] = parse_dates(w[w.columns[0]])
        cols = [c for c in w.columns if c.startswith("SIE_")]
        long = w.melt(id_vars="Date", value_vars=cols,
                      var_name="sector", value_name="val").dropna()
        # DEMO ONLY: anomaly from a fixed day-of-year climatology, NOT the APAC
        # residual. Shown so you can see the output shape before running on the
        # real thing; the dates will differ.
        long["doy"] = long["Date"].dt.dayofyear
        clim = long.groupby(["sector", "doy"])["val"].transform("mean")
        long["resid"] = long["val"] - clim
        daily, valcol = long, "resid"
        print("DEMO MODE: anomaly from a FIXED climatology, not the APAC residual.\n"
              "Use this to check the output shape only. The real run uses\n"
              "residual_apac from daily_fitted.csv.\n")
    else:
        from ch3_config import DAILY_CSV
        daily = pd.read_csv(os.environ.get("DAILY_CSV", DAILY_CSV))
        if "period" in daily.columns:
            daily = daily[daily["period"] == "FULL"]
        dcol = "Date" if "Date" in daily.columns else daily.columns[0]
        daily["Date"] = parse_dates(daily[dcol])
        daily = daily.dropna(subset=["Date"])
        valcol = "residual_apac"
        if valcol not in daily.columns:
            sys.exit("no residual_apac column; found %s" % list(daily.columns)[:8])

    sectors = sorted(daily["sector"].dropna().unique())
    zs = build(daily, sectors, valcol)
    print("sectors: %s" % ", ".join(str(s).replace("SIE_", "") for s in zs))

    ev_l = report(zs, "z_lev", top, year,
                  "LEVEL events -- the edge sat far from its modulated cycle")
    ev_t = report(zs, "z_tend", top, year,
                  "TENDENCY events -- the edge moved fast relative to the cycle"
                  "  (closest to Eabry et al.'s episodes)")

    outdir = os.environ.get("OUTDIR", ".")
    for ev, name in ((ev_l, "residual_events_level.csv"),
                     (ev_t, "residual_events_tendency.csv")):
        if len(ev):
            p = os.path.join(outdir, name)
            ev.to_csv(p, index=False)
            print("\nwrote %s  (%d events)" % (p, len(ev)))

    print("\nNext step: take the peak dates to the SIC maps and look at what the")
    print("field was doing. If 2016 does not appear near the top of the tendency")
    print("list, the detector is not finding what Eabry et al. found by hand.")


if __name__ == "__main__":
    main()
