#!/usr/bin/env python3
"""
check_constructive_null.py -- does a change in the LEVEL of the cycle, on its own,
reproduce what the regime-shift literature points at?

Sect. 4.3 currently argues that the recent change sits in the level of the cycle rather
than in its size or in the departures from it, on the evidence of variance ratios
(check_level_variance.py). A variance ratio is suggestive: it says the level got more
variable and the amplitude did not. This script asks the constructive version of the same
question, after Hwangbo and McKinnon (2026, GRL, doi:10.1029/2026GL123725), who test how
much of the observed change in heatwave metrics a shift in the mean alone reproduces.

The decomposition is additive and telescopes, so no refitting is needed. In
daily_fitted.csv (written by 01_fit_apac.R):

    Extent = fitted_invariant + trend_component + amplitude_component
                              + phase_component + raw_anomaly

Three reconstructions, all keeping the observed raw anomaly so that the only thing that
differs between them is which component is allowed to vary after the split year:

    FULL    everything observed                          (sanity check: must equal Extent)
    LEVEL   trend observed;  amplitude and phase frozen at pre-split climatology
    SHAPE   trend frozen at pre-split climatology;  amplitude and phase observed

"Frozen" means replaced, position in the cycle by position in the cycle, with the mean
over the pre-split years. Each reconstruction then goes through the same summary
statistics, and the question is which one moves the way the observations moved.

The statistics are deliberately ones that are NOT linear in the level, because a linear
one is decided by arithmetic before the script runs:
    var_ratio    variance of the summer-mean extent, after/before  (linear in the level --
                                                       reported for reference, not as
                                                       evidence; it ties to Table S5)
    n_records    running record-low summers after the split        (threshold: nonlinear)
    n_coherent   years with >= COHERENT_K of 5 sectors below their pre-split 10th
                 percentile at once                                (threshold: nonlinear)

How to read it:
  LEVEL reproduces most of the observed change and SHAPE little  -> Sect. 4.3 holds, and
        now holds constructively rather than by inference from variance ratios.
  Both reproduce some of it                                      -> the change is shared;
        soften 4.3 to say the level carries the larger part.
  Neither                                                        -> the reconstruction or
        the split year is wrong; check the SANITY line first.

Standalone: numpy, pandas, and the path to daily_fitted.csv. Nothing else is imported, so
it does not care where the rest of the pipeline lives.

Usage:
    python check_constructive_null.py [SPLIT] [LAST_YEAR] [path/to/daily_fitted.csv]
    python check_constructive_null.py 2007 /user/geog/.../Ch3/data/daily_fitted.csv
    python check_constructive_null.py 2016 2023      # drop the noisy 2024-25 record
The path may also come from SEAICE_DAILY_FITTED. Arguments can be in any order: the one
ending in .csv is the path, the bare numbers are the split year and the last year.
Default split 2007 (Hobbs et al. 2024); try 2016 as well.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

_args = [a for a in sys.argv[1:] if not a.lower().endswith(".csv")]
_path_arg = next((a for a in sys.argv[1:] if a.lower().endswith(".csv")), None)
SPLIT = int(_args[0]) if len(_args) > 0 else 2007
LAST_YEAR = int(_args[1]) if len(_args) > 1 else 2025

SUMMER = (1, 2)      # calendar months defining the summer mean (the cycle's low end)
COHERENT_K = 4       # how many of the five sectors must be low simultaneously

_CANDIDATES = [
    _path_arg,
    os.environ.get("SEAICE_DAILY_FITTED"),
    os.path.join(HERE, "daily_fitted.csv"),
    os.path.join(HERE, "data", "daily_fitted.csv"),
    os.path.join(HERE, "..", "data", "daily_fitted.csv"),
    os.path.join(os.getcwd(), "daily_fitted.csv"),
]
DAILY_FITTED = next((p for p in _CANDIDATES if p and os.path.isfile(p)), None)
if DAILY_FITTED is None:
    raise SystemExit(
        "Could not find daily_fitted.csv. Tried:\n  "
        + "\n  ".join(str(p) for p in _CANDIDATES if p)
        + "\nPass the full path as an argument, or set SEAICE_DAILY_FITTED."
    )

# The four additive pieces of the fitted cycle, as 01_fit_apac.R names them.
COMPONENTS = ["fitted_invariant", "trend_component",
              "amplitude_component", "phase_component"]
LEVEL_PART = ["trend_component"]                                   # how high it sits
SHAPE_PART = ["amplitude_component", "phase_component"]            # its size and its clock
FROZEN = {"FULL": [], "LEVEL": SHAPE_PART, "SHAPE": LEVEL_PART}


def load():
    d = pd.read_csv(DAILY_FITTED)
    if "period" in d.columns:                      # older files carry one block per period
        d = d[d["period"] == "FULL"].copy()

    need = COMPONENTS + ["raw_anomaly", "Date", "sector"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise SystemExit(f"daily_fitted.csv is missing: {missing}\n"
                         f"columns present: {list(d.columns)}")

    d["Date"] = pd.to_datetime(d["Date"])
    d["cyear"] = d["Date"].dt.year        # calendar year, for the summer statistics
    d["month"] = d["Date"].dt.month

    # Position within the cycle, used to freeze a component day by day. t runs from the
    # cycle's own start (t_min) to its end (t_max), so t - t_min is days into the cycle.
    if "t" in d.columns and "t_min" in d.columns:
        d["cycle_day"] = (d["t"] - d["t_min"]).round().astype(int)
        how = "t - t_min"
    elif "cycle_day" in d.columns:
        how = "cycle_day column"
    else:                                  # last resort: rank of the day within its cycle
        d = d.sort_values(["sector", "Year", "Date"])
        d["cycle_day"] = d.groupby(["sector", "Year"]).cumcount()
        how = "rank within (sector, Year)"

    lo, hi = d["cycle_day"].min(), d["cycle_day"].max()
    print(f"cycle position from {how}; range {lo} to {hi} "
          f"({'looks like days in a cycle' if 300 <= hi <= 400 else 'CHECK THIS'})")

    return d[d["cyear"].between(1979, LAST_YEAR)].copy()


def reconstruct(d, which):
    """Rebuild daily extent with the named components frozen at pre-split climatology."""
    out = d.copy()
    frozen = FROZEN[which]
    if frozen:
        pre = d[d["cyear"] < SPLIT]
        clim = pre.groupby(["sector", "cycle_day"])[frozen].mean().reset_index()
        out = out.drop(columns=frozen).merge(clim, on=["sector", "cycle_day"], how="left")
        # a cycle position never seen before the split has no climatology; drop those days
        out = out.dropna(subset=frozen)
    out["extent"] = out[COMPONENTS].sum(axis=1) + out["raw_anomaly"]
    return out


def summarise(out):
    """Per sector and calendar year: the summer-mean extent, then the statistics."""
    s = (out[out["month"].isin(SUMMER)]
         .groupby(["sector", "cyear"])["extent"].mean()
         .rename("summer").reset_index())

    rows = []
    for sec, g in s.groupby("sector"):
        g = g.sort_values("cyear")
        v, yrs = g["summer"].values, g["cyear"].values
        pre, post = v[yrs < SPLIT], v[yrs >= SPLIT]
        running_min = np.minimum.accumulate(np.r_[np.inf, v[:-1]])
        rows.append(dict(
            sector=str(sec).replace("SIE_", ""),
            var_pre=pre.var(ddof=1), var_post=post.var(ddof=1),
            var_ratio=post.var(ddof=1) / pre.var(ddof=1),
            n_records=int(((v < running_min) & (yrs >= SPLIT)).sum()),
        ))

    sect = s[~s["sector"].astype(str).str.contains("circumpolar", case=False)]
    thr = (sect[sect["cyear"] < SPLIT].groupby("sector")["summer"]
           .quantile(0.10).rename("thr"))
    j = sect.join(thr, on="sector")
    low = j.assign(below=j["summer"] < j["thr"]).groupby("cyear")["below"].sum()
    n_coherent = int(((low >= COHERENT_K) & (low.index >= SPLIT)).sum())

    return pd.DataFrame(rows).set_index("sector"), n_coherent


def main():
    d = load()
    print(f"Constructive null, split {SPLIT}, record through {LAST_YEAR}, "
          f"summer = calendar months {SUMMER}")
    print(f"reading {DAILY_FITTED}\n")

    full = reconstruct(d, "FULL")
    if "Extent" in d.columns:
        gap = (full["extent"] - full["Extent"]).abs().max()
        print(f"SANITY  max |reconstruction - observed Extent| = {gap:.3e}   "
              f"({'ok' if gap < 1e-6 else 'FAILS -- the components do not telescope; stop here'})\n")
    else:
        print("SANITY  no Extent column to check against.\n")

    tabs, coh = {}, {}
    for which in ("FULL", "LEVEL", "SHAPE"):
        tabs[which], coh[which] = summarise(reconstruct(d, which))

    for stat, label in (("var_ratio", "variance ratio of summer extent, after/before"),
                        ("n_records", "running record-low summers after the split")):
        print(label)
        print(pd.DataFrame({w: tabs[w][stat] for w in ("FULL", "LEVEL", "SHAPE")})
              .round(2).to_string(), "\n")

    print(f"years with >= {COHERENT_K} of 5 sectors below their pre-{SPLIT} 10th percentile")
    print("  " + "   ".join(f"{w} {coh[w]}" for w in ("FULL", "LEVEL", "SHAPE")), "\n")

    cp = next((i for i in tabs["FULL"].index if "circumpolar" in i.lower()), None)
    if cp is not None:
        f = tabs["FULL"].loc[cp]
        den = f["var_post"] - f["var_pre"]
        print("Circumpolar, share of the observed change in variance reproduced:")
        if abs(den) < 0.10 * f["var_pre"]:
            print("  n/a -- the observed variance barely changed at this split, so there "
                  "is no change to apportion")
        else:
            for name, w in (("LEVEL only", "LEVEL"), ("SHAPE only", "SHAPE")):
                t = tabs[w].loc[cp]
                print(f"  {name:11s} {(t['var_post'] - f['var_pre']) / den:+.0%}")

    print("\nvar_ratio is linear in the level -- reference only. The evidence is in the "
          "record counts and the coherence line.")


if __name__ == "__main__":
    main()

