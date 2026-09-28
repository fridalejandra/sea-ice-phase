#!/usr/bin/env python3
"""
check_constructive_null.py -- does a change in the LEVEL of the cycle, on its own,
reproduce what the regime-shift literature points at?

Sect. 4.3 argues that the recent change sits in the level of the cycle rather than in its
size or in the departures from it, on the evidence of variance ratios
(check_level_variance.py). A variance ratio is suggestive: it says the level got more
variable and the amplitude did not. This is the constructive version of the same question,
after Hwangbo and McKinnon (2026, GRL, doi:10.1029/2026GL123725), who test how much of the
observed change in heatwave metrics a shift in the mean alone reproduces.

COMPONENTS ARE RECOMPUTED, NOT READ FROM THE FILE. The stored amplitude_component and
phase_component columns are normalised against each row's calendar-year amplitude, which
steps on 1 January -- partway through a cycle that runs 21 Feb to 20 Feb. plot_fig7_sectors.R
fixes this by rebuilding the fits against an amplitude blended across the retreat limb, and
the chapter's figures use the blended version. This script does the same thing, so that it
tests the decomposition the chapter actually plots. The blend is lifted directly from
components_for() in that script:

    ampb  = ampY (1-w) + ampN w ,   w = clip((day - day_of_max) / (365 - day_of_max), 0, 1)
    famp  = u_amp  * ampb + minb
    fapac = u_apac * ampb + minb

    trend     = trend_component
    amplitude = famp  - iac_notrend - trend_component
    phase     = fapac - famp
    raw       = Extent - fapac

and these satisfy Extent = iac_notrend + trend + amplitude + phase + raw exactly, by
construction, in extent units (not percent -- the /amplitude*100 of the figure script is
dropped here because the test needs 10^6 km^2).

Three reconstructions, all keeping the observed raw anomaly, so the only thing that differs
is which component is allowed to vary after the split year:

    FULL    everything observed                          (must equal observed Extent)
    LEVEL   trend observed;  amplitude and phase frozen at pre-split climatology
    SHAPE   trend frozen at pre-split climatology;  amplitude and phase observed

"Frozen" means replaced, cycle day by cycle day, with the mean over the pre-split cycles.

The statistics are deliberately ones that are NOT linear in the level, because a linear one
is decided by arithmetic before the script runs:
    var_ratio    variance of the summer-mean extent, after/before   (linear -- reference
                                                        only; it ties to Table S5)
    n_records    running record-low summers after the split         (threshold: nonlinear)
    n_coherent   years with >= COHERENT_K of 5 sectors below their pre-split 10th
                 percentile at once                                 (threshold: nonlinear)

How to read it:
  LEVEL reproduces most of the observed change and SHAPE little  -> Sect. 4.3 holds, and
        now holds constructively rather than by inference from variance ratios.
  Both reproduce some of it                                      -> the change is shared;
        soften 4.3 to say the level carries the larger part.
  Neither                                                        -> check the DIAGNOSTICS
        block before believing anything below it.

Needs, from daily_fitted.csv: Date, sector, Extent, u_amp, u_apac, amplitude, min_extent,
iac_notrend, trend_component (and period, if the file has one).

Usage:
    python check_constructive_null.py [SPLIT] [LAST_YEAR] [path/to/daily_fitted.csv]
    python check_constructive_null.py 2007 ~/Research/repos/sea-ice-phase/data/ch3/daily_fitted.csv
    python check_constructive_null.py 2016 2023      # drop the noisy 2024-25 record
Arguments in any order: the one ending in .csv is the path, bare numbers are the split year
and the last year. The path may also come from SEAICE_DAILY_FITTED.
"""
import os
import sys

import numpy as np
import pandas as pd

_args = [a for a in sys.argv[1:] if not a.lower().endswith(".csv")]
_path = next((a for a in sys.argv[1:] if a.lower().endswith(".csv")), None)
SPLIT = int(_args[0]) if len(_args) > 0 else 2007
LAST_YEAR = int(_args[1]) if len(_args) > 1 else 2025

SUMMER = (1, 2)        # calendar months defining the summer mean (the cycle's low end)
COHERENT_K = 4         # how many of the five sectors must be low simultaneously
PERIOD = "FULL"        # the chapter's full-record fit, as in plot_fig7_sectors.R
MIN_ROWS = 300         # a cycle with fewer days than this is incomplete; skip it

HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    _path, os.environ.get("SEAICE_DAILY_FITTED"),
    os.path.join(HERE, "daily_fitted.csv"),
    os.path.join(HERE, "data", "ch3", "daily_fitted.csv"),
    os.path.expanduser("~/Research/repos/sea-ice-phase/data/ch3/daily_fitted.csv"),
    os.path.join(os.getcwd(), "daily_fitted.csv"),
]
DAILY = next((p for p in _CANDIDATES if p and os.path.isfile(p)), None)
if DAILY is None:
    raise SystemExit("Could not find daily_fitted.csv. Tried:\n  "
                     + "\n  ".join(str(p) for p in _CANDIDATES if p)
                     + "\nPass the full path as an argument.")

NEEDED = ["Date", "sector", "Extent", "u_amp", "u_apac", "amplitude",
          "min_extent", "iac_notrend", "trend_component"]
PARTS = ["trend", "amp", "phase", "raw"]
FROZEN = {"FULL": [], "LEVEL": ["amp", "phase"], "SHAPE": ["trend"]}


def build():
    """Recompute the four components, cycle by cycle, with the retreat-limb blend."""
    d = pd.read_csv(DAILY)
    missing = [c for c in NEEDED if c not in d.columns]
    if missing:
        raise SystemExit(f"{DAILY}\nis missing: {missing}\n"
                         f"columns present: {list(d.columns)}\n"
                         "This is probably the wrong daily_fitted.csv -- the one the figure "
                         "scripts read carries u_amp, u_apac, amplitude, min_extent and "
                         "iac_notrend.")
    d["Date"] = pd.to_datetime(d["Date"])
    if "period" in d.columns:
        if PERIOD not in set(d["period"]):
            raise SystemExit(f"period '{PERIOD}' not in file; have: {sorted(set(d['period']))}")
        d = d[d["period"] == PERIOD]

    out, skipped = [], []
    for sec, gs in d.groupby("sector", sort=False):
        gs = gs.sort_values("Date")
        for y in range(1979, LAST_YEAR + 1):
            t0 = pd.Timestamp(year=y, month=2, day=21)
            t1 = pd.Timestamp(year=y + 1, month=2, day=20)
            g = gs[(gs["Date"] >= t0) & (gs["Date"] <= t1)]
            if len(g) < MIN_ROWS:
                skipped.append((sec, y, len(g)))
                continue
            cd = (g["Date"] - t0).dt.days.to_numpy()

            ampY, minY = g["amplitude"].iloc[0], g["min_extent"].iloc[0]
            ampN, minN = g["amplitude"].iloc[-1], g["min_extent"].iloc[-1]
            dmax = cd[np.argmax(g["Extent"].to_numpy())]
            w = np.clip((cd - dmax) / max(365 - dmax, 1), 0, 1)
            ampb = ampY * (1 - w) + ampN * w
            minb = minY * (1 - w) + minN * w

            famp = g["u_amp"].to_numpy() * ampb + minb
            fapac = g["u_apac"].to_numpy() * ampb + minb
            iac = g["iac_notrend"].to_numpy()
            tr = g["trend_component"].to_numpy()

            out.append(pd.DataFrame(dict(
                sector=sec, cycle=y, cycle_day=cd, Date=g["Date"].to_numpy(),
                Extent=g["Extent"].to_numpy(), iac=iac,
                trend=tr, amp=famp - iac - tr, phase=fapac - famp,
                raw=g["Extent"].to_numpy() - fapac,
                fapac_blend=fapac,
                fapac_stored=g["fitted_apac"].to_numpy() if "fitted_apac" in g else np.nan,
            )))
    if skipped:
        print(f"skipped {len(skipped)} incomplete cycles "
              f"(e.g. {skipped[:3]}{' ...' if len(skipped) > 3 else ''})")
    f = pd.concat(out, ignore_index=True)
    f["cyear"] = f["Date"].dt.year
    f["month"] = f["Date"].dt.month
    return f


def reconstruct(f, which):
    out = f.copy()
    frozen = FROZEN[which]
    if frozen:
        pre = f[f["cycle"] < SPLIT]
        clim = pre.groupby(["sector", "cycle_day"])[frozen].mean().reset_index()
        out = out.drop(columns=frozen).merge(clim, on=["sector", "cycle_day"], how="left")
        out = out.dropna(subset=frozen)
    out["extent"] = out["iac"] + out[PARTS].sum(axis=1)
    return out


def summarise(out):
    s = (out[out["month"].isin(SUMMER)]
         .groupby(["sector", "cyear"])["extent"].mean().rename("summer").reset_index())
    rows = []
    for sec, g in s.groupby("sector"):
        g = g.sort_values("cyear")
        v, yrs = g["summer"].to_numpy(), g["cyear"].to_numpy()
        pre, post = v[yrs < SPLIT], v[yrs >= SPLIT]
        running_min = np.minimum.accumulate(np.r_[np.inf, v[:-1]])
        rows.append(dict(sector=str(sec).replace("SIE_", ""),
                         var_pre=pre.var(ddof=1), var_post=post.var(ddof=1),
                         var_ratio=post.var(ddof=1) / pre.var(ddof=1),
                         n_records=int(((v < running_min) & (yrs >= SPLIT)).sum())))
    sect = s[~s["sector"].astype(str).str.contains("circumpolar", case=False)]
    thr = sect[sect["cyear"] < SPLIT].groupby("sector")["summer"].quantile(0.10).rename("thr")
    j = sect.join(thr, on="sector")
    low = j.assign(below=j["summer"] < j["thr"]).groupby("cyear")["below"].sum()
    return (pd.DataFrame(rows).set_index("sector"),
            int(((low >= COHERENT_K) & (low.index >= SPLIT)).sum()))


def main():
    f = build()
    print(f"\nreading {DAILY}")
    print(f"constructive null, split {SPLIT}, through {LAST_YEAR}, "
          f"summer = calendar months {SUMMER}")
    print(f"{f['cycle'].nunique()} cycles x {f['sector'].nunique()} sectors, "
          f"{f['Date'].min().date()} to {f['Date'].max().date()}\n")

    print("DIAGNOSTICS")
    full = reconstruct(f, "FULL")
    gap = (full["extent"] - full["Extent"]).abs().max()
    print(f"  reconstruction vs observed Extent   max {gap:.3e}   "
          f"({'ok, exact by construction' if gap < 1e-8 else 'FAILS -- stop'})")
    if full["fapac_stored"].notna().any():
        b = (full["fapac_blend"] - full["fapac_stored"]).abs()
        print(f"  blended APAC fit vs stored column   max {b.max():.4f}, median {b.median():.4f}")
        print("  (this is the calendar-year amplitude step the blend removes; "
              "nonzero is expected)")
    print()

    tabs, coh = {}, {}
    for which in ("FULL", "LEVEL", "SHAPE"):
        tabs[which], coh[which] = summarise(reconstruct(f, which))

    for stat, label in (("var_ratio", "variance ratio of summer extent, after/before"),
                        ("n_records", "running record-low summers after the split")):
        print(label)
        print(pd.DataFrame({w: tabs[w][stat] for w in ("FULL", "LEVEL", "SHAPE")})
              .round(2).to_string(), "\n")

    print(f"years with >= {COHERENT_K} of 5 sectors below their pre-{SPLIT} 10th percentile")
    print("  " + "   ".join(f"{w} {coh[w]}" for w in ("FULL", "LEVEL", "SHAPE")), "\n")

    cp = next((i for i in tabs["FULL"].index if "circumpolar" in i.lower()), None)
    if cp is not None:
        fr = tabs["FULL"].loc[cp]
        den = fr["var_post"] - fr["var_pre"]
        print("circumpolar, share of the observed change in variance reproduced:")
        if abs(den) < 0.10 * fr["var_pre"]:
            print("  n/a -- the observed variance barely changed at this split")
        else:
            for name, w in (("LEVEL only", "LEVEL"), ("SHAPE only", "SHAPE")):
                print(f"  {name:11s} {(tabs[w].loc[cp]['var_post'] - fr['var_pre']) / den:+.0%}")

    print("\nvar_ratio is linear in the level -- reference only. The evidence is in the "
          "record counts and the coherence line.")


if __name__ == "__main__":
    main()
