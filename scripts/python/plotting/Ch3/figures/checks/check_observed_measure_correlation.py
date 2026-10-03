#!/usr/bin/env python3
"""
check_observed_measure_correlation.py -- the corrected Check 2.

CORRECTION (2026-10-03): check_asl_ross_departure.py's "Check 2" used the
GAM's FITTED trend/amplitude/phase/residual_apac components, from
t32_attribution_by_cycle.csv -- ch3_config.py says directly these are
"used ONLY for the per-year attribution (Sect. 3.2 / Fig 7)". But Table 3
and Fig. 4 never use those fitted terms at all: per ch3_stats.py's own
methods 2.2.3 rule, every correlation in the chapter uses the OBSERVED
scalars (amplitude_raw_anom, max_doy_raw_anom, plus anomaly_from_iac /
residual_apac at the daily level for departure / raw anomaly). So the
earlier Check 2 answered "is the fitted GAM decomposition internally
entangled" (a real, separate, useful finding -- keep it as evidence for
the observed-over-fitted methods choice) but NOT "are the measures Table 3
actually reports on entangled with each other" -- that's what THIS script
checks, on the correct, matching variable set.

For each of the five sectors (circumpolar excluded -- not one of Table 3's
seven), at the annual level, 1979-2023: correlate all four measures Fig. 4
/ Table S2 use --
    departure    = anomaly_from_iac   (daily, averaged over the year)
    amplitude    = amplitude_raw_anom (ANNUAL_CSV)
    phase        = max_doy_raw_anom   (ANNUAL_CSV)
    raw anomaly  = residual_apac      (daily, averaged over the year)
-- with each other. Same |r| > 0.3 flag as the fitted-component check, so
the two tables read side by side.

Reads   DAILY_CSV, ANNUAL_CSV (via ch3_config)
Writes  results/ch3/tables/t_observed_measure_correlation_matrix.csv
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR, DAILY_CSV, ANNUAL_CSV, SECTORS_COMPUTE

YEAR_LO, YEAR_HI = 1979, 2023
MEAS_LABELS = {"departure": "departure", "amplitude": "amplitude",
               "phase": "phase", "raw_anom": "raw anomaly"}


def detrend(x):
    x = np.asarray(x, float); ok = np.isfinite(x)
    if ok.sum() < 3:
        return x
    t = np.arange(len(x), dtype=float); b, a = np.polyfit(t[ok], x[ok], 1)
    out = x.copy(); out[ok] = x[ok] - (a + b * t[ok]); return out


def pear(x, y):
    x, y = detrend(np.asarray(x, float)), detrend(np.asarray(y, float))
    ok = np.isfinite(x) & np.isfinite(y); n = int(ok.sum())
    if n < 6:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x[ok], y[ok]); return float(r), float(p), n


daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
if "period" in daily.columns:
    daily = daily[daily["period"] == "FULL"]
daily = daily[(daily["Year"] >= YEAR_LO) & (daily["Year"] <= YEAR_HI)].copy()
_dl = {str(x).lower(): x for x in daily["sector"].unique()}

ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"] == "FULL"]

rows = []
for sec, lab in SECTORS_COMPUTE.items():
    cands = [lab, sec, lab.replace(" ", "_"), "SIE_" + lab.replace(" ", "_")]
    key = next((c for c in cands if c.lower() in _dl), None)
    if key is None:
        print(f"  WARNING: could not find {lab!r} in DAILY_CSV's sector column -- skipping.")
        continue
    d = daily[daily["sector"] == _dl[key.lower()]]
    g = d.groupby("Year")
    dep = pd.DataFrame({"Year": g.size().index, "departure": g["anomaly_from_iac"].mean().values,
                        "raw_anom": g["residual_apac"].mean().values, "nd": g.size().values})
    dep = dep[dep["nd"] >= 0.35 * dep["nd"].median()]

    a = ann[ann["sector"] == lab]
    if a.empty:
        a = ann[ann["sector"] == sec]
    a = a[a.Year.between(YEAR_LO, YEAR_HI)][["Year", "amplitude_raw_anom", "max_doy_raw_anom"]].rename(
        columns={"amplitude_raw_anom": "amplitude", "max_doy_raw_anom": "phase"})

    m = dep.merge(a, on="Year", how="inner")
    if len(m) < 10:
        print(f"  WARNING: only {len(m)} overlapping years for {lab} -- skipping.")
        continue

    for c1, c2 in itertools.combinations(["departure", "amplitude", "phase", "raw_anom"], 2):
        r, p, n = pear(m[c1], m[c2])
        rows.append(dict(sector=lab, pair=f"{MEAS_LABELS[c1]}~{MEAS_LABELS[c2]}", r=r, p=p, n=n))

CC = pd.DataFrame(rows)
out = os.path.join(TABLES_DIR, "t_observed_measure_correlation_matrix.csv")
CC.to_csv(out, index=False)
print(f"wrote {out}\n")
print("(|r| > 0.3 flagged -- these are the measures Table 3 / Fig. 4 actually use;")
print(" compare against t_component_correlation_matrix.csv, the fitted-component version)\n")
for sec, g in CC.groupby("sector", sort=False):
    flagged = g[g["r"].abs() > 0.3]
    line = "  ".join(f"{row.pair}: {row.r:+.2f}{'*' if abs(row.r) > 0.3 else ''}" for row in g.itertuples())
    print(f"   {sec:16s} {line}")
print(f"\n{(CC['r'].abs() > 0.3).sum()} of {len(CC)} pairs flagged across {CC['sector'].nunique()} sectors.")
