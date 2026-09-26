#!/usr/bin/env python3
"""
diagnose_telescope.py -- which columns of daily_fitted.csv actually add back up?

check_constructive_null.py assumed

    Extent = fitted_invariant + trend_component + amplitude_component
                              + phase_component + raw_anomaly

and its sanity check failed by 1.358, so that identity is wrong. This tries every
plausible version and reports how badly each one misses, overall and by sector, then
shows where the worst errors sit in time for the best candidate. Whichever line comes
back at ~1e-12 is the identity the reconstruction should be built on.

Usage:
    python diagnose_telescope.py [path/to/daily_fitted.csv]
"""
import os
import sys

import numpy as np
import pandas as pd

path = next((a for a in sys.argv[1:] if a.lower().endswith(".csv")), None) \
    or os.environ.get("SEAICE_DAILY_FITTED") or "daily_fitted.csv"
d = pd.read_csv(path)
d["Date"] = pd.to_datetime(d["Date"])
print(f"reading {path}   ({len(d):,} rows, {d['sector'].nunique()} sectors)\n")

C = set(d.columns)


def col(name):
    return d[name] if name in C else None


def total(*names):
    """Sum these columns, or None if any is absent."""
    parts = [col(n) for n in names]
    return None if any(p is None for p in parts) else sum(parts)


CANDIDATES = {
    "Extent = invariant + trend + amp + phase + raw_anomaly":
        (col("Extent"), total("fitted_invariant", "trend_component",
                              "amplitude_component", "phase_component", "raw_anomaly")),
    "Extent = invariant + trend + amp + phase + est_anomaly":
        (col("Extent"), total("fitted_invariant", "trend_component",
                              "amplitude_component", "phase_component", "est_anomaly")),
    "Extent = invariant + trend + amp + phase   (no residual term)":
        (col("Extent"), total("fitted_invariant", "trend_component",
                              "amplitude_component", "phase_component")),
    "Extent = invariant + anomaly_from_iac":
        (col("Extent"), total("fitted_invariant", "anomaly_from_iac")),
    "Extent = fitted_apac + residual_apac":
        (col("Extent"), total("fitted_apac", "residual_apac")),
    "anomaly_from_iac = trend + amp + phase + raw_anomaly":
        (col("anomaly_from_iac"), total("trend_component", "amplitude_component",
                                        "phase_component", "raw_anomaly")),
    "anomaly_from_iac = trend + amp + phase + est_anomaly":
        (col("anomaly_from_iac"), total("trend_component", "amplitude_component",
                                        "phase_component", "est_anomaly")),
    "anomaly_from_iac = trend + amp + phase   (no residual term)":
        (col("anomaly_from_iac"), total("trend_component", "amplitude_component",
                                        "phase_component")),
    "raw_anomaly == residual_apac":
        (col("raw_anomaly"), col("residual_apac")),
    "fitted_apac = invariant + trend + amp + phase":
        (col("fitted_apac"), total("fitted_invariant", "trend_component",
                                   "amplitude_component", "phase_component")),
}

rows = []
for label, (lhs, rhs) in CANDIDATES.items():
    if lhs is None or rhs is None:
        rows.append(dict(identity=label, max_abs=np.nan, median_abs=np.nan,
                         note="column missing"))
        continue
    e = (lhs - rhs).abs()
    rows.append(dict(identity=label, max_abs=e.max(), median_abs=e.median(),
                     note="EXACT" if e.max() < 1e-8 else ""))

res = pd.DataFrame(rows).sort_values("max_abs", na_position="last")
pd.set_option("display.width", 200, "display.max_colwidth", 62)
print(res.to_string(index=False, float_format=lambda v: f"{v:.3e}"), "\n")

best = res.dropna(subset=["max_abs"]).iloc[0]
if best["max_abs"] < 1e-8:
    print(f"USE THIS ONE:  {best['identity']}\n")
    raise SystemExit(0)

print(f"Nothing is exact. Closest: {best['identity']}  (max {best['max_abs']:.3e})")
lhs, rhs = CANDIDATES[best["identity"]]
d["err"] = (lhs - rhs).abs()

print("\nworst error by sector")
print(d.groupby("sector")["err"].agg(["max", "median"]).round(6).to_string())

print("\nworst error by year (top 10)")
print(d.groupby(d["Date"].dt.year)["err"].max().sort_values(ascending=False)
      .head(10).round(6).to_string())

big = d[d["err"] > 0.01]
print(f"\n{len(big):,} of {len(d):,} rows miss by more than 0.01 "
      f"({100 * len(big) / len(d):.2f}%)")
if len(big):
    print("dates of the worst 10 rows:")
    print(big.nlargest(10, "err")[["Date", "sector", "err"]].to_string(index=False))
    if "t" in C and "t_min" in C and "t_max" in C:
        pos = (big["t"] - big["t_min"]) / (big["t_max"] - big["t_min"])
        print(f"\nposition of the bad rows within the cycle: "
              f"{pos.min():.2f} to {pos.max():.2f} "
              f"(near 0 or 1 would mean the problem is only at the cycle edges)")
