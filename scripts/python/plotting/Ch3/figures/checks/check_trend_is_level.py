#!/usr/bin/env python3
"""
check_trend_is_level.py -- is the Trend term of the decomposition the LEVEL of the cycle?

The chapter treats the observed annual-mean extent as the level and says the Trend term carries its
"slowly varying part". This check makes that precise. For each series and year, the level departure is
the annual mean of anomaly_from_iac, which is exactly the sum of the annual means of the four
components (trend + amplitude + phase + raw). So the level can be split, year by year, into what each
term carries.

  A. CORRELATION. r between the level and the Trend term's annual mean, and the share of the level's
     variance the Trend term explains on its own (R^2), full record.
  B. VARIANCE BUDGET. var(level) = sum of the four components' variances + 2 x covariances; each shown as
     a share of var(level). Tells whether the year-to-year level lives in the Trend term or elsewhere.
  C. WHAT THE TREND MISSES. level minus the Trend term's annual mean is the part of the level the smooth
     term cannot follow. Its variance as a share of var(level), and how it splits among Amplitude, Phase
     and Raw. This is the 'leak' measured directly on the level, not inferred.
  D. TIMESCALES. The same budget for the slow part of the level (9-yr centred running mean) and the fast
     part (level minus that). If the Trend term is the level's low-pass, it should own the slow part and
     little of the fast part.
  E. THE PHASE TERM'S MEAN. The Phase term is -delta C'(t) to first order and should average to ~0 over a
     year; the Beta warp, anchored at the minimum, need not. Mean and SD of its annual mean, and r with
     the level: a non-zero, level-correlated Phase mean is the warp carrying level.

Reads   DAILY_CSV (via ch3_config)
Writes  results/ch3/tables/t_trend_is_level_<START>_<END>.csv
Run     python checks/check_trend_is_level.py [END]      (START=1988 env var to drop the SMMR era)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from ch3_config import DAILY_CSV, TABLES_DIR, SECTORS, SECTOR_LABELS, YEAR_MIN

END = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
START = int(os.environ.get("START", YEAR_MIN))
WIN = 9
COMP = {"trend": "trend_component", "amplitude": "amplitude_component",
        "phase": "phase_component", "raw": "residual_apac"}


def budget(t, cols, total):
    """Variance shares of each column and of the covariances, relative to var(total)."""
    v = t[total].var(ddof=1)
    own = {c: t[c].var(ddof=1) / v for c in cols}
    cov = 1 - sum(own.values())
    return own, cov, v


cols = ["Date", "sector", "period", "anomaly_from_iac"] + list(COMP.values())
d = pd.read_csv(DAILY_CSV, usecols=lambda c: c in cols, dtype={"Date": str})
if "period" in d.columns:
    d = d[d["period"] == "FULL"]
d["Y"] = d["Date"].str[:4].astype(int)
d = d[d["Y"].between(START, END)]
missing = [c for c in list(COMP.values()) + ["anomaly_from_iac"] if c not in d.columns]
if missing:
    sys.exit(f"DAILY_CSV lacks {missing}")

rows = []
print(f"Is the Trend term the level?  annual means, {START}-{END}\n")
for sec in SECTORS:
    g = d[d.sector == sec].groupby("Y")
    t = pd.DataFrame({k: g[v].mean() for k, v in COMP.items()})
    t["level"] = g["anomaly_from_iac"].mean()
    t = t.dropna()
    closure = (t["level"] - t[list(COMP)].sum(axis=1)).abs().max()
    lab = SECTOR_LABELS.get(sec, sec)
    print(f"=== {lab}   (closure: max |level - sum of terms| = {closure:.2e})")

    # A
    r, p = stats.pearsonr(t["level"], t["trend"])
    print(f"  A. r(level, Trend) = {r:+.2f}  (R^2 = {r*r:.2f})")

    # B
    own, cov, v = budget(t, list(COMP), "level")
    print(f"  B. var(level) = {v:.4f}; shares: " + ", ".join(f"{k} {own[k]:.0%}" for k in COMP) + f", covariances {cov:+.0%}")

    # C
    t["miss"] = t["level"] - t["trend"]
    own_m, cov_m, v_m = budget(t, ["amplitude", "phase", "raw"], "miss")
    print(f"  C. level - Trend: variance {v_m / v:.0%} of var(level); carried by "
          + ", ".join(f"{k} {own_m[k]:.0%}" for k in ("amplitude", "phase", "raw")) + f", covariances {cov_m:+.0%}")

    # D
    slow = t.rolling(WIN, center=True, min_periods=WIN // 2 + 1).mean()
    fast = t - slow
    shares = {}
    for name, part in (("slow", slow), ("fast", fast)):
        part = part.dropna()
        own_p, cov_p, v_p = budget(part, list(COMP), "level")
        shares[name] = (own_p, cov_p, v_p)
        print(f"  D. {name:4s} part of level ({WIN}-yr): var {v_p / v:.0%} of var(level); shares "
              + ", ".join(f"{k} {own_p[k]:.0%}" for k in COMP) + f", cov {cov_p:+.0%}")

    # E
    r_ph, p_ph = stats.pearsonr(t["level"], t["phase"])
    print(f"  E. Phase term annual mean: {t['phase'].mean():+.3f} +/- {t['phase'].std(ddof=1):.3f}; "
          f"r with level {r_ph:+.2f} (p={p_ph:.3f})\n")

    rows.append(dict(series=lab, start=START, end=END, closure=closure, r_level_trend=r, R2_trend=r * r,
                     var_level=v, **{f"share_{k}": own[k] for k in COMP}, share_cov=cov,
                     miss_var_share=v_m / v, **{f"miss_share_{k}": own_m[k] for k in ("amplitude", "phase", "raw")},
                     slow_var_share=shares["slow"][2] / v, slow_trend_share=shares["slow"][0]["trend"],
                     fast_var_share=shares["fast"][2] / v, fast_trend_share=shares["fast"][0]["trend"],
                     fast_amp_share=shares["fast"][0]["amplitude"], fast_phase_share=shares["fast"][0]["phase"],
                     fast_raw_share=shares["fast"][0]["raw"],
                     phase_mean=t["phase"].mean(), phase_sd=t["phase"].std(ddof=1), r_level_phase=r_ph))

out = os.path.join(TABLES_DIR, f"t_trend_is_level_{START}_{END}.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {out}")
print("Reads as 'Trend = low-pass level' if: A is high, D-slow is owned by trend, D-fast is owned by the other\n"
      "terms. C says where the year-to-year level goes instead; E says whether the Phase term carries level.")
