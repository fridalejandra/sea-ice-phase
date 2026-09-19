"""
ch3_data.py — load, validate, and derive. Figures never touch CSVs directly.
============================================================================
Every loader validates its input and fails loudly rather than producing a
plausible-looking wrong figure. In particular, load_daily() refuses Pipeline A
output, because in Pipeline A the decomposition components did not sum to the
anomaly (trend double-counted; phase_component built from the wrong curve).

PERIOD COLUMN (added when 01_fit_apac.R moved to a dual-period fit):
daily_fitted_E.csv and annual_params_E.csv now contain one full copy of the
output per entry in R's PERIODS list ("HR2018" = through 2018-12-31, "FULL" =
through 2023-12-31), so years 1979-2018 appear under BOTH periods as
independent refits. load_daily()/load_annual() default to period="FULL" —
every normal figure/table should use that default. Only the H&R-comparison
validation table should pass period="HR2018" explicitly. Pass period=None to
get every period's rows back (e.g. to build that comparison table yourself).

rmse_summary_E.csv is NOT filtered by load_rmse() — the whole point of that
file is to compare both periods side by side, so it always returns every
period's rows; filter it yourself where you use it.

NOTE: any script that reads DAILY_CSV / ANNUAL_CSV with pd.read_csv() directly
instead of through these loaders will silently get both periods' rows mixed
together. compute_atmospheric_correlations.py currently does this (reads
ANNUAL_CSV directly) and needs its own period filter — grep the repo for
other direct pd.read_csv(ANNUAL_CSV) / pd.read_csv(DAILY_CSV) call sites.
"""

import os
import numpy as np
import pandas as pd

from ch3_config import (
    DAILY_CSV, ANNUAL_CSV, RMSE_CSV,
    SECTORS, SECTOR_LABELS, COMPONENT_COLS,
    YEAR_START, YEAR_END, BREAK_YEAR,
    DECADE_BINS, DECADE_LABELS,
)

_SUM_TOL = 1e-6
DEFAULT_PERIOD = "FULL"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_daily(validate=True, period=DEFAULT_PERIOD):
    """Daily fitted series with decomposition components (Pipeline E).

    period: "FULL" (default), "HR2018", or None for every period's rows
    unfiltered (both periods stacked; you'll want to filter yourself).
    """
    if not os.path.exists(DAILY_CSV):
        raise SystemExit(
            f"Not found: {DAILY_CSV}\n"
            "Run R/ch3/01_fit_apac.R, or update DAILY_CSV in ch3_config.py."
        )
    d = pd.read_csv(DAILY_CSV, parse_dates=["Date"])

    if validate:
        need = ["anomaly_from_iac", "iac_notrend", "est_anomaly",
                *COMPONENT_COLS.values()]
        miss = [c for c in need if c not in d.columns]
        if miss:
            raise SystemExit(
                f"Missing columns {miss}.\n"
                "Pre-E output (anomaly_from_iac missing). Run R/ch3/01_fit_apac.R."
            )
        err = decomposition_error(d)
        if err > _SUM_TOL:
            raise SystemExit(
                f"Decomposition does not sum (mean |error| = {err:.3e}).\n"
                "Regenerate with R/ch3/01_fit_apac.R (E: anomaly_from_iac = trend+amp+phase+residual)."
            )

    if period is not None and "period" in d.columns:
        d = d[d["period"] == period].reset_index(drop=True)
        if d.empty:
            raise SystemExit(
                f"{os.path.basename(DAILY_CSV)} has a `period` column but no "
                f"rows with period == {period!r}. Available: "
                f"{sorted(pd.read_csv(DAILY_CSV, usecols=['period'])['period'].unique())}"
            )
    return d


def load_annual(validate=True, period=DEFAULT_PERIOD):
    """Annual scalar parameters (Pipeline E, 1979-2023).

    period: "FULL" (default), "HR2018", or None for every period's rows
    unfiltered (both periods stacked; you'll want to filter yourself).
    """
    if not os.path.exists(ANNUAL_CSV):
        raise SystemExit(f"Not found: {ANNUAL_CSV} — run R/ch3/01_fit_apac.R.")
    a = pd.read_csv(ANNUAL_CSV)

    if validate:
        need = ["min_doy_raw_anom", "max_doy_raw_anom", "amplitude_raw_anom"]
        miss = [c for c in need if c not in a.columns]
        if miss:
            raise SystemExit(f"Missing columns {miss} — run Pipeline B.")
        # Wrap fix present? Pipeline A had min-DOY anomalies of ~+305 days.
        if a["Year"].min() < 1979:
            raise SystemExit(
                "annual_params contains a pre-1979 row (partial 1978 cycle). "
                "This is not the ch3-pipeline-v1 output; rerun R/ch3/01_fit_apac.R.")
        if a["min_doy_raw_anom"].abs().max() > 150:
            raise SystemExit(
                "min_doy_raw_anom exceeds 150 days — the DOY wrap fix is not "
                "applied. This is Pipeline A output."
            )

    if period is not None and "period" in a.columns:
        a = a[a["period"] == period].reset_index(drop=True)
        if a.empty:
            raise SystemExit(
                f"{os.path.basename(ANNUAL_CSV)} has a `period` column but no "
                f"rows with period == {period!r}. Available: "
                f"{sorted(pd.read_csv(ANNUAL_CSV, usecols=['period'])['period'].unique())}"
            )
    return a


def load_rmse():
    """RMSE/pct-improvement summary, all periods (deliberately NOT filtered —
    this is the one table meant to compare periods side by side)."""
    if not os.path.exists(RMSE_CSV):
        raise SystemExit(f"Not found: {RMSE_CSV} — run R/ch3/01_fit_apac.R.")
    return pd.read_csv(RMSE_CSV)


def load_correlations(path, required_cols=None):
    """Generic loader for the correlation-pipeline CSVs, with a warning.

    These are built from annual_params. If they predate Pipeline B they encode
    the OLD anomalies (no wrap fix, no cycle-centred differencing) and must be
    regenerated before use.
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"Not found: {path}\n"
            "Run processing/compute_atmospheric_correlations.py first."
        )
    df = pd.read_csv(path)
    if required_cols:
        miss = [c for c in required_cols if c not in df.columns]
        if miss:
            raise SystemExit(f"{os.path.basename(path)} missing columns {miss}")

    if os.path.getmtime(path) < os.path.getmtime(ANNUAL_CSV):
        print(f"  WARNING: {os.path.basename(path)} is older than "
              f"{os.path.basename(ANNUAL_CSV)} — it may encode the old anomalies.")
    return df


# ── Checks ────────────────────────────────────────────────────────────────────

def decomposition_error(daily):
    """Mean |anomaly - sum(components)|. Should be ~1e-19 under Pipeline B."""
    parts = sum(daily[c] for c in COMPONENT_COLS.values())
    return float(np.nanmean(np.abs(daily["anomaly_from_iac"] - parts)))


# ── Derived quantities ────────────────────────────────────────────────────────

def component_dominance(daily):
    """Per sector-year: variance share of each component and which dominates.

    Feeds the dot-timeline figure. Shares are of total component variance
    within that year, so they sum to 1 by construction.
    """
    rows = []
    for (sec, yr), g in daily.groupby(["sector", "Year"]):
        var = {k: float(np.nanvar(g[col])) for k, col in COMPONENT_COLS.items()}
        tot = sum(var.values())
        if not np.isfinite(tot) or tot <= 0:
            continue
        frac = {k: v / tot for k, v in var.items()}
        dom = max(frac, key=frac.get)
        rows.append(dict(sector=sec, Year=int(yr),
                         dominant=dom, share=frac[dom],
                         **{f"f_{k}": frac[k] for k in var}))
    return pd.DataFrame(rows)


def rolling_stat(annual, col, window, func="std", min_frac=0.8):
    """Centred rolling statistic of an annual metric, per sector.

    Returns long-format: sector, Year, value. `func` is 'std' or 'mean'.
    """
    out = []
    minp = int(np.ceil(window * min_frac))
    for sec, g in annual.groupby("sector"):
        g = g.sort_values("Year")
        r = g[col].rolling(window, center=True, min_periods=minp)
        val = r.std() if func == "std" else r.mean()
        out.append(pd.DataFrame(dict(sector=sec, Year=g["Year"].values,
                                     value=val.values)))
    return pd.concat(out, ignore_index=True)


def rolling_corr(annual, col_a, col_b, window, min_frac=0.8,
                 method="spearman", center=False):
    """Rolling correlation between two annual metrics, per sector.

    Default is a TRAILING window (value plotted at the window's last year) and
    Spearman rho — the statistic quoted in the chapter text and written by
    ch3_stats.py (t33_phase_amp_rolling10.csv). Pass method="pearson" or
    center=True only for a figure that says so in its caption.

    NOTE: with 45 years and a 10-15 year window these wander substantially by
    sampling variability alone; consecutive windows share 9 of 10 years. The
    inference lives in the whole-era split tests, not in the curve.
    """
    out = []
    minp = int(np.ceil(window * min_frac))
    for sec, g in annual.groupby("sector"):
        g = g.sort_values("Year")
        if method == "spearman":
            ra, rb = g[col_a].rank(), g[col_b].rank()
        else:
            ra, rb = g[col_a], g[col_b]
        r = ra.rolling(window, center=center, min_periods=minp).corr(rb)
        out.append(pd.DataFrame(dict(sector=sec, Year=g["Year"].values,
                                     value=r.values)))
    return pd.concat(out, ignore_index=True)


def era_split(df, break_year=BREAK_YEAR):
    """Add a pre/post regime-shift label."""
    df = df.copy()
    df["era"] = np.where(df["Year"] < break_year,
                         f"pre-{break_year}", f"{break_year}+")
    return df


def add_decade(df):
    df = df.copy()
    df["decade"] = pd.cut(df["Year"], bins=DECADE_BINS, labels=DECADE_LABELS)
    return df


def zscore(series):
    s = pd.Series(series).astype(float)
    return (s - s.mean()) / s.std(ddof=1)


def sector_order(df, col="sector"):
    """Return df ordered by the canonical sector order."""
    return df.assign(
        _o=df[col].map({s: i for i, s in enumerate(SECTORS)})
    ).sort_values(["_o", "Year"] if "Year" in df.columns else "_o").drop(columns="_o")


# ── Convenience ───────────────────────────────────────────────────────────────

def summary(daily=None, annual=None):
    """Print a short provenance report. Call at the top of any figure script."""
    if daily is not None:
        print(f"  daily : {len(daily):,} rows, "
              f"{daily['Year'].min()}–{daily['Year'].max()}, "
              f"{daily['sector'].nunique()} sectors")
        print(f"          decomposition sum error = {decomposition_error(daily):.2e}")
    if annual is not None:
        print(f"  annual: {len(annual)} rows, "
              f"{annual['Year'].min()}–{annual['Year'].max()}")