"""
diagnose_climatology_drift.py

More direct, higher-powered alternative to diagnose_acf_annual_bump.py.

That script tried to detect residual seasonal contamination by looking
for an echo in the ACF around lag ~365 days. Calibration against
realistic sample sizes showed this has real power problems: at a
threshold that keeps the false-positive rate low (~3%) on pure noise, it
also failed to detect even a SEVERE synthetic contamination signal
(15/15 misses). A single realization's autocorrelation at lag 300-400
days is just too noisy to reliably see a seasonal echo this way.

This script checks the same underlying question - did the seasonal
cycle deseasonalize_sia_and_wind.py removes actually stay constant
across the whole 1979-2024 record, or did its amplitude/timing drift -
directly and with much more power: split the record into sub-periods,
compute a separate day-of-year climatology for EACH sub-period, and
compare their shapes. If the climatologies disagree meaningfully, a
single full-record climatology (what's actually used) systematically
over/under-subtracts in different sub-periods, leaving residual
seasonal structure in what's supposed to be a pure anomaly - directly
explaining spuriously long apparent memory, without needing to detect
a faint echo in noisy long-lag autocorrelation.

Method: split into 3 sub-periods (adjust SPLIT_YEARS if you want a
different split), compute day-of-year mean SIA per sub-period per
sector, and report:
  - the max absolute difference between any two sub-periods' climatology
    curves, in the same units as SIA_anomaly (km^2)
  - that difference as a fraction of the FULL seasonal cycle's own
    range (max climatology - min climatology) - a large fraction means
    the climatology genuinely moved a meaningful amount relative to the
    seasonal cycle itself, not just a small wobble
"""

import numpy as np
import pandas as pd

IN_CSV = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/"
    "analysis_table_daily_anomaly.csv"
)
OUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_results/"

DATE_COL = "date"
SECTOR_COL = "sector"
SIA_COL = "SIA"   # RAW SIA (not the anomaly) - we're recomputing the
                   # climatology from scratch per sub-period, so we need
                   # the raw series the same way deseasonalize_sia_and_wind.py
                   # started from, not its already-full-record-adjusted output

SPLIT_YEARS = [1979, 1998, 2016, 2025]  # 3 sub-periods: 1979-1997, 1998-2015, 2016-2024
SPLIT_LABELS = ["1979-1997", "1998-2015", "2016-2024"]


def compute_climatology(df_subset):
    """Day-of-year mean SIA for one sub-period."""
    df_subset = df_subset.copy()
    df_subset["doy"] = df_subset[DATE_COL].dt.dayofyear
    return df_subset.groupby("doy")[SIA_COL].mean()


def run_drift_diagnostic(df):
    results = []
    climatology_curves = []

    for sector in df[SECTOR_COL].unique():
        sub = df[df[SECTOR_COL] == sector].dropna(subset=[SIA_COL]).copy()

        climatologies = {}
        for i in range(len(SPLIT_YEARS) - 1):
            y_start, y_end = SPLIT_YEARS[i], SPLIT_YEARS[i + 1]
            label = SPLIT_LABELS[i]
            period_df = sub[(sub[DATE_COL].dt.year >= y_start) & (sub[DATE_COL].dt.year < y_end)]
            if len(period_df) < 300:
                print(f"  [{sector}, {label}] SKIPPED - not enough data")
                continue
            clim = compute_climatology(period_df)
            climatologies[label] = clim
            for doy, val in clim.items():
                climatology_curves.append({"sector": sector, "sub_period": label,
                                            "doy": doy, "climatology_SIA": val})

        if len(climatologies) < 2:
            continue

        # align on shared day-of-year values, compare all pairs
        common_doy = set.intersection(*[set(c.index) for c in climatologies.values()])
        common_doy = sorted(common_doy)

        aligned = pd.DataFrame({label: clim.reindex(common_doy) for label, clim in climatologies.items()})
        max_diff = (aligned.max(axis=1) - aligned.min(axis=1)).max()

        full_range = pd.concat(climatologies.values()).max() - pd.concat(climatologies.values()).min()
        frac_of_seasonal_range = max_diff / full_range if full_range > 0 else np.nan

        print(f"  [{sector}] max climatology difference between sub-periods = "
              f"{max_diff:,.0f} km^2 ({100*frac_of_seasonal_range:.1f}% of the full "
              f"seasonal cycle's own range)")

        results.append({
            "sector": sector,
            "max_climatology_diff_km2": max_diff,
            "seasonal_cycle_range_km2": full_range,
            "frac_of_seasonal_range": frac_of_seasonal_range,
        })

    return pd.DataFrame(results), pd.DataFrame(climatology_curves)


if __name__ == "__main__":
    df = pd.read_csv(IN_CSV, parse_dates=[DATE_COL])

    print(f"Comparing day-of-year climatology across sub-periods: {SPLIT_LABELS}\n")
    results_df, curves_df = run_drift_diagnostic(df)

    results_df.to_csv(OUT_DIR + "climatology_drift_diagnostic.csv", index=False)
    curves_df.to_csv(OUT_DIR + "climatology_drift_curves.csv", index=False)

    print("\n--- Interpretation guide ---")
    print("frac_of_seasonal_range > ~0.15-0.20: the climatology moved enough that a")
    print("single full-record climatology likely leaves real residual seasonal")
    print("structure in SIA_anomaly - worth considering a period-specific or")
    print("smoothly time-varying climatology instead of one fixed curve.")
    print("frac_of_seasonal_range small (a few percent): the seasonal cycle's shape")
    print("has been fairly stable, and the long e-folding times from the ACF test")
    print("are more likely to reflect genuine physical memory.")
    print(f"\nSaved summary to: {OUT_DIR}climatology_drift_diagnostic.csv")
    print(f"Saved full climatology curves (for plotting all 3 sub-periods overlaid) "
          f"to: {OUT_DIR}climatology_drift_curves.csv")