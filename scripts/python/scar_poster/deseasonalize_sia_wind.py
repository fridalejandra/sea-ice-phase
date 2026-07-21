"""
deseasonalize_sia_and_wind_v2_periodclim.py

Fixes a real problem surfaced by diagnose_climatology_drift.py: the
original deseasonalize_sia_and_wind.py used ONE day-of-year climatology
computed across the full 1979-2024 record. That climatology drifted
meaningfully across the record - 12-19% of the seasonal cycle's own
range, consistent across all 5 sectors (not noise-looking; a systematic
signal present everywhere). The likely cause: the record spans a real
regime shift (2016), and a single full-record climatology represents
neither period well, systematically over/under-subtracting in each -
leaving residual seasonal structure in what's supposed to be a pure
stochastic anomaly. This is a strong candidate explanation for the
implausibly long e-folding memory timescales found in
persistence_efold_test.py (multiple sectors never decorrelating within
60+ days).

Fix: compute SEPARATE day-of-year climatologies for the pre-2016
(1979-2015) and post-2016 (2016-2024) periods, per sector, for both SIA
and wind_stress, and apply each row's own period's climatology - the
same two-regime framework already used everywhere else in this pipeline
(interaction-term regression, trend analysis), now applied to the
deseasonalization step itself.

TRANSITION-DAY HANDLING: delta_SIA_anomaly (the day-over-day change) is
set to NaN on the single day each sector crosses from pre- to
post-climatology (Jan 1, 2016), rather than computed as a naive diff.
That diff would otherwise combine a real physical daily change with the
DISCONTINUITY between the two climatologies at that calendar day - up to
~735,000 km^2 for King Haakon VII per the drift diagnostic - which would
inject one severe, non-physical outlier per sector into every downstream
test. Dropping 5 rows total (one per sector) is a small, principled cost
compared to letting that artifact sit in the data.

REMAINING CAVEAT: this addresses drift AT THE 2016 BOUNDARY specifically
- it does not rule out additional slower drift WITHIN the pre-2016
period itself (the original drift diagnostic's 3-way split included a
1979-1997 vs 1998-2015 comparison that could reflect either real
within-period drift or just noise). If you want to check that
specifically, re-run diagnose_climatology_drift.py comparing only the
two pre-2016 sub-periods.

Output columns match the original deseasonalize_sia_and_wind.py exactly
(same names), so downstream scripts just need IN_CSV repointed at this
script's OUT_CSV - no other changes needed in
wind_sensitivity_interaction_test.py, trend_analysis_sector_month_season.py,
or persistence_efold_test.py.
"""

import numpy as np
import pandas as pd

IN_CSV = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/"
    "analysis_table_daily.csv"
)
OUT_CSV = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/"
    "analysis_table_daily_anomaly_periodclim.csv"
)

DATE_COL = "date"
SECTOR_COL = "sector"
REGIME_SHIFT_YEAR = 2016

df = pd.read_csv(IN_CSV, parse_dates=[DATE_COL])
df["doy"] = df[DATE_COL].dt.dayofyear
df["period"] = np.where(df[DATE_COL].dt.year >= REGIME_SHIFT_YEAR, "post", "pre")

# ---------------- SIA: period-specific climatology + anomaly ----------------
print("Computing PERIOD-SPECIFIC day-of-year climatology per sector: SIA...")
sia_clim = df.groupby([SECTOR_COL, "period", "doy"])["SIA"].mean().reset_index()
sia_clim = sia_clim.rename(columns={"SIA": "SIA_climatology"})
df = df.merge(sia_clim, on=[SECTOR_COL, "period", "doy"], how="left")
df["SIA_anomaly"] = df["SIA"] - df["SIA_climatology"]

df = df.sort_values([SECTOR_COL, DATE_COL]).reset_index(drop=True)
df["delta_SIA_anomaly"] = df.groupby(SECTOR_COL)["SIA_anomaly"].diff()

# --- blank out the transition-day diff (crosses from pre- to post-climatology) ---
transition_date = pd.Timestamp(f"{REGIME_SHIFT_YEAR}-01-01")
is_transition_day = df[DATE_COL] == transition_date
n_blanked = is_transition_day.sum()
df.loc[is_transition_day, "delta_SIA_anomaly"] = np.nan
print(f"Blanked delta_SIA_anomaly on {n_blanked} transition-day rows "
      f"(one per sector, {transition_date.date()}) - see docstring for why.")

# ---------------- wind_stress: period-specific climatology + anomaly ----------------
print("Computing PERIOD-SPECIFIC day-of-year climatology per sector: wind_stress...")
wind_clim = df.groupby([SECTOR_COL, "period", "doy"])["wind_stress"].mean().reset_index()
wind_clim = wind_clim.rename(columns={"wind_stress": "wind_stress_climatology"})
df = df.merge(wind_clim, on=[SECTOR_COL, "period", "doy"], how="left")
df["wind_stress_anomaly"] = df["wind_stress"] - df["wind_stress_climatology"]
# NOTE: wind_stress itself (raw) has no transition-day artifact issue since
# it's not differenced anywhere in this pipeline - only SIA_anomaly is.

# ---------------- Sanity checks ----------------
print("\nSIA anomaly mean by sector x period (should be ~0 within each):")
print(df.groupby([SECTOR_COL, "period"])["SIA_anomaly"].mean())

print("\nWind stress anomaly mean by sector x period (should be ~0 within each):")
print(df.groupby([SECTOR_COL, "period"])["wind_stress_anomaly"].mean())

print(f"\ndelta_SIA_anomaly non-null count: {df['delta_SIA_anomaly'].notna().sum()} "
      f"(should be total rows minus ~5*number_of_sectors: 1 transition day + "
      f"1 first-day-per-sector-per-period diff() edge, x5 sectors)")

# ---------------- Save (drop helper columns to match original schema) ----------------
df = df.drop(columns=["period"])
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved to: {OUT_CSV}")
print("New/changed columns: SIA_climatology, SIA_anomaly, delta_SIA_anomaly, "
      "wind_stress_climatology, wind_stress_anomaly (all now period-specific)")