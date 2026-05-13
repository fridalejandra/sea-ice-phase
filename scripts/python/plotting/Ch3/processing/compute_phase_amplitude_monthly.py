"""
Derives monthly phase and amplitude proxies from the daily APAC fitted curves.

The annual APAC scalars (in annual_params.csv) capture the full-year phase
and amplitude as single numbers. This script asks a finer question: how does
the fitted SIE cycle behave within each calendar month, and how does that
vary year to year?

Three monthly variables are computed for each sector/year/month:

    monthly_amplitude
        Max minus min of the fitted_apac curve within the month.
        Captures how much the ice is changing within that month.
        Near zero in flat parts of the cycle, large during active advance/retreat.

    monthly_mean_sie
        Mean of the raw Extent within the month.
        The traditional monthly SIE used in most studies — kept here for
        comparison against the APAC-derived metrics.

    monthly_apac_anomaly
        Mean of (fitted_apac - fitted_invariant) within the month.
        The APAC model's estimate of how anomalous that month is relative
        to the climatological cycle — separates the year-specific signal
        from the climatological mean.

    monthly_phase_offset
        Only computed for months near the sector-specific peak season
        (within ±2 months of the climatological max DOY). Defined as the
        DOY of the fitted_apac maximum within the month minus the DOY of
        the fitted_invariant maximum within the month. Positive = later
        than climatology, negative = earlier.
        Set to NaN outside the peak window — phase is not meaningful
        when the curve is flat or still ascending/descending far from peak.

    monthly_amplitude_anomaly
        monthly_amplitude minus the climatological monthly amplitude
        (computed as the 1979-2015 median for that sector/month).
        This is the anomaly form used for correlations.

A quality flag (near_peak) marks months within ±2 months of the
climatological peak for each sector — phase_offset is only reliable there.

Output: monthly_params.csv
    One row per sector/year/month.
    Used by compute_monthly_lagged_correlations.py and
    figS_phase_amp_monthly_corr_matrix.py.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR    = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
DAILY_CSV   = os.path.join(DATA_DIR, "daily_fitted.csv")
ANNUAL_CSV  = os.path.join(DATA_DIR, "annual_params.csv")
OUTPUT_CSV  = os.path.join(DATA_DIR, "monthly_params.csv")

YEAR_MIN = 1979
YEAR_MAX = 2023

# Months within this window of the climatological peak DOY are flagged as
# near_peak — phase_offset is only meaningful here.
PEAK_WINDOW_MONTHS = 2

SECTORS = [
    "SIE_Weddell",
    "SIE_Amundsen_Bellingshausen",
    "SIE_Ross",
    "SIE_East_Antarctica",
    "SIE_King_Haakon",
]


# --- Load -----------------------------------------------------------------

print("Loading daily fitted data...")
daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
daily = daily[daily["Year"].between(YEAR_MIN, YEAR_MAX)]
daily["Month"] = daily["Date"].dt.month
print(f"  {len(daily)} rows | {daily['Year'].min()}–{daily['Year'].max()}")

print("Loading annual params for climatological peak DOY...")
annual = pd.read_csv(ANNUAL_CSV)


# --- Climatological peak DOY per sector -----------------------------------
# Used to define the near_peak window and for phase_offset computation.
# We use the median of fitted_apac peak DOY over 1979-2015 as the reference.

clim_peak = {}
for sec in SECTORS:
    sec_daily = daily[(daily["sector"] == sec) &
                      (daily["Year"].between(1979, 2015))]
    # Peak of the invariant curve — same for all years by construction
    inv_by_doy = sec_daily.groupby("DOY")["fitted_invariant"].mean()
    clim_peak[sec] = int(inv_by_doy.idxmax())
    print(f"  Climatological peak DOY — {sec}: {clim_peak[sec]}")


def doy_to_month(doy, year):
    """Convert DOY to month number, handling leap years."""
    try:
        return pd.Timestamp(year=int(year), month=1, day=1) + \
               pd.Timedelta(days=int(doy) - 1)
    except Exception:
        return pd.NaT


def peak_month(doy):
    """Approximate month of a given DOY (ignoring leap year edge cases)."""
    return pd.Timestamp(year=2001, month=1, day=1) + \
           pd.Timedelta(days=int(doy) - 1)


# --- Compute monthly metrics ----------------------------------------------

print("\nComputing monthly metrics...")
records = []

for sec in SECTORS:
    print(f"  {sec}...")
    sec_data = daily[daily["sector"] == sec].copy()
    peak_doy = clim_peak[sec]

    # Approximate which months are within the peak window
    # Convert peak_doy to a reference month, then flag ±PEAK_WINDOW_MONTHS
    peak_month_num = peak_month(peak_doy).month
    peak_months = set()
    for offset in range(-PEAK_WINDOW_MONTHS, PEAK_WINDOW_MONTHS + 1):
        m = ((peak_month_num - 1 + offset) % 12) + 1
        peak_months.add(m)

    for year in range(YEAR_MIN, YEAR_MAX + 1):
        yr_data = sec_data[sec_data["Year"] == year]
        if len(yr_data) == 0:
            continue

        for month in range(1, 13):
            mo_data = yr_data[yr_data["Month"] == month]
            if len(mo_data) < 5:
                # Too few observations — skip (e.g. missing data)
                continue

            apac    = mo_data["fitted_apac"].values
            invar   = mo_data["fitted_invariant"].values
            extent  = mo_data["Extent"].values
            doys    = mo_data["DOY"].values

            # Monthly amplitude: range of fitted curve within the month
            monthly_amp = float(np.nanmax(apac) - np.nanmin(apac))

            # Monthly mean SIE (raw)
            monthly_mean = float(np.nanmean(extent))

            # Monthly APAC anomaly: mean departure from climatological curve
            monthly_anom = float(np.nanmean(apac - invar))

            # Phase offset: only near the peak season
            near_peak = month in peak_months
            if near_peak:
                # DOY of fitted_apac max vs fitted_invariant max within month
                apac_peak_doy  = float(doys[np.argmax(apac)])
                invar_peak_doy = float(doys[np.argmax(invar)])
                phase_offset   = apac_peak_doy - invar_peak_doy
            else:
                phase_offset = np.nan

            records.append({
                "sector"        : sec,
                "Year"          : year,
                "Month"         : month,
                "monthly_amp"   : monthly_amp,
                "monthly_mean"  : monthly_mean,
                "monthly_anom"  : monthly_anom,
                "phase_offset"  : phase_offset,
                "near_peak"     : near_peak,
            })

monthly = pd.DataFrame(records)
print(f"\n  {len(monthly)} monthly records computed")


# --- Compute anomalies relative to 1979-2015 baseline ---------------------
# Subtract the climatological monthly median so we have anomaly form
# for the correlation analysis.

print("Computing anomalies from 1979–2015 baseline...")

baseline = (monthly[monthly["Year"].between(1979, 2015)]
            .groupby(["sector", "Month"])[["monthly_amp", "monthly_mean"]]
            .median()
            .reset_index()
            .rename(columns={
                "monthly_amp" : "clim_amp",
                "monthly_mean": "clim_mean",
            }))

monthly = monthly.merge(baseline, on=["sector", "Month"], how="left")

monthly["monthly_amp_anom"]  = monthly["monthly_amp"]  - monthly["clim_amp"]
monthly["monthly_mean_anom"] = monthly["monthly_mean"] - monthly["clim_mean"]


# --- Dipole diagnostic ----------------------------------------------------
# Flag months where Ross and Weddell amplitude anomalies have opposite signs
# (Ross-Weddell dipole). This is the counterintuitive amplitude gain pattern.

print("Computing Ross-Weddell dipole flag...")

ross    = monthly[monthly["sector"] == "SIE_Ross"][
            ["Year","Month","monthly_amp_anom"]].rename(
            columns={"monthly_amp_anom": "ross_amp_anom"})
weddell = monthly[monthly["sector"] == "SIE_Weddell"][
            ["Year","Month","monthly_amp_anom"]].rename(
            columns={"monthly_amp_anom": "weddell_amp_anom"})

dipole = ross.merge(weddell, on=["Year","Month"], how="inner")
dipole["dipole_active"] = (
    (dipole["ross_amp_anom"] * dipole["weddell_amp_anom"]) < 0
)

# Merge dipole flag back — only meaningful for Ross and Weddell rows
monthly = monthly.merge(
    dipole[["Year","Month","dipole_active"]],
    on=["Year","Month"], how="left"
)

n_dipole = dipole["dipole_active"].sum()
print(f"  Dipole active in {n_dipole} of {len(dipole)} Ross-Weddell month pairs "
      f"({100*n_dipole/len(dipole):.1f}%)")


# --- Save -----------------------------------------------------------------

monthly.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
print(f"  {len(monthly)} rows | columns: {list(monthly.columns)}")

# Quick summary
print("\nMonthly amplitude anomaly stats by sector:")
for sec in SECTORS:
    sub = monthly[monthly["sector"] == sec]["monthly_amp_anom"].dropna()
    print(f"  {sec:<32} mean={sub.mean():+.4f}  std={sub.std():.4f}  "
          f"n={len(sub)}")