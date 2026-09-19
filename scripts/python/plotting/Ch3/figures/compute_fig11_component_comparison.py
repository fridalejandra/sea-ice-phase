#!/usr/bin/env python3
"""
compute_component_comparison.py

For each of the seven pre-registered sector x atmospheric-index pairs
(ch3_config.PRIMARY_PAIRS), computes the correlation between that index
(at the pair's specified season) and FOUR different ways of summarizing
that sector's sea ice that season:

  1. raw_sie    -- seasonal SIE extent (sie_<season> from annual_params.csv),
                   linearly detrended so it's on the same footing as the
                   already-detrended atmospheric index and the *_anom scalars.
                   NEW -- not in any existing table, computed here.
  2. amplitude  -- amplitude_raw_anom. Pulled directly from
                   t35_index_scan_raw.csv (target=='amplitude_raw_anom'),
                   your own canonical 420-cell scan -- NOT recomputed here,
                   so it matches Fig 10 / t35_primary_pairs.csv exactly.
  3. phase      -- max_doy_raw_anom. Same source, target=='max_doy_raw_anom'.
                   (An earlier version of this script recomputed amp/phase
                   from a fresh merge of annual_params.csv + master_index_
                   detrended.csv -- that gave numbers close to but not
                   identical to the canonical ones, e.g. Ross came out 0.378
                   here vs 0.347 in t35_primary_pairs.csv. Reading them
                   straight from your own scan avoids that discrepancy
                   entirely.)
  4. residual   -- within-season standard deviation of residual_apac
                   (the leftover after trend+amplitude+phase are removed),
                   one scalar per sector-year. NEW quantity, not previously
                   in any of your tables -- computed fresh here from
                   daily_fitted.csv.

This directly answers "how much better does the index correlate with phase
than with raw SIE or the residual" -- side by side, same sector/index/season.
Only raw_sie and residual are new computation; amplitude and phase are your
own already-verified numbers, just pulled into the same table for comparison.

ASSUMPTIONS TO VERIFY (flagging rather than silently guessing):
  - ADV = Mar-Aug, RET = Oct-Jan. ch3_config doesn't expose these explicitly;
    this matches what we used earlier for the season-mismatch reruns. If your
    actual pipeline defines them differently, only rows using ADV/RET are
    affected -- annual/DJF/MAM/JJA/SON rows are unaffected.
  - For RET (wraps the year boundary: Oct/Nov/Dec of year Y + Jan of Y+1),
    the residual scalar for "Year Y" pools Oct-Dec of Y with Jan of Y+1.
    Same convention as DJF (Dec of Y pools with Jan/Feb of Y+1 under "Year Y+1").
    If annual_params.csv's own Year-labeling convention for sie_DJF differs,
    the raw_sie vs residual columns for that one row could be off by a year
    from each other -- check the printed n and a spot pearsonr by hand if the
    residual number for East Antarctica/SAM_RET looks surprising.
  - Residual volatility = SD of residual_apac within the season window. This
    is a new metric (not the same as the 3.4c pre/post-2016 ratio), designed
    to be comparable in *kind* to amplitude/phase for this specific heatmap.

Run from the same directory as ch3_config.py, after the full pipeline has
run. Writes results/ch3/tables/t37_component_comparison.csv and prints a
summary table -- paste both back.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import (
    ANNUAL_CSV, DAILY_CSV, INDEX_CSV, TABLES_DIR,
    PRIMARY_PAIRS, SECTORS_COMPUTE,
)

SCAN_PATH = os.path.join(TABLES_DIR, "t35_index_scan_raw.csv")

SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "ADV": [3, 4, 5, 6, 7, 8],
    "RET": [10, 11, 12, 1],
    "annual": list(range(1, 13)),
}


def detrend(years, values):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    out = values.copy()
    mask = ~np.isnan(values)
    if mask.sum() < 3:
        return out
    m, b = np.polyfit(years[mask], values[mask], 1)
    out[mask] = values[mask] - (m * years[mask] + b)
    return out


def season_year(month, calendar_year, season):
    """Bucket a daily row's calendar (month, year) into the season-year
    label used for grouping, for the two seasons that cross the year
    boundary. See the RET/DJF note in the module docstring."""
    if season == "RET" and month == 1:
        return calendar_year - 1
    if season == "DJF" and month == 12:
        return calendar_year + 1
    return calendar_year


def safe_pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 6:
        return np.nan, np.nan, int(mask.sum())
    r, p = stats.pearsonr(x[mask], y[mask])
    return r, p, int(mask.sum())


print("Loading annual_params.csv, daily_fitted.csv, master_index_detrended.csv, t35_index_scan_raw.csv ...")
scan = pd.read_csv(SCAN_PATH)


def scan_lookup(sector_label, index_col, target):
    hit = scan[(scan["sector"] == sector_label) & (scan["index"] == index_col) & (scan["target"] == target)]
    if hit.empty:
        return np.nan, np.nan, 0
    row = hit.iloc[0]
    return float(row["r"]), float(row["p"]), int(row["n"])


ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"] == "FULL"]
daily = pd.read_csv(DAILY_CSV)
if "period" in daily.columns:
    daily = daily[daily["period"] == "FULL"]
idx = pd.read_csv(INDEX_CSV)

daily["Date"] = pd.to_datetime(daily["Date"])
daily["month"] = daily["Date"].dt.month

rows = []
for sie_sector, target_var, index_col, basis in PRIMARY_PAIRS:
    sector_label = SECTORS_COMPUTE.get(sie_sector, sie_sector)

    m = re.match(r"^(.*)_(annual|DJF|MAM|JJA|SON|ADV|RET)$", index_col)
    if not m:
        print(f"  SKIP {index_col}: couldn't parse a season suffix off the index name")
        continue
    index_base, season = m.groups()

    a = ann[ann["sector"] == sector_label].copy()
    if a.empty:
        a = ann[ann["sector"] == sie_sector].copy()
    if a.empty:
        print(f"  SKIP {sector_label}: no rows in annual_params.csv under that sector label")
        continue
    a = a.sort_values("Year")

    merged = a.merge(idx[["Year", index_col]], on="Year", how="inner")
    index_vals = merged[index_col]

    # 1. amplitude -- canonical, from t35_index_scan_raw.csv (not recomputed)
    r_amp, p_amp, n_amp = scan_lookup(sector_label, index_col, "amplitude_raw_anom")

    # 2. phase -- canonical, from t35_index_scan_raw.csv. The scan only has
    # max_doy_raw_anom (no min_doy pair among the seven), so that's what
    # "phase" means here regardless of the pair's own official target.
    phase_col = "max_doy_raw_anom"
    r_phase, p_phase, n_phase = scan_lookup(sector_label, index_col, phase_col)

    # 3. raw seasonal SIE, detrended -- NEW, computed here
    sie_col = f"sie_{season}"
    if sie_col in merged.columns and merged[sie_col].notna().sum() > 5:
        sie_dt = detrend(merged["Year"].values, merged[sie_col].values)
        r_sie, p_sie, n_sie = safe_pearsonr(index_vals, sie_dt)
        sie_note = ""
    else:
        r_sie, p_sie, n_sie = np.nan, np.nan, 0
        sie_note = f"no {sie_col} column in annual_params.csv"

    # 4. residual volatility: SD of residual_apac within season, per sector-year
    dsec = daily[daily["sector"] == sector_label]
    if dsec.empty:
        dsec = daily[daily["sector"] == sie_sector]
    months = SEASON_MONTHS[season]
    dseason = dsec[dsec["month"].isin(months)].copy()
    dseason["season_year"] = dseason.apply(
        lambda row: season_year(row["month"], row["Year"], season), axis=1
    )
    resid_by_year = (
        dseason.groupby("season_year")["residual_apac"].std().rename("resid_sd")
    )
    merged2 = merged.merge(resid_by_year, left_on="Year", right_index=True, how="inner")
    if len(merged2) > 5:
        r_resid, p_resid, n_resid = safe_pearsonr(merged2[index_col], merged2["resid_sd"])
    else:
        r_resid, p_resid, n_resid = np.nan, np.nan, 0

    def fmt(r):
        return "  NaN" if pd.isna(r) else f"{r:+.3f}"

    print(
        f"{sector_label:16s} {index_col:14s} (n={len(merged):2d})  "
        f"raw_sie={fmt(r_sie)}  amp={fmt(r_amp)}  phase={fmt(r_phase)}  resid={fmt(r_resid)}"
        + (f"   [{sie_note}]" if sie_note else "")
    )

    rows.append(dict(
        sector=sector_label, index=index_col, index_base=index_base, season=season,
        basis=basis, n=len(merged),
        r_raw_sie=r_sie, p_raw_sie=p_sie, n_raw_sie=n_sie, note_raw_sie=sie_note,
        r_amplitude=r_amp, p_amplitude=p_amp, n_amplitude=n_amp,
        r_phase=r_phase, p_phase=p_phase, n_phase=n_phase, phase_var=phase_col,
        r_residual=r_resid, p_residual=p_resid, n_residual=n_resid,
    ))

out = pd.DataFrame(rows)
out_path = os.path.join(TABLES_DIR, "t37_component_comparison.csv")
out.to_csv(out_path, index=False)
print(f"\nwrote {out_path}  ({len(out)} rows)")
print("\n" + out.to_string(index=False))