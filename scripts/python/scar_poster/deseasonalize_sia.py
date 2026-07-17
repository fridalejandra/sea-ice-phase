"""
deseasonalize_sia.py

Computes X_t = SIA anomaly (deviation from day-of-year climatology), per
sector, to replace raw SIA in the AR(1) persistence model and the wind
stress regression - per the Chapter 4 framework, X_t should be a deviation
from the seasonal cycle, not the raw state.

Method: simple day-of-year climatological mean, leave-one-year-out NOT used
(full-record climatology) for simplicity - flag if you want to switch to a
smoother fit (e.g. harmonic/cyclic spline, matching the Ch3 IAC machinery)
later.

Handles Feb 29 by mapping to day-of-year using a fixed 366-day calendar
(pandas dayofyear), so leap years don't misalign the DOY climatology.
"""

import pandas as pd
import numpy as np

IN_CSV = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_table_daily.csv"
OUT_CSV = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_table_daily_anomaly.csv"

df = pd.read_csv(IN_CSV, parse_dates=["date"])
df["doy"] = df["date"].dt.dayofyear

# ---------------- Compute day-of-year climatology per sector ----------------
print("Computing day-of-year climatology per sector...")
climatology = df.groupby(["sector", "doy"])["SIA"].mean().reset_index()
climatology = climatology.rename(columns={"SIA": "SIA_climatology"})

print(climatology.head())

# ---------------- Merge climatology back and compute anomaly ----------------
df = df.merge(climatology, on=["sector", "doy"], how="left")
df["SIA_anomaly"] = df["SIA"] - df["SIA_climatology"]

# ---------------- Recompute delta_X on the anomaly, not raw SIA ----------------
df = df.sort_values(["sector", "date"])
df["delta_SIA_anomaly"] = df.groupby("sector")["SIA_anomaly"].diff()

# ---------------- Sanity checks ----------------
print("\nAnomaly summary by sector:")
print(df.groupby("sector")["SIA_anomaly"].describe())

print("\nChecking anomaly is roughly mean-zero per sector (should be near 0):")
print(df.groupby("sector")["SIA_anomaly"].mean())

# ---------------- Save ----------------
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved to: {OUT_CSV}")
print(f"New columns: SIA_climatology, SIA_anomaly, delta_SIA_anomaly")