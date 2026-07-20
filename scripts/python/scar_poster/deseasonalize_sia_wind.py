"""
deseasonalize_sia_and_wind.py

Extends deseasonalize_sia_wind.py: computes day-of-year climatology and anomaly
for BOTH SIA and wind stress, per sector. Previously only SIA was
deseasonalized - wind_stress passed through build_forcing_sector_table.py
unchanged, as raw magnitude. That's fine for the regression itself (delta_
SIA_anomaly ~ wind_stress uses raw wind stress as the predictor, which is
correct - wind stress IS the forcing, you're not trying to remove its own
seasonality from the regression). But for the Section 2 overview figure -
"did forcing change, did response change" - both panels should get the
same treatment (deseasonalized anomaly) so they're visually comparable on
the same terms, and so wind stress's OWN seasonal cycle (much stronger in
winter) doesn't swamp the plot the way raw SIA's seasonal cycle would have.

Method matches deseasonalize_sia_wind.py exactly: full-record day-of-year mean
climatology, no leave-one-year-out, pandas dayofyear (366-day calendar,
handles Feb 29 the same way).
"""

import pandas as pd
import numpy as np

IN_CSV = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_table_daily.csv"
OUT_CSV = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_table_daily_anomaly.csv"

df = pd.read_csv(IN_CSV, parse_dates=["date"])
df["doy"] = df["date"].dt.dayofyear

# ---------------- SIA climatology + anomaly (unchanged from original) ----------------
print("Computing day-of-year climatology per sector: SIA...")
sia_climatology = df.groupby(["sector", "doy"])["SIA"].mean().reset_index()
sia_climatology = sia_climatology.rename(columns={"SIA": "SIA_climatology"})

df = df.merge(sia_climatology, on=["sector", "doy"], how="left")
df["SIA_anomaly"] = df["SIA"] - df["SIA_climatology"]

df = df.sort_values(["sector", "date"])
df["delta_SIA_anomaly"] = df.groupby("sector")["SIA_anomaly"].diff()

# ---------------- NEW: wind stress climatology + anomaly ----------------
print("Computing day-of-year climatology per sector: wind_stress...")
wind_climatology = df.groupby(["sector", "doy"])["wind_stress"].mean().reset_index()
wind_climatology = wind_climatology.rename(columns={"wind_stress": "wind_stress_climatology"})

df = df.merge(wind_climatology, on=["sector", "doy"], how="left")
df["wind_stress_anomaly"] = df["wind_stress"] - df["wind_stress_climatology"]

# NOTE: the regression itself (delta_SIA_anomaly ~ wind_stress) should
# keep using raw wind_stress as the predictor, NOT wind_stress_anomaly -
# wind stress is the forcing variable, and its magnitude (not its
# deviation from typical-for-that-day) is what mechanically drives ice
# motion/divergence. wind_stress_anomaly is for the Section 2 overview
# figure only, so both panels share the same "deviation from normal"
# visual language - don't swap it into the regression by mistake.

# ---------------- Sanity checks ----------------
print("\nSIA anomaly summary by sector:")
print(df.groupby("sector")["SIA_anomaly"].describe())

print("\nWind stress anomaly summary by sector:")
print(df.groupby("sector")["wind_stress_anomaly"].describe())

print("\nChecking both anomalies are roughly mean-zero per sector (should be near 0):")
print(df.groupby("sector")[["SIA_anomaly", "wind_stress_anomaly"]].mean())

# ---------------- Save ----------------
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved to: {OUT_CSV}")
print("New columns: SIA_climatology, SIA_anomaly, delta_SIA_anomaly, "
      "wind_stress_climatology, wind_stress_anomaly")