"""
compute_ZW3_goyal.py
Computes Goyal et al. (2022) ZW3 index from ERA5 v500 monthly data.
Requires: pip install eofs
Reference: Goyal et al. (2022), J. Climate, 35, 15.
"""

import numpy as np
import xarray as xr
import pandas as pd
from eofs.standard import Eof
import os
import glob

V500_DIR   = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/v500_monthly/"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/indices/"

# =============================================================================
# 1. LOAD ALL YEARS INTO ONE DATASET
# =============================================================================

files = sorted(glob.glob(os.path.join(V500_DIR, "*.nc")))
print(f"Found {len(files)} files")
ds = xr.open_mfdataset(files, combine="by_coords")

# =============================================================================
# 2. SUBSET AND COMPUTE ANOMALIES (matching Rishav's notebook exactly)
# =============================================================================

# Subset to 500 hPa, 40S-70S, 1979-2023
# Note: latitude stored decreasing so slice(-40,-70) works
v = (
    ds["v"]
    .sel(time=slice("1979", "2023"))
    .sel(latitude=slice(-40, -70))
)

# Monthly anomalies
v_anom = v.groupby("time.month") - v.groupby("time.month").mean(dim="time")

print(f"v anomaly shape: {v_anom.shape}")

# =============================================================================
# 3. EOF ANALYSIS WITH AREA WEIGHTING
# =============================================================================

lat    = v_anom.latitude
coslat = np.cos(np.deg2rad(lat.values)).clip(0., 1.)
wgts   = np.sqrt(coslat)[..., np.newaxis]   # (nlat, 1)

solver = Eof(v_anom.values, weights=wgts)
pcs    = solver.pcs(npcs=6, pcscaling=1)
var    = solver.varianceFraction()

print(f"Variance explained: PC1={var[0]*100:.1f}%, PC2={var[1]*100:.1f}%")

# =============================================================================
# 4. ZW3 MAGNITUDE AND PHASE (Rishav's quadrant-aware implementation)
# =============================================================================

zw3magnitude = (pcs[:, 0]**2 + pcs[:, 1]**2)**0.5
zw3phase     = np.full(len(v_anom.time), np.nan)

for i in range(len(v_anom.time)):
    p1, p2 = pcs[i, 0], pcs[i, 1]
    angle  = np.arctan(p2 / p1) * 180 / np.pi
    if   p1 > 0 and p2 > 0:
        zw3phase[i] = angle
    elif p1 < 0 and p2 > 0:
        zw3phase[i] = angle + 180
    elif p1 > 0 and p2 < 0:
        zw3phase[i] = angle
    elif p1 < 0 and p2 < 0:
        zw3phase[i] = angle - 180

# =============================================================================
# 5. SAVE
# =============================================================================

times = v_anom.time.values

df = pd.DataFrame({
    "year":          pd.DatetimeIndex(times).year,
    "month":         pd.DatetimeIndex(times).month,
    "ZW3_PC1":       pcs[:, 0],
    "ZW3_PC2":       pcs[:, 1],
    "ZW3_magnitude": zw3magnitude,
    "ZW3_phase_deg": zw3phase,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(os.path.join(OUTPUT_DIR, "ZW3_goyal_monthly.csv"), index=False)

annual = df.groupby("year")[
    ["ZW3_PC1", "ZW3_PC2", "ZW3_magnitude", "ZW3_phase_deg"]
].mean().reset_index()
annual.to_csv(os.path.join(OUTPUT_DIR, "ZW3_goyal_annual.csv"), index=False)

print("\n=== Done ===")
print(df.head(12))