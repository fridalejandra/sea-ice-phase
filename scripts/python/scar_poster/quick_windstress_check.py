"""
quick_wind_stress_check.py

FAST, meeting-ready check: has wind stress itself increased since 2016?
Uses the existing yearly wind_stress files directly (domain-averaged over the
Southern Ocean study region), no regridding to sectors needed. This is meant
to get you a real number/figure quickly - the full sector-level regridded
version can come later.
"""

import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

WIND_STRESS_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/wind_stress"
START_YEAR = 1979
END_YEAR = 2024
ACCUM_SECONDS = 86400

records = []

for year in range(START_YEAR, END_YEAR + 1):
    tx_file = os.path.join(WIND_STRESS_DIR, str(year), f"era5_windstress_tau_x_{year}.nc")
    ty_file = os.path.join(WIND_STRESS_DIR, str(year), f"era5_windstress_tau_y_{year}.nc")
    if not (os.path.exists(tx_file) and os.path.exists(ty_file)):
        print(f"  {year}: missing file(s), skipping")
        continue

    tx_ds = xr.open_dataset(tx_file)
    ty_ds = xr.open_dataset(ty_file)

    tau_x = tx_ds["ewss"] / ACCUM_SECONDS
    tau_y = ty_ds["nsss"] / ACCUM_SECONDS
    tau_mag = np.sqrt(tau_x**2 + tau_y**2)

    # domain-mean wind stress magnitude per day (mean over lat/lon)
    daily_mean = tau_mag.mean(dim=["latitude", "longitude"])

    df = pd.DataFrame({
        "date": pd.to_datetime(daily_mean.valid_time.values),
        "wind_stress": daily_mean.values,
    })
    records.append(df)
    tx_ds.close()
    ty_ds.close()

full_df = pd.concat(records, ignore_index=True)
full_df["year"] = full_df["date"].dt.year
full_df["period"] = np.where(full_df["year"] < 2016, "pre_2016", "post_2016")

print("=" * 60)
print("DOMAIN-WIDE WIND STRESS MAGNITUDE: pre vs. post 2016")
print("=" * 60)
summary = full_df.groupby("period")["wind_stress"].describe()
print(summary)

pre = full_df[full_df["period"] == "pre_2016"]["wind_stress"]
post = full_df[full_df["period"] == "post_2016"]["wind_stress"]

pct_change = 100 * (post.mean() - pre.mean()) / pre.mean()
print(f"\nMean pre-2016: {pre.mean():.5f} N/m^2")
print(f"Mean post-2016: {post.mean():.5f} N/m^2")
print(f"Percent change: {pct_change:+.2f}%")

# simple t-test
from scipy import stats
t_stat, p_val = stats.ttest_ind(pre, post, equal_var=False)
print(f"\nWelch's t-test: t={t_stat:.3f}, p={p_val:.4f}")
if p_val < 0.05:
    print("  -> statistically significant difference between periods")
else:
    print("  -> NOT statistically significant - no clear evidence of a shift")

# ---------------- Plot for the meeting ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# annual mean time series
annual = full_df.groupby("year")["wind_stress"].mean()
axes[0].plot(annual.index, annual.values, marker='o')
axes[0].axvline(2016, color='red', linestyle='--', label='2016')
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Mean wind stress (N/m^2)")
axes[0].set_title("Annual mean wind stress magnitude")
axes[0].legend()

# distribution comparison
axes[1].hist(pre, bins=50, alpha=0.5, label=f"Pre-2016 (mean={pre.mean():.4f})", density=True)
axes[1].hist(post, bins=50, alpha=0.5, label=f"Post-2016 (mean={post.mean():.4f})", density=True)
axes[1].set_xlabel("Daily wind stress (N/m^2)")
axes[1].set_ylabel("Density")
axes[1].set_title("Distribution: pre vs. post 2016")
axes[1].legend()

plt.tight_layout()
plt.savefig("/user/geog/falejandraperez/sea-ice-phase/data/merged/wind_stress_pre_post_2016.png", dpi=150)
print("\nSaved figure to wind_stress_pre_post_2016.png")
plt.show()