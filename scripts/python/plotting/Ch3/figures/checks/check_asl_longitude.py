#!/usr/bin/env python3
"""
check_asl_longitude.py -- has the Amundsen Sea Low moved east, and in which season?

Sect. 3.3 says the ASL's longitude "did not move" between 1979-2000 and 2001-2025,
but that was the ANNUAL mean. The low has a strong seasonal migration (west in
winter, east in summer), so a spring shift can hide in an annual mean. For each
season this prints the mean longitude and latitude before and after the split,
the difference with a two-sample t-test, the linear trend over the record, and
the correlation of that season's longitude with the Ross amplitude in each period.

Reads  data/indices/asli_era5_v3-latest.csv   (Hosking ASL v3: time, lon, lat, RelCenPres)
       annual_params.csv                      (period == FULL; Ross amplitude)
Run from the figures directory.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import INDEX_DIR, INDEX_FILES, ANNUAL_CSV

SPLIT = 2001
SEASONS = {"annual": list(range(1, 13)), "DJF": [12, 1, 2], "MAM": [3, 4, 5],
           "JJA": [6, 7, 8], "SON": [9, 10, 11], "Oct-Jan": [10, 11, 12, 1]}

a = pd.read_csv(os.path.join(INDEX_DIR, INDEX_FILES["ASL"]), comment="#")
a["time"] = pd.to_datetime(a["time"])
a["year"], a["month"] = a.time.dt.year, a.time.dt.month
# DJF and Oct-Jan: December (and Oct-Dec) belong to the following / same season-year
a["sy_djf"] = np.where(a.month == 12, a.year + 1, a.year)
a["sy_oj"] = np.where(a.month == 1, a.year - 1, a.year)

ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"].astype(str) == "FULL"]
ross = ann[ann.sector == "SIE_Ross"].set_index("Year")
AMP = next(c for c in ("amplitude_raw_yr", "amplitude_raw", "amplitude") if c in ross.columns)


def detrend(x):
    t = np.arange(len(x), dtype=float); b, c = np.polyfit(t, x, 1); return x - (c + b * t)


print(f"ASL file: {INDEX_FILES['ASL']}, {a.year.min()}-{a.year.max()}; split at {SPLIT}\n")
print(f"{'season':8s} {'lon pre':>8s} {'lon post':>9s} {'shift':>6s} {'p':>6s} "
      f"{'trend/dec':>10s} {'lat pre':>8s} {'lat post':>9s} {'p':>6s}   r(lon, Ross amp) pre / post")
for name, months in SEASONS.items():
    key = "sy_djf" if name == "DJF" else ("sy_oj" if name == "Oct-Jan" else "year")
    s = (a[a.month.isin(months)].groupby(key)[["lon", "lat"]].mean())
    s = s[(s.index >= 1979) & (s.index <= 2025)]
    pre, post = s[s.index < SPLIT], s[s.index >= SPLIT]
    t_lon = stats.ttest_ind(pre.lon, post.lon, equal_var=False)
    t_lat = stats.ttest_ind(pre.lat, post.lat, equal_var=False)
    slope = np.polyfit(s.index.values, s.lon.values, 1)[0] * 10
    rs = []
    for part in (pre, post):
        j = part.join(ross[[AMP]], how="inner").dropna()
        rs.append(stats.pearsonr(detrend(j.lon.values), detrend(j[AMP].values))[0] if len(j) > 5 else np.nan)
    print(f"{name:8s} {pre.lon.mean():8.1f} {post.lon.mean():9.1f} {post.lon.mean()-pre.lon.mean():+6.1f} "
          f"{t_lon.pvalue:6.3f} {slope:+10.2f} {pre.lat.mean():8.1f} {post.lat.mean():9.1f} {t_lat.pvalue:6.3f}"
          f"   {rs[0]:+.2f} / {rs[1]:+.2f}")
print("\nlongitude in degrees east (Ross Sea ~ 160E-230E, Amundsen ~ 230E-260E); positive shift = eastward.")
print("If the spring (SON) or Oct-Jan low moved east while the annual mean barely did, 3.3 should say so.")
