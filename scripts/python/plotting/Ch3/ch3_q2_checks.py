#!/usr/bin/env python
# ch3_q2_checks.py -- robustness of the post-2016 coupling between the annual mean and the cycle's shape.
#  (1) leave-one-year-out: r(annual mean, X) in 2016-2024 recomputed with each year dropped, for
#      X = amplitude, day of max, and the APAC Beta-warp parameters beta1, beta2, beta_asym
#  (2) where the amplitude coupling comes from: slope of each seasonal mean (DJF, MAM, JJA, SON) on the
#      annual mean, before and after 2016.  slope > 1 in a season = that season moves MORE than the mean.
import os, numpy as np, pandas as pd
ROOT = os.environ.get("SIP_ROOT", "/user/geog/falejandraperez/sea-ice-phase")
a = pd.read_csv(f"{ROOT}/data/ch3/annual_params.csv"); a = a[a.period == "FULL"] if "period" in a else a
OUT = f"{ROOT}/results/ch3/tables"; os.makedirs(OUT, exist_ok=True)
SEC = {"WS": "SIE_Weddell", "KH": "SIE_King_Haakon", "EA": "SIE_East_Antarctica",
       "RA": "SIE_Ross", "ABS": "SIE_Amundsen_Bellingshausen", "Total": "SIE_circumpolar"}
PRE, POST = (1979, 2015), (2016, 2024)
X = {"amplitude": "amplitude_raw_yr", "day of max": "max_doy_raw",
     "beta1": "beta1", "beta2": "beta2", "beta_asym": "beta_asym"}
def detr(v, t):
    return v - np.polyval(np.polyfit(t, v, 1), t)
def r(df, x, y):
    t = df.index.values.astype(float); return np.corrcoef(detr(df[x].values, t), detr(df[y].values, t))[0, 1]
def slope(df, x, y):
    t = df.index.values.astype(float); return np.polyfit(detr(df[x].values, t), detr(df[y].values, t), 1)[0]
pd.set_option("display.width", 200)
rows1, rows2 = [], []
for s, name in SEC.items():
    g = a[a.sector == name].set_index("Year").sort_index()
    pre, post = g.loc[PRE[0]:PRE[1]], g.loc[POST[0]:POST[1]]
    for lab, col in X.items():
        if col not in g: continue
        d = post[["sie_annual", col]].dropna()
        loo = {yr: r(d.drop(yr), "sie_annual", col) for yr in d.index}
        lo_yr = min(loo, key=lambda k: abs(loo[k]))
        rows1.append(dict(sector=s, X=lab, r_pre=r(pre[["sie_annual", col]].dropna(), "sie_annual", col),
                          r_post=r(d, "sie_annual", col), loo_min=min(loo.values()), loo_max=max(loo.values()),
                          weakest_when_dropping=lo_yr))
    for seas in ("DJF", "MAM", "JJA", "SON"):
        col = f"sie_{seas}"
        if col not in g: continue
        rows2.append(dict(sector=s, season=seas,
                          slope_pre=slope(pre[["sie_annual", col]].dropna(), "sie_annual", col),
                          slope_post=slope(post[["sie_annual", col]].dropna(), "sie_annual", col)))
T1 = pd.DataFrame(rows1); T2 = pd.DataFrame(rows2)
T1.to_csv(f"{OUT}/ch3_q2_loyo.csv", index=False); T2.to_csv(f"{OUT}/ch3_q2_seasonal_slopes.csv", index=False)
print("\n(1) r(annual mean, X): before, after, and range after dropping each post-2016 year in turn")
print(T1.round(2).to_string(index=False))
print("\n(2) slope of each seasonal mean on the annual mean (detrended).  1 = moves with the mean;")
print("    >1 = moves more than the mean.  Amplitude coupling via winter -> SON/JJA slopes rise after 2016")
print(T2.pivot(index="sector", columns="season", values=["slope_pre", "slope_post"]).round(2).to_string())
print(f"\nwrote {OUT}/ch3_q2_loyo.csv and ch3_q2_seasonal_slopes.csv")
