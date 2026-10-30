#!/usr/bin/env python
# ch3_beta_check.py -- do the APAC warp parameters track observed timing?
# Observed advance duration = max_doy_raw - min_doy_raw (days from minimum to maximum)
# Observed retreat duration = next year's min_doy_raw + 365 - max_doy_raw (days from maximum to next minimum)
# Detrended correlations over 1979-2024, per sector.  |r| > ~0.29 is p<0.05 with ~45 years.
import os, numpy as np, pandas as pd
ROOT = os.environ.get("SIP_ROOT", "/user/geog/falejandraperez/sea-ice-phase")
a = pd.read_csv(f"{ROOT}/data/ch3/annual_params.csv"); a = a[a.period == "FULL"] if "period" in a else a
SEC = {"WS": "SIE_Weddell", "KH": "SIE_King_Haakon", "EA": "SIE_East_Antarctica",
       "RA": "SIE_Ross", "ABS": "SIE_Amundsen_Bellingshausen", "Total": "SIE_circumpolar"}
def r(x, y):
    d = pd.concat([x, y], axis=1).dropna(); t = d.index.values.astype(float)
    if len(d) < 10: return np.nan
    dt = lambda v: v - np.polyval(np.polyfit(t, v, 1), t)
    return np.corrcoef(dt(d.iloc[:, 0].values), dt(d.iloc[:, 1].values))[0, 1]
rows = []
for s, name in SEC.items():
    g = a[a.sector == name].set_index("Year").sort_index().loc[1979:2024]
    adv = (g.max_doy_raw - g.min_doy_raw).rename("adv")
    ret = (g.min_doy_raw.shift(-1) + 365 - g.max_doy_raw).rename("ret")
    for lab, col in (("beta1", "beta1"), ("beta2", "beta2"), ("beta_asym", "beta_asym"),
                     ("fitted day of max", "max_doy_fitted"), ("observed day of max", "max_doy_raw")):
        rows.append(dict(sector=s, measure=lab, r_advance=r(g[col], adv), r_retreat=r(g[col], ret)))
    rows.append(dict(sector=s, measure="fitted vs observed day of max",
                     r_advance=r(g.max_doy_fitted, g.max_doy_raw), r_retreat=np.nan))
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\nDetrended r with observed advance duration (min -> max) and retreat duration (max -> next min), 1979-2024")
print(T.round(2).to_string(index=False))
