#!/usr/bin/env python3
"""Fig-4-consistent numbers for 4.3: sector-mean change (2016-2024 minus 1979-2015) in
Freeze Start, Melt Start and season length, static and dynamic, over active (80%) pixels."""
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
A, S = ROOT / "data/anomalies/SMMR", ROOT / "data/canonical_sectors.nc"
SEC = {1: "ABS", 2: "WS", 3: "KHV", 4: "EA", 5: "RAS", 0: "ALL"}
T = {("FS","static"):"FS_static_thr15_k5",("FS","dynamic"):"FS_dynamic_k5_q70",
     ("MS","static"):"MS_static_thr15_k5",("MS","dynamic"):"MS_dynamic_k5_q70"}
def load(ph, m):
    t = T[(ph, m)]; v = "_dsa" if ph == "MS" else ""
    c = xr.open_dataset(A/f"{t}_climatology.nc", decode_times=False)
    a = xr.open_dataset(A/f"{t}_anomalies.nc", decode_times=False)
    cn = f"{t}_clim{v}" if f"{t}_clim{v}" in c else f"{t}_clim"
    an = f"{t}_anom{v}" if f"{t}_anom{v}" in a else f"{t}_anom"
    return c[cn].values[None] + a[an].values, a["year"].values
sid = xr.open_dataset(S)["sector_id"].values
d = {k: load(*k) for k in T}; yrs = d[("FS","static")][1]
pre, post = yrs <= 2015, yrs >= 2016
rows = []
for m in ["static", "dynamic"]:
    fs, ms = d[("FS", m)][0], d[("MS", m)][0]
    act = {ph: (np.isfinite(d[(ph,"static")][0]).mean(0) >= .8) & (np.isfinite(d[(ph,"dynamic")][0]).mean(0) >= .8) for ph in ["FS","MS"]}
    step = lambda x: np.nanmean(x[post], 0) - np.nanmean(x[pre], 0)
    sfs, sms = step(fs), step(ms)
    dur = (ms + 227) - fs                      # season length (days), both detected
    sdur = step(dur)
    for s, lab in SEC.items():
        sel = lambda mk: mk & ((sid == s) if s else True)
        both = act["FS"] & act["MS"]
        rows.append(dict(method=m, sector=lab,
            FS_step=round(np.nanmean(sfs[sel(act["FS"])]), 1),
            MS_step=round(np.nanmean(sms[sel(act["MS"])]), 1),
            season_step=round(np.nanmean(sdur[sel(both)]), 1)))
df = pd.DataFrame(rows)
print("Change 2016-2024 minus 1979-2015 (days); FS +later, MS -earlier, season -shorter")
print(df.pivot(index="sector", columns="method").reindex(list(SEC.values())).to_string())
