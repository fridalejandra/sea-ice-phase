#!/usr/bin/env python
"""
23_motion_trend_diagnostic.py -- why is our SO 1982-2024 pan-Antarctic speed trend (+1.23 cm/s/dec)
~1.8x Webster et al. (2026) (+0.69), and why does starting in 1988 drop it to +0.42?
Per year (Sep-Oct): days with any drift, n valid cells, and the pan-Antarctic mean speed computed
several ways; then trends for 1982- and 1988- starts, and a 1982-87 vs 1988-93 step.
"""
import glob, os, re, sys, warnings
import numpy as np, pandas as pd, xarray as xr
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/user/geog/falejandraperez/sea-ice-phase"
DRIFT_DIR = f"{ROOT}/data/drift_nsidc/"
DRIFT_GLOB = "icemotion_daily_sh_25km_*_v4.1.nc"
OUT = f"{ROOT}/results/ch4/tables/motion_trend_diagnostic_SO.csv"
MONTHS, YRS = (9, 10), np.arange(1982, 2025)


def drift_file(year):
    hits = [f for f in glob.glob(os.path.join(DRIFT_DIR, DRIFT_GLOB)) if re.search(rf"_{year}\d{{4}}", os.path.basename(f))]
    return sorted(hits)[0] if hits else None


def times(ds, year):
    try:
        return pd.DatetimeIndex(ds.indexes["time"].to_datetimeindex())
    except Exception:
        try:
            return pd.DatetimeIndex(pd.to_datetime([str(t)[:10] for t in ds.time.values]))
        except Exception:
            return pd.date_range(f"{year}-01-01", periods=ds.sizes["time"], freq="D")


def tr(y, v, y0):
    ok = np.isfinite(v) & (y >= y0)
    r = stats.linregress(y[ok], v[ok])
    return r.slope * 10, r.pvalue


rows, S50, S10 = [], [], []
for y in YRS:
    f = drift_file(y)
    if f is None:
        continue
    ds = xr.open_dataset(f)
    j = np.nonzero(np.isin(times(ds, y).month, MONTHS))[0]
    u = ds["u"].isel(time=j).values.astype("float32")
    v = ds["v"].isel(time=j).values.astype("float32")
    ok = np.isfinite(u) & np.isfinite(v)
    sp = np.where(ok, np.hypot(u, v), np.nan)
    frac = ok.mean(0)
    cs = np.nanmean(sp, 0)
    s50, s10 = np.where(frac >= 0.5, cs, np.nan), np.where(frac >= 0.1, cs, np.nan)
    vec = np.where(frac >= 0.5, np.hypot(np.nanmean(u, 0), np.nanmean(v, 0)), np.nan)
    S50.append(s50); S10.append(s10)
    rows.append(dict(year=y, days_with_data=int(ok.any(axis=(1, 2)).sum()), days=len(j),
                     n_cells_50=int(np.isfinite(s50).sum()), n_cells_10=int(np.isfinite(s10).sum()),
                     pan_50=np.nanmean(s50), pan_10=np.nanmean(s10), pooled=np.nanmean(sp),
                     vecspeed_50=np.nanmean(vec)))
    print(f"  {y}", flush=True)
d = pd.DataFrame(rows)
S50 = np.array(S50)
fixed = np.isfinite(S50).all(0)
d["pan_fixed"] = np.nanmean(S50[:, fixed], 1)
yy = d.year.values.astype(float)
for y0 in (1982, 1988):
    m = yy >= y0
    Z = S50[m][:, fixed]
    x = (yy[m] - yy[m].mean()) / 10
    b = (x @ (Z - Z.mean(0))) / (x @ x)
    print(f"\nSO {y0}-2024 trend (cm/s per decade):  mean of per-cell trends (fixed cells) {b.mean():+.2f}")
    for c in ("pan_50", "pan_10", "pooled", "pan_fixed", "vecspeed_50"):
        s, p = tr(yy, d[c].values, y0)
        print(f"   {c:12s} {s:+.2f}  (p={p:.3f})")
e, l = d[d.year.between(1982, 1987)], d[d.year.between(1988, 1993)]
print(f"\nfixed cells: {fixed.sum()}")
print("step 1982-87 -> 1988-93: pan_50 {:.2f} -> {:.2f};  pan_fixed {:.2f} -> {:.2f};  n_cells_50 {:.0f} -> {:.0f}".format(
    e.pan_50.mean(), l.pan_50.mean(), e.pan_fixed.mean(), l.pan_fixed.mean(), e.n_cells_50.mean(), l.n_cells_50.mean()))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
d.to_csv(OUT, index=False, float_format="%.4g")
pd.set_option("display.width", 200)
print(f"\nwrite {OUT}\n")
print(d.round(2).to_string(index=False))
