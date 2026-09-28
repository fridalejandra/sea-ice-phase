"""
Quick-look: pre/post-2016 linear slopes of FS and MS under the STATIC method.
Compare printed medians against the dynamic run:
  dynamic FS pre=-0.375  FS post=+1.583  MS pre=+0.083  MS post=+0.067
Inspection figure only - not publication styling.
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANOM = "/user/geog/falejandraperez/sea-ice-phase/data/anomalies/SMMR"
FILES = {"FS": f"{ANOM}/FS_static_thr15_k5_anomalies.nc",
         "MS": f"{ANOM}/MS_static_thr15_k5_anomalies.nc"}
PRE  = (1979, 2015)
POST = (2016, 2024)
MIN_YEARS = {"pre": 15, "post": 6}   # min valid years per pixel for a slope

def load_anom(path):
    ds = xr.open_dataset(path, decode_times=False)
    v = list(ds.data_vars)[0]
    da = ds[v]
    ycoord = next(c for c in da.coords if "year" in c.lower() or "time" in c.lower())
    years = np.asarray(ds[ycoord].values).astype(int)
    return da.values.astype(float), years   # (nyear, ny, nx), (nyear,)

def nan_slope(arr, years, y0, y1, min_n):
    """Vectorized per-pixel OLS slope over years in [y0, y1], NaN-aware."""
    sel = (years >= y0) & (years <= y1)
    a = arr[sel]                       # (t, ny, nx)
    t = years[sel].astype(float)[:, None, None] * np.ones_like(a)
    m = np.isfinite(a)
    n = m.sum(axis=0)
    t = np.where(m, t, np.nan); a = np.where(m, a, np.nan)
    tm = np.nanmean(t, axis=0); am = np.nanmean(a, axis=0)
    cov = np.nanmean((t - tm) * (a - am), axis=0)
    var = np.nanmean((t - tm) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = cov / var
    s[n < min_n] = np.nan
    return s

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
print("STATIC method sub-period slope medians (days/yr):")
for i, phase in enumerate(["FS", "MS"]):
    arr, years = load_anom(FILES[phase])
    for j, (tag, (y0, y1)) in enumerate({"pre": PRE, "post": POST}.items()):
        s = nan_slope(arr, years, y0, y1, MIN_YEARS[tag])
        med = np.nanmedian(s)
        print(f"  {phase} {tag}-2016: median = {med:+.3f}  "
              f"(valid pixels: {int(np.isfinite(s).sum())})")
        ax = axes[i, j]
        im = ax.imshow(s, origin="lower", cmap="RdBu_r", vmin=-4, vmax=4)
        ax.set_title(f"{phase} {tag}-2016  (median {med:+.2f} d/yr)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, shrink=0.7, label="slope (days/yr), red = later")
# fig.suptitle("STATIC method (thr15_k5): pre/post-2016 linear trends - quick look")
out = "Fig08_static_check.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved -> {out}")
