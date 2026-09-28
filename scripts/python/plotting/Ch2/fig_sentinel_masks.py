"""
Sentinel masks at thr=0.15 (Fig. S[X]):
  FS sentinel: climatological min SIC within advance window (DOY 46-273) never
               drops below 0.15  -> perennial ice, no genuine freeze onset.
  MS sentinel: climatological max SIC within retreat window (DOY >=227 or <=59)
               never reaches 0.15 -> open ocean, never meaningfully ice-covered.
1987 excluded, matching the detection pipeline.
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

MERGED = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
SECTOR = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"
OUT    = Path("/user/geog/falejandraperez/sea-ice-phase/results/Ch2_Figures/FigSX_sentinel_masks.png")
THR    = 0.15
BAD_YEARS = [1987]

ds = xr.open_dataset(MERGED)
var = [v for v in ds.data_vars if v.endswith("ICECON")][0]
years = np.unique(ds.time.dt.year.values)
years = [int(y) for y in years if 1979 <= y <= 2024 and y not in BAD_YEARS]

fs_min = None; ms_max = None
for y in years:                                   # year-by-year: low memory
    sl = ds[var].sel(time=str(y))
    a = sl.values.astype(float)
    a = np.where((a >= 0) & (a <= 1.0), a, np.nan)   # already fractional; drop 1.1/1.2 flags
    doy = sl.time.dt.dayofyear.values
    feb29 = (sl.time.dt.month.values == 2) & (sl.time.dt.day.values == 29)
    fs_w = (doy >= 46) & (doy <= 273) & ~feb29
    ms_w = ((doy >= 227) | (doy <= 59)) & ~feb29
    if fs_w.any():
        m = np.nanmin(a[fs_w], axis=0)
        fs_min = m if fs_min is None else np.fmin(fs_min, m)
    if ms_w.any():
        m = np.nanmax(a[ms_w], axis=0)
        ms_max = m if ms_max is None else np.fmax(ms_max, m)
    print(f"  {y} done", end="\r")

sm = xr.open_dataset(SECTOR)
valid = sm["valid_ocean"].astype(bool).values
x = sm["x"].values if "x" in sm else ds["x"].values
yy = sm["y"].values if "y" in sm else ds["y"].values

fs_sent = np.isfinite(fs_min) & (fs_min >= THR) & valid
ms_sent = (~np.isfinite(ms_max) | (ms_max < THR)) & valid & np.isfinite(ms_max)
print(f"\nSentinel counts at thr={THR}:  FS = {int(fs_sent.sum())}   MS = {int(ms_sent.sum())}")
print("(receipt: pipeline log said FS=81, MS=38,499 - small differences fine,")
print(" cite the pipeline's numbers in the text)")

proj = ccrs.SouthPolarStereo(); pc = ccrs.PlateCarree()
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), subplot_kw={"projection": proj})
specs = [(fs_sent, "(a) Freeze Start sentinels\nperennial ice: min SIC \u2265 15% in advance window", "#b2182b"),
         (ms_sent, "(b) Melt Start sentinels\nopen ocean: max SIC < 15% in retreat window", "#2166ac")]
for ax, (mask, title, color) in zip(axes, specs):
    ax.set_extent([-180, 180, -90, -50], pc)
    ax.add_feature(cfeature.LAND, facecolor="0.75", edgecolor="none", zorder=3)
    ax.set_facecolor("white")
    shade = np.where(valid, 0.0, np.nan)          # valid ocean = light background
    ax.pcolormesh(x, yy, shade, transform=proj, cmap="Greys", vmin=-0.15, vmax=1, zorder=1)
    mm = np.where(mask, 1.0, np.nan)
    ax.pcolormesh(x, yy, mm, transform=proj,
                  cmap=mcolors.ListedColormap([color]), zorder=2)
    ax.set_title(title, fontsize=9)
fig.suptitle("Sentinel masks (excluded from phase detection), 15% threshold", fontsize=10, y=1.04)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved -> {OUT}")
