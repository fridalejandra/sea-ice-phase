import os, numpy as np, xarray as xr, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap

NC = "/user/geog/falejandraperez/sea-ice-phase/results/NSIDC0116/NSIDC0116_driftSpeed_season_year.nc"
SEC = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"
LAND = "/user/geog/falejandraperez/sea-ice-phase/data/landmask_sic.nc"
OUT = "/user/geog/falejandraperez/sea-ice-phase/results/wind_drift_norm"
PRE, POST = (1992, 2015), (2016, 2024)
SEASONS = ["JJA", "SON"]
MIN_PRE, MIN_YEARS, VMAX = 0.03, 15, 40

d = xr.open_dataset(NC)["drift_speed_mps"]
x, y = d.x.values, d.y.values
sector = xr.open_dataset(SEC)["sector_id"]
try:
    lm = xr.open_dataset(LAND); v = [k for k in lm.data_vars if lm[k].ndim == 2][0]
    land = (lm[v].values != 0) & np.isfinite(lm[v].values)
except Exception: land = None

pre  = d.sel(year=slice(*PRE)); post = d.sel(year=slice(*POST))
npre = np.isfinite(pre).sum("year"); npost = np.isfinite(post).sum("year")
pre_m, post_m = pre.mean("year", skipna=True), post.mean("year", skipna=True)
valid = (pre_m > MIN_PRE) & (npre >= MIN_YEARS) & (npost >= 5)
pct = (100 * (post_m - pre_m) / pre_m).where(valid)

fig, axes = plt.subplots(1, len(SEASONS), figsize=(5.2 * len(SEASONS), 5.4), dpi=150)
norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0, vmax=VMAX)
for ax, s in zip(np.atleast_1d(axes), SEASONS):
    m = ax.pcolormesh(x, y, pct.sel(season=s).values, cmap="RdBu_r", norm=norm, shading="auto", zorder=1)
    if land is not None:
        ax.pcolormesh(x, y, np.where(land, 1.0, np.nan), cmap=ListedColormap(["#c8c8c8"]), shading="auto", zorder=2)
    ax.contour(x, y, np.nan_to_num(sector.values, nan=0), levels=[1.5,2.5,3.5,4.5], colors="#333", linewidths=0.5, zorder=3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlim(x.min()*0.78, x.max()*0.78); ax.set_ylim(y.min()*0.78, y.max()*0.78)
    ax.set_title({"JJA": "Winter (JJA)", "SON": "Spring (SON)"}.get(s, s), fontsize=12)
cb = fig.colorbar(m, ax=axes, orientation="horizontal", fraction=0.045, pad=0.04, extend="both")
cb.set_label(f"Ice-drift-speed change, {POST[0]}–{POST[1]} vs {PRE[0]}–{PRE[1]} (%)", fontsize=10)
fig.suptitle("Where did ice drift speed up after 2016?  (NSIDC-0116; cells with ≥15 baseline years)", fontsize=12, y=0.98)
plt.savefig(f"{OUT}/drift_change_map_{POST[0]}-{POST[1]}_vs_{PRE[0]}-{PRE[1]}.png", bbox_inches="tight")
plt.savefig(f"{OUT}/drift_change_map_{POST[0]}-{POST[1]}_vs_{PRE[0]}-{PRE[1]}.pdf", bbox_inches="tight")
print("saved")
