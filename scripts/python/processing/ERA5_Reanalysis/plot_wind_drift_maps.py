# =====================================================
# Maps: % change (pre-2016 -> post-2016) in wind speed, drift speed,
# and the "excess" (drift% - wind%) — cell by cell on the SIE grid.
# Rows = seasons, cols = wind | drift | excess.
# =====================================================

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

WIND_NC  = "/user/geog/falejandraperez/sea-ice-phase/results/ERA5/ERA5_windSpeed_period_season_climatology.nc"
DRIFT_NC = "/user/geog/falejandraperez/sea-ice-phase/results/NSIDC0116/NSIDC0116_driftSpeed_period_season_climatology.nc"
OUT_DIR  = "/user/geog/falejandraperez/sea-ice-phase/results/wind_drift_norm"

SEASONS = ["JJA", "SON", "MAM", "DJF"]   # trim to ["JJA", "SON"] for a tighter figure
MIN_PRE_DRIFT = 0.01   # m/s; skip cells with near-zero baseline (blows up %)
VMAX = 50              # colour limit, % (symmetric)

os.makedirs(OUT_DIR, exist_ok=True)

wind  = xr.open_dataset(WIND_NC)["wind_speed_mps"]
drift = xr.open_dataset(DRIFT_NC)["drift_speed_mps"]
x, y = wind.x.values, wind.y.values
lat = wind.lat.values

def pct(da):
    pre, post = da.sel(period="pre"), da.sel(period="post")
    return 100.0 * (post - pre) / pre

wind_pct  = pct(wind)
drift_pct = pct(drift)

# valid where drift has a usable baseline AND a post value
valid = (drift.sel(period="pre") > MIN_PRE_DRIFT) & np.isfinite(drift.sel(period="post"))
drift_pct = drift_pct.where(valid)
wind_pct_masked = wind_pct.where(valid)       # wind only where ice drift exists
excess = drift_pct - wind_pct_masked

# optional cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
    proj = ccrs.SouthPolarStereo()
    data_crs = ccrs.PlateCarree()
    lon = wind.lon.values
except Exception:
    HAVE_CARTOPY = False

cols = [
    ("Wind speed change (%)",  wind_pct_masked),
    ("Drift speed change (%)", drift_pct),
    ("Excess: drift − wind (pp)", excess),
]
norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0, vmax=VMAX)
cmap = "RdBu_r"   # diverging: blue = decrease, red = increase, white = no change

nrow, ncol = len(SEASONS), len(cols)
fig = plt.figure(figsize=(4.2 * ncol, 4.0 * nrow), dpi=130)

for i, season in enumerate(SEASONS):
    for j, (title, field) in enumerate(cols):
        k = i * ncol + j + 1
        f = field.sel(season=season).values
        if HAVE_CARTOPY:
            ax = fig.add_subplot(nrow, ncol, k, projection=proj)
            ax.set_extent([-180, 180, -90, -50], crs=data_crs)
            ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", zorder=2)
            ax.coastlines(lw=0.4, color="#555", zorder=3)
            m = ax.pcolormesh(lon, lat, f, transform=data_crs, cmap=cmap, norm=norm,
                              shading="auto", zorder=1)
        else:
            ax = fig.add_subplot(nrow, ncol, k)
            m = ax.pcolormesh(x, y, f, cmap=cmap, norm=norm, shading="auto")
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        if i == 0:
            ax.set_title(title, fontsize=11)
        if j == 0:
            ax.text(-0.02, 0.5, season, transform=ax.transAxes, rotation=90,
                    ha="right", va="center", fontsize=12, fontweight="bold")

cb = fig.colorbar(m, ax=fig.axes, orientation="horizontal", fraction=0.03, pad=0.03,
                  extend="both")
cb.set_label("Change post-2016 relative to 1979–2015 mean", fontsize=10)
fig.suptitle("Wind vs. ice-drift change, cell by cell (same ice-covered cells in all columns)",
             fontsize=13, y=0.995)

out_png = os.path.join(OUT_DIR, "wind_drift_pct_change_maps.png")
out_pdf = os.path.join(OUT_DIR, "wind_drift_pct_change_maps.pdf")
plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}\nSaved: {out_pdf}\ncartopy: {HAVE_CARTOPY}")

# ---- bonus: sector-season summary recomputed on a FIXED mask (same cells both periods)
# This tests whether the scatter result survives when the sampled region can't change.
sec = xr.open_dataset("/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc")["sector_id"]
rows = []
for season in SEASONS:
    v = valid.sel(season=season)
    for sid in [1, 2, 3, 4, 5]:
        m_ = (sec == sid) & v
        dpre  = float(drift.sel(period="pre",  season=season).where(m_).mean())
        dpost = float(drift.sel(period="post", season=season).where(m_).mean())
        wpre  = float(wind.sel(period="pre",   season=season).where(m_).mean())
        wpost = float(wind.sel(period="post",  season=season).where(m_).mean())
        rows.append((season, sid, 100*(wpost-wpre)/wpre, 100*(dpost-dpre)/dpre, int(m_.sum())))
print("\nFixed-mask check (same cells in both periods):")
print(f"{'season':>6} {'sector':>6} {'wind%':>8} {'drift%':>8} {'ncells':>7}")
for r in rows:
    print(f"{r[0]:>6} {r[1]:>6} {r[2]:>8.1f} {r[3]:>8.1f} {r[4]:>7d}")

