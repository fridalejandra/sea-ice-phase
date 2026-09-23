"""
fig_wind_sia_djf_mam.py
Top row: wind stress magnitude, DJF (left) + MAM (right), with vectors.
Bottom row: SIA change, DJF (left) + MAM (right), no vectors.
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

SIC_PATH = "sic_bootstrap_on_ease_sh.nc"
WIND_PATH = "wind_stress_on_ease_sh.nc"
LATLON_PATH = "ease_divergence_with_latlon.nc"
SIC_VAR = "sic"
SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5]}
QUIVER_SKIP = 20
QUIVER_SCALE = 0.5
QUIVER_COLOR = "0.15"
QUIVER_ALPHA = 0.75

EASE_CRS = ccrs.LambertAzimuthalEqualArea(central_latitude=-90.0, central_longitude=0.0)
PLATE = ccrs.PlateCarree()
OUT = "fig_wind_sia_djf_mam.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def _tname(da):
    for c in ("time", "valid_time"):
        if c in da.dims:
            return c
    raise KeyError(f"No time dim in {list(da.dims)}")


def seasonal_prepost(da, months):
    tn = _tname(da)
    yrs = da[tn].dt.year
    da = da.sel({tn: ~yrs.isin(EXCLUDE_YEARS)})
    # shift December into the following meteorological-summer year, so
    # DJF season-years group correctly (Dec 2015 + Jan/Feb 2016 = one season)
    adj_year = da[tn].dt.year + (da[tn].dt.month == 12).astype(int)
    da = da.assign_coords(_adj_year=(tn, adj_year.values))
    sub = da.sel({tn: da[tn].dt.month.isin(months)})
    # flat pooled mean over all days (not yearly-mean-of-means) --
    # matches the originally verified aggregation method
    adj_years = sub["_adj_year"].values
    pre = np.nanmean(sub.values[adj_years < SPLIT_YEAR], axis=0)
    post = np.nanmean(sub.values[adj_years >= SPLIT_YEAR], axis=0)
    return pre, post


def main():
    sic_ds = xr.open_dataset(SIC_PATH)
    wind_ds = xr.open_dataset(WIND_PATH)
    latlon_ds = xr.open_dataset(LATLON_PATH, decode_times=False)
    x = sic_ds["x"].values
    y = sic_ds["y"].values
    lat2d = latlon_ds["lat"].values
    lon2d = latlon_ds["lon"].values

    wind_cache = {}
    wind_diffs = []
    sia_cache = {}
    sia_diffs = []

    for season, months in SEASONS.items():
        tx_pre, tx_post = seasonal_prepost(wind_ds["tau_x"], months)
        ty_pre, ty_post = seasonal_prepost(wind_ds["tau_y"], months)
        mag_pre = np.hypot(tx_pre, ty_pre)
        mag_post = np.hypot(tx_post, ty_post)
        wdiff = mag_post - mag_pre
        band = (lat2d < -50) & (lat2d > -65)
        pct = 100 * (np.nanmean(mag_post[band]) - np.nanmean(mag_pre[band])) / np.nanmean(mag_pre[band])
        wind_cache[season] = (wdiff, tx_post - tx_pre, ty_post - ty_pre, pct)
        wind_diffs.append(wdiff)

        sic_pre, sic_post = seasonal_prepost(sic_ds[SIC_VAR], months)
        sdiff = (sic_post - sic_pre) * 625.0
        sia_cache[season] = sdiff
        sia_diffs.append(sdiff)

    vmax_wind = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in wind_diffs])), 98)
    vmax_sia = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in sia_diffs])), 98)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 13),
                             subplot_kw={"projection": EASE_CRS},
                             constrained_layout=True)

    for col, season in enumerate(SEASONS.keys()):
        # top row: wind
        ax = axes[0, col]
        wdiff, dtx, dty, pct = wind_cache[season]
        im_wind = ax.pcolormesh(x, y, wdiff, transform=EASE_CRS, cmap="RdBu_r",
                                vmin=-vmax_wind, vmax=vmax_wind, shading="auto", zorder=1)

        s = QUIVER_SKIP
        lon_sub = lon2d[::s, ::s]
        lat_sub = lat2d[::s, ::s]
        u_sub = dtx[::s, ::s].copy()
        v_sub = dty[::s, ::s].copy()
        mag_sub = np.hypot(u_sub, v_sub)
        cap = np.nanpercentile(mag_sub, 95)
        scale_factor = np.where(mag_sub > cap, cap / np.maximum(mag_sub, 1e-12), 1.0)
        u_sub *= scale_factor
        v_sub *= scale_factor
        valid = np.isfinite(lon_sub) & np.isfinite(lat_sub) & np.isfinite(u_sub) & np.isfinite(v_sub)
        lon_sub = np.where(valid, lon_sub, np.nan)
        lat_sub = np.where(valid, lat_sub, np.nan)
        q = ax.quiver(lon_sub, lat_sub, u_sub, v_sub, transform=PLATE,
                     color=QUIVER_COLOR, alpha=QUIVER_ALPHA, scale=QUIVER_SCALE,
                     width=0.0032, headwidth=4, headlength=4, zorder=3)

        ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)
        ax.set_title(f"{season}  ({pct:+.1f}%)", fontsize=16, fontweight="bold")
        ax.quiverkey(q, 0.82, -0.06, cap, f"{cap:.3f} Pa", labelpos="E",
                    coordinates="axes", fontproperties={"size": 9})

        if col == 1:
            fig.colorbar(im_wind, ax=axes[0, :].tolist(), orientation="vertical",
                        fraction=0.025, pad=0.06, label="\u0394 wind stress (Pa)")

        # bottom row: SIA
        ax = axes[1, col]
        sdiff = sia_cache[season]
        im_sia = ax.pcolormesh(x, y, sdiff, transform=EASE_CRS, cmap="RdBu_r",
                               vmin=-vmax_sia, vmax=vmax_sia, shading="auto", zorder=1)
        ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)

        if col == 1:
            fig.colorbar(im_sia, ax=axes[1, :].tolist(), orientation="vertical",
                        fraction=0.025, pad=0.06, label="\u0394SIA (km\u00b2)")

    axes[0, 0].text(-0.12, 0.5, "Wind stress\nmagnitude \u0394", transform=axes[0, 0].transAxes,
                   fontsize=14, fontweight="bold", va="center", ha="right", rotation=90)
    axes[1, 0].text(-0.12, 0.5, "Sea ice\narea \u0394", transform=axes[1, 0].transAxes,
                   fontsize=14, fontweight="bold", va="center", ha="right", rotation=90)

    fig.suptitle("Wind stress strengthened; sea ice area declined broadly", fontsize=17, y=0.995)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()
