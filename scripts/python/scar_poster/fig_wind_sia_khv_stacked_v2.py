"""
fig_wind_sia_khv_stacked_v2.py
MAM only, verified method: mean-of-magnitudes, 50-65S band, Dec-adjusted.
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
SEASON_MONTHS = [3, 4, 5]
QUIVER_SKIP = 20
QUIVER_SCALE = 0.5
QUIVER_COLOR = "0.15"
QUIVER_ALPHA = 0.75

EASE_CRS = ccrs.LambertAzimuthalEqualArea(central_latitude=-90.0, central_longitude=0.0)
PLATE = ccrs.PlateCarree()
OUT = "fig_wind_sia_khv_stacked_v2.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def _tname(da):
    for c in ("time", "valid_time"):
        if c in da.dims:
            return c
    raise KeyError(f"No time dim in {list(da.dims)}")


def load_masked(da, months):
    tn = _tname(da)
    yrs = da[tn].dt.year
    da = da.sel({tn: ~yrs.isin(EXCLUDE_YEARS)})
    sub = da.sel({tn: da[tn].dt.month.isin(months)})
    adj_years = (sub[tn].dt.year.values +
                 (sub[tn].dt.month.values == 12).astype(int))
    return sub.values, adj_years


def wind_mag_prepost(wind_ds, months, band=None):
    tx_vals, adj_years = load_masked(wind_ds["tau_x"], months)
    ty_vals, _ = load_masked(wind_ds["tau_y"], months)
    mag = np.hypot(tx_vals, ty_vals)

    pre_field = np.nanmean(mag[adj_years < SPLIT_YEAR], axis=0)
    post_field = np.nanmean(mag[adj_years >= SPLIT_YEAR], axis=0)

    if band is not None:
        pct = 100 * (np.nanmean(post_field[band]) - np.nanmean(pre_field[band])) / np.nanmean(pre_field[band])
    else:
        pct = 100 * (np.nanmean(post_field) - np.nanmean(pre_field)) / np.nanmean(pre_field)

    return pre_field, post_field, pct


def wind_vector_prepost(wind_ds, months):
    tx_vals, adj_years = load_masked(wind_ds["tau_x"], months)
    ty_vals, _ = load_masked(wind_ds["tau_y"], months)
    tx_pre = np.nanmean(tx_vals[adj_years < SPLIT_YEAR], axis=0)
    tx_post = np.nanmean(tx_vals[adj_years >= SPLIT_YEAR], axis=0)
    ty_pre = np.nanmean(ty_vals[adj_years < SPLIT_YEAR], axis=0)
    ty_post = np.nanmean(ty_vals[adj_years >= SPLIT_YEAR], axis=0)
    return tx_pre, tx_post, ty_pre, ty_post


def sic_prepost(sic_ds, months):
    vals, adj_years = load_masked(sic_ds[SIC_VAR], months)
    pre = np.nanmean(vals[adj_years < SPLIT_YEAR], axis=0)
    post = np.nanmean(vals[adj_years >= SPLIT_YEAR], axis=0)
    return pre, post


def main():
    sic_ds = xr.open_dataset(SIC_PATH)
    wind_ds = xr.open_dataset(WIND_PATH)
    latlon_ds = xr.open_dataset(LATLON_PATH, decode_times=False)

    x = sic_ds["x"].values
    y = sic_ds["y"].values
    lat2d = latlon_ds["lat"].values
    lon2d = latlon_ds["lon"].values
    band = (lat2d < -50) & (lat2d > -65)

    mag_pre, mag_post, pct = wind_mag_prepost(wind_ds, SEASON_MONTHS, band=band)
    tx_pre, tx_post, ty_pre, ty_post = wind_vector_prepost(wind_ds, SEASON_MONTHS)
    mag_diff = mag_post - mag_pre
    dtx, dty = tx_post - tx_pre, ty_post - ty_pre
    vmax_wind = np.nanpercentile(np.abs(mag_diff), 98)

    print(f"  MAM: {pct:+.2f}%  (50-65S band, mean-of-magnitudes, Dec-adjusted)")

    sic_pre, sic_post = sic_prepost(sic_ds, SEASON_MONTHS)
    sia_diff = (sic_post - sic_pre) * 625.0
    vmax_sia = np.nanpercentile(np.abs(sia_diff), 98)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5),
                             subplot_kw={"projection": EASE_CRS})

    ax = axes[0]
    im_wind = ax.pcolormesh(x, y, mag_diff, transform=EASE_CRS, cmap="RdBu_r",
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
    ax.set_title(f"Wind stress magnitude \u0394 \u2014 MAM  ({pct:+.1f}%)",
                fontsize=15, fontweight="bold")
    fig.colorbar(im_wind, ax=ax, orientation="vertical", fraction=0.045,
                pad=0.03, label="\u0394 stress (Pa)")
    ax.quiverkey(q, 0.82, -0.06, cap, f"{cap:.3f} Pa", labelpos="E",
                coordinates="axes", fontproperties={"size": 10})

    ax = axes[1]
    im_sia = ax.pcolormesh(x, y, sia_diff, transform=EASE_CRS, cmap="RdBu_r",
                           vmin=-vmax_sia, vmax=vmax_sia, shading="auto", zorder=1)
    ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
    ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
    ax.set_extent([-180, 180, -90, -50], crs=PLATE)
    ax.set_title("Sea ice area \u0394 \u2014 MAM", fontsize=15, fontweight="bold")
    fig.colorbar(im_sia, ax=ax, orientation="vertical", fraction=0.045,
                pad=0.03, label="\u0394SIA (km\u00b2)")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()
