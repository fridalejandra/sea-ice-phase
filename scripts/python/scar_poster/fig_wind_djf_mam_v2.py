"""
fig_wind_djf_mam_v2.py
Mean-of-magnitudes, band-masked, Dec-adjusted, single clean version.
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

WIND_PATH = "wind_stress_on_ease_sh.nc"
LATLON_PATH = "ease_divergence_with_latlon.nc"
SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5]}
QUIVER_SKIP = 20
QUIVER_SCALE = 0.5
QUIVER_COLOR = "0.15"
QUIVER_ALPHA = 0.75

EASE_CRS = ccrs.LambertAzimuthalEqualArea(central_latitude=-90.0, central_longitude=0.0)
PLATE = ccrs.PlateCarree()
OUT = "fig_wind_djf_mam_v2.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def _tname(da):
    for c in ("time", "valid_time"):
        if c in da.dims:
            return c
    raise KeyError(f"No time dim in {list(da.dims)}")


def load_masked(wind_ds, months):
    tx = wind_ds["tau_x"]
    ty = wind_ds["tau_y"]
    tn = _tname(tx)

    yrs = tx[tn].dt.year
    keep = ~yrs.isin(EXCLUDE_YEARS)
    tx = tx.sel({tn: keep})
    ty = ty.sel({tn: keep})

    m = tx[tn].dt.month.isin(months)
    tx_sub = tx.sel({tn: m})
    ty_sub = ty.sel({tn: m})

    adj_years = (tx_sub[tn].dt.year.values +
                 (tx_sub[tn].dt.month.values == 12).astype(int))

    return tx_sub.values, ty_sub.values, adj_years


def seasonal_prepost_mag(wind_ds, months, band=None):
    tx_vals, ty_vals, adj_years = load_masked(wind_ds, months)
    mag = np.hypot(tx_vals, ty_vals)

    pre_field = np.nanmean(mag[adj_years < SPLIT_YEAR], axis=0)
    post_field = np.nanmean(mag[adj_years >= SPLIT_YEAR], axis=0)

    if band is not None:
        pct = 100 * (np.nanmean(post_field[band]) - np.nanmean(pre_field[band])) / np.nanmean(pre_field[band])
    else:
        pct = 100 * (np.nanmean(post_field) - np.nanmean(pre_field)) / np.nanmean(pre_field)

    return pre_field, post_field, pct


def vector_prepost(wind_ds, months):
    tx_vals, ty_vals, adj_years = load_masked(wind_ds, months)
    tx_pre = np.nanmean(tx_vals[adj_years < SPLIT_YEAR], axis=0)
    tx_post = np.nanmean(tx_vals[adj_years >= SPLIT_YEAR], axis=0)
    ty_pre = np.nanmean(ty_vals[adj_years < SPLIT_YEAR], axis=0)
    ty_post = np.nanmean(ty_vals[adj_years >= SPLIT_YEAR], axis=0)
    return tx_pre, tx_post, ty_pre, ty_post


def main():
    wind_ds = xr.open_dataset(WIND_PATH)
    latlon_ds = xr.open_dataset(LATLON_PATH, decode_times=False)
    x = wind_ds["x"].values
    y = wind_ds["y"].values
    lat2d = latlon_ds["lat"].values
    lon2d = latlon_ds["lon"].values

    band = (lat2d < -50) & (lat2d > -65)

    cache = {}
    diffs = []
    for season, months in SEASONS.items():
        pre_field, post_field, pct = seasonal_prepost_mag(wind_ds, months, band=band)
        tx_pre, tx_post, ty_pre, ty_post = vector_prepost(wind_ds, months)
        diff = post_field - pre_field
        cache[season] = (diff, tx_post - tx_pre, ty_post - ty_pre, pct)
        diffs.append(diff)
        print(f"  {season}: {pct:+.2f}%  (50-65S band, mean-of-magnitudes, Dec-adjusted)")

    vmax = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in diffs])), 98)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             subplot_kw={"projection": EASE_CRS})

    for ax, season in zip(axes, SEASONS.keys()):
        diff, dtx, dty, pct = cache[season]
        im = ax.pcolormesh(x, y, diff, transform=EASE_CRS, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto", zorder=1)

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
                    coordinates="axes", fontproperties={"size": 10})

    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.02,
                label="\u0394 wind stress magnitude (Pa)")
    fig.suptitle("Wind stress strengthened in both seasons (50\u201365\u00b0S band)",
                 fontsize=17, y=0.99)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()
