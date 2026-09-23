"""
fig_wind_strengthening_v2.py
Wind stress magnitude: pre | post | difference, DJF and MAM rows.
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
SECTORS = {
    "Weddell": (-60.0, 20.0), "King Haakon": (20.0, 90.0),
    "East Antarctica": (90.0, 160.0), "Ross": (160.0, 230.0),
    "ABS": (230.0, 300.0),
}
QUIVER_SKIP = 18
QUIVER_SCALE = 0.8
EASE_CRS = ccrs.LambertAzimuthalEqualArea(central_latitude=-90.0, central_longitude=0.0)
PLATE = ccrs.PlateCarree()
OUT = "fig_wind_strengthening_v2.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def _tname(da):
    for c in ("time", "valid_time"):
        if c in da.dims:
            return c
    raise KeyError(f"No time dim in {list(da.dims)}")


def seasonal_prepost(ds, var, months):
    da = ds[var]
    tn = _tname(da)
    yrs = da[tn].dt.year
    da = da.sel({tn: ~yrs.isin(EXCLUDE_YEARS)})
    sub = da.sel({tn: da[tn].dt.month.isin(months)})
    ym = sub.groupby(sub[tn].dt.year).mean(dim=tn).load()
    years = ym["year"].values
    pre = np.nanmean(ym.values[years < SPLIT_YEAR], axis=0)
    post = np.nanmean(ym.values[years >= SPLIT_YEAR], axis=0)
    return pre, post


def draw_sector_boundaries(ax):
    for name, (lon_min, lon_max) in SECTORS.items():
        lm = ((lon_min + 180) % 360) - 180
        ax.plot([lm, lm], [-90, -50], transform=PLATE,
                color="0.25", linewidth=1.0, linestyle="--", alpha=0.6, zorder=6)


def main():
    wind_ds = xr.open_dataset(WIND_PATH)
    latlon_ds = xr.open_dataset(LATLON_PATH, decode_times=False)
    x = wind_ds["x"].values
    y = wind_ds["y"].values
    lat2d = latlon_ds["lat"].values
    lon2d = latlon_ds["lon"].values

    mean_vals, diff_vals = [], []
    cache = {}
    for season, months in SEASONS.items():
        tx_pre, tx_post = seasonal_prepost(wind_ds, "tau_x", months)
        ty_pre, ty_post = seasonal_prepost(wind_ds, "tau_y", months)
        mag_pre = np.hypot(tx_pre, ty_pre)
        mag_post = np.hypot(tx_post, ty_post)
        cache[season] = (mag_pre, mag_post, tx_post - tx_pre, ty_post - ty_pre)
        mean_vals.extend([mag_pre, mag_post])
        diff_vals.append(mag_post - mag_pre)
    vmax_mean = np.nanpercentile(np.concatenate([d.ravel() for d in mean_vals]), 98)
    vmax_diff = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in diff_vals])), 98)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12.5),
                             subplot_kw={"projection": EASE_CRS})

    for row, season in enumerate(SEASONS.keys()):
        mag_pre, mag_post, dtx, dty = cache[season]
        mag_diff = mag_post - mag_pre

        panels = [(mag_pre, f"{season}\npre-2016 mean", "YlOrRd", 0, vmax_mean, False),
                  (mag_post, "post-2016 mean", "YlOrRd", 0, vmax_mean, False),
                  (mag_diff, "difference (post \u2212 pre)", "RdBu_r", -vmax_diff, vmax_diff, True)]

        for col, (field, title, cmap, vmin, vmax, show_vec) in enumerate(panels):
            ax = axes[row, col]
            im = ax.pcolormesh(x, y, field, transform=EASE_CRS, cmap=cmap,
                               vmin=vmin, vmax=vmax, shading="auto", zorder=1)

            if show_vec:
                s = QUIVER_SKIP
                lon_sub = lon2d[::s, ::s]
                lat_sub = lat2d[::s, ::s]
                u_sub = dtx[::s, ::s]
                v_sub = dty[::s, ::s]
                valid = np.isfinite(lon_sub) & np.isfinite(lat_sub) & np.isfinite(u_sub) & np.isfinite(v_sub)
                lon_sub = np.where(valid, lon_sub, np.nan)
                lat_sub = np.where(valid, lat_sub, np.nan)
                ax.quiver(lon_sub, lat_sub, u_sub, v_sub, transform=PLATE,
                         color="0.1", alpha=0.8, scale=QUIVER_SCALE, width=0.0032,
                         headwidth=4, headlength=4, zorder=3)

            draw_sector_boundaries(ax)
            ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
            ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
            ax.set_extent([-180, 180, -90, -50], crs=PLATE)
            ax.set_title(title, fontsize=13)

            if col == 2:
                fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.045,
                            pad=0.03, label="\u0394 stress (Pa)")
            elif col == 1:
                fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.045,
                            pad=0.03, label="stress mag. (Pa)")

    fig.suptitle("Westerlies strengthened ~11% (50\u201365\u00b0S band)",
                 fontsize=18, y=0.99)
    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()
