"""
fig_monthly_wind_convergence.py
Monthly (not seasonal) wind + convergence diagnostic.
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

CONV_PATH = "ice_divergence_daily_sh.nc"
CONV_VAR = "divergence"
WIND_PATH = "wind_stress_on_ease_sh.nc"
SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
MONTHS = list(range(1, 13))
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

EASE_CRS = ccrs.LambertAzimuthalEqualArea(central_latitude=-90.0, central_longitude=0.0)
PLATE = ccrs.PlateCarree()
OUT_WIND = "fig_monthly_wind.png"
OUT_CONV = "fig_monthly_convergence.png"
RCLONE_REMOTE = "gdrive:scar_poster/supplementary/"


def _tname(da):
    for c in ("time", "valid_time"):
        if c in da.dims:
            return c
    raise KeyError(f"No time dim in {list(da.dims)}")


def monthly_prepost(da, month, negate=False):
    if negate:
        da = -da
    tn = _tname(da)
    yrs = da[tn].dt.year
    da = da.sel({tn: ~yrs.isin(EXCLUDE_YEARS)})
    sub = da.sel({tn: da[tn].dt.month == month})
    post_mask = sub[tn].dt.year.values >= SPLIT_YEAR
    pre_data = sub.isel({tn: ~post_mask}).values
    post_data = sub.isel({tn: post_mask}).values
    valid_pre = np.isfinite(pre_data).mean()
    valid_post = np.isfinite(post_data).mean()
    pre_mean = np.nanmean(pre_data, axis=0)
    post_mean = np.nanmean(post_data, axis=0)
    return pre_mean, post_mean, valid_pre, valid_post


def main():
    conv_ds = xr.open_dataset(CONV_PATH)
    wind_ds = xr.open_dataset(WIND_PATH)
    x = conv_ds["x"].values
    y = conv_ds["y"].values

    print("=== Wind stress magnitude, monthly ===")
    wind_diffs = []
    for m in MONTHS:
        tx_pre, tx_post, _, _ = monthly_prepost(wind_ds["tau_x"], m)
        ty_pre, ty_post, _, _ = monthly_prepost(wind_ds["tau_y"], m)
        mag_pre = np.hypot(tx_pre, ty_pre)
        mag_post = np.hypot(tx_post, ty_post)
        diff = mag_post - mag_pre
        pct = 100 * (np.nanmean(mag_post) - np.nanmean(mag_pre)) / np.nanmean(mag_pre)
        wind_diffs.append(diff)
        print(f"  {MONTH_NAMES[m-1]}: mean|tau| pre={np.nanmean(mag_pre):.4f} "
              f"post={np.nanmean(mag_post):.4f} pct={pct:+.1f}%")

    print("\n=== Convergence, monthly (ice-presence check) ===")
    conv_diffs = []
    for m in MONTHS:
        pre, post, vpre, vpost = monthly_prepost(conv_ds[CONV_VAR], m, negate=True)
        diff = post - pre
        conv_diffs.append(diff)
        print(f"  {MONTH_NAMES[m-1]}: valid_frac pre={vpre:.2f} post={vpost:.2f}  "
              f"mean_diff={np.nanmean(diff):.3e}  max|diff|={np.nanmax(np.abs(diff)):.3e}")

    vmax_w = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in wind_diffs])), 98)
    fig, axes = plt.subplots(3, 4, figsize=(18, 14),
                             subplot_kw={"projection": EASE_CRS})
    for i, m in enumerate(MONTHS):
        ax = axes.flat[i]
        im = ax.pcolormesh(x, y, wind_diffs[i], transform=EASE_CRS, cmap="RdBu_r",
                           vmin=-vmax_w, vmax=vmax_w, shading="auto")
        ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="none", zorder=4)
        ax.coastlines(resolution="50m", linewidth=0.4, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)
        ax.set_title(MONTH_NAMES[i], fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical",
                fraction=0.02, pad=0.02, label="\u0394 wind stress (Pa)")
    fig.suptitle("Wind stress magnitude difference, post \u2212 pre 2016, by month", fontsize=16)
    fig.savefig(OUT_WIND, dpi=140, bbox_inches="tight")
    print(f"\n-> {OUT_WIND}")

    vmax_c = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in conv_diffs])), 98)
    fig, axes = plt.subplots(3, 4, figsize=(18, 14),
                             subplot_kw={"projection": EASE_CRS})
    for i, m in enumerate(MONTHS):
        ax = axes.flat[i]
        im = ax.pcolormesh(x, y, conv_diffs[i], transform=EASE_CRS, cmap="RdBu_r",
                           vmin=-vmax_c, vmax=vmax_c, shading="auto")
        ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="none", zorder=4)
        ax.coastlines(resolution="50m", linewidth=0.4, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)
        ax.set_title(MONTH_NAMES[i], fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical",
                fraction=0.02, pad=0.02, label="\u0394 convergence (s\u207b\u00b9)")
    fig.suptitle("Convergence difference, post \u2212 pre 2016, by month (unmasked)", fontsize=16)
    fig.savefig(OUT_CONV, dpi=140, bbox_inches="tight")
    print(f"-> {OUT_CONV}")

    os.system(f"rclone mkdir {RCLONE_REMOTE}")
    os.system(f"rclone copy {OUT_WIND} {RCLONE_REMOTE}")
    os.system(f"rclone copy {OUT_CONV} {RCLONE_REMOTE}")
    print("uploaded both to supplementary.")


if __name__ == "__main__":
    main()
