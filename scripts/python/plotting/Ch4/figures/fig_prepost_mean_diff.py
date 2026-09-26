import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

REPO = "/user/geog/falejandraperez/sea-ice-phase"
DIV_PATH = f"{REPO}/results/ch4/derived_nc/ease_divergence_with_latlon.nc"
OUT_DIR = f"{REPO}/results/ch4/figures"
os.makedirs(OUT_DIR, exist_ok=True)

REGIME_SHIFT_YEAR = 2016
EXCLUDE_YEARS = [1978]
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}

def season_year(time_da, months):
    y = time_da.dt.year
    if 12 in months:
        y = y + (time_da.dt.month == 12).astype(int)
    return y

def load(var_name):
    ds = xr.open_dataset(DIV_PATH)
    da = ds[var_name]
    if var_name == "div_negative":
        da = -da
    yrs = da["time"].dt.year
    da = da.sel(time=~yrs.isin(EXCLUDE_YEARS))
    return da, ds["lat"], ds["lon"]

def pre_post_diff(da, months):
    sub = da.sel(time=da["time"].dt.month.isin(months))
    sy = season_year(sub["time"], months)
    n_pre = int((sy.values < REGIME_SHIFT_YEAR).sum())
    n_post = int((sy.values >= REGIME_SHIFT_YEAR).sum())
    print(f"    n_pre={n_pre} n_post={n_post}", flush=True)
    if n_pre == 0 or n_post == 0:
        print("    [WARN] empty pre or post group -- skipping", flush=True)
        return None, None, None
    pre_mask = sy < REGIME_SHIFT_YEAR
    post_mask = sy >= REGIME_SHIFT_YEAR
    pre_mean = sub.where(pre_mask, drop=False).mean(dim="time", skipna=True)
    post_mean = sub.where(post_mask, drop=False).mean(dim="time", skipna=True)
    return pre_mean.values, post_mean.values, (post_mean - pre_mean).values

def plot_var(var_name, label):
    print(f"\n=== {var_name} ({label}) ===", flush=True)
    da, lat, lon = load(var_name)
    diffs = {}
    for season, months in SEASONS.items():
        print(f"  {season}:", flush=True)
        pre, post, diff = pre_post_diff(da, months)
        diffs[season] = diff
    finite_vals = np.concatenate([d.ravel()[np.isfinite(d.ravel())] for d in diffs.values() if d is not None])
    vmax = np.nanpercentile(np.abs(finite_vals), 98) if finite_vals.size else 1.0
    fig = plt.figure(figsize=(18, 5))
    for i, season in enumerate(SEASONS):
        ax = fig.add_subplot(1, 4, i + 1, projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=1)
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=2)
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)
        if diffs[season] is not None:
            im = ax.pcolormesh(lon.values, lat.values, diffs[season], transform=ccrs.PlateCarree(),
                               cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", zorder=0)
        ax.set_title(season)
    fig.suptitle(f"{label}: post-2016 minus pre-2016 mean (native 25 km)", y=1.02)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.6])
    fig.colorbar(im, cax=cbar_ax, label="Delta (s-1)")
    out = f"{OUT_DIR}/prepost_mean_diff_{var_name}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}", flush=True)

if __name__ == "__main__":
    plot_var("divergence", "Net divergence")
    plot_var("div_positive", "Divergence (opening)")
    plot_var("div_negative", "Convergence (closing, shown positive)")
    print("\nDone.")
