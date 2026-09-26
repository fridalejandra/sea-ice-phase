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
BLOCK_SIZES = [75, 100, 150]
EASE_CELL_KM = 25.0
SEASONS = ["DJF", "MAM", "JJA", "SON"]
OUT_DIR = f"{REPO}/results/ch4/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def block_average_latlon(factor):
    ds = xr.open_dataset(DIV_PATH)
    lat = ds["lat"].coarsen({"x": factor, "y": factor}, boundary="trim").mean()
    lon = ds["lon"].coarsen({"x": factor, "y": factor}, boundary="trim").mean()
    return lat.values, lon.values

def load_all():
    ds = {}
    for km in BLOCK_SIZES:
        path = f"{REPO}/results/ch4/derived_nc/spatial_coupling_block{km}km.nc"
        d = xr.open_dataset(path)
        factor = int(round(km / EASE_CELL_KM))
        lat, lon = block_average_latlon(factor)
        if lat.shape != d["beta3"].isel(season=0).shape:
            print(f"  [warn] {km}km: lat/lon shape {lat.shape} != beta3 shape {d['beta3'].isel(season=0).shape} -- trimming")
            ny = min(lat.shape[0], d.sizes['y'])
            nx = min(lat.shape[1], d.sizes['x'])
            lat, lon = lat[:ny, :nx], lon[:ny, :nx]
            d = d.isel(y=slice(0, ny), x=slice(0, nx))
        d = d.assign_coords(lat2d=(("y", "x"), lat), lon2d=(("y", "x"), lon))
        ds[km] = d
        n_fin = int(np.isfinite(d["beta3"].values).sum())
        print(f"loaded {km} km: beta3 finite count {n_fin}")
    return ds

def plot_season(ds, season):
    fig = plt.figure(figsize=(15, 5.5))
    all_vals = np.concatenate([ds[km]["beta3"].sel(season=season).values.flatten() for km in BLOCK_SIZES])
    finite = all_vals[np.isfinite(all_vals)]
    vmax = np.nanpercentile(np.abs(finite), 95) if finite.size else 1.0
    for i, km in enumerate(BLOCK_SIZES):
        ax = fig.add_subplot(1, 3, i + 1, projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=1)
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=2)
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)
        field = ds[km]["beta3"].sel(season=season)
        lat2d = ds[km]["lat2d"].values
        lon2d = ds[km]["lon2d"].values
        im = ax.pcolormesh(lon2d, lat2d, field.values, transform=ccrs.PlateCarree(),
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", zorder=0)
        n_fin = int(np.isfinite(field.values).sum())
        ax.set_title(f"{km} km  (n={n_fin})")
    fig.suptitle(f"Delta sensitivity of sea-ice divergence to wind stress (post-2016 - pre-2016) - {season}", y=1.02)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.6])
    fig.colorbar(im, cax=cbar_ax, label="beta3 (s-1 Pa-1)")
    out = f"{OUT_DIR}/beta3_polar_blocksize_compare_{season}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")

if __name__ == "__main__":
    ds = load_all()
    for season in SEASONS:
        plot_season(ds, season)
    print("\nDone.")
