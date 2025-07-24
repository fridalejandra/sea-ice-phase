import os
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# === USER CONFIGURATION === #
phase = "retreat"
years = range(2012, 2024)
output_dir = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/sensor_bias/"
smmr_base = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/seaice_phases_SMMR_{year}.nc"
amsre_base = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/AMSRE_phase/seaice_phases_AMSRE_{year}.nc"

smmr_list = []
amsre_list = []

# === LOAD + COARSEN === #
for year in years:
    varname = f"{phase}_{year}"
    try:
        ds_smmr = xr.open_dataset(smmr_base.format(year=year))
        ds_amsre = xr.open_dataset(amsre_base.format(year=year))

        smmr = ds_smmr[varname].load()
        amsre = ds_amsre[varname].load()

        # Match grid shape
        ny, nx = amsre.shape
        if ny % 2 != 0:
            amsre = amsre.isel(y=slice(0, ny - 1))
        if nx % 2 != 0:
            amsre = amsre.isel(x=slice(0, nx - 1))
        amsre_coarse = amsre.coarsen(y=2, x=2, boundary="trim").mean()

        # Append
        smmr_list.append(smmr)
        amsre_list.append(amsre_coarse)

    except Exception as e:
        print(f"Skipping {year} due to error: {e}")

# === STACK INTO 3D ARRAYS === #
smmr_stack = xr.concat(smmr_list, dim="year")
amsre_stack = xr.concat(amsre_list, dim="year")

# === INITIALIZE CORRELATION ARRAY === #
ny, nx = smmr_stack.shape[1], smmr_stack.shape[2]
correlation_map = np.full((ny, nx), np.nan)

# === LOOP THROUGH PIXELS AND COMPUTE CORRELATION === #
for j in range(ny):
    for i in range(nx):
        smmr_ts = smmr_stack[:, j, i].values
        amsre_ts = amsre_stack[:, j, i].values

        valid_mask = ~np.isnan(smmr_ts) & ~np.isnan(amsre_ts)
        if np.sum(valid_mask) >= 5:
            r, _ = pearsonr(smmr_ts[valid_mask], amsre_ts[valid_mask])
            correlation_map[j, i] = r

# === WRAP IN XARRAY === #
correlation_da = xr.DataArray(
    data=correlation_map,
    coords={"x": smmr_stack.x, "y": smmr_stack.y},
    dims=("y", "x")
)

# === PLOT === #
def plot_corr(corr, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})
    im = ax.pcolormesh(
        corr.x, corr.y, corr,
        transform=ccrs.SouthPolarStereo(),
        cmap="PuOr", vmin=-1, vmax=1, shading="auto"
    )
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, zorder=100, facecolor="gray")
    ax.coastlines(linewidth=0.4)
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_label("Pearson r (Phase Timing)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

# === SAVE CORRELATION MAP === #
corr_path = os.path.join(output_dir, f"correlation_map_{phase}_2012-2023.png")
plot_corr(correlation_da, f"{phase.capitalize()} Timing Correlation (2012–2023)", corr_path)
