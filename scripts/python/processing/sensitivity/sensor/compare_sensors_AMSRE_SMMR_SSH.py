import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# === USER CONFIGURATION === #
phase = "retreat"
years = range(2012, 2024)
output_dir = f"/user/geog/falejandraperez/sea-ice-phase/results/figures/sensor_bias_{phase}/"
os.makedirs(output_dir, exist_ok=True)

# === PLOTTING FUNCTION === #
def plot_bias_map(bias, title, save_path, vmin=-20, vmax=20):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})
    im = ax.pcolormesh(
        bias.x, bias.y, bias,
        transform=ccrs.SouthPolarStereo(),
        cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto"
    )
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, zorder=100, facecolor="gray")
    ax.coastlines(linewidth=0.4)
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_label("Bias (AMSRE - SMMR, days)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

# === WRAPPED DIFFERENCE FUNCTION === #
def wrapped_difference(amsre, smmr):
    raw_diff = amsre - smmr
    return ((raw_diff + 183) % 366) - 183

# === CLIMATOLOGY DICTIONARY === #
clim_dict = {}

# === MAIN LOOP OVER YEARS === #
for year in years:
    varname = f"{phase}_{year}"
    smmr_path = f"/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase/seaice_phases_SMMR_{year}.nc"
    amsre_path = f"/user/geog/falejandraperez/sea-ice-phase/results/AMSRE_phase/seaice_phases_AMSRE_{year}.nc"

    try:
        ds_smmr = xr.open_dataset(smmr_path)
        ds_amsre = xr.open_dataset(amsre_path)

        smmr = ds_smmr[varname].load()
        amsre = ds_amsre[varname].load()

        ny, nx = amsre.shape
        if ny % 2 != 0:
            amsre = amsre.isel(y=slice(0, ny - 1))
        if nx % 2 != 0:
            amsre = amsre.isel(x=slice(0, nx - 1))

        amsre_coarse = amsre.coarsen(y=2, x=2, boundary="trim").mean()

        bias = xr.DataArray(
            data=wrapped_difference(amsre_coarse.values, smmr.values),
            coords={"x": smmr.x, "y": smmr.y},
            dims=("y", "x")
        )

        title = f"{phase.capitalize()} Wrapped Bias (AMSRE - SMMR) — {year}"
        save_path = os.path.join(output_dir, f"bias_map_wrapped_{phase}_{year}.png")
        plot_bias_map(bias, title, save_path)

        clim_dict[year] = bias

    except Exception as e:
        print(f"Skipping {year} due to error: {e}")

# === CLIMATOLOGY MEAN BIAS MAP === #
if clim_dict:
    bias_stack = xr.concat(list(clim_dict.values()), dim="year")
    mean_bias = bias_stack.mean(dim="year")
    clim_path = os.path.join(output_dir, f"bias_map_wrapped_{phase}_climatology.png")
    plot_bias_map(mean_bias, f"{phase.capitalize()} Mean Wrapped Bias (2012–2023)", clim_path)

# === HISTOGRAM === #
bias_stack = xr.concat(list(clim_dict.values()), dim="year")
flat_bias = bias_stack.values.flatten()
valid_bias = flat_bias[~np.isnan(flat_bias)]

plt.figure(figsize=(6, 4))
plt.hist(valid_bias, bins=50, color="steelblue", edgecolor="black")
plt.axvline(0, linestyle="--", color="black", linewidth=1)
plt.title(f"Histogram of Wrapped {phase.capitalize()} Timing Differences (AMSRE - SMMR)\n2012–2023")
plt.xlabel("Wrapped DOY Difference")
plt.ylabel("Number of Pixels")
plt.tight_layout()
hist_path = os.path.join(output_dir, f"histogram_wrapped_bias_{phase}.png")
plt.savefig(hist_path, dpi=300)
plt.close()

# === UPLOAD WITH RCLONE === #
gdrive_path = "gdrive:sea-ice-phase/results/figures/sensor_bias_retreat/"

print("\nUploading climatology map to Google Drive...")
os.system(f"rclone copy '{clim_path}' '{gdrive_path}'")

print("Uploading histogram to Google Drive...")
os.system(f"rclone copy '{hist_path}' '{gdrive_path}'")

print("Uploading all yearly bias maps to Google Drive...")
os.system(f"rclone copy '{output_dir}' '{gdrive_path}' --include 'bias_map_wrapped_{phase}_*.png'")

print("✅ All plots uploaded successfully.")
