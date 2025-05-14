import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os

# === CONFIG === #
BASE_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/"
SAVE_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/anomalies_pairs/"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Define year pairs === #
pairs = [
    (2016, 2017),
    (2013, 2014),
    (2022, 2023),
]

# === Load grid === #
grid = xr.open_dataset("/Users/fridaperez/Developer/repos/sea-ice-phase/data/merged/SMMR_merged_1979_06302024.nc")
lon = grid["x"]
lat = grid["y"]

# === Loop over each pair === #
for retreat_year, advance_year in pairs:
    print(f"🔵 Plotting Retreat {retreat_year} anomaly and Advance {advance_year} timing...")

    # Load datasets
    ds_retreat = xr.open_dataset(f"{BASE_DIR}/seaice_phases_SMMR_{retreat_year}.nc")
    ds_advance = xr.open_dataset(f"{BASE_DIR}/seaice_phases_SMMR_{advance_year}.nc")

    retreat = ds_retreat[f"retreat_{retreat_year}"]
    advance = ds_advance[f"advance_{advance_year}"]

    # Build climatology across all years (1979–2024)
    retreat_files = sorted([f for f in os.listdir(BASE_DIR) if "seaice_phases_SMMR_" in f])
    retreat_stack = []
    for f in retreat_files:
        year = int(f.split("_")[-1].split(".")[0])
        if 1979 <= year <= 2024:
            temp = xr.open_dataset(os.path.join(BASE_DIR, f))
            var_name = f"retreat_{year}"
            if var_name in temp:
                retreat_stack.append(temp[var_name])
    retreat_clim = xr.concat(retreat_stack, dim="year").mean(dim="year", skipna=True)

    # Retreat Anomaly
    retreat_anomaly = retreat - retreat_clim

    # === PLOTTING === #
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})

    anomaly_norm = plt.Normalize(vmin=-50, vmax=50)
    advance_norm = plt.Normalize(vmin=32, vmax=274)

    cmap_anomaly = plt.cm.RdBu_r
    cmap_advance = plt.cm.viridis

    # Retreat anomaly
    axs[0].set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    mesh1 = axs[0].pcolormesh(lon, lat, retreat_anomaly, transform=ccrs.PlateCarree(),
                              cmap=cmap_anomaly, norm=anomaly_norm, shading="auto")
    axs[0].set_title(f"Retreat Anomaly {retreat_year}", fontsize=12)
    axs[0].add_feature(cfeature.LAND, zorder=100, facecolor="gray")
    axs[0].coastlines(linewidth=0.4)
    cbar1 = plt.colorbar(mesh1, ax=axs[0], orientation="horizontal", pad=0.05)
    cbar1.set_label("Retreat Anomaly (days)")

    # Advance
    axs[1].set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    mesh2 = axs[1].pcolormesh(lon, lat, advance, transform=ccrs.PlateCarree(),
                              cmap=cmap_advance, norm=advance_norm, shading="auto")
    axs[1].set_title(f"Advance Timing {advance_year}", fontsize=12)
    axs[1].add_feature(cfeature.LAND, zorder=100, facecolor="gray")
    axs[1].coastlines(linewidth=0.4)
    cbar2 = plt.colorbar(mesh2, ax=axs[1], orientation="horizontal", pad=0.05)
    cbar2.set_label("Advance Day of Year")

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"retreat{retreat_year}_advance{advance_year}.png")
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {save_path}")



