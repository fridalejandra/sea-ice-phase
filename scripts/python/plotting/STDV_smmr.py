import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
from glob import glob

# === CONFIG === #
INPUT_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/"
MERGED_GRID = "/Users/fridaperez/Developer/repos/sea-ice-phase/data/merged/SMMR_merged_1979_06302024.nc"
SAVE_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/variability/"
YEAR_START = 1979
YEAR_END = 2024

# === Load and stack === #
files = sorted(glob(f"{INPUT_DIR}/seaice_phases_SMMR_*.nc"))
advance_stack, retreat_stack = [], []

for f in files:
    year = os.path.basename(f).split("_")[-1].split(".")[0]
    if not year.isdigit() or not (YEAR_START <= int(year) <= YEAR_END):
        continue

    ds = xr.open_dataset(f)
    adv_var = f"advance_{year}"
    ret_var = f"retreat_{year}"
    if adv_var in ds and ret_var in ds:
        advance_stack.append(ds[adv_var])
        retreat_stack.append(ds[ret_var])

if not advance_stack or not retreat_stack:
    raise RuntimeError("No valid advance/retreat variables found to concatenate.")

advance_stack = xr.concat(advance_stack, dim="year")
retreat_stack = xr.concat(retreat_stack, dim="year")

# === Calculate Means and Standard Deviations === #
advance_std = advance_stack.std(dim="year", skipna=True)
retreat_std = retreat_stack.std(dim="year", skipna=True)
duration_std = (retreat_stack - advance_stack).std(dim="year", skipna=True)

# === Load grid === #
grid = xr.open_dataset(MERGED_GRID)
lat = grid["y"]
lon = grid["x"]

# === Helper plotting function === #
def plot_std_map(data, title, save_name, vmin=0, vmax=60):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    mesh = ax.pcolormesh(lon, lat, data, transform=ccrs.SouthPolarStereo(),
                         cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    ax.add_feature(cfeature.LAND, facecolor="gray", zorder=100)
    ax.coastlines(linewidth=0.4)
    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.07)
    cbar.set_label("Standard Deviation (days)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    ax.set_title(title, fontsize=12, fontweight="bold")
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig.savefig(os.path.join(SAVE_DIR, save_name), dpi=300, bbox_inches="tight")
    plt.close()

# === Plot and save === #
plot_std_map(advance_std, f"Advance Timing Variability ({YEAR_START}-{YEAR_END})", "Advance_std_SMMR.png")
plot_std_map(retreat_std, f"Retreat Timing Variability ({YEAR_START}-{YEAR_END})", "Retreat_std_SMMR.png")
plot_std_map(duration_std, f"Duration Variability ({YEAR_START}-{YEAR_END})", "Duration_std_SMMR.png")

print("✅ Standard deviation plots saved successfully!")
