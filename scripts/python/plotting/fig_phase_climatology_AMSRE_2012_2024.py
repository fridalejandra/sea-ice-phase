import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
from glob import glob
from matplotlib.colors import Normalize

# === CONFIGURATION === #
INPUT_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/"
SAVE_PATH = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/climatology/fig_phase_climatology_SMMR_1979_2024.png"
YEAR_START = 1979
YEAR_END = 2024

# === Load and stack valid files === #
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
    else:
        print(f"⚠️ Skipping {f} — missing {adv_var} or {ret_var}")

# === Safety check === #
if not advance_stack or not retreat_stack:
    raise RuntimeError("❌ No valid advance/retreat variables found to concatenate.")

advance_stack = xr.concat(advance_stack, dim="year")
retreat_stack = xr.concat(retreat_stack, dim="year")

advance_mean = advance_stack.mean(dim="year", skipna=True)
retreat_mean = retreat_stack.mean(dim="year", skipna=True)
duration_mean = retreat_mean - advance_mean

# === Wrap retreat for plotting === #
retreat_wrapped = retreat_mean.where(retreat_mean >= 100, retreat_mean + 365)

# === PLOTTING === #
fig, axs = plt.subplots(1, 3, figsize=(15, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})
data_list = [advance_mean, retreat_wrapped, duration_mean]
cbar_titles = ['Advance (DOY)', 'Retreat (DOY)', 'Duration (days)']

# === Colorbar settings === #
norms = [
    Normalize(vmin=32, vmax=274),
    Normalize(vmin=274, vmax=424),
    Normalize(vmin=0, vmax=300)
]
tick_locs = [
    [32, 60, 91, 121, 152, 182, 213, 244],   # Advance: Feb–Sep
    [274, 305, 335, 366, 395, 424],           # Retreat: Oct–Mar
    [50, 100, 150, 200, 250]                  # Duration
]
tick_labels = [
    ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
    ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'],
    ['50', '100', '150', '200', '250']
]

cmap = plt.cm.viridis.copy()
cmap.set_bad("white")

# === Plot each panel === #
for ax, data, norm, ticks, labels, title in zip(axs, data_list, norms, tick_locs, tick_labels, cbar_titles):
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    mesh = ax.pcolormesh(data.x, data.y, data, transform=ccrs.SouthPolarStereo(),
                         cmap=cmap, norm=norm, shading="auto")
    ax.add_feature(cfeature.LAND, zorder=100, facecolor="gray")
    ax.coastlines(linewidth=0.4)

    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    cbar.set_label(title, fontsize=9, labelpad=3)

    # Turn off axes
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=500, bbox_inches="tight")
plt.show()
