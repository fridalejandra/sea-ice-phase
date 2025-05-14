import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
from glob import glob
from matplotlib.colors import Normalize
from matplotlib.path import Path
import matplotlib.patches as mpatches

# === CONFIG === #
INPUT_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/"
SAVE_PATH = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/climatology/fig_phase_climatology_SMMR_2012_2024_round.png"
YEAR_START = 2012
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

if not advance_stack or not retreat_stack:
    raise RuntimeError("❌ No valid advance/retreat variables found to concatenate.")

advance_stack = xr.concat(advance_stack, dim="year")
retreat_stack = xr.concat(retreat_stack, dim="year")

advance_mean = advance_stack.mean(dim="year", skipna=True)
retreat_mean = retreat_stack.mean(dim="year", skipna=True)
duration_mean = retreat_mean - advance_mean

# === Wrap RETREAT for plotting === #
retreat_wrapped = retreat_mean.where(retreat_mean >= 100, retreat_mean + 365)

# === Coordinates === #
grid = xr.open_dataset("/Users/fridaperez/Developer/repos/sea-ice-phase/data/merged/SMMR_merged_1979_06302024.nc")
lat = grid["y"]
lon = grid["x"]

# === PLOTTING === #
fig, axs = plt.subplots(1, 3, figsize=(15, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})
data_list = [advance_mean, retreat_wrapped, duration_mean]
cbar_titles = ['Advance (DOY)', 'Retreat (DOY)', 'Duration (days)']

norms = [
    Normalize(vmin=32, vmax=274),
    Normalize(vmin=274, vmax=424),
    Normalize(vmin=0, vmax=300)
]
tick_locs = [
    [32, 60, 91, 121, 152, 182, 213, 244],
    [274, 305, 335, 366, 395, 424],
    [50, 100, 150, 200, 250]
]
tick_labels = [
    ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
    ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'],
    ['50', '100', '150', '200', '250']
]

cmap = plt.cm.viridis.copy()
cmap.set_bad("white")

# === Create circular boundary === #
theta = np.linspace(0, 2 * np.pi, 100)
circle_verts = np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5
circle_path = Path(circle_verts)

# === Plotting each panel === #
for ax, data, norm, ticks, labels, title in zip(axs, data_list, norms, tick_locs, tick_labels, cbar_titles):
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.set_boundary(circle_path, transform=ax.transAxes)
    mesh = ax.pcolormesh(lon, lat, data, transform=ccrs.SouthPolarStereo(),
                         cmap=cmap, norm=norm, shading="auto")
    ax.add_feature(cfeature.LAND, zorder=100, facecolor="gray")
    ax.coastlines(linewidth=0.4)
    ax.set_title("", fontsize=1)  # Hide titles

    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    cbar.set_label(title, fontsize=9, labelpad=3)

plt.tight_layout()
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=500, bbox_inches="tight")
plt.show()
