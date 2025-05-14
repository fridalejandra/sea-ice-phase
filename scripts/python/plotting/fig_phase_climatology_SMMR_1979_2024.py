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
OUT_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/yearly_phase_maps/"
YEAR_START, YEAR_END = 1979, 2023
os.makedirs(OUT_DIR, exist_ok=True)

# === Plot Settings === #
norms = [
    Normalize(vmin=32, vmax=274),    # Advance
    Normalize(vmin=274, vmax=424),   # Retreat (wrapped)
    Normalize(vmin=0, vmax=300)      # Duration
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
cbar_titles = ['ADVANCE (DOY)', 'RETREAT (DOY)', 'DURATION (days)']
cmap = plt.cm.viridis.copy()
cmap.set_bad("white")

# === Circular Mask === #
theta = np.linspace(0, 2 * np.pi, 100)
circle_verts = np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5
circle_path = Path(circle_verts)

# === Loop through years === #
for f in sorted(glob(f"{INPUT_DIR}/seaice_phases_SMMR_*.nc")):
    year = os.path.basename(f).split("_")[-1].split(".")[0]
    if not year.isdigit() or not (YEAR_START <= int(year) <= YEAR_END):
        continue

    ds = xr.open_dataset(f)
    adv_var, ret_var = f"advance_{year}", f"retreat_{year}"
    if adv_var not in ds or ret_var not in ds:
        print(f"⚠️ Skipping {year}: missing variables")
        continue

    # Load and process variables
    advance = ds[adv_var].where(ds[adv_var] >= 32)
    retreat = ds[ret_var]
    retreat_wrapped = retreat.where(retreat >= 100, retreat + 365)
    duration = retreat_wrapped - advance

    data_list = [advance, retreat_wrapped, duration]

    # === Plotting === #
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})

    for ax, data, norm, ticks, labels, cbar_label in zip(axs, data_list, norms, tick_locs, tick_labels, cbar_titles):
        ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
        ax.set_boundary(circle_path, transform=ax.transAxes)
        mesh = ax.pcolormesh(data.x, data.y, data, transform=ccrs.SouthPolarStereo(),
                             cmap=cmap, norm=norm, shading="auto")
        ax.add_feature(cfeature.LAND, zorder=100, facecolor="gray")
        ax.coastlines(linewidth=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")  # Optional: add title here if desired

        cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(False)
        cbar.set_label(cbar_label, fontsize=9, labelpad=3)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"phase_map_{year}.png")
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")
