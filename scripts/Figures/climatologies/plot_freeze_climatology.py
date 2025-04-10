import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import os
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from pathlib import Path

# ---- CONFIG ----
phase_dir = "/data/bootstrap_smmr/test_downloads/"
output_path = "/figures/sea_ice_phase_climatologies.png"
files = sorted(glob.glob(os.path.join(phase_dir, "seaice_phases_SMMR_*.nc")))

phases = ['advance', 'retreat', 'early_melt']
titles = ['ADVANCE', 'RETREAT', 'EARLY MELT']


# ---- Compute Climatologies ----
climatologies = {}

for phase in phases:
    da_list = []
    for f in files:
        try:
            ds = xr.open_dataset(f)
            da = ds[phase]
            da_list.append(da)
        except Exception as e:
            print(f"Skipping {f} due to error with {phase}: {e}")
    if da_list:
        stacked = xr.concat(da_list, dim="year")
        clim = stacked.mean(dim="year", skipna=True)
        climatologies[phase] = clim

# ---- Plotting Setup ----
fig, axs = plt.subplots(1, 3, figsize=(20, 9), subplot_kw={'projection': ccrs.SouthPolarStereo()})
levels = np.arange(0, 366, 30)
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
cmap = cm.viridis
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

im = None  # placeholder for colorbar reference

for i, (phase, ax) in enumerate(zip(phases, axs)):
    if phase not in climatologies:
        continue
    data = climatologies[phase]

    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.8)
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black', facecolor='lightgray')
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', linestyle='--')

    im = ax.pcolormesh(data['x'], data['y'], data, transform=ccrs.SouthPolarStereo(), cmap=cmap, norm=norm)
    ax.set_title(titles[i], fontsize=15, fontweight='bold', fontfamily='sans-serif')

# ---- Colorbar ----
if im is not None:
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal', pad=0.07, aspect=40, shrink=0.9)
    cbar.set_ticks(levels)
    cbar.set_ticklabels(month_labels)
    cbar.set_label("DAY OF YEAR", fontsize=12, fontweight='bold', fontfamily='sans-serif')

# ---- Title + Save ----
plt.suptitle("SEA ICE PHASE CLIMATOLOGIES (1980–2024)", fontsize=18, fontweight='bold', fontfamily='sans-serif')
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"✅ Saved: {output_path}")
plt.show()
