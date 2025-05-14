import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# === Load dataset ===
ds = xr.open_dataset("/Users/fridaperez/Developer/repos/sea-ice-phase/data/phase_SMMR_1980_2023_combined.nc")

# Extract retreat and advance variables
retreat_vars = [v for v in ds.data_vars if "retreat_" in v]
advance_vars = [v for v in ds.data_vars if "advance_" in v]

# Pre-2016
retreat_pre = [ds[v] for v in retreat_vars if int(v.split("_")[-1]) <= 2015]
advance_pre = [ds[v] for v in advance_vars if int(v.split("_")[-1]) <= 2015]

# Post-2016
retreat_post = [ds[v] for v in retreat_vars if int(v.split("_")[-1]) >= 2016]
advance_post = [ds[v] for v in advance_vars if int(v.split("_")[-1]) >= 2016]

# Stack and compute duration (apply masking based on valid DOY ranges)
retreat_pre_stack = xr.concat(retreat_pre, dim="year")
advance_pre_stack = xr.concat(advance_pre, dim="year")
duration_pre = (retreat_pre_stack.where((retreat_pre_stack >= 274) & (retreat_pre_stack <= 424)) -
                advance_pre_stack.where((advance_pre_stack >= 1) & (advance_pre_stack <= 181))).mean(dim="year", skipna=True)

retreat_post_stack = xr.concat(retreat_post, dim="year")
advance_post_stack = xr.concat(advance_post, dim="year")
duration_post = (retreat_post_stack.where((retreat_post_stack >= 274) & (retreat_post_stack <= 424)) -
                 advance_post_stack.where((advance_post_stack >= 1) & (advance_post_stack <= 181))).mean(dim="year", skipna=True)

duration_diff = duration_post - duration_pre

# === Plotting ===
fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})
titles = ["Duration (1980–2015)", "Duration (2016–2023)", "Δ Duration (Post - Pre)"]
data_list = [duration_pre, duration_post, duration_diff]

cmaps = [plt.cm.inferno, plt.cm.inferno, plt.cm.coolwarm]
vmins = [0, 0, -40]
vmaxs = [300, 300, 40]

for ax, data, title, cmap, vmin_, vmax_ in zip(axs, data_list, titles, cmaps, vmins, vmaxs):
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    mesh = ax.pcolormesh(data.x, data.y, data, transform=ccrs.SouthPolarStereo(),
                         cmap=cmap, vmin=vmin_, vmax=vmax_, shading='auto')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.coastlines(linewidth=0.4)
    ax.add_feature(cfeature.LAND, zorder=100, facecolor='gray')
    ax.spines['geo'].set_visible(False)

    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05)
    if "Δ" in title:
        cbar.set_label("Change in Duration (days)", fontsize=10)
    else:
        cbar.set_label("Melt Season Duration (days)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

plt.tight_layout()
plt.savefig("/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/fig_duration_pre_post2016.png", dpi=300, bbox_inches="tight")
plt.show()
