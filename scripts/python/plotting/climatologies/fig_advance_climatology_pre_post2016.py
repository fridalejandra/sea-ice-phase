import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# === Load updated cross-year phase dataset === #
ds = xr.open_dataset("/Users/fridaperez/Developer/repos/sea-ice-phase/data/phase_SMMR_1980_2023_combined.nc")

# === Extract advance variables === #
advance_vars = [v for v in ds.data_vars if "advance_" in v]

# === Split into pre- and post-2016 === #
advance_pre = [ds[v] for v in advance_vars if int(v.split("_")[-1]) <= 2015]
advance_post = [ds[v] for v in advance_vars if int(v.split("_")[-1]) >= 2016]

# === Stack and compute mean === #
advance_pre_stack = xr.concat(advance_pre, dim="year")
advance_post_stack = xr.concat(advance_post, dim="year")

advance_pre_mean = advance_pre_stack.mean(dim="year", skipna=True).where((advance_pre_stack.mean(dim="year") >= 1) & (advance_pre_stack.mean(dim="year") <= 181))
advance_post_mean = advance_post_stack.mean(dim="year", skipna=True).where((advance_post_stack.mean(dim="year") >= 1) & (advance_post_stack.mean(dim="year") <= 181))

# === Difference (Post - Pre) === #
advance_diff = advance_post_mean - advance_pre_mean

# === Plotting === #
fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})
titles = ["Advance (1980–2015)", "Advance (2016–2023)", "Δ Advance (Post - Pre)"]
data_list = [advance_pre_mean, advance_post_mean, advance_diff]

# Color scale setup
vmin = 1
vmax = 181
diff_lim = 30

cmaps = [plt.cm.viridis, plt.cm.viridis, plt.cm.coolwarm]
vmins = [vmin, vmin, -diff_lim]
vmaxs = [vmax, vmax, diff_lim]
tick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

for ax, data, title, cmap, vmin_, vmax_ in zip(axs, data_list, titles, cmaps, vmins, vmaxs):
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    mesh = ax.pcolormesh(data.x, data.y, data, transform=ccrs.SouthPolarStereo(),
                         cmap=cmap, vmin=vmin_, vmax=vmax_, shading='auto')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.add_feature(cfeature.LAND, zorder=100, facecolor='gray')
    ax.coastlines(linewidth=0.4)
    ax.spines['geo'].set_visible(False)

    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05)
    if "Δ" in title:
        cbar.set_label("Change in Days", fontsize=10)
    else:
        cbar.set_ticks(np.linspace(vmin, vmax, len(tick_labels)))
        cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

plt.tight_layout()
plt.savefig("/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/fig_advance_pre_post2016_updated.png", dpi=300, bbox_inches="tight")
plt.show()
