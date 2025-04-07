import xarray as xr
import numpy as np
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
from matplotlib.colors import BoundaryNorm

# Directory where your phase NetCDFs are stored
phase_dir = "/user/geog/falejandraperez/sea-ice-phase/results/bootstrap_smmr/"
files = sorted(glob.glob(f"{phase_dir}/seaice_phases_SMMR_*.nc"))

# Load all freeze_start DataArrays into a list
da_list = [xr.open_dataset(f).freeze_start for f in files]

# Stack into a single DataArray along a new time dim
dstack = xr.concat(da_list, dim="year")
dstack["year"] = [int(f[-8:-3]) for f in files]  # Parse year from filename

# Compute climatology (mean over years)
freeze_clim = dstack.mean(dim="year", skipna=True)

# Plot settings
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -55], ccrs.PlateCarree())
ax.coastlines(resolution='110m', linewidth=0.8)
ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black', facecolor='lightgray')
ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', linestyle='--')

# Define color levels for Day-of-Year
levels = np.arange(0, 366, 30)
cmap = cm.viridis
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# Plot
freeze_plot = ax.pcolormesh(freeze_clim.x, freeze_clim.y, freeze_clim,
                             transform=ccrs.SouthPolarStereo(), cmap=cmap, norm=norm)
cbar = plt.colorbar(freeze_plot, orientation='horizontal', pad=0.05, aspect=50, shrink=0.7)
cbar.set_label('Freeze Start Day of Year')
cbar.set_ticks(levels)

plt.title("Mean Freeze Start (Climatology)", fontsize=14)
plt.tight_layout()
plt.show()
