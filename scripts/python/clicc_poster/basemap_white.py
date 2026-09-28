#!/usr/bin/env python
# ==================================================
# Antarctic sector reference map (black style)
# Cartopy, South Polar Stereo, round disk
# ==================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --------------------------------------------------
# Paths
# --------------------------------------------------
SECTOR_MASK_FILE = (
    "/user/geog/falejandraperez/sea-ice-phase/data/"
    "canonical_sectors.nc"
)

# --------------------------------------------------
# Load data
# --------------------------------------------------
ds = xr.open_dataset(SECTOR_MASK_FILE)

sector = ds["sector_id"]
lon    = ds["lon"]
lat    = ds["lat"]

sector_ids = [1, 2, 3, 4, 5]

# --------------------------------------------------
# Figure / projection
# --------------------------------------------------
proj = ccrs.SouthPolarStereo()
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=proj)

ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

# --------------------------------------------------
# TRUE circular boundary (Cartopy way)
# --------------------------------------------------
theta = np.linspace(0, 2*np.pi, 400)
center = np.array([0.5, 0.5])
radius = 0.5

verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

ax.set_boundary(circle, transform=ax.transAxes)

# --------------------------------------------------
# Dark background (ocean)
# --------------------------------------------------
ax.set_facecolor("#0b0b0b")   # near-black ocean

# --------------------------------------------------
# Land + coast (subtle)
# --------------------------------------------------
ax.add_feature(
    cfeature.LAND,
    facecolor="0.35",
    edgecolor="none",
    zorder=2,
)

ax.add_feature(
    cfeature.COASTLINE,
    linewidth=0.5,
    edgecolor="0.7",
    zorder=3,
)

# --------------------------------------------------
# Sector fills — single blue hue, high opacity
# --------------------------------------------------
for sid in sector_ids:
    mask = xr.where(sector == sid, 1, np.nan)

    ax.pcolormesh(
        lon,
        lat,
        mask,
        transform=ccrs.PlateCarree(),
        color="#1f6fb2",   # muted blue
        alpha=0.65,
        shading="nearest",  # IMPORTANT: kills checkerboard look
        zorder=4,
    )

# --------------------------------------------------
# Sector boundaries — cyan/teal like reference map
# --------------------------------------------------
ax.contour(
    lon,
    lat,
    sector,
    levels=[1.5, 2.5, 3.5, 4.5],
    colors="#00bcd4",
    linewidths=1.2,
    transform=ccrs.PlateCarree(),
    zorder=5,
)

# --------------------------------------------------
# Final formatting
# --------------------------------------------------
ax.axis("off")

plt.tight_layout(pad=0)
plt.savefig(
    "antarctic_sector_reference_map.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="black",
)
plt.show()
