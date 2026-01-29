#!/usr/bin/env python
# ==================================================
# Antarctic sector reference map
# - True circular polar disk
# - Sector masks from cluster
# - Distinct colors, ~70% opacity
# - No labels (added manually later)
# ==================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --------------------------------------------------
# Paths (cluster)
# --------------------------------------------------
SECTOR_MASK_FILE = (
    "/user/geog/falejandraperez/sea-ice-phase/data/"
    "canonical_sectors.nc"
)

# --------------------------------------------------
# Load data
# --------------------------------------------------
ds = xr.open_dataset(SECTOR_MASK_FILE)

sector = ds["sector"]   # integer sector IDs
lon    = ds["lon"]
lat    = ds["lat"]

# --------------------------------------------------
# Sector definitions
# --------------------------------------------------
sector_ids = [1, 2, 3, 4, 5]

sector_colors = {
    1: "#1f78b4",  # ABS – blue
    2: "#33a02c",  # WED – green
    3: "#6a3d9a",  # KH – purple
    4: "#ff7f00",  # EA – orange
    5: "#e31a1c",  # RA – red
}

# --------------------------------------------------
# Figure / projection
# --------------------------------------------------
proj = ccrs.SouthPolarStereo()
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=proj)

ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

# --------------------------------------------------
# Force circular boundary (TRUE polar disk)
# --------------------------------------------------
theta = np.linspace(0, 2 * np.pi, 300)
center = np.array([0.5, 0.5])
radius = 0.5

verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

ax.set_boundary(circle, transform=ax.transAxes)

# --------------------------------------------------
# Background features
# --------------------------------------------------
ax.add_feature(
    cfeature.LAND,
    facecolor="0.65",   # light, non-dominant
    edgecolor="none",
    zorder=1,
)

ax.add_feature(
    cfeature.COASTLINE,
    linewidth=0.6,
    zorder=2,
)

# --------------------------------------------------
# Plot sector fills (distinct colors, high opacity)
# --------------------------------------------------
for sid in sector_ids:
    mask = xr.where(sector == sid, 1, np.nan)

    ax.pcolormesh(
        lon,
        lat,
        mask,
        transform=ccrs.PlateCarree(),
        color=sector_colors[sid],
        alpha=0.70,          # key choice
        shading="auto",
        zorder=3,
    )

# --------------------------------------------------
# Sector boundaries (quiet, not dominant)
# --------------------------------------------------
ax.contour(
    lon,
    lat,
    sector,
    levels=[1.5, 2.5, 3.5, 4.5],
    colors="0.25",
    linewidths=0.7,
    transform=ccrs.PlateCarree(),
    zorder=4,
)

# --------------------------------------------------
# Final formatting
# --------------------------------------------------
ax.axis("off")

plt.tight_layout()
plt.savefig(
    "antarctic_sector_reference_map_round.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()
