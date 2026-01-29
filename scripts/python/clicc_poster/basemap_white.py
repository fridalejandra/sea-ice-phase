#!/usr/bin/env python
# --------------------------------------------------
# Reference map: Antarctic sector definitions
# Poster-safe, abbreviations only
# --------------------------------------------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ------------------------
# Paths (cluster)
# ------------------------
SECTOR_MASK_FILE = (
    "/user/geog/falejandraperez/sea-ice-phase/data/"
    "canonical_sectors.nc"
)

# ------------------------
# Load data
# ------------------------
ds = xr.open_dataset(SECTOR_MASK_FILE)

sector = ds["sector"]   # integer mask
lon    = ds["lon"]
lat    = ds["lat"]

# ------------------------
# Sector definitions
# ------------------------
sector_ids = [1, 2, 3, 4, 5]

sector_abbr = {
    1: "ABS",
    2: "WED",
    3: "KH",
    4: "EA",
    5: "RA",
}

sector_colors = {
    1: "#1b9e77",  # ABS
    2: "#66a61e",  # WED
    3: "#7570b3",  # KH
    4: "#e6ab02",  # EA
    5: "#d95f02",  # RA
}

# Manually tuned for visual centering (poster)
label_positions = {
    1: (-115, -72),  # ABS
    2: (-35,  -66),  # WED
    3: (40,   -66),  # KH
    4: (110,  -72),  # EA
    5: (165,  -77),  # RA
}

# ------------------------
# Figure setup
# ------------------------
proj = ccrs.SouthPolarStereo()
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=proj)

ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor="0.45", zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)

# ------------------------
# Plot sector fills
# ------------------------
for sid in sector_ids:
    mask = xr.where(sector == sid, 1, np.nan)

    ax.pcolormesh(
        lon,
        lat,
        mask,
        transform=ccrs.PlateCarree(),
        color=sector_colors[sid],
        alpha=0.30,
        shading="auto",
        zorder=3,
    )

# ------------------------
# Sector boundaries
# ------------------------
ax.contour(
    lon,
    lat,
    sector,
    levels=[1.5, 2.5, 3.5, 4.5],
    colors="0.3",
    linewidths=0.8,
    transform=ccrs.PlateCarree(),
    zorder=4,
)

# ------------------------
# Abbreviation labels
# ------------------------
for sid, (x, y) in label_positions.items():
    ax.text(
        x,
        y,
        sector_abbr[sid],
        transform=ccrs.PlateCarree(),
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
        color="white",
        zorder=5,
    )

# ------------------------
# Final formatting
# ------------------------
ax.axis("off")

plt.tight_layout()
plt.savefig(
    "antarctic_sector_reference_map.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()
