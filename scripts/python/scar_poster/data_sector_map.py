"""
fig_sector_map_poster.py

Compact, single-panel version of fig_sector_map.py, sized for a poster
corner thumbnail (paired with a separate legend, not on-map labels).
Reuses the real sector longitude boundaries from the Ch3 script.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings("ignore")
import subprocess

OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/poster/figures"
GDRIVE     = "gdrive:My Drive/scar_poster/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Canonical Raphael & Hobbs (2014) sector boundaries - matches the
# convention cited throughout the poster text.
SECTORS = {
    "Weddell":         {"lon_min": -65.0, "lon_max":  -25.0, "color": "#F44336"},  # red
    "King Haakon":     {"lon_min":  -25.0, "lon_max":  70.0, "color": "#FFC107"},  # yellow
    "East Antarctica": {"lon_min":  70.0, "lon_max": 165.0, "color": "#FF9800"},  # orange
    "Ross":            {"lon_min": 165.0, "lon_max": 250.0, "color": "#4CAF50"},  # green
    "ABS":             {"lon_min": 250.0, "lon_max": 295.0, "color": "#2196F3"},  # blue
}

ALPHA = 0.55


def sector_polygon(lon_min, lon_max, lat_min=-90, lat_max=-50, n=100):
    if lon_min < 0:
        lon_min += 360
    if lon_max < 0:
        lon_max += 360
    lons_top = np.linspace(lon_min, lon_max, n)
    lons_bot = np.linspace(lon_max, lon_min, n)
    lats_top = np.full(n, lat_max)
    lats_bot = np.full(n, lat_min)
    lons = np.concatenate([lons_top, lons_bot])
    lats = np.concatenate([lats_top, lats_bot])
    lons = np.where(lons > 180, lons - 360, lons)
    return lons, lats


fig = plt.figure(figsize=(3.2, 3.2))
ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -35], crs=ccrs.PlateCarree())

theta = np.linspace(0, 2 * np.pi, 100)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)
ax.set_boundary(circle, transform=ax.transAxes)

# light neutral land/ocean - reads clean at small thumbnail size
ax.add_feature(cfeature.OCEAN, color="#F1EFE8", zorder=0)
ax.add_feature(cfeature.LAND, color="#D3D1C7", zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#888780", zorder=3)

for name, props in SECTORS.items():
    lons, lats = sector_polygon(props["lon_min"], props["lon_max"])
    ax.fill(lons, lats, transform=ccrs.PlateCarree(),
            color=props["color"], alpha=ALPHA, zorder=1)
    ax.plot(np.append(lons, lons[0]), np.append(lats, lats[0]),
            transform=ccrs.PlateCarree(),
            color=props["color"], linewidth=1.0, alpha=0.9, zorder=3)

# longitude gridlines + labels every 30 deg, matching the reference style
ax.gridlines(draw_labels=False, linewidth=0.4, linestyle="--",
             color="#888780", alpha=0.7, zorder=4,
             xlocs=range(-180, 181, 30), ylocs=range(-80, -49, 10))

LON_LABEL_RADIUS_LAT = -40  # sits in the margin ring outside the sector fill (-50) and outside the frame's inner gridlines
for lon_deg in range(-180, 181, 30):
    if lon_deg == 180 or lon_deg == -180:
        label = "180"
    elif lon_deg == 0:
        label = "0"
    elif lon_deg > 0:
        label = f"{lon_deg}E"
    else:
        label = f"{abs(lon_deg)}W"
    ax.text(lon_deg, LON_LABEL_RADIUS_LAT, label,
            transform=ccrs.PlateCarree(),
            ha="center", va="center", fontsize=7, color="#52514e", zorder=5)

# no on-map sector-name labels - this is a thumbnail meant to be paired
# with a legend + the SIA/wind timeseries panel, not read alone

fpath = os.path.join(OUTPUT_DIR, "sector_map_poster.png")
fig.savefig(fpath, dpi=300, bbox_inches="tight", transparent=True)
plt.close()
print(f"Saved -> {fpath}")

result = subprocess.run(
    ["rclone", "copy", fpath, GDRIVE],
    capture_output=True, text=True
)
if result.returncode == 0:
    print(f"Synced -> {GDRIVE}")
else:
    print(f"rclone failed: {result.stderr.strip()}")