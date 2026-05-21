"""
fig_sector_map.py  (v4 — reverted to original, legend removed)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings("ignore")
import subprocess

OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch3/figures"
GDRIVE     = "gdrive:results/Ch3_Figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SECTORS = {
    "Weddell"        : {"lon_min": -65.0,  "lon_max": -25.0,  "color": "#2196F3"},
    "King Haakon"    : {"lon_min": -25.0,  "lon_max":  70.0,  "color": "#9C27B0"},
    "East Antarctica": {"lon_min":  70.0,  "lon_max": 165.0,  "color": "#FF9800"},
    "Ross"           : {"lon_min": 165.0,  "lon_max": 250.0,  "color": "#4CAF50"},
    "ABS"            : {"lon_min": 250.0,  "lon_max": 295.0,  "color": "#F44336"},
}

LAT_MIN = -90.0
LAT_MAX = -40.0
ALPHA   = 0.45


def sector_polygon(lon_min, lon_max, lat_min=-90, lat_max=-40, n=100):
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


fig = plt.figure(figsize=(8, 11))

ax_polar = fig.add_axes([0.05, 0.40, 0.90, 0.52],
                         projection=ccrs.SouthPolarStereo())
ax_rect  = fig.add_axes([0.05, 0.04, 0.90, 0.34],
                         projection=ccrs.PlateCarree(central_longitude=0))

# ── Panel (a): polar ──────────────────────────────────────────────────────────
ax_polar.set_extent([-180, 180, -90, -40], crs=ccrs.PlateCarree())

theta  = np.linspace(0, 2*np.pi, 100)
verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)
ax_polar.set_boundary(circle, transform=ax_polar.transAxes)

ax_polar.add_feature(cfeature.OCEAN,     color="#D6EAF8", zorder=0)
ax_polar.add_feature(cfeature.LAND,      color="#2C2C2A", zorder=2)
ax_polar.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#888888", zorder=3)

for name, props in SECTORS.items():
    lons, lats = sector_polygon(props["lon_min"], props["lon_max"])
    ax_polar.fill(lons, lats, transform=ccrs.PlateCarree(),
                  color=props["color"], alpha=ALPHA, zorder=1)
    ax_polar.plot(np.append(lons, lons[0]), np.append(lats, lats[0]),
                  transform=ccrs.PlateCarree(),
                  color=props["color"], linewidth=1.2, alpha=0.9, zorder=3)

ax_polar.gridlines(draw_labels=False, linewidth=0.4,
                   color="white", alpha=0.6, zorder=4)

label_lons = {
    "Weddell"        : -45,
    "King Haakon"    :  22,
    "East Antarctica": 117,
    "Ross"           : -152,
    "ABS"            : -87,
}
for name, lon in label_lons.items():
    color = SECTORS[name]["color"]
    ax_polar.text(lon, -52, name,
                  transform=ccrs.PlateCarree(),
                  ha="center", va="center",
                  fontsize=8.5, fontweight="bold", color="white",
                  bbox=dict(boxstyle="round,pad=0.2",
                            facecolor=color, alpha=0.85, edgecolor="none"),
                  zorder=5)

ax_polar.set_title("(a)  Antarctic sea ice sectors",
                   fontsize=11, fontweight="bold", pad=8)

# ── Panel (b): rectangular ────────────────────────────────────────────────────
ax_rect.set_extent([-180, 180, -80, -40], crs=ccrs.PlateCarree())

ax_rect.add_feature(cfeature.OCEAN,     color="#D6EAF8", zorder=0)
ax_rect.add_feature(cfeature.LAND,      color="#2C2C2A", zorder=2)
ax_rect.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#888888", zorder=3)

for name, props in SECTORS.items():
    lons, lats = sector_polygon(props["lon_min"], props["lon_max"],
                                lat_min=-80, lat_max=-40)
    ax_rect.fill(lons, lats, transform=ccrs.PlateCarree(),
                 color=props["color"], alpha=ALPHA, zorder=1)
    ax_rect.plot(np.append(lons, lons[0]), np.append(lats, lats[0]),
                 transform=ccrs.PlateCarree(),
                 color=props["color"], linewidth=1.2, alpha=0.9, zorder=3)

gl2 = ax_rect.gridlines(draw_labels=True, linewidth=0.4,
                         color="grey", alpha=0.5, zorder=4,
                         x_inline=False, y_inline=False)
gl2.top_labels   = False
gl2.right_labels = False
gl2.xlocator     = plt.FixedLocator(range(-180, 181, 30))
gl2.ylocator     = plt.FixedLocator([-70, -60, -50])
gl2.xlabel_style = {"size": 8}
gl2.ylabel_style = {"size": 8}

ocean_labels = [
    (-150, -45, "Pacific Ocean"),
    ( -30, -45, "Atlantic Ocean"),
    (  70, -45, "Indian Ocean"),
]
for lon, lat, label in ocean_labels:
    ax_rect.text(lon, lat, label,
                 transform=ccrs.PlateCarree(),
                 ha="center", va="center",
                 fontsize=9, fontstyle="italic",
                 color="#1a5276", zorder=6)

ax_rect.set_title("(b)  Southern Ocean view",
                   fontsize=11, fontweight="bold", pad=8)

# ── Save ──────────────────────────────────────────────────────────────────────
fpath = os.path.join(OUTPUT_DIR, "fig02_sector_map.png")
fig.savefig(fpath, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {fpath}")

result = subprocess.run(
    ["rclone", "copy", fpath, GDRIVE],
    capture_output=True, text=True
)
if result.returncode == 0:
    print(f"✓ Synced → {GDRIVE}")
else:
    print(f"✗ rclone failed: {result.stderr.strip()}")

print("Done.")