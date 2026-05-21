"""
fig_sector_map.py  (v3 — updated May 2026)
==========================================
Antarctic sea ice sector boundary map.

CHANGES FROM v2:
- Colors: blue, purple, orange, yellow, red
- Panel (b) taller, no sector labels (fills only)
- Ocean labels restored from v1 style, repositioned for extended lat range
- Continent tips kept

Two-panel figure:
  (a) — circumpolar polar stereographic projection
  (b) — Southern Ocean rectangular view with geographic context
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

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch3/figures"
GDRIVE     = "gdrive:results/Ch3_Figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Sector definitions ────────────────────────────────────────────────────────
SECTORS = {
    "Weddell"        : {"lon_min": -65.0, "lon_max": -25.0, "color": "#4A90D9"},  # blue
    "King Haakon"    : {"lon_min": -25.0, "lon_max":  70.0, "color": "#9B59B6"},  # purple
    "East Antarctica": {"lon_min":  70.0, "lon_max": 165.0, "color": "#E67E22"},  # orange
    "Ross"           : {"lon_min": 165.0, "lon_max": 250.0, "color": "#F1C40F"},  # yellow
    "ABS"            : {"lon_min": 250.0, "lon_max": 295.0, "color": "#E74C3C"},  # red
}

OCEAN_COLOR = "#EAF2FB"
LAND_COLOR  = "#D4CEBC"
COAST_COLOR = "#9A9080"
ALPHA       = 0.50


def sector_polygon(lon_min, lon_max, lat_min=-90, lat_max=-40, n=200):
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


# ── Figure layout ─────────────────────────────────────────────────────────────
# panel (a) takes top 52%, panel (b) takes bottom 40% with gap in between
fig = plt.figure(figsize=(8, 13))

ax_polar = fig.add_axes([0.05, 0.42, 0.90, 0.52],
                         projection=ccrs.SouthPolarStereo())
ax_rect  = fig.add_axes([0.05, 0.02, 0.90, 0.37],
                         projection=ccrs.PlateCarree(central_longitude=0))

# ── Panel (a): polar view ─────────────────────────────────────────────────────
ax_polar.set_extent([-180, 180, -90, -40], crs=ccrs.PlateCarree())

theta  = np.linspace(0, 2*np.pi, 100)
verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)
ax_polar.set_boundary(circle, transform=ax_polar.transAxes)

ax_polar.add_feature(cfeature.OCEAN,     color=OCEAN_COLOR, zorder=0)
ax_polar.add_feature(cfeature.LAND,      color=LAND_COLOR,  zorder=2)
ax_polar.add_feature(cfeature.COASTLINE, linewidth=0.5,
                     color=COAST_COLOR, zorder=3)

for name, props in SECTORS.items():
    lons, lats = sector_polygon(props["lon_min"], props["lon_max"])
    ax_polar.fill(lons, lats,
                  transform=ccrs.PlateCarree(),
                  color=props["color"], alpha=ALPHA, zorder=1)
    ax_polar.plot(np.append(lons, lons[0]),
                  np.append(lats, lats[0]),
                  transform=ccrs.PlateCarree(),
                  color=props["color"], linewidth=1.4,
                  alpha=0.95, zorder=3)

ax_polar.gridlines(draw_labels=False, linewidth=0.3,
                   color="white", alpha=0.7, zorder=4)

# Sector labels on polar panel
label_lons = {
    "Weddell"        : -45,
    "King Haakon"    :  22,
    "East Antarctica": 117,
    "Ross"           : -152,
    "ABS"            : -87,
}
for name, lon in label_lons.items():
    color = SECTORS[name]["color"]
    # yellow needs dark text
    txt_color = "#333333" if name == "Ross" else "white"
    ax_polar.text(lon, -54, name,
                  transform=ccrs.PlateCarree(),
                  ha="center", va="center",
                  fontsize=8.5, fontweight="bold",
                  color=txt_color,
                  bbox=dict(boxstyle="round,pad=0.25",
                            facecolor=color, alpha=0.90,
                            edgecolor="none"),
                  zorder=5)

ax_polar.set_title("(a)  Antarctic sea ice sectors",
                   fontsize=11, fontweight="bold", pad=8,
                   color="#2C2C2A")

# ── Panel (b): rectangular Southern Ocean view ────────────────────────────────
ax_rect.set_extent([-180, 180, -78, -35], crs=ccrs.PlateCarree())

ax_rect.add_feature(cfeature.OCEAN,     color=OCEAN_COLOR, zorder=0)
ax_rect.add_feature(cfeature.LAND,      color=LAND_COLOR,  zorder=2)
ax_rect.add_feature(
    cfeature.NaturalEarthFeature("physical", "land", "50m",
                                  facecolor=LAND_COLOR,
                                  edgecolor=COAST_COLOR,
                                  linewidth=0.5),
    zorder=2
)

# Sector fills only — no boundary lines, no labels
for name, props in SECTORS.items():
    lons, lats = sector_polygon(props["lon_min"], props["lon_max"],
                                lat_min=-78, lat_max=-38)
    ax_rect.fill(lons, lats,
                 transform=ccrs.PlateCarree(),
                 color=props["color"], alpha=ALPHA, zorder=1)

# Sector boundary lines only (no labels)
for name, props in SECTORS.items():
    lons, lats = sector_polygon(props["lon_min"], props["lon_max"],
                                lat_min=-78, lat_max=-38)
    ax_rect.plot(np.append(lons, lons[0]),
                 np.append(lats, lats[0]),
                 transform=ccrs.PlateCarree(),
                 color=props["color"], linewidth=1.2,
                 alpha=0.85, zorder=3)

# Gridlines
gl2 = ax_rect.gridlines(draw_labels=True, linewidth=0.3,
                         color="#AAAAAA", alpha=0.6, zorder=4,
                         x_inline=False, y_inline=False)
gl2.top_labels   = False
gl2.right_labels = False
gl2.xlocator     = plt.FixedLocator(range(-180, 181, 30))
gl2.ylocator     = plt.FixedLocator([-70, -60, -50, -40])
gl2.xlabel_style = {"size": 7.5, "color": "#555555"}
gl2.ylabel_style = {"size": 7.5, "color": "#555555"}

# Ocean basin labels — italicised, well-spaced
ocean_labels = [
    (-145, -57, "South Pacific Ocean"),
    ( -25, -62, "South Atlantic Ocean"),
    (  75, -62, "Indian Ocean"),
]
for lon, lat, label in ocean_labels:
    ax_rect.text(lon, lat, label,
                 transform=ccrs.PlateCarree(),
                 ha="center", va="center",
                 fontsize=8.5, fontstyle="italic",
                 color="#1a5276", zorder=6)



ax_rect.set_title("(b)  Southern Ocean context",
                   fontsize=11, fontweight="bold", pad=8,
                   color="#2C2C2A")

# ── Save ──────────────────────────────────────────────────────────────────────
fpath = os.path.join(OUTPUT_DIR, "fig_sector_map.png")
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