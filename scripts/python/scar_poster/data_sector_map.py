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
import matplotlib.patheffects as pe
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
    "Weddell":         {"lon_min": -60.0, "lon_max":  20.0, "color": "#F44336", "abbrev": "WS"},
    "King Haakon":     {"lon_min":  20.0, "lon_max":  90.0, "color": "#FFC107", "abbrev": "KH"},
    "East Antarctica": {"lon_min":  90.0, "lon_max": 160.0, "color": "#FF9800", "abbrev": "EA"},
    "Ross":            {"lon_min": 160.0, "lon_max": 230.0, "color": "#4CAF50", "abbrev": "RS"},
    "ABS":             {"lon_min": 230.0, "lon_max": 300.0, "color": "#2196F3", "abbrev": "ABS"},
}

ALPHA = 0.55


def sector_polygon(lon_min, lon_max, lat_min=-90, lat_max=-42, n=100):
    # Normalize both bounds to 0-360 first, then detect wraparound
    # generically (lon_max <= lon_min after normalization means the
    # sector crosses 0deg/360deg, e.g. Weddell's -60 to 20). This fixes
    # a bug where only lon_min was checked for negativity, causing
    # np.linspace to sweep the wrong (280deg) way around for any sector
    # straddling the prime meridian.
    lon_min = lon_min % 360
    lon_max = lon_max % 360
    if lon_max <= lon_min:
        lon_max += 360

    lons_top = np.linspace(lon_min, lon_max, n)
    lons_bot = np.linspace(lon_max, lon_min, n)
    lats_top = np.full(n, lat_max)
    lats_bot = np.full(n, lat_min)
    lons = np.concatenate([lons_top, lons_bot])
    lats = np.concatenate([lats_top, lats_bot])
    lons = lons % 360
    lons = np.where(lons > 180, lons - 360, lons)
    return lons, lats


def sector_midpoint_lon(lon_min, lon_max):
    """Wraparound-safe midpoint longitude for label placement."""
    lon_min_n = lon_min % 360
    lon_max_n = lon_max % 360
    if lon_max_n <= lon_min_n:
        lon_max_n += 360
    mid = (lon_min_n + lon_max_n) / 2
    mid = mid % 360
    return mid - 360 if mid > 180 else mid


ADD_MEAN_SIA_OUTLINE = True  # set False to skip if you haven't computed mean_sic yet
MEAN_SIC_PATH = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/"
    "merged_bootstrap_SH_latest.nc"
)
SIC_LEVEL = 15  # standard 15% SIC ice-edge threshold, matches Ch2 convention
MAX_EXTENT_MONTH = 9  # September = canonical Antarctic SIE annual maximum
                       # (Parkinson & Cavalieri, Parkinson 2019). Check
                       # against your own record if you want the exact
                       # month of max extent rather than the canonical one.


def add_mean_sia_outline(ax, nc_path=MEAN_SIC_PATH, level=SIC_LEVEL,
                          month=MAX_EXTENT_MONTH):
    """
    Overlays the climatological winter-maximum sea ice edge as a black
    contour line only (no fill), at the standard 15% SIC threshold.
    "Winter max" = climatological mean SIC for `month` across all years
    (default September, the canonical Antarctic SIE annual maximum),
    not an all-months/all-time average.

    Expects an xarray-readable file with a SIC variable and 2D (or 1D,
    auto-broadcast) longitude/latitude coordinates. Adjust VAR_NAME,
    LON_NAME, LAT_NAME below to match your actual merged file's variable
    names - these are guesses based on typical NSIDC Bootstrap conventions
    and may need correcting against your real file.
    """
    try:
        import xarray as xr
    except ImportError:
        print("xarray not installed - skipping mean SIA outline "
              "(pip install xarray --break-system-packages)")
        return

    VAR_NAME = "SIC"        # <- confirm against your actual merged file
    LON_NAME = "longitude"  # <- confirm against your actual merged file
    LAT_NAME = "latitude"   # <- confirm against your actual merged file
    TIME_NAME = "time"      # <- confirm against your actual merged file

    try:
        ds = xr.open_dataset(nc_path)
        winter_only = ds[VAR_NAME].sel(
            {TIME_NAME: ds[TIME_NAME].dt.month == month}
        )
        mean_sic = winter_only.mean(dim=TIME_NAME, skipna=True)
        lon2d = ds[LON_NAME].values
        lat2d = ds[LAT_NAME].values
        if lon2d.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon2d, lat2d)

        ax.contour(lon2d, lat2d, mean_sic.values, levels=[level],
                   colors="black", linewidths=1.1,
                   transform=ccrs.PlateCarree(), zorder=6)
    except Exception as e:
        print(f"Could not add mean SIA outline: {e}")
        print("Check VAR_NAME / LON_NAME / LAT_NAME / TIME_NAME against your actual file.")


fig = plt.figure(figsize=(3.6, 3.6))
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

    # abbreviated sector name, mid-wedge, wraparound-safe
    mid_lon = sector_midpoint_lon(props["lon_min"], props["lon_max"])
    ax.text(mid_lon, -65, props["abbrev"],
            transform=ccrs.PlateCarree(),
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="#2b2a28", zorder=7,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

if ADD_MEAN_SIA_OUTLINE:
    add_mean_sia_outline(ax)

# longitude gridlines + labels every 30 deg, matching the reference style
ax.gridlines(draw_labels=False, linewidth=0.4, linestyle="--",
             color="#888780", alpha=0.7, zorder=4,
             xlocs=range(-180, 181, 30), ylocs=range(-80, -49, 10))

# labels placed genuinely outside the frame: clip_on=False lets them
# render past the circular boundary patch instead of competing with the
# sector fill for the same ring of space
LON_LABEL_LAT = -32
for lon_deg in range(-180, 181, 30):
    if lon_deg == 180 or lon_deg == -180:
        label = "180"
    elif lon_deg == 0:
        label = "0"
    elif lon_deg > 0:
        label = f"{lon_deg}E"
    else:
        label = f"{abs(lon_deg)}W"
    ax.text(lon_deg, LON_LABEL_LAT, label,
            transform=ccrs.PlateCarree(), clip_on=False,
            ha="center", va="center", fontsize=7, color="#52514e", zorder=5)

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