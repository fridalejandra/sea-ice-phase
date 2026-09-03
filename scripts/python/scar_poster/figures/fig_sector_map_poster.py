"""
fig_sector_map_poster.py

Sector map for the poster. Sectors drawn as thin longitude strips to avoid
dateline-wrapping polygon bugs. Fully opaque, distinct colors matching all
other poster figures.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings("ignore")

SECTORS = {
    "Weddell":         {"lon_min": -60.0, "lon_max":  20.0, "color": "#F44336", "abbrev": "WS"},
    "King Haakon":     {"lon_min":  20.0, "lon_max":  90.0, "color": "#FFC107", "abbrev": "KH"},
    "East Antarctica": {"lon_min":  90.0, "lon_max": 160.0, "color": "#FF9800", "abbrev": "EA"},
    "Ross":            {"lon_min": 160.0, "lon_max": 230.0, "color": "#4CAF50", "abbrev": "RS"},
    "ABS":             {"lon_min": 230.0, "lon_max": 300.0, "color": "#2196F3", "abbrev": "ABS"},
}

OUT = "fig_sector_map_poster.png"
RCLONE_REMOTE = "gdrive:scar_poster/"

MEAN_SIC_PATH = ("/user/geog/falejandraperez/sea-ice-phase/data/merged/"
                 "merged_bootstrap_SH_latest.nc")
SIC_LEVEL = 0.15
MAX_EXTENT_MONTH = 9
ADD_MEAN_SIA_OUTLINE = True

NSIDC_SH_CRS = ccrs.Stereographic(
    central_latitude=-90, central_longitude=0, true_scale_latitude=-70,
    globe=ccrs.Globe(semimajor_axis=6378273, semiminor_axis=6356889.449),
)


def draw_sector(ax, lon_min, lon_max, color, lat_min=-90, lat_max=-35):
    """Fill a sector as many thin longitude strips — avoids dateline
    polygon-wrapping bugs entirely."""
    lon_min = lon_min % 360
    lon_max = lon_max % 360
    if lon_max <= lon_min:
        lon_max += 360
    edges = np.linspace(lon_min, lon_max, 60)
    for a, b in zip(edges[:-1], edges[1:]):
        a_ = ((a + 180) % 360) - 180
        b_ = ((b + 180) % 360) - 180
        if b_ < a_:  # strip crosses dateline; split it
            ax.fill([a_, 180, 180, a_],
                    [lat_max, lat_max, lat_min, lat_min],
                    transform=ccrs.PlateCarree(), color=color, lw=0,
                    alpha=0.55, zorder=1)
            ax.fill([-180, b_, b_, -180],
                    [lat_max, lat_max, lat_min, lat_min],
                    transform=ccrs.PlateCarree(), color=color, lw=0,
                    alpha=0.55, zorder=1)
        else:
            ax.fill([a_, b_, b_, a_],
                    [lat_max, lat_max, lat_min, lat_min],
                    transform=ccrs.PlateCarree(), color=color, lw=0,
                    alpha=0.55, zorder=1)


def sector_midpoint_lon(lon_min, lon_max):
    lon_min_n = lon_min % 360
    lon_max_n = lon_max % 360
    if lon_max_n <= lon_min_n:
        lon_max_n += 360
    mid = (lon_min_n + lon_max_n) / 2
    mid = mid % 360
    return mid - 360 if mid > 180 else mid


def add_mean_sia_outline(ax):
    try:
        import xarray as xr
        ds = xr.open_dataset(MEAN_SIC_PATH)
        icecon_vars = [v for v in ds.data_vars if v.endswith("_ICECON")]
        if not icecon_vars:
            print(f"No *_ICECON variable found. Available: {list(ds.data_vars)}")
            return
        sic = ds[icecon_vars[0]]
        for v in icecon_vars[1:]:
            sic = sic.combine_first(ds[v])
        sic = sic.where(sic <= 1.0)
        winter = sic.sel(time=ds["time"].dt.month == MAX_EXTENT_MONTH)
        mean_sic = winter.mean(dim="time", skipna=True)
        ax.contour(ds["x"].values, ds["y"].values, mean_sic.values,
                   levels=[SIC_LEVEL], colors="black", linewidths=1.5,
                   transform=NSIDC_SH_CRS, zorder=6)
    except Exception as e:
        print(f"Could not add SIA outline: {e}")


def main():
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96],
                      projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -35], crs=ccrs.PlateCarree())

    # circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="0.92", edgecolor="none", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color="0.25", zorder=5)

    # sectors — strip fill, fully opaque
    for name, props in SECTORS.items():
        draw_sector(ax, props["lon_min"], props["lon_max"], props["color"])

        # radial white boundary line at each sector's start
        lm = ((props["lon_min"] + 180) % 360) - 180
        ax.plot([lm, lm], [-90, -35], transform=ccrs.PlateCarree(),
                color="white", linewidth=1.5, zorder=2)

        # label
        mid_lon = sector_midpoint_lon(props["lon_min"], props["lon_max"])
        ax.text(mid_lon, -65, props["abbrev"],
                transform=ccrs.PlateCarree(),
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="white", zorder=7,
                path_effects=[pe.withStroke(linewidth=3, foreground="0.2")])

    if ADD_MEAN_SIA_OUTLINE:
        add_mean_sia_outline(ax)

    ax.gridlines(draw_labels=False, linewidth=0.3, linestyle="--",
                 color="0.5", alpha=0.5, zorder=4,
                 xlocs=range(-180, 181, 30), ylocs=range(-80, -49, 10))

    for lon_deg in range(-150, 181, 30):
        if lon_deg == 180 or lon_deg == -180:
            label = "180"
        elif lon_deg == 0:
            label = "0"
        elif lon_deg > 0:
            label = f"{lon_deg}E"
        else:
            label = f"{abs(lon_deg)}W"
        ax.text(lon_deg, -32, label,
                transform=ccrs.PlateCarree(), clip_on=False,
                ha="center", va="center", fontsize=8, color="0.3", zorder=8)

    fig.savefig(OUT, dpi=300, bbox_inches="tight",
                transparent=False, facecolor="white")
    plt.close()
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()