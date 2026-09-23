"""
fig_convergence_sector_map.py
Sector-resolution convergence change, using verified sector x season CSV.
FIXED: sectors are already abbreviated (ABS/EA/KHV/RA/WED), season already
in file, convergence = -div_negative (already signed negative).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs

IN_CSV = "ice_divergence_by_sector_season.csv"
SPLIT_YEAR = 2016
SEASONS_TO_PLOT = ["DJF", "MAM"]

SECTORS = {
    "WED": {"lon_min": -60.0, "lon_max":  20.0, "abbrev": "WS"},
    "KHV": {"lon_min":  20.0, "lon_max":  90.0, "abbrev": "KH"},
    "EA":  {"lon_min":  90.0, "lon_max": 160.0, "abbrev": "EA"},
    "RA":  {"lon_min": 160.0, "lon_max": 230.0, "abbrev": "RS"},
    "ABS": {"lon_min": 230.0, "lon_max": 300.0, "abbrev": "ABS"},
}

OUT = "fig_convergence_sector_map.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def sector_polygon(lon_min, lon_max, lat_min=-90, lat_max=-35, n=100):
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
    lon_min_n = lon_min % 360
    lon_max_n = lon_max % 360
    if lon_max_n <= lon_min_n:
        lon_max_n += 360
    mid = (lon_min_n + lon_max_n) / 2
    mid = mid % 360
    return mid - 360 if mid > 180 else mid


def main():
    d = pd.read_csv(IN_CSV, parse_dates=["date"])
    d["post"] = (d["date"].dt.year >= SPLIT_YEAR).astype(int)
    d["convergence"] = -d["div_negative"]

    values = {season: {} for season in SEASONS_TO_PLOT}
    for season in SEASONS_TO_PLOT:
        for sec in SECTORS.keys():
            sub = d[(d["sector"] == sec) & (d["season"] == season)]
            pre = sub[sub["post"] == 0]["convergence"].mean()
            post = sub[sub["post"] == 1]["convergence"].mean()
            diff = post - pre
            values[season][sec] = diff
            print(f"{season} {sec}: pre={pre:.4e} post={post:.4e} diff={diff:+.4e}")

    fig, axes = plt.subplots(1, len(SEASONS_TO_PLOT), figsize=(6 * len(SEASONS_TO_PLOT), 6.5),
                             subplot_kw={"projection": ccrs.SouthPolarStereo()})
    if len(SEASONS_TO_PLOT) == 1:
        axes = [axes]

    vmax = max(abs(v) for season in values.values() for v in season.values())
    cmap = plt.cm.RdBu_r

    for ax, season in zip(axes, SEASONS_TO_PLOT):
        ax.set_extent([-180, 180, -90, -35], crs=ccrs.PlateCarree())
        theta = np.linspace(0, 2 * np.pi, 100)
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * 0.5 + 0.5)
        ax.set_boundary(circle, transform=ax.transAxes)

        for sec, props in SECTORS.items():
            val = values[season][sec]
            color = cmap((val + vmax) / (2 * vmax))
            lons, lats = sector_polygon(props["lon_min"], props["lon_max"])
            ax.fill(lons, lats, transform=ccrs.PlateCarree(), color=color,
                    alpha=1.0, zorder=1)
            ax.plot(np.append(lons, lons[0]), np.append(lats, lats[0]),
                    transform=ccrs.PlateCarree(), color="white", linewidth=1.5, zorder=2)
            mid_lon = sector_midpoint_lon(props["lon_min"], props["lon_max"])
            ax.text(mid_lon, -65, f"{props['abbrev']}\n{val:+.2e}",
                    transform=ccrs.PlateCarree(), ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white", zorder=3)

        ax.set_title(season, fontsize=16, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.03, pad=0.05,
                label="mean convergence change (post \u2212 pre, s\u207b\u00b9)")

    fig.suptitle("Convergence change by sector \u2014 King Haakon MAM stands out",
                 fontsize=15, y=1.02)

    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()
