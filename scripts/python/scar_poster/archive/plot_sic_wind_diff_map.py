"""
plot_sic_wind_diff_map.py

Reads the NetCDF from compute_gridded_sic_wind_diff.py and produces the
actual map figure: SIC difference as a filled contour/pcolormesh
(post-2016 minus pre-2016), wind stress difference vectors overlaid as
quiver arrows - the same visual grammar as Feba et al. 2026 Figure 2.

Saves to the same poster figures directory as the other poster scripts
and rclone-syncs it to the shared Drive folder, matching the pattern
already used in fig_sector_map_poster.py and build_poster_data_figures.py.
"""

import os
import subprocess
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings("ignore")

IN_NC = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_results/gridded_sic_wind_diff.nc"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/poster/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GDRIVE = "gdrive:My Drive/scar_poster/"

# subsample the quiver so arrows are readable, not a solid mat of ink -
# every Nth grid cell in each direction
QUIVER_STRIDE = 8


def make_diff_map(outpath=OUTPUT_DIR + "sic_wind_diff_map.png"):
    ds = xr.open_dataset(IN_NC)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())

    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)

    # SIC difference as a filled field, diverging colormap centered at 0
    sic_diff_pct = ds["SIC_diff"] * 100  # fractional -> percent, matches Feba's convention
    vmax = float(np.nanmax(np.abs(sic_diff_pct.values)))
    mesh = ax.pcolormesh(
        ds["lon"], ds["lat"], sic_diff_pct,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        shading="auto", zorder=1,
    )
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cbar.set_label("Difference in SIC [%]  (post-2016 minus pre-2016)")

    # wind stress difference vectors, subsampled
    lon_sub = ds["lon"].values[::QUIVER_STRIDE, ::QUIVER_STRIDE]
    lat_sub = ds["lat"].values[::QUIVER_STRIDE, ::QUIVER_STRIDE]
    u_sub = ds["tau_x_diff"].values[::QUIVER_STRIDE, ::QUIVER_STRIDE]
    v_sub = ds["tau_y_diff"].values[::QUIVER_STRIDE, ::QUIVER_STRIDE]

    q = ax.quiver(
        lon_sub, lat_sub, u_sub, v_sub,
        transform=ccrs.PlateCarree(),
        scale=0.01, scale_units="inches", width=0.0015,
        headwidth=2.5, headlength=3, headaxislength=2.5,
        color="#2b2a28", zorder=3,
    )
    # reference arrow calibrated to the actual MEDIAN magnitude (~0.0005
    # N/m^2), not an arbitrary round number - the field spans an 18x range
    # (median 0.0005, max 0.0092), so a single key can't represent both
    # ends, but anchoring to the median gives an honest sense of "typical"
    ax.quiverkey(q, X=0.85, Y=-0.05, U=0.0005, label="0.0005 N/m$^2$ (typical)",
                 labelpos="E", coordinates="axes", fontproperties={"size": 8})

    try:
        ax.add_feature(cfeature.LAND, facecolor="#e8e6dd", zorder=2)
        ax.coastlines(resolution="50m", linewidth=0.4, zorder=4)
    except Exception as e:
        print(f"Coastline/land features unavailable ({e}) - "
              f"figure will render without them, data is unaffected.")

    ax.gridlines(draw_labels=False, linewidth=0.3, linestyle="--",
                 color="#888780", alpha=0.6, zorder=5)

    ax.set_title("SIC difference and wind stress vector difference,\n"
                  "post-2016 minus pre-2016", fontsize=10)

    fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved {outpath}")
    return outpath


if __name__ == "__main__":
    fpath = make_diff_map()

    result = subprocess.run(
        ["rclone", "copy", fpath, GDRIVE],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"Synced -> {GDRIVE}")
    else:
        print(f"rclone failed: {result.stderr.strip()}")