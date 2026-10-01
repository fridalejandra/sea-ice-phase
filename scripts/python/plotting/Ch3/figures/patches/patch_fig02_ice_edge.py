"""
patch_fig02_ice_edge.py — adds the mean ice-edge contour to fig_02_sector_map.py.
================================================================================
Draws the climatological (1978-11-01 to 2025-12-31) 0.15-concentration ice-edge
contour on BOTH panels, reprojected from its native NSIDC polar-stereographic
grid via a cartopy CRS built from ch3_config.NSIDC_SH_PS_PROJ. Reads the small
cached climatology (ch3_config.MEAN_ICE_CONC_NC) built by
compute/compute_mean_ice_edge_climatology.py -- never touches the 13GB source
directly. Line color is CBLIND_COLORS["blue"], unused by any sector in this
figure. Also updates the DEPENDENCIES docstring block and the sector legend.

Run from figures/patches/: python3 patch_fig02_ice_edge.py
Then: py_compile, git diff, run fig_02_sector_map.py.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.dirname(HERE)


def patch_file(relname, replacements, skip_if_present=None):
    path = os.path.join(FIGURES_DIR, relname)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if skip_if_present is not None and skip_if_present in content:
        print(f"Skipping {relname} — already patched (found {skip_if_present!r}).")
        return

    for label, old, new in replacements:
        n = content.count(old)
        if n != 1:
            raise SystemExit(
                f"{relname}: expected exactly 1 match for {label!r}, found {n}. "
                "File has drifted from what this patch expects -- aborting "
                "without writing anything to this file. Diff it by hand."
            )
        content = content.replace(old, new, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Patched {relname} ({len(replacements)} edit(s))")


patch_file(
    "fig_02_sector_map.py",
    [
        ("imports + CRS/data setup + draw_ice_edge() helper",
         '''import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ch3_config import SECTOR_LABELS, SECTOR_COLORS, SECTOR_BOUNDS_DEG, SECTOR_ORDER_BY_LONGITUDE
from ch3_plot import figsize, save
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    raise SystemExit(
        "cartopy is required for fig_sector_map.py (pip install cartopy / "
        "conda install -c conda-forge cartopy). See the DEPENDENCY note at "
        "the top of this file if you're on an offline cluster node."
    )''',
         '''import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.lines as mlines

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ch3_config import (
    SECTOR_LABELS, SECTOR_COLORS, SECTOR_BOUNDS_DEG, SECTOR_ORDER_BY_LONGITUDE,
    MEAN_ICE_CONC_NC, NSIDC_SH_PS_PROJ, ICE_EDGE_CONCENTRATION, CBLIND_COLORS,
)
from ch3_plot import figsize, save
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    raise SystemExit(
        "cartopy is required for fig_sector_map.py (pip install cartopy / "
        "conda install -c conda-forge cartopy). See the DEPENDENCY note at "
        "the top of this file if you're on an offline cluster node."
    )

# -- mean ice-edge contour (added 2026-10-01) ------------------------------------
# Climatological (1978-11-01 to 2025-12-31) mean sea-ice concentration, on its
# native NSIDC polar-stereographic grid -- built once by
# compute/compute_mean_ice_edge_climatology.py from the 13GB source, so this
# figure only ever reads the small (~1MB) cached result below. The 0.15
# contour is the standard "ice edge" definition. NSIDC_CRS lets cartopy
# reproject this straight onto either panel's own projection (SouthPolarStereo
# / PlateCarree) without computing lat/lon per cell by hand.
if not os.path.exists(MEAN_ICE_CONC_NC):
    raise SystemExit(
        f"Not found: {MEAN_ICE_CONC_NC}\n"
        "Run compute/compute_mean_ice_edge_climatology.py first (one-off, "
        "reads the 13GB merged concentration file and caches a small result)."
    )
_ice_ds = xr.open_dataset(MEAN_ICE_CONC_NC)
ICE_X = _ice_ds["x"].values
ICE_Y = _ice_ds["y"].values
ICE_MEAN_CONC = _ice_ds["mean_concentration"].values
_ice_ds.close()
ICE_X2D, ICE_Y2D = np.meshgrid(ICE_X, ICE_Y)

NSIDC_CRS = ccrs.Stereographic(
    central_latitude=NSIDC_SH_PS_PROJ["central_latitude"],
    central_longitude=NSIDC_SH_PS_PROJ["central_longitude"],
    true_scale_latitude=NSIDC_SH_PS_PROJ["true_scale_latitude"],
    globe=ccrs.Globe(semimajor_axis=NSIDC_SH_PS_PROJ["semimajor_axis"],
                      semiminor_axis=NSIDC_SH_PS_PROJ["semiminor_axis"]),
)
ICE_EDGE_COLOR = CBLIND_COLORS["blue"]


def draw_ice_edge(ax, zorder=4.5):
    """Climatological mean ice-edge contour (0.15 concentration), reprojected
    from its native NSIDC grid onto ax's own projection. zorder=4.5 sits just
    above coastlines (4) so it reads clearly, below the sector-boundary
    dashed meridians (6) and their labels (7) so those stay legible on top."""
    ax.contour(ICE_X2D, ICE_Y2D, ICE_MEAN_CONC, levels=[ICE_EDGE_CONCENTRATION],
               transform=NSIDC_CRS, colors=[ICE_EDGE_COLOR], linewidths=1.3,
               linestyles="-", zorder=zorder)'''),
        ("docstring DEPENDENCIES block",
         '''Data:   none from the pipeline -- sector/ocean boundaries are hardcoded
        constants in ch3_config.py (SECTOR_BOUNDS_DEG, OCEAN_BOUNDS_DEG),
        not derived from any CSV. Coastline/land geometry comes from
        cartopy's Natural Earth shapefiles (downloaded via pooch on first
        use; needs internet -- see the DEPENDENCY / WHERE TO RUN THIS note
        above).
Tables: none -- reads no results/ch3/tables/*.csv and writes none.''',
         '''Data:   sector/ocean boundaries are hardcoded constants in ch3_config.py
        (SECTOR_BOUNDS_DEG, OCEAN_BOUNDS_DEG), not derived from any CSV.
        Coastline/land geometry comes from cartopy's Natural Earth
        shapefiles (downloaded via pooch on first use; needs internet --
        see the DEPENDENCY / WHERE TO RUN THIS note above).
        Added 2026-10-01: the mean ice-edge contour reads
        ch3_config.MEAN_ICE_CONC_NC (data/ch3/mean_ice_concentration_
        climatology.nc), a small cached climatology built once by
        compute/compute_mean_ice_edge_climatology.py from
        ch3_config.MERGED_ICECON_NC (Bootstrap SH concentration, ~13GB --
        never read directly by this script). Re-run that compute script
        only if the source concentration file is updated.
Tables: none -- reads no results/ch3/tables/*.csv and writes none.'''),
        ("polar panel: draw ice edge after coastlines",
         '''    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8e4d8", zorder=3)
    ax.coastlines(resolution="50m", linewidth=0.8, color="#3a3a3a", zorder=4)''',
         '''    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8e4d8", zorder=3)
    ax.coastlines(resolution="50m", linewidth=0.8, color="#3a3a3a", zorder=4)
    draw_ice_edge(ax)'''),
        ("rect panel: draw ice edge after coastlines",
         '''    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8e4d8", zorder=3)
    ax.coastlines(resolution="50m", linewidth=0.7, color="#3a3a3a", zorder=4)''',
         '''    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8e4d8", zorder=3)
    ax.coastlines(resolution="50m", linewidth=0.7, color="#3a3a3a", zorder=4)
    draw_ice_edge(ax)'''),
        ("legend: add ice-edge line handle",
         '''    handles = [mpatches.Patch(facecolor=SECTOR_COLORS[s], edgecolor=SECTOR_COLORS[s],
                               alpha=0.7, label=SECTOR_LABELS[s])
               for s in SECTOR_ORDER_BY_LONGITUDE]
    fig.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               frameon=False, fontsize=10)''',
         '''    handles = [mpatches.Patch(facecolor=SECTOR_COLORS[s], edgecolor=SECTOR_COLORS[s],
                               alpha=0.7, label=SECTOR_LABELS[s])
               for s in SECTOR_ORDER_BY_LONGITUDE]
    handles.append(mlines.Line2D([], [], color=ICE_EDGE_COLOR, lw=1.3,
                                  label="Mean ice edge (1979\u20132025)"))
    fig.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               frameon=False, fontsize=10)'''),
    ],
    skip_if_present="draw_ice_edge(ax, zorder=4.5)",
)

print("\nDone. Now: py_compile, git diff, then run fig_02_sector_map.py.")
