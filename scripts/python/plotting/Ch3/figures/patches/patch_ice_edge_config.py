"""
patch_ice_edge_config.py — adds paths/constants for the mean ice-edge contour
(Fig 2 enhancement, 2026-10-01). Run from figures/patches/.
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
    "ch3_config.py",
    [
        ("DAILY_RAW_CSV block",
         '''DAILY_RAW_CSV = os.path.join(RAW_DATA_DIR, "SIE_daily_sector_and_circumpolar_million_km2.csv")

# Detrended seasonal index table (written by processing/compute_atmospheric_correlations.py)
INDEX_CSV  = os.path.join(DATA_DIR, "master_index_detrended.csv")''',
         '''DAILY_RAW_CSV = os.path.join(RAW_DATA_DIR, "SIE_daily_sector_and_circumpolar_million_km2.csv")

# ── Gridded ice concentration (mean ice-edge contour, added 2026-10-01) ─────
# Source: Bootstrap SH concentration, NSIDC polar-stereographic 25km grid,
# 1978-11-01 to 2025-12-31 (15,589 daily timesteps, ~13GB -- never read this
# directly in a figure script; use MEAN_ICE_CONC_NC below instead, built once
# by compute/compute_mean_ice_edge_climatology.py).
MERGED_ICECON_NC = os.path.join(ROOT, "data", "merged", "merged_bootstrap_SH_latest.nc")
# Grid geometry / CRS reference for the same grid (cell areas; also where the
# exact NSIDC_SH_PS_PROJ parameters below were confirmed, via its `crs` var).
CELL_AREA_NC = os.path.join(ROOT, "data", "NSIDC0771_CellArea_PS_S25km_v1.1.nc")
# Small cached output (one float grid, ~1MB): the all-time mean concentration
# per cell, built once by compute_mean_ice_edge_climatology.py so figures
# never need to touch the 13GB source above.
MEAN_ICE_CONC_NC = os.path.join(DATA_DIR, "mean_ice_concentration_climatology.nc")

# NSIDC Sea Ice Polar Stereographic South (EPSG:3412), read from
# CELL_AREA_NC's `crs` variable -- lets cartopy reproject the concentration
# grid onto any other projection (SouthPolarStereo, PlateCarree, ...) without
# needing to compute lat/lon per cell by hand.
NSIDC_SH_PS_PROJ = dict(
    central_latitude=-90.0,
    central_longitude=0.0,
    true_scale_latitude=-70.0,
    semimajor_axis=6378273.0,
    semiminor_axis=6356889.449,
)

# Standard "ice edge" definition: the 15% concentration contour.
ICE_EDGE_CONCENTRATION = 0.15

# Detrended seasonal index table (written by processing/compute_atmospheric_correlations.py)
INDEX_CSV  = os.path.join(DATA_DIR, "master_index_detrended.csv")'''),
    ],
    skip_if_present="MEAN_ICE_CONC_NC = os.path.join(DATA_DIR,",
)

print("\nDone.")
