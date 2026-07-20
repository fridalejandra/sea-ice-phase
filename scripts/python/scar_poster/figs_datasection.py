""""""
Poster Section 2 figures - built from real data, not mockup values
=====================================================================

Produces two files:
  1. sector_map.png              - polar-stereographic map, 5 real sectors
  2. sia_wind_stacked_panels.png - SIA anomaly + wind stress anomaly by
                                   sector, stacked, 2016 marked

Requires: matplotlib, numpy, pandas, cartopy (for the real coastline map)
    pip install cartopy --break-system-packages   # if not already installed

Sector longitude boundaries follow Raphael & Hobbs (2014), the same
convention already used for your 5-sector mask:
    Weddell:              60W - 20E
    King Haakon VII:      20E  - 90E
    East Antarctica:      90E  - 160E
    Ross-Amundsen:        160E - 130W
    Amundsen-Bellingshausen: 130W - 60W

Adjust COLUMN NAMES / FILE PATHS in the CONFIG block below to match your
actual analysis_table_daily_anomaly.csv.
"""

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------
# CONFIG - adjust to match your actual files/columns
# ---------------------------------------------------------------------

ANALYSIS_TABLE_PATH = (
    '/user/geog/falejandraperez/sea-ice-phase/data/merged/'
    'analysis_table_daily_anomaly.csv'
)
DATE_COL = 'date'
SECTOR_COL = 'sector'
SIA_ANOMALY_COL = 'SIA_anomaly'           # deseasonalized SIA anomaly (km^2)
WIND_STRESS_ANOMALY_COL = 'wind_stress_anomaly'  # deseasonalized wind stress anomaly (N/m^2)
                                            # NOTE: this is the anomaly, for
                                            # THIS FIGURE ONLY - the regression
                                            # itself uses raw wind_stress, not
                                            # this column. Don't mix them up.
REGIME_SHIFT_YEAR = 2016

OUTPUT_DIR = '/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/poster/figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
GDRIVE = 'gdrive:My Drive/scar_poster/'

# Sector display order + colors (Cove categorical palette, matches
# earlier mockup so the poster stays visually consistent). Names match
# EXACTLY what's actually in your analysis_table_daily_anomaly.csv -
# confirmed from your deseasonalize script output ("King Haakon VII",
# "Ross-Amundsen", not the shorter "King Haakon"/"Ross" used elsewhere -
# e.g. the sector map script's abbreviations. Worth reconciling naming
# across the poster at some point, but this figure needs the exact match
# to filter the dataframe correctly.
SECTORS = ['Weddell', 'King Haakon VII', 'East Antarctica',
           'Ross-Amundsen', 'Amundsen-Bellingshausen']
SECTOR_COLORS = {
    'Weddell':                  '#2a78d6',
    'King Haakon VII':          '#eb6834',
    'East Antarctica':          '#1baf7a',
    'Ross-Amundsen':            '#eda100',
    'Amundsen-Bellingshausen':  '#e87ba4',
}

# Longitude boundaries per sector, degrees East, 0-360 convention
# (matches the convention fix already applied in build_forcing_sector_table.py)
SECTOR_LON_BOUNDS = {
    'Weddell':                  (300, 380),   # 60W (300E) to 20E (wraps past 360)
    'King Haakon VII':          (20, 90),
    'East Antarctica':          (90, 160),
    'Ross-Amundsen':            (160, 230),  # 160E to 130W (230E)
    'Amundsen-Bellingshausen':  (230, 300),  # 130W to 60W
}


# ---------------------------------------------------------------------
# Figure 1: real sector map
# ---------------------------------------------------------------------

def make_sector_map(outpath=OUTPUT_DIR + 'sector_map.png'):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("cartopy not installed - run: pip install cartopy --break-system-packages")
        return

    fig = plt.figure(figsize=(4, 4))
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    for sector in SECTORS:
        lon_start, lon_end = SECTOR_LON_BOUNDS[sector]
        # build a wedge from the pole out to the plotted latitude limit
        lons = np.linspace(lon_start, lon_end, 50)
        lons_wrapped = ((lons + 180) % 360) - 180  # to -180..180 for plotting
        lats_outer = np.full_like(lons, -50)
        # wedge polygon: pole -> outer arc -> back to pole
        poly_lons = np.concatenate([[lons_wrapped[0]], lons_wrapped, [lons_wrapped[-1]]])
        poly_lats = np.concatenate([[-90], lats_outer, [-90]])
        ax.fill(poly_lons, poly_lats, transform=ccrs.PlateCarree(),
                 color=SECTOR_COLORS[sector], alpha=0.55, edgecolor='none')

    ax.add_feature(cfeature.LAND, facecolor='#e8e6dd', zorder=2)
    ax.coastlines(resolution='50m', linewidth=0.4, zorder=3)
    ax.set_boundary(_polar_boundary_circle(ax), transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved {outpath}")


def _polar_boundary_circle(ax):
    """Clip the polar stereo axes to a circle instead of a square."""
    import matplotlib.path as mpath
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    return mpath.Path(verts * radius + center)


# ---------------------------------------------------------------------
# Figure 2: stacked SIA anomaly + wind stress anomaly by sector
# ---------------------------------------------------------------------

ROLLING_WINDOW_DAYS = 90  # smooths day-to-day/storm-scale noise so the
                           # underlying 2016 shift (or lack thereof, for
                           # wind) is actually visible. This is purely for
                           # THIS OVERVIEW FIGURE - the regression itself
                           # still uses unsmoothed daily values.


def make_stacked_panels(outpath=OUTPUT_DIR + 'sia_wind_stacked_panels.png'):
    df = pd.read_csv(ANALYSIS_TABLE_PATH, parse_dates=[DATE_COL])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    for sector in SECTORS:
        sub = df[df[SECTOR_COL] == sector].sort_values(DATE_COL).set_index(DATE_COL)
        color = SECTOR_COLORS[sector]

        sia_smooth = (sub[SIA_ANOMALY_COL] / 1e6).rolling(
            ROLLING_WINDOW_DAYS, min_periods=ROLLING_WINDOW_DAYS // 2, center=True
        ).mean()
        wind_smooth = sub[WIND_STRESS_ANOMALY_COL].rolling(
            ROLLING_WINDOW_DAYS, min_periods=ROLLING_WINDOW_DAYS // 2, center=True
        ).mean()

        axes[0].plot(sia_smooth.index, sia_smooth.values,
                      color=color, linewidth=1.1, alpha=0.9, label=sector)
        axes[1].plot(wind_smooth.index, wind_smooth.values,
                      color=color, linewidth=1.1, alpha=0.9, label=sector)

    shift_date = pd.Timestamp(f'{REGIME_SHIFT_YEAR}-01-01')
    for ax in axes:
        ax.axvline(shift_date, color='#52514e', linestyle='--', linewidth=1)
        ax.grid(True, color='#e1e0d9', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('SIA anomaly (10\u2076 km\u00b2)')
    axes[0].set_title(f'Sea ice area anomaly by sector ({ROLLING_WINDOW_DAYS}-day mean)',
                       fontsize=11, loc='left')
    axes[1].set_ylabel('Wind stress anomaly (N/m\u00b2)')
    axes[1].set_title(f'Wind stress anomaly by sector, deseasonalized ({ROLLING_WINDOW_DAYS}-day mean)',
                       fontsize=11, loc='left')

    axes[1].xaxis.set_major_locator(mdates.YearLocator(8))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved {outpath}")


# ---------------------------------------------------------------------

if __name__ == '__main__':
    make_sector_map()
    make_stacked_panels()

    result = subprocess.run(
        ["rclone", "copy", OUTPUT_DIR, GDRIVE],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"Synced -> {GDRIVE}")
    else:
        print(f"rclone failed: {result.stderr.strip()}")