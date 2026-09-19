"""
fig_sector_map.py — Figure 2: the five sectors, redrawn post label-fix.

Two panels, one figure:
  (a) South Polar Stereographic map — the sectors as they're actually used
      throughout the chapter (this is the standard projection for Antarctic
      sea-ice work, but it reads as an unfamiliar circular map to anyone
      outside the field).
  (b) A plain PlateCarree (rectangular lon/lat) strip of the Southern Ocean,
      same sectors, same colors, same boundaries — an orientation panel for
      readers who don't read polar stereographic maps fluently. This is the
      "something like this is helpful" panel: ordinary x/y axes, ticks in
      familiar degrees, nothing about the projection to decode first.

Both panels use Raphael & Hobbs (2014) boundaries (SECTOR_BOUNDS_DEG in
ch3_config.py — confirmed against canonical_sectors.nc during the
2026-09-14 Weddell/ABS label-swap fix) and SECTOR_COLORS, so they match each
other and every other sector-colored figure in the chapter. Ocean-basin
names (Atlantic/Indian/Pacific, OCEAN_BOUNDS_DEG below) are layered on top
of both panels as italic text labels in a consistently muted color — this is
informational context for orientation, not a fifth data boundary, and never
competes with the sector fills for visual weight.

Geometry note: sector boundaries are stored in ch3_config as a LOCAL (lon0,
lon1) pair per sector, not wrapped to -180..180 — e.g. King Haakon is
(346, 431), sweeping across 0 deg, and Ross is (162, 250), sweeping across
the antimeridian.
  - Panel (a) (polar): np.linspace(lon0, lon1, n) sweeps the short way for
    every sector without special-casing, because cartopy's PlateCarree
    source CRS accepts longitudes outside -180..180 and a polar stereo
    target has no cut except at the pole itself.
  - Panel (b) (rectangular): a flat lon/lat axis DOES have a cut, at the
    antimeridian, so a wedge whose local range crosses it (only Ross, in
    -180..180 terms: 162E -> 180/-180 -> 110W) has to be split into two
    pieces before filling, or it silently wraps the wrong way across the
    whole map. split_for_rect() below does that split generically for any
    sector/ocean-band range, not just Ross, so this keeps working if the
    boundaries ever change.
Both the wedge_polygon() and split_for_rect() constructions were unit-tested
standalone (no cartopy) against SECTOR_BOUNDS_DEG: all five sectors tile the
full circle with no gaps or overlaps, and the wrap cases (King Haakon at 0
deg, Ross at 180 deg) render correctly. That check does not exercise
cartopy's actual projection math, only the input geometry.

DEPENDENCY / WHERE TO RUN THIS
  Needs cartopy. On first use it downloads Natural Earth coastline/land
  shapefiles (~a few MB) via pooch -- that needs internet access. If your
  cluster compute nodes are offline, run this on your laptop instead (or
  `python -c "import cartopy.feature as cf; cf.NaturalEarthFeature(...).geometries()"`
  once on a machine with internet to populate the cartopy data cache, then
  copy ~/.local/share/cartopy to the cluster).

Font fix 2026-09-18: this script set no font at all before (no rcParams
call, no ch3_style import) -- it was rendering in whatever matplotlib's
installed default happened to be. Added `import ch3_style` below, same as
every other figure script, so this one matches too.

Polar-panel fix 2026-09-18: the rendered figure showed a stray, oddly-shaped
landmass near the Weddell/King Haakon boundary, overlapping the "14°W" and
"Atlantic Ocean" labels. Diagnosis (unverified against a real render --
cartopy isn't available in this sandbox; PLEASE re-check the panel after
running this): ax.set_extent() on a SouthPolarStereo axes, without an
explicit circular boundary, leaves the axes as a SQUARE bounding box that
tightly circumscribes the circular lat/lon extent you actually asked for.
The square's four corners sit further from the pole (in the projection's
own x/y space) than the true LAT_MAP_EDGE circle does, so land at
latitudes well north of LAT_MAP_EDGE -- southern Africa, southern South
America, Tasmania, New Zealand, depending on which corner -- can poke into
those corners even though the intended extent excludes it. This is a
well-documented cartopy gotcha for polar stereographic plots; the standard
fix is to clip the axes to an actual circle, added below as
circular_boundary(). LAT_MAP_EDGE was also tightened (-38 -> -42) and the
two label rings (sector-boundary degree tags, ocean-basin names) given more
radial separation, as a second line of defense either way.

Run from scripts/python/plotting/Ch3/figures/ :  python fig_sector_map.py
Output: results/ch3/figures/fig02_sector_map.png
"""
import os
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
    )

# -- panel (a): polar -----------------------------------------------------------
LAT_OUTER = -50          # circumpolar edge of the map / wedges (deg)
LAT_INNER = -90          # pole
# Tightened -38 -> -42 (see the polar-panel fix note at the top of this
# file) -- less margin for stray land to appear in even with the circular
# boundary clip below, still enough room for the two label rings.
LAT_MAP_EDGE = -42       # map extent a bit past the wedges, for label room
LAT_OCEAN_LABEL = -44.5  # radius (within the outer ring) for ocean-basin labels

# -- panel (b): rectangular ------------------------------------------------------
RECT_LAT_MIN = -78
RECT_LAT_MAX = -45
RECT_OCEAN_LABEL_LAT = -47.5   # near the top of the strip

# Conventional three-basin division of the Southern Ocean (Atlantic/Indian
# boundary at 20E, Cape Agulhas meridian; Indian/Pacific at 147E, South East
# Cape; Pacific/Atlantic at 70W, Drake Passage -- this last one happens to
# coincide with the Weddell/ABS sector boundary, which is why panel (a)
# doesn't draw a separate divider line for it). Same local, non-wrapped
# degree convention as SECTOR_BOUNDS_DEG.
OCEAN_BOUNDS_DEG = {
    "Atlantic Ocean": (-70, 20),
    "Indian Ocean":   (20, 147),
    "Pacific Ocean":  (147, 290),
}
OCEAN_LABEL_COLOR = "#1a5276"

WEDGE_ALPHA = 0.35


def wedge_polygon(lon0, lon1, lat_inner=LAT_INNER, lat_outer=LAT_OUTER, n=200):
    """Pole -> along boundary meridian lon0 -> arc at lat_outer -> boundary
    meridian lon1 -> pole. lon0/lon1 are the LOCAL (possibly >180 or <-180,
    non-wrapped) bounds from SECTOR_BOUNDS_DEG; cartopy handles the wrap."""
    lons_arc = np.linspace(lon0, lon1, n)
    lats_arc = np.full(n, lat_outer)
    lons = np.concatenate([[lon0], lons_arc, [lon1]])
    lats = np.concatenate([[lat_inner], lats_arc, [lat_inner]])
    return lons, lats


def split_for_rect(lon0, lon1):
    """Convert a LOCAL (possibly >180 or <-180) (lon0, lon1) range into one
    or two (a, b) sub-ranges, each fully inside -180..180, for a flat
    PlateCarree axis where the antimeridian is a real cut. Two pieces come
    back only when the range actually straddles it (Ross, and the Pacific
    ocean band); everything else returns a single piece."""
    w0 = ((lon0 + 180) % 360) - 180
    w1 = ((lon1 + 180) % 360) - 180
    if w0 <= w1:
        return [(w0, w1)]
    return [(w0, 180.0), (-180.0, w1)]


def boundary_meridians():
    """The distinct sector-boundary longitudes (deg, wrapped to -180..180)
    and their conventional label, e.g. 14W, 71E."""
    seen = {}
    for sec in SECTOR_ORDER_BY_LONGITUDE:
        lo, hi = SECTOR_BOUNDS_DEG[sec]
        for lon in (lo, hi):
            w = ((lon + 180) % 360) - 180
            key = round(w, 3)
            if key not in seen:
                lab = f"{abs(w):.0f}°{'W' if w < 0 else 'E'}" if w != 0 else "0°"
                seen[key] = lab
    return seen


def circular_boundary(ax):
    """Clip a polar-stereographic axes to a true circle in display space.

    Without this, ax.set_extent(...) on a SouthPolarStereo axes leaves the
    axes as a SQUARE bounding box that tightly circumscribes the circular
    lat/lon extent you actually asked for -- the square's corners sit
    further from the pole (in the projection's own x/y space) than the
    LAT_MAP_EDGE circle does, so land at latitudes well north of
    LAT_MAP_EDGE can appear in those corners. Standard fix for cartopy
    polar stereographic plots."""
    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])
    ax.set_boundary(circle, transform=ax.transAxes)


def draw_polar_panel(ax):
    ax.set_extent([-180, 180, -90, LAT_MAP_EDGE], crs=ccrs.PlateCarree())
    circular_boundary(ax)

    for sec in SECTOR_ORDER_BY_LONGITUDE:
        lo, hi = SECTOR_BOUNDS_DEG[sec]
        lons, lats = wedge_polygon(lo, hi)
        ax.fill(lons, lats, transform=ccrs.PlateCarree(),
                 color=SECTOR_COLORS[sec], alpha=WEDGE_ALPHA, zorder=2,
                 edgecolor=SECTOR_COLORS[sec], linewidth=1.4)
        mid_lon = lo + (hi - lo) / 2.0
        label_lat = (LAT_INNER + LAT_OUTER) / 2.0 + 5  # nudge outward from pole for legibility
        ax.text(mid_lon, label_lat, SECTOR_LABELS[sec], transform=ccrs.PlateCarree(),
                 ha="center", va="center", fontsize=11, fontweight="bold",
                 color=SECTOR_COLORS[sec], zorder=5)

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8e4d8", zorder=3)
    ax.coastlines(resolution="50m", linewidth=0.8, color="#3a3a3a", zorder=4)

    for w, lab in boundary_meridians().items():
        ax.plot([w, w], [LAT_INNER, LAT_OUTER], transform=ccrs.PlateCarree(),
                 color="black", lw=0.9, ls=(0, (4, 3)), zorder=6, alpha=0.75)
        # Kept close to the wedge ring (LAT_OUTER + 1.5, was +3) so it's
        # clearly on its own inner ring, well short of the ocean-basin
        # labels' ring further out at LAT_OCEAN_LABEL -- previously these
        # two label rings were only ~4 deg apart (-43 vs -47) and collided.
        # Opaque background (alpha 1.0, was 0.75) so this tag stays legible
        # over the land/gridline colors either way.
        ax.text(w, LAT_OUTER + 1.5, lab, transform=ccrs.PlateCarree(),
                 ha="center", va="center", fontsize=8.5, color="black", zorder=7,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=1.0))

    theta = np.linspace(0, 360, 400)
    ax.plot(theta, np.full_like(theta, LAT_OUTER), transform=ccrs.PlateCarree(),
             color="0.3", lw=0.8, zorder=1)

    # ocean-basin ring: divider ticks only where one isn't already drawn by a
    # sector boundary (20E, 147E; 70W is the Weddell/ABS line, above)
    for div_lon in (20, 147):
        ax.plot([div_lon, div_lon], [LAT_OUTER, LAT_MAP_EDGE], transform=ccrs.PlateCarree(),
                 color=OCEAN_LABEL_COLOR, lw=0.7, ls=":", alpha=0.7, zorder=6)
    for name, (lo, hi) in OCEAN_BOUNDS_DEG.items():
        mid_lon = lo + (hi - lo) / 2.0
        ax.text(mid_lon, LAT_OCEAN_LABEL, name, transform=ccrs.PlateCarree(),
                 ha="center", va="center", fontsize=9.5, fontstyle="italic",
                 color=OCEAN_LABEL_COLOR, zorder=6)

    ax.set_title("(a)  Antarctic sea-ice sectors", fontsize=12, fontweight="bold",
                 pad=10, loc="left")
    ax.set_facecolor("#eaf3fb")
    ax.gridlines(color="0.6", lw=0.4, ls=":", zorder=0)


def draw_rect_panel(ax):
    ax.set_extent([-180, 180, RECT_LAT_MIN, RECT_LAT_MAX], crs=ccrs.PlateCarree())

    for sec in SECTOR_ORDER_BY_LONGITUDE:
        lo, hi = SECTOR_BOUNDS_DEG[sec]
        for a, b in split_for_rect(lo, hi):
            ax.fill([a, b, b, a], [RECT_LAT_MIN, RECT_LAT_MIN, RECT_LAT_MAX, RECT_LAT_MAX],
                     transform=ccrs.PlateCarree(), color=SECTOR_COLORS[sec],
                     alpha=WEDGE_ALPHA, zorder=2, edgecolor=SECTOR_COLORS[sec], linewidth=1.2)

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8e4d8", zorder=3)
    ax.coastlines(resolution="50m", linewidth=0.7, color="#3a3a3a", zorder=4)

    for w, lab in boundary_meridians().items():
        ax.plot([w, w], [RECT_LAT_MIN, RECT_LAT_MAX], transform=ccrs.PlateCarree(),
                 color="black", lw=0.8, ls=(0, (4, 3)), zorder=6, alpha=0.7)

    for name, (lo, hi) in OCEAN_BOUNDS_DEG.items():
        for a, b in split_for_rect(lo, hi):
            mid = a + (b - a) / 2.0
            ax.text(mid, RECT_OCEAN_LABEL_LAT, name, transform=ccrs.PlateCarree(),
                     ha="center", va="center", fontsize=9, fontstyle="italic",
                     color=OCEAN_LABEL_COLOR, zorder=6)

    ax.set_title("(b)  Southern Ocean view", fontsize=12, fontweight="bold",
                 pad=8, loc="left")
    ax.set_facecolor("#eaf3fb")
    gl = ax.gridlines(draw_labels=True, color="0.6", lw=0.4, ls=":", zorder=0,
                       x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = plt.FixedLocator(range(-180, 181, 30))
    gl.ylocator = plt.FixedLocator([-70, -60, -50])
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}


def main():
    # This was the figure eating a full dissertation page: figsize=(8, 11) is
    # nearly the physical size of a Letter page. Switched to the shared
    # "stack2" preset (TEXT_WIDTH_IN x 5.0in) so it obeys the same page-fit
    # rule as every other figure. The add_axes fractions below are unchanged
    # (they're proportions of the figure, not absolute inches) — cartopy's
    # GeoAxes hold the maps' true aspect ratio internally, so the polar
    # circle and the rectangular strip won't distort, they'll just render
    # more compactly with a bit more surrounding whitespace than before,
    # which bbox_inches="tight" then crops off. Cartopy isn't available in
    # this sandbox to render a preview, so it's worth a visual check after
    # running this — if the panels end up cramped, nudge the height fractions
    # (0.54 / 0.22 below) rather than the overall figsize.
    fig = plt.figure(figsize=figsize("stack2"))
    ax_polar = fig.add_axes([0.06, 0.40, 0.88, 0.54], projection=ccrs.SouthPolarStereo())
    ax_rect = fig.add_axes([0.06, 0.12, 0.88, 0.22], projection=ccrs.PlateCarree())

    draw_polar_panel(ax_polar)
    draw_rect_panel(ax_rect)

    handles = [mpatches.Patch(facecolor=SECTOR_COLORS[s], edgecolor=SECTOR_COLORS[s],
                               alpha=0.7, label=SECTOR_LABELS[s])
               for s in SECTOR_ORDER_BY_LONGITUDE]
    fig.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               frameon=False, fontsize=10)

    # dpi bumped 220 -> 300 to meet the Graduate Division's >=300dpi requirement
    # (this script predates that check; save() also re-warns if width creeps
    # past TEXT_WIDTH_IN again in the future).
    out = save(fig, "fig02_sector_map.png", dpi=300, sync=False)
    print("wrote", out)
    print("\nSector boundaries plotted (deg, local / display form):")
    for sec in SECTOR_ORDER_BY_LONGITUDE:
        lo, hi = SECTOR_BOUNDS_DEG[sec]
        print(f"  {SECTOR_LABELS[sec]:16s} {lo:6.0f} -> {hi:6.0f}  "
              f"(width {hi - lo:.0f} deg)  rect pieces: {split_for_rect(lo, hi)}")
    print("\nOcean-basin labels (conventional 3-way split, informational only "
          "-- not a data boundary):")
    for name, (lo, hi) in OCEAN_BOUNDS_DEG.items():
        print(f"  {name:16s} {lo:6.0f} -> {hi:6.0f}  rect pieces: {split_for_rect(lo, hi)}")


if __name__ == "__main__":
    main()