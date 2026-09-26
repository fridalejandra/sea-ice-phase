#!/usr/bin/env python
"""
22b_fig1_two_panel.py -- Fig 1 as one figure: (a) September-October and (b) June-August sea-ice
motion trends, 1988-2024, from the fields 22_motion_trend_map.py already saved (runs in seconds).
Shared colour scale and arrow scale; sector boundaries as in Figs 2 and 4.
"""
import os
import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_style as st  # noqa: E402

ROOT = os.environ.get("CH4_ROOT", "/user/geog/falejandraperez/sea-ice-phase")
NC = f"{ROOT}/results/ch4/derived_nc/motion_trend_{{season}}_1988.nc"
OUT = f"{ROOT}/results/ch4/figures/fig1_motion_trend_1988_2024.png"
PANELS = [("SO", "(a) September–October"), ("JJA", "(b) June–August")]
VMAX, STRIDE, ARROW_SCALE, EXTENT = 1.5, 7, 40, 4.0e6
LAND = "0.88"
SECTORS = {"WS": (300, 20), "KH": (20, 90), "EA": (90, 160), "RA": (160, 230), "ABS": (230, 300)}
EASE = ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=-90,
                                      globe=ccrs.Globe(ellipse="WGS84"))


def panel(ax, ds, title):
    x, y = ds.x.values, ds.y.values
    X, Y = np.meshgrid(x, y)
    ax.set_extent([-EXTENT, EXTENT, -EXTENT, EXTENT], crs=EASE)
    im = ax.pcolormesh(x, y, ds.speed_trend.values, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX,
                       shading="auto", transform=EASE, zorder=1)
    k = (slice(None, None, STRIDE), slice(None, None, STRIDE))
    bu, bv = ds.u_trend.values[k], ds.v_trend.values[k]
    sig = ds.vector_sig.values[k].astype(bool)
    m = np.isfinite(bu) & np.isfinite(bv)
    q = None
    for mask, col in ((m & ~sig, "0.72"), (m & sig, "k")):
        if mask.any():
            q = ax.quiver(X[k][mask], Y[k][mask], bu[mask], bv[mask], transform=EASE, color=col,
                          scale=ARROW_SCALE, width=0.0032, headwidth=4, zorder=2)
    if os.environ.get("CH4_NO_LAND") != "1":
        ax.add_feature(cfeature.LAND, facecolor=LAND, zorder=3)
        ax.coastlines(lw=0.3, color=st.INK, zorder=4)
    for sec, (lo, hi) in SECTORS.items():
        ax.plot([lo, lo], [-90, -57], color="0.55", lw=0.6, ls=(0, (3, 2)),
                transform=ccrs.PlateCarree(), zorder=5)
        mid = (lo + ((hi - lo) % 360) / 2) % 360
        ax.text(mid, -55.5, sec, color=st.SECTOR_COLORS[sec], ha="center", va="center",
                fontproperties=st.bold_font_properties(size=8), transform=ccrs.PlateCarree(), zorder=6)
    ax.spines["geo"].set_edgecolor("0.85")
    ax.set_title(title, loc="left", fontsize=10)
    return im, q


def main():
    fig = plt.figure(figsize=(10.2, 5.2))
    axes = [fig.add_subplot(1, 2, i + 1, projection=EASE) for i in range(2)]
    im = q = None
    for ax, (season, title) in zip(axes, PANELS):
        im, q_ = panel(ax, xr.open_dataset(NC.format(season=season)), title)
        q = q_ or q
    fig.subplots_adjust(wspace=0.04, bottom=0.14)
    cax = fig.add_axes([0.30, 0.07, 0.40, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("Trend in drift speed, 1988–2024 (cm s⁻¹ per decade)", fontsize=9)
    cb.outline.set_visible(False)
    if q is not None:
        q.axes.quiverkey(q, 0.75, 0.083, 1, "1 cm s⁻¹ per decade", labelpos="E",
                         fontproperties={"size": 8}, coordinates="figure")
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
