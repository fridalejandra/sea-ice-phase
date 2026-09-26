#!/usr/bin/env python
"""
18_context_figures.py -- the Fig 3 options, from real data (run 17_prep_context_fields.py first).

  fig3_optionA_{JJA,SON}.png        (a) normal state: divergence rate + ice drift, 1988-2015
                                    (b) trend in divergence rate, interior ice, 1988-2024
                                    (c) by sector: median block trend + IQR
  slide_drift_wind_{JJA,SON}.png    normal state with ice drift (black) and wind stress (blue)
  slide_paradox_{JJA,SON}.png       per sector: area variability (colour) vs interior divergence
                                    rate (grey), both relative to 1988-2015

Maps are drawn in the EASE-Grid 2.0 South projection itself (Lambert azimuthal equal-area,
pole-centred, WGS84), so NSIDC-0116 u/v (along grid x/y) plot as vectors without rotation.
ERA5 stress (east/north) is passed with the PlateCarree transform so cartopy rotates it.
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_style as st  # noqa: E402

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
CLIM_NC = f"{ROOT}/results/ch4/derived_nc/context_climatology.nc"
TREND_NC = f"{ROOT}/results/ch4/derived_nc/opening_closing_trend_blocks100km.nc"
RATES_CSV = f"{ROOT}/results/ch4/tables/interior_sector_rates.csv"
SIA_CSV = f"{ROOT}/data/merged/analysis_table_daily_anomaly_clean.csv"
FIG = f"{ROOT}/results/ch4/figures"
EASE = ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=-90,
                                      globe=ccrs.Globe(ellipse="WGS84"))
SECTORS = ["WS", "KH", "EA", "RA", "ABS"]
BOUNDS = {"WS": (300, 20), "KH": (20, 90), "EA": (90, 160), "RA": (160, 230), "ABS": (230, 300)}
LONG = {"WS": "Weddell", "KH": "King Haakon VII", "EA": "East Antarctica",
        "RA": "Ross-Amundsen", "ABS": "Amundsen-Bellingshausen"}
EXTENT = 4.2e6          # half-width of the map in metres
STRIDE, WSTRIDE = 8, 12  # vector subsampling (25 km cells): ~200 km drift, ~300 km wind


def base_map(ax):
    ax.set_extent([-EXTENT, EXTENT, -EXTENT, EXTENT], crs=EASE)
    ax.add_feature(cfeature.LAND, facecolor="0.88", zorder=3)
    ax.coastlines(lw=0.3, color=st.INK, zorder=4)
    for sec, (lo, hi) in BOUNDS.items():
        ax.plot([lo, lo], [-90, -55], color="0.65", lw=0.5, transform=ccrs.PlateCarree(), zorder=5)
        mid = (lo + ((hi - lo) % 360) / 2) % 360
        ax.text(mid, -54, sec, color=st.SECTOR_COLORS[sec], ha="center", va="center",
                fontproperties=st.bold_font_properties(size=8), transform=ccrs.PlateCarree(), zorder=6)
    ax.spines["geo"].set_edgecolor("0.85")


def normal_state(ax, c, s, wind=False):
    X, Y = np.meshgrid(c.x.values, c.y.values)
    ice = c[f"ice_{s}"].values >= 0.5
    d = np.where(ice, c[f"div_rate_{s}"].values * 1e7, np.nan)
    im = ax.pcolormesh(c.x.values, c.y.values, d, cmap="Purples", vmin=0, vmax=np.nanpercentile(d, 98),
                       shading="auto", transform=EASE, zorder=1)
    if wind:
        k = (slice(None, None, WSTRIDE), slice(None, None, WSTRIDE))
        q2 = ax.quiver(c.lon.values[k], c.lat.values[k], c[f"tau_x_{s}"].values[k], c[f"tau_y_{s}"].values[k],
                       transform=ccrs.PlateCarree(), color="#1f77b4", alpha=0.6, scale=1.6,
                       width=0.004, headwidth=4, zorder=2)
        ax.quiverkey(q2, 0.80, 0.02, 0.1, "0.1 Pa", labelpos="E", color="#1f77b4",
                     fontproperties={"size": 7}, coordinates="axes")
    k = (slice(None, None, STRIDE), slice(None, None, STRIDE))
    u, v = c[f"u_{s}"].values, c[f"v_{s}"].values
    m = ice[k] & np.isfinite(u[k])
    q = ax.quiver(X[k][m], Y[k][m], u[k][m], v[k][m], transform=EASE, color="k", scale=250,
                  width=0.0035, headwidth=4, zorder=2.5)
    ax.quiverkey(q, 0.80, 0.08 if wind else 0.02, 10, "10 cm s⁻¹", labelpos="E",
                 fontproperties={"size": 7}, coordinates="axes")
    return im


def sector_panel(axc, tr_ds):
    blon = tr_ds.block_lon.values
    ypos, y = {}, 0.0
    for sec in SECTORS:
        for s in ("JJA", "SON"):
            ypos[(sec, s)] = -y
            y += 1
        y += 0.7
    for (sec, s), yy in ypos.items():
        col = st.SECTOR_COLORS[sec]
        lo, hi = BOUNDS[sec]
        lon = blon % 360
        M = ((lon >= lo) | (lon < hi)) if lo > hi else ((lon >= lo) & (lon < hi))
        for name, off, face in (("opening", -0.14, col), ("closing", 0.14, "white")):
            tr = tr_ds[f"{name}_trend_pct_dec_{s}_interior"].values[M]
            tr = tr[np.isfinite(tr)]
            if tr.size < 5:
                continue
            q1, med, q3 = np.percentile(tr, [25, 50, 75])
            axc.plot([q1, q3], [yy + off] * 2, color=col, lw=1.4)
            axc.plot(med, yy + off, "o", ms=5, color=face, mec=col, mew=1.2, zorder=3)
    axc.axvline(0, color=st.INK, lw=0.8, zorder=0)
    axc.set_yticks(list(ypos.values()))
    axc.set_yticklabels([f"{sec}  {s}" for (sec, s) in ypos])
    for lab, (sec, _) in zip(axc.get_yticklabels(), ypos):
        lab.set_color(st.SECTOR_COLORS[sec])
    axc.spines["left"].set_visible(False)
    axc.tick_params(axis="y", length=0)
    axc.yaxis.tick_right()
    axc.set_xlabel("Trend (% per decade)\n● divergence   ○ convergence   bars: interquartile range", fontsize=8)


def option_a(c, tr, s):
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.85], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0], projection=EASE)
    base_map(ax)
    im = normal_state(ax, c, s)
    ax.set_title(f"(a) Normal state, {s} 1988–2015", loc="left")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03)
    cb.set_label("Divergence rate (10⁻⁷ s⁻¹) · arrows: mean ice drift", fontsize=8)
    cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[0, 1], projection=EASE)
    base_map(ax)
    P = EASE.transform_points(ccrs.PlateCarree(), tr.block_lon.values, tr.block_lat.values)
    t = tr[f"opening_trend_pct_dec_{s}_interior"].values
    sig = tr[f"opening_sig_fdr_{s}_interior"].values.astype(bool)
    im2 = ax.pcolormesh(P[..., 0], P[..., 1], t, cmap="RdBu_r", vmin=-10, vmax=10, shading="nearest",
                        transform=EASE, zorder=1)
    ax.scatter(P[..., 0][sig], P[..., 1][sig], s=0.8, c="k", lw=0, transform=EASE, zorder=2)
    ax.set_title(f"(b) Change: trend 1988–2024, interior ice", loc="left")
    cb = fig.colorbar(im2, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03, extend="both")
    cb.set_label("% per decade · dots: FDR-significant", fontsize=8)
    cb.outline.set_visible(False)

    axc = fig.add_subplot(gs[0, 2])
    sector_panel(axc, tr)
    axc.set_title("(c) By sector", loc="left")
    out = f"{FIG}/fig3_optionA_{s}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def slide_wind(c, s):
    fig = plt.figure(figsize=(5.2, 5.6))
    ax = fig.add_subplot(1, 1, 1, projection=EASE)
    base_map(ax)
    im = normal_state(ax, c, s, wind=True)
    ax.set_title(f"{s} 1988–2015: ice drift (black), wind stress (blue)", loc="left")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03)
    cb.set_label("Divergence rate (10⁻⁷ s⁻¹)", fontsize=8)
    cb.outline.set_visible(False)
    out = f"{FIG}/slide_drift_wind_{s}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def slide_paradox(s, months):
    sia = pd.read_csv(SIA_CSV, parse_dates=["date"])
    sia = sia[(sia.date.dt.year >= 1988) & sia.date.dt.month.isin(months)]
    rates = pd.read_csv(RATES_CSV)
    rates = rates[rates.season == s]
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.9), sharey=True)
    for ax, sec in zip(axes, SECTORS):
        col = st.SECTOR_COLORS[sec]
        g = sia[sia.sector == LONG[sec]]
        v = g.groupby(g.date.dt.year).delta_SIA_anomaly.var()
        v = v / v[v.index < 2016].mean()
        r = rates[rates.sector == sec].set_index("year").div_rate
        r = r / r[(r.index >= 1988) & (r.index < 2016)].mean()
        ax.axhline(1, color=st.GRID, lw=0.8)
        ax.axvline(2015.5, color=st.INK, lw=0.6, ls=(0, (2, 2)))
        ax.plot(v.index, v.values, color=col, lw=1.8)
        ax.plot(r.index, r.values, color="0.25", lw=1.4)
        ax.set_yscale("log")
        ax.set_ylim(0.12, 3)
        ax.set_yticks([0.25, 0.5, 1, 2])
        ax.set_yticklabels(["¼", "½", "1", "2"])
        ax.minorticks_off()
        ax.set_xticks([1990, 2005, 2020])
        ax.text(0.02, 0.97, LONG[sec].replace("-", "–"), transform=ax.transAxes, color=col, va="top",
                fontproperties=st.bold_font_properties(size=8.5))
    axes[0].text(0.02, 0.84, "divergence rate (interior ice)", transform=axes[0].transAxes, color="0.25", fontsize=7.5)
    axes[0].text(0.02, 0.04, "day-to-day area variability", transform=axes[0].transAxes,
                 color=st.SECTOR_COLORS["WS"], fontsize=7.5)
    axes[0].set_ylabel(f"Relative to 1988–2015 ({s})")
    fig.tight_layout()
    out = f"{FIG}/slide_paradox_{s}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    c = xr.open_dataset(CLIM_NC)
    tr = xr.open_dataset(TREND_NC)
    for s, months in (("JJA", (6, 7, 8)), ("SON", (9, 10, 11))):
        option_a(c, tr, s)
        slide_wind(c, s)
        slide_paradox(s, months)


if __name__ == "__main__":
    main()
