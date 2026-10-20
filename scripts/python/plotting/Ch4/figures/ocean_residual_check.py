#!/usr/bin/env python
"""
ocean_residual_check.py -- does the change in wind-independent ice drift (dC, from
windfactor_analysis.py) match the change in GLORYS12 surface current over the same cells?

Steps
  1. GLORYS12 monthly surface uo/vo (lat/lon, 1/12 deg) -> SIE 25 km grid (xesmf bilinear, weights cached)
  2. season x year means; sector means over the SAME fixed cells as the drift analysis
  3. pre (1993-2015) vs post (2016-2020) change vector dU_ocean, year-block bootstrap
  4. table + fig08: per sector-season, dC (ice, sector colour) and dU_ocean (grey) arrows,
     with the bootstrap cloud of dC; angle between them and magnitude ratio.

Inputs : data/glorys/glorys12_surface_uv_monthly_1993-2020.nc   (copernicusmarine subset)
         results/ch4/tables/windfactor_prepost.csv (+ _dc_boots.npz)
         results/NSIDC0116/NSIDC0116_uv_season_year.nc (fixed-cell definition)
Outputs: results/ch4/tables/ocean_residual_check.csv
         results/ch4/figures/{manuscript,slides}/fig08_ocean_residual.*
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_figstyle as fs  # noqa: E402

ROOT = fs.ROOT
GLORYS = sorted(glob.glob(f"{ROOT}/data/glorys/*.nc"))
SIE_GRID_FILE = f"{ROOT}/data/merged/merged_bootstrap_SH_latest.nc"
SECTOR_NC = f"{ROOT}/data/canonical_sectors.nc"
DRIFT_NC = f"{ROOT}/results/NSIDC0116/NSIDC0116_uv_season_year.nc"
WF_TAB = f"{ROOT}/results/ch4/tables/windfactor_prepost.csv"
WEIGHTS = f"{ROOT}/scripts/python/processing/ERA5_Reanalysis/glorys_to_sie_bilinear_weights.nc"
OUT_TAB = f"{ROOT}/results/ch4/tables/ocean_residual_check.csv"
SERIES = f"{ROOT}/results/ch4/tables/glorys_surface_uv_season_year_fixedcells.csv"
BASE0, BASE1, POST0, POST1 = 1993, 2015, 2016, 2020     # GLORYS starts 1993
SEASONS = ["MAM", "JJA", "SON"]
N_BOOT, SEED = 2000, 42


def season_of(m):
    return np.select([m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])], ["DJF", "MAM", "JJA"], "SON")


def build_series():
    """Season x year sector means of GLORYS surface u, v over the drift fixed cells."""
    if os.path.exists(SERIES):
        return pd.read_csv(SERIES)
    import xesmf as xe
    from pyproj import CRS, Transformer
    sie = xr.open_dataset(SIE_GRID_FILE)
    to_wgs = Transformer.from_crs(CRS.from_epsg(3412), CRS.from_epsg(4326), always_xy=True)
    sxx, syy = np.meshgrid(sie.x.values, sie.y.values)
    lon, lat = to_wgs.transform(sxx, syy)
    sie = sie.assign_coords(lon=(("y", "x"), lon), lat=(("y", "x"), lat))
    sector = xr.open_dataset(SECTOR_NC)["sector_id"].values

    d = xr.open_dataset(DRIFT_NC)["speed"]
    dpre, dpost = d.sel(year=slice(1992, 2015)), d.sel(year=slice(2016, 2020))
    fixed = ((dpre.mean("year") > 0.03) & (np.isfinite(dpre).sum("year") >= 15)
             & (np.isfinite(dpost).sum("year") >= 3))

    g = xr.open_mfdataset(GLORYS, combine="by_coords") if len(GLORYS) > 1 else xr.open_dataset(GLORYS[0])
    if "depth" in g.dims:
        g = g.isel(depth=0)
    g = g.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in g.dims})
    print("GLORYS:", dict(g.sizes), "\n  vars:", list(g.data_vars))
    rg = xe.Regridder(g, sie, method="bilinear", reuse_weights=os.path.exists(WEIGHTS), filename=WEIGHTS,
                      periodic=True)
    # monthly -> season-year means first (cheap on the source grid), then regrid
    m = g.time.dt.month
    g = g.assign_coords(season=("time", season_of(m).astype(str)),
                        syear=("time", (g.time.dt.year + (m == 12)).values))
    rows = []
    seasons, years = ["DJF", "MAM", "JJA", "SON"], sorted(set(int(y) for y in g.syear.values))
    for sea in seasons:
        fm = fixed.sel(season=sea).values
        for yr in years:
            sel = (g.season.values == sea) & (g.syear.values == yr)
            if sel.sum() < 2:
                continue
            uo = rg(g.uo.isel(time=sel).mean("time").load()).values
            vo = rg(g.vo.isel(time=sel).mean("time").load()).values
            for sid, code in fs.SECTOR_ID.items():
                mk = fm & (sector == sid) & np.isfinite(uo) & np.isfinite(vo)
                if mk.sum() == 0:
                    continue
                rows.append(dict(sector=code, season=sea, syear=yr, uo=float(uo[mk].mean()),
                                 vo=float(vo[mk].mean()), n_cells=int(mk.sum())))
        print(f"  {sea} done")
    s = pd.DataFrame(rows)
    s.to_csv(SERIES, index=False)
    print(f"wrote {SERIES}")
    return s


def compare(s, wf):
    rng = np.random.default_rng(SEED)
    rows = []
    for (sec, sea), g in s[s.season.isin(SEASONS)].groupby(["sector", "season"]):
        pre = g[(g.syear >= BASE0) & (g.syear <= BASE1)]
        post = g[(g.syear >= POST0) & (g.syear <= POST1)]
        if len(pre) < 10 or len(post) < 3:
            continue
        upre, upost = (pre.uo + 1j * pre.vo).values, (post.uo + 1j * post.vo).values
        du = upost.mean() - upre.mean()
        dub = np.array([upost[rng.integers(0, len(upost), len(upost))].mean()
                        - upre[rng.integers(0, len(upre), len(upre))].mean() for _ in range(N_BOOT)])
        w = wf[(wf.sector == sec) & (wf.season == sea)]
        if w.empty:
            continue
        w = w.iloc[0]
        dc = complex(w.dc_e, w.dc_n)
        ang = np.degrees(np.angle(du * np.conj(dc))) if abs(dc) > 0 else np.nan   # 0 = same heading
        proj = (du * np.conj(dc)).real / abs(dc) if abs(dc) > 0 else np.nan        # dU along dC heading
        rows.append(dict(sector=sec, season=sea,
                         u_ocean_pre=upre.mean().real, v_ocean_pre=upre.mean().imag,
                         du_e=du.real, du_n=du.imag, du_mag=abs(du),
                         du_mag_lo=np.percentile(np.abs(dub), 2.5), du_mag_hi=np.percentile(np.abs(dub), 97.5),
                         du_e_lo=np.percentile(dub.real, 2.5), du_e_hi=np.percentile(dub.real, 97.5),
                         du_n_lo=np.percentile(dub.imag, 2.5), du_n_hi=np.percentile(dub.imag, 97.5),
                         dc_e=dc.real, dc_n=dc.imag, dc_mag=abs(dc),
                         angle_deg=ang, du_along_dc=proj, ratio=abs(du) / abs(dc) if abs(dc) > 0 else np.nan,
                         resid_sig=bool(w.resid_sig)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_TAB, index=False)
    print(f"wrote {OUT_TAB}")
    return t


def fig08(t, boots):
    nrow, ncol = len(fs.SECTOR_ORDER), len(SEASONS)
    fig, axes = plt.subplots(nrow, ncol, figsize=fs.size(0.8, 1.25) if fs.mode() == "manuscript" else fs.size(0.6, 0.95),
                             sharex=True, sharey=True)
    lim = float(np.nanmax(np.abs(t[["du_e", "du_n", "dc_e", "dc_n"]].values))) * 1.25
    for i, sec in enumerate(fs.SECTOR_ORDER):
        for j, sea in enumerate(SEASONS):
            ax = axes[i, j]
            c = fs.SECTOR_COLORS[sec]
            r = t[(t.sector == sec) & (t.season == sea)]
            ax.axhline(0, color=fs.BAND, lw=0.8, zorder=0)
            ax.axvline(0, color=fs.BAND, lw=0.8, zorder=0)
            key = f"{sec}_{sea}"
            if key in boots:
                b = boots[key][::10]
                ax.scatter(b.real, b.imag, s=3, color=c, alpha=0.12, lw=0, zorder=1)
            if not r.empty:
                r = r.iloc[0]
                a = 1.0 if r.resid_sig else 0.45
                ax.annotate("", xy=(r.dc_e, r.dc_n), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=fs.lw(0.8), alpha=a, mutation_scale=10))
                ax.annotate("", xy=(r.du_e, r.du_n), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color="#444", lw=fs.lw(0.8), mutation_scale=10))
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(sea, loc="center")
            if j == 0:
                ax.text(-0.08, 0.5, fs.SECTOR_NAME[sec], transform=ax.transAxes, ha="right", va="center",
                        color=c, fontproperties=fs.st.bold_font_properties(plt.rcParams["axes.labelsize"]))
    # scale bar
    ax = axes[-1, -1]
    sb = 0.01 if lim > 0.02 else 0.005
    ax.plot([lim * 0.15, lim * 0.15 + sb], [-lim * 0.85, -lim * 0.85], color=fs.INK, lw=fs.lw(0.8))
    ax.text(lim * 0.15 + sb / 2, -lim * 0.8, f"{sb * 100:g} cm s$^{{-1}}$", ha="center", va="bottom",
            color=fs.INK, fontsize=plt.rcParams["legend.fontsize"])
    h = [Line2D([0], [0], color="#666", lw=fs.lw(0.8), label="Δ wind-independent ice drift (ΔC)"),
         Line2D([0], [0], color="#444", lw=fs.lw(0.8), label="Δ GLORYS12 surface current"),
         Line2D([0], [0], marker="o", ls="none", color="#666", alpha=0.3, ms=4, label="ΔC bootstrap")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(hspace=0.15, wspace=0.1, bottom=0.07)
    fs.save(fig, "fig08_ocean_residual")


if __name__ == "__main__":
    if not GLORYS:
        sys.exit(f"no GLORYS file under {ROOT}/data/glorys/")
    wf = pd.read_csv(WF_TAB)
    boots = dict(np.load(WF_TAB.replace(".csv", "_dc_boots.npz")))
    s = build_series()
    t = compare(s, wf) if not os.path.exists(OUT_TAB) else pd.read_csv(OUT_TAB)
    pd.set_option("display.width", 200)
    print(t[["sector", "season", "dc_mag", "du_mag", "du_mag_lo", "du_mag_hi", "angle_deg", "du_along_dc",
             "ratio", "resid_sig"]].round(4).to_string(index=False))
    print("\nRead: angle_deg near 0 and ratio near 1 = the ocean reanalysis moved the same way as the\n"
          "      wind-independent drift. du_along_dc > 0 with du_mag CI excluding 0 = agreement in sign.")
    for m in ("manuscript", "slide"):
        fs.set_mode(m)
        fig08(t, boots)
