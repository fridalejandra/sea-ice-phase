#!/usr/bin/env python
"""
make_ch4_figures.py -- every Chapter 4 figure, in manuscript AND slide form.

    python make_ch4_figures.py                 # all figures, both modes
    python make_ch4_figures.py --mode slide    # one mode
    python make_ch4_figures.py --only 3 4      # subset by figure number (1-6)

Outputs:  results/ch4/figures/manuscript/*.png|pdf
          results/ch4/figures/slides/*.png            (transparent)

Figure list (6):
  fig01_beta_ratio                 forest plot, post/pre beta ratio, 95% CI          (Ch3 beta CSV)
  fig02_drift_wind_vectors         drift-speed change (colour) + wind change (arrows), JJA/SON
  fig03_wind_vs_drift_two_products forest: NSIDC all-cell, OSI all-cell, NSIDC fixed-mask; wind tick
  fig04_ab_timeseries              AB Sep-Oct: wind, NSIDC, OSI, normalised, 2016-20 band
  fig05_product_timeseries         JJA by sector, small multiples, two products
  fig06_appendix                   (a) DJF forest  (b) all-cell vs fixed-mask excess, with CIs

Inputs are the sector-daily CSVs already on the cluster (ERA5 v2, NSIDC-0116,
OSI-455), the Ch3 beta-shift CSV, and the per-year seasonal u/v/speed NetCDFs.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_figstyle as fs  # noqa: E402

ROOT = fs.ROOT
WIND_CSV = f"{ROOT}/results/ERA5/ERA5_windSpeed_daily_sector_v2.csv"
DRIFT_CSV = {"NSIDC-0116": f"{ROOT}/results/NSIDC0116/NSIDC0116_driftSpeed_daily_sector.csv",
             "OSI-455": f"{ROOT}/results/OSI455/OSI455_driftSpeed_daily_sector.csv"}
BETA_CSV = f"{ROOT}/data/merged/analysis_results_from1988/beta_shift_seasonal_fdr.csv"
# per-year seasonal u, v, speed on the SIE grid (used by fig02 and the fixed-mask check)
WIND_NC = f"{ROOT}/results/ERA5/ERA5_uv_season_year.nc"
DRIFT_NC = f"{ROOT}/results/NSIDC0116/NSIDC0116_uv_season_year.nc"
LAND_NC = f"{ROOT}/data/landmask_sic.nc"
SECTOR_NC = f"{ROOT}/data/canonical_sectors.nc"
TABLE_DIR = f"{ROOT}/results/ch4/tables"

BASE0, BASE1 = 1992, 2015
POST0, POST1 = 2016, 2020
N_BOOT, SEED = 2000, 42
MAIN_SEASONS = ["MAM", "JJA", "SON"]
MASK_SEASONS = ["MAM", "JJA", "SON"]
SHORT = {"Weddell": "WS", "King Haakon VII": "KH", "East Antarctica": "EA",
         "Ross-Amundsen": "RA", "Ross": "RA", "Amundsen-Bellingshausen": "ABS"}

os.makedirs(TABLE_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def _season_of(month):
    return np.select([month.isin([12, 1, 2]), month.isin([3, 4, 5]),
                      month.isin([6, 7, 8]), month.isin([9, 10, 11])],
                     ["DJF", "MAM", "JJA", "SON"], default="NA")


def seasonal_means(path, col, months):
    """Sector x season x season-year means. December counts toward the next year's DJF."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"].dt.month.isin(months)].copy()
    m = df["date"].dt.month
    df["season"] = _season_of(m)
    df["syear"] = df["date"].dt.year + (m == 12).astype(int)
    out = df.groupby(["sector", "season", "syear"])[col].mean().rename("value").reset_index()
    out["sector"] = out["sector"].map(fs.SECTOR_ID)
    return out


_cache = {}


def joined(product):
    """wind + drift seasonal means for one product, restricted to the months OSI-455 has."""
    if product in _cache:
        return _cache[product]
    osi_months = sorted(pd.read_csv(DRIFT_CSV["OSI-455"], parse_dates=["date"])["date"].dt.month.unique())
    w = seasonal_means(WIND_CSV, "wind_speed_mps", osi_months)
    d = seasonal_means(DRIFT_CSV[product], "drift_speed_mps", osi_months)
    j = w.merge(d, on=["sector", "season", "syear"], suffixes=("_w", "_d")).dropna()
    _cache[product] = j
    return j


FIXED_LABEL = "NSIDC-0116 fixed"     # NSIDC drift + ERA5 wind averaged over the same cells in both periods


def fixed_mask_series():
    """Per-year sector means of NSIDC drift speed and ERA5 wind speed over a FIXED set of cells:
    pre-period mean drift > 0.03 m/s, >= 15 valid pre years, >= 3 valid post years (as in fig02).
    Removes the ice-edge-retreat sampling effect that the all-cell sector CSVs carry."""
    if FIXED_LABEL in _cache:
        return _cache[FIXED_LABEL]
    import xarray as xr
    sector = xr.open_dataset(SECTOR_NC)["sector_id"]
    d = xr.open_dataset(DRIFT_NC)["speed"]
    w = xr.open_dataset(WIND_NC)["speed"]
    dpre, dpost = d.sel(year=slice(BASE0, BASE1)), d.sel(year=slice(POST0, POST1))
    fixed = ((dpre.mean("year") > 0.03) & (np.isfinite(dpre).sum("year") >= 15)
             & (np.isfinite(dpost).sum("year") >= 3))
    rows = []
    for season in ["DJF", "MAM", "JJA", "SON"]:
        fm = fixed.sel(season=season)
        for sid, code in fs.SECTOR_ID.items():
            m = (sector == sid) & fm
            if int(m.sum()) == 0:
                continue
            dser = d.sel(season=season).where(m).mean(("y", "x"))
            wser = w.sel(season=season).where(m).mean(("y", "x"))
            for yr in d.year.values:
                rows.append(dict(sector=code, season=season, syear=int(yr),
                                 value_w=float(wser.sel(year=yr)), value_d=float(dser.sel(year=yr))))
    j = pd.DataFrame(rows).dropna()
    _cache[FIXED_LABEL] = j
    return j


def sources():
    return {"NSIDC-0116": joined("NSIDC-0116"), "OSI-455": joined("OSI-455"),
            FIXED_LABEL: fixed_mask_series()}


def bootstrap_table(seasons, tag):
    """Pre->post % change in wind and drift with block-bootstrap (years) 95% CIs,
    for NSIDC all-cell, OSI all-cell and NSIDC fixed-mask series."""
    cache = f"{TABLE_DIR}/bootstrap_{tag}_fm_{BASE0}-{BASE1}_vs_{POST0}-{POST1}.csv"  # delete to recompute
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rng = np.random.default_rng(SEED)
    rows = []
    for product, j in sources().items():
        for (sec, sea), g in j[j["season"].isin(seasons)].groupby(["sector", "season"]):
            pre = g[(g.syear >= BASE0) & (g.syear <= BASE1)]
            post = g[(g.syear >= POST0) & (g.syear <= POST1)]
            if len(pre) < 10 or len(post) < 3:
                continue
            pw, po = pre.value_w.values, post.value_w.values
            pd_, qd = pre.value_d.values, post.value_d.values
            wb, db = np.empty(N_BOOT), np.empty(N_BOOT)
            for b in range(N_BOOT):
                ip = rng.integers(0, len(pw), len(pw))
                iq = rng.integers(0, len(po), len(po))
                wb[b] = 100 * (po[iq].mean() - pw[ip].mean()) / pw[ip].mean()
                db[b] = 100 * (qd[iq].mean() - pd_[ip].mean()) / pd_[ip].mean()
            eb = db - wb
            ci = lambda a: np.percentile(a, [2.5, 97.5])
            w_obs = 100 * (po.mean() - pw.mean()) / pw.mean()
            d_obs = 100 * (qd.mean() - pd_.mean()) / pd_.mean()
            rows.append(dict(product=product, sector=sec, season=sea, n_pre=len(pre), n_post=len(post),
                             wind_pct=w_obs, wind_lo=ci(wb)[0], wind_hi=ci(wb)[1],
                             drift_pct=d_obs, drift_lo=ci(db)[0], drift_hi=ci(db)[1],
                             excess_pp=d_obs - w_obs, excess_lo=ci(eb)[0], excess_hi=ci(eb)[1]))
    tab = pd.DataFrame(rows)
    tab["sig"] = (tab.excess_lo > 0) | (tab.excess_hi < 0)
    tab.to_csv(cache, index=False)
    print(f"  wrote {cache}")
    return tab


# ----------------------------------------------------------------------------
# shared panels
# ----------------------------------------------------------------------------
def forest_panel(ax, tab, seasons, title=None):
    """Rows = sector x season. Drift % change with 95% CI: NSIDC all-cell (filled, upper), OSI all-cell
    (hollow, middle), NSIDC fixed-mask (dark edge, lower). Grey tick = ERA5 wind % change. Faded = excess CI spans 0."""
    ypos, y = {}, 0.0
    for s in fs.SECTOR_ORDER:
        for se in seasons:
            ypos[(s, se)] = -y
            y += 1.15
        y += 0.6
    ax.axvline(0, color=fs.INK, lw=0.8, zorder=1)
    series = (("NSIDC-0116", +0.28, "filled"), ("OSI-455", 0.0, "hollow"), (FIXED_LABEL, -0.28, "edged"))
    for (s, se), yy in ypos.items():
        c = fs.SECTOR_COLORS[s]
        for prod, dy, style in series:
            r = tab[(tab.sector == s) & (tab.season == se) & (tab["product"] == prod)]
            if r.empty:
                continue
            r = r.iloc[0]
            a = 1.0 if r.sig else 0.4
            ax.plot([r.drift_lo, r.drift_hi], [yy + dy, yy + dy], color=c, lw=fs.lw(0.8), alpha=a,
                    solid_capstyle="round", zorder=2)
            ax.plot(r.drift_pct, yy + dy, marker=fs.SEASON_MARKERS[se], ls="none",
                    markerfacecolor="none" if style == "hollow" else c,
                    markeredgecolor="#222" if style == "edged" else c,
                    markeredgewidth=fs.lw(0.9 if style != "edged" else 0.7), ms=fs.ms(0.75), alpha=a, zorder=3)
            if prod == "NSIDC-0116":
                ax.plot(r.wind_pct, yy, marker="|", ls="none", color=fs.GREY, ms=fs.ms(1.3),
                        markeredgewidth=fs.lw(0.9), zorder=4)
        ax.text(-0.01, yy, se, ha="right", va="center", color=fs.INK, transform=ax.get_yaxis_transform(),
                fontsize=plt.rcParams["ytick.labelsize"])
    for s in fs.SECTOR_ORDER:
        ys = [ypos[(s, se)] for se in seasons]
        ax.text(-0.09, np.mean(ys), fs.SECTOR_NAME[s], transform=ax.get_yaxis_transform(), ha="right",
                va="center", color=fs.SECTOR_COLORS[s],
                fontproperties=fs.st.bold_font_properties(plt.rcParams["axes.labelsize"]))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Change, {POST0}–{POST1} vs {BASE0}–{BASE1} (%)")
    if title:
        ax.set_title(title, loc="left")
    h = [Line2D([0], [0], marker="o", ls="none", color="#666", ms=fs.ms(0.75), label="drift, NSIDC-0116"),
         Line2D([0], [0], marker="o", ls="none", markerfacecolor="none", markeredgecolor="#666",
                markeredgewidth=fs.lw(0.9), ms=fs.ms(0.75), label="drift, OSI-455"),
         Line2D([0], [0], marker="o", ls="none", markerfacecolor="#666", markeredgecolor="#222",
                markeredgewidth=fs.lw(0.7), ms=fs.ms(0.75), label="drift, NSIDC-0116, fixed cells"),
         Line2D([0], [0], marker="|", ls="none", color=fs.GREY, ms=fs.ms(1.3), markeredgewidth=fs.lw(0.9),
                label="wind, ERA5"),
         Line2D([0], [0], marker="o", ls="none", color="#666", ms=fs.ms(0.75), alpha=0.4,
                label="faded: excess CI spans 0")]
    ax.legend(handles=h, loc="lower right", ncol=1)
    lo = min(float(tab.drift_lo.min()), float(tab.wind_pct.min()))
    hi = max(float(tab.drift_hi.max()), float(tab.wind_pct.max()))
    ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.45 * (hi - lo))

PLOT_START = 1991   # first year shown in time series; pre-1991 NSIDC drift is SMMR-era


def normalised_series(j, sector, season):
    g = j[(j.sector == sector) & (j.season == season) & (j.syear >= PLOT_START)].sort_values("syear")
    base = g[(g.syear >= BASE0) & (g.syear <= BASE1)]
    return g.syear.values, 100 * g.value_w.values / base.value_w.mean(), 100 * g.value_d.values / base.value_d.mean()


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def fig01():
    b = pd.read_csv(BETA_CSV)
    b["sector"] = b.sector.map(SHORT).fillna(b.sector)
    pre = b.beta_pre
    b["ratio"] = (pre + b.beta_shift) / pre
    lo, hi = (pre + b.ci_low) / pre, (pre + b.ci_high) / pre
    b["lo"], b["hi"] = np.minimum(lo, hi), np.maximum(lo, hi)
    qcol = next((c for c in b.columns if c.lower() in ("q", "q_fdr", "qval", "q_value", "fdr_q", "p_value_fdr")), None)
    seasons = ["DJF", "MAM", "JJA", "SON"]
    b = b.set_index(["sector", "season"])

    ypos, y = {}, 0.0
    for s in fs.SECTOR_ORDER:
        for se in seasons:
            ypos[(s, se)] = -y
            y += 1
        y += 0.8

    fig, ax = plt.subplots(figsize=fs.size(0.75 if fs.mode() == "manuscript" else 0.55, 0.95))
    ax.axvspan(0.9, 1.1, color=fs.BAND, zorder=0, lw=0)
    ax.axvline(1, color=fs.INK, lw=0.8, zorder=1)
    for (s, se), yy in ypos.items():
        if (s, se) not in b.index:
            continue
        r = b.loc[(s, se)]
        c = fs.SECTOR_COLORS[s]
        ax.plot([r.lo, r.hi], [yy, yy], color=c, lw=fs.lw(0.9), solid_capstyle="round", zorder=2)
        ax.plot(r.ratio, yy, marker=fs.SEASON_MARKERS[se], color=c, ms=fs.ms(0.75),
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(-0.01, yy, se, ha="right", va="center", color=fs.INK,
                fontsize=plt.rcParams["ytick.labelsize"], transform=ax.get_yaxis_transform())
    for s in fs.SECTOR_ORDER:
        ys = [ypos[(s, se)] for se in seasons]
        ax.text(-0.11, np.mean(ys), fs.SECTOR_NAME[s], transform=ax.get_yaxis_transform(),
                ha="right", va="center", color=fs.SECTOR_COLORS[s],
                fontproperties=fs.st.bold_font_properties(plt.rcParams["axes.labelsize"]))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("β ratio, post-2016 / pre-2016")
    xmax = min(4, float(np.nanmax(b.hi)) * 1.05)
    ax.set_xlim(max(-1, float(np.nanmin(b.lo)) * 1.05), xmax)
    if qcol is not None:
        ax.text(0.99, 0.01, f"min FDR q = {b[qcol].min():.2f} (20 tests)", transform=ax.transAxes,
                ha="right", va="bottom", color=fs.INK, fontsize=plt.rcParams["legend.fontsize"])
    fs.save(fig, "fig01_beta_ratio")



FIG02_POST = (2016, 2024)      # NSIDC-only figure; full post-2016 record
FIG02_SEASONS = ["JJA", "SON"]
FIG02_STEP, FIG02_MIN_PRE, FIG02_MIN_YEARS, FIG02_VMAX = 8, 0.03, 15, 3.0
LAND_COLOR = "#3b3b3b"


def _bird_cmap():
    try:
        import cblind as cb
        from matplotlib.colors import LinearSegmentedColormap, to_rgb
        cols, _ = cb.Colorplots().bird(9)
        cols = [to_rgb(c) for c in cols]
        if cols[0][0] > cols[-1][0]:
            cols = cols[::-1]
        return LinearSegmentedColormap.from_list("bird", cols)
    except Exception as e:
        print(f"  cblind unavailable ({e}); using RdBu_r")
        return "RdBu_r"


def fig02():
    import xarray as xr
    from matplotlib.colors import TwoSlopeNorm, ListedColormap
    pre, post = (BASE0, BASE1), FIG02_POST

    def load(nc):
        d = xr.open_dataset(nc)
        p0, p1 = d.sel(year=slice(*pre)), d.sel(year=slice(*post))
        valid = ((p0["speed"].mean("year") > FIG02_MIN_PRE)
                 & (np.isfinite(p0["speed"]).sum("year") >= FIG02_MIN_YEARS)
                 & (np.isfinite(p1["speed"]).sum("year") >= 5))
        ch = {k: (p1[k].mean("year") - p0[k].mean("year")) for k in ("u", "v", "speed")}
        return d, ch, valid

    dd, dch, dvalid = load(DRIFT_NC)
    _, wch, _ = load(WIND_NC)
    x, y = dd.x.values, dd.y.values
    X, Y = np.meshgrid(x, y)
    sector = xr.open_dataset(SECTOR_NC)["sector_id"]
    try:
        lm = xr.open_dataset(LAND_NC)
        v = [k for k in lm.data_vars if lm[k].ndim == 2][0]
        land = (lm[v].values != 0) & np.isfinite(lm[v].values)
    except Exception:
        land = None

    anyvalid = np.zeros_like(X, bool)
    for s_ in FIG02_SEASONS:
        anyvalid |= dvalid.sel(season=s_).values
    if land is not None:
        anyvalid |= land
    xs, ys = X[anyvalid], Y[anyvalid]
    dx, dy = (xs.max() - xs.min()) * 0.04, (ys.max() - ys.min()) * 0.04
    xlim, ylim = (xs.min() - dx, xs.max() + dx), (ys.min() - dy, ys.max() + dy)

    n = len(FIG02_SEASONS)
    fig, axes = plt.subplots(1, n, figsize=fs.size(1.0, 0.55 if fs.mode() == "manuscript" else 0.5))
    axes = np.atleast_1d(axes)
    cmap = _bird_cmap()
    norm = TwoSlopeNorm(vmin=-FIG02_VMAX, vcenter=0, vmax=FIG02_VMAX)
    sl = (slice(None, None, FIG02_STEP), slice(None, None, FIG02_STEP))
    key_len = 0.5
    for ax, s_ in zip(axes, FIG02_SEASONS):
        C = (dch["speed"].sel(season=s_) * 100.0).where(dvalid.sel(season=s_)).values
        m = ax.pcolormesh(x, y, C, cmap=cmap, norm=norm, shading="auto", zorder=1, rasterized=True)
        if land is not None:
            ax.pcolormesh(x, y, np.where(land, 1.0, np.nan), cmap=ListedColormap([LAND_COLOR]),
                          shading="auto", zorder=2)
        ax.contour(x, y, np.nan_to_num(sector.values, nan=0), levels=[1.5, 2.5, 3.5, 4.5],
                   colors="#555", linewidths=0.5, zorder=3)
        U = wch["u"].sel(season=s_).where(dvalid.sel(season=s_)).values
        V = wch["v"].sel(season=s_).where(dvalid.sel(season=s_)).values
        q = ax.quiver(X[sl], Y[sl], U[sl], V[sl], color="#111", scale_units="xy", angles="xy",
                      scale=key_len / (FIG02_STEP * 25000 * 0.9), width=0.0028, headwidth=3.5,
                      headlength=4.5, zorder=4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title({"JJA": "Winter (JJA)", "SON": "Spring (SON)"}.get(s_, s_), loc="left")
    ax.quiverkey(q, 0.88, 0.04, key_len, f"{key_len:g} m s$^{{-1}}$", labelpos="E", coordinates="axes",
                 fontproperties={"size": plt.rcParams["legend.fontsize"]})
    cb_ = fig.colorbar(m, ax=axes, orientation="horizontal", fraction=0.045, pad=0.03, extend="both")
    cb_.set_label(f"Ice-drift speed change, {post[0]}–{post[1]} vs {pre[0]}–{pre[1]} (cm s$^{{-1}}$)")
    cb_.outline.set_visible(False)
    fs.save(fig, "fig02_drift_wind_vectors")


def fig03():
    tab = bootstrap_table(MAIN_SEASONS, "main")
    fig, ax = plt.subplots(figsize=fs.size(0.9 if fs.mode() == "manuscript" else 0.62, 1.25 if fs.mode() == "manuscript" else 1.0))
    forest_panel(ax, tab, MAIN_SEASONS)
    fs.save(fig, "fig03_wind_vs_drift_two_products")


def fig04():
    fig, ax = plt.subplots(figsize=fs.size(1.0, 0.42 if fs.mode() == "manuscript" else 0.48))
    ax.axvspan(POST0 - 0.5, POST1 + 0.5, color=fs.BAND, zorder=0, lw=0)
    ax.axhline(100, color=fs.INK, lw=0.6, zorder=1)
    c = fs.SECTOR_COLORS["WS"]
    yrs, wind, nsidc = normalised_series(joined("NSIDC-0116"), "WS", "SON")
    yrs_o, _, osi = normalised_series(joined("OSI-455"), "WS", "SON")
    ax.plot(yrs, wind, color=fs.GREY, lw=fs.lw(0.8), label="ERA5 wind speed", zorder=2)
    ax.plot(yrs, nsidc, color=c, lw=fs.lw(1.0), label="Drift speed, NSIDC-0116", zorder=3)
    ax.plot(yrs_o, osi, color=c, lw=fs.lw(0.9), ls=(0, (1.5, 1.5)), label="Drift speed, OSI-455", zorder=3)
    # direct labels at the right edge instead of a boxed legend
    for xv, yv, lab, col, dy in ((yrs[-1], nsidc[-1], "NSIDC-0116", c, 0),
                                 (yrs_o[-1], osi[-1], "OSI-455", c, 10),
                                 (yrs[-1], wind[-1], "ERA5 wind", fs.GREY, -10)):
        ax.annotate(lab, (xv, yv), xytext=(6, dy), textcoords="offset points", va="center",
                    color=col, fontsize=plt.rcParams["legend.fontsize"])
    ax.text(POST0 - 0.3, ax.get_ylim()[1], f"{POST0}–{POST1}", ha="left", va="top", color=fs.INK,
            fontsize=plt.rcParams["legend.fontsize"])
    ax.set_ylabel(f"% of {BASE0}–{BASE1} mean")
    ax.set_xlabel("Year")
    ax.set_title("Weddell, September–October", loc="left")
    ax.set_xlim(yrs.min() - 0.5, yrs.max() + 0.5)
    fs.save(fig, "fig04_weddell_timeseries")


def fig05():
    season = "JJA"
    jn, jo = joined("NSIDC-0116"), joined("OSI-455")
    if fs.mode() == "manuscript":
        fig, axes = plt.subplots(5, 1, figsize=fs.size(1.0, 1.05), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(1, 5, figsize=fs.size(1.0, 0.33), sharex=True, sharey=True)
    for ax, s in zip(axes, fs.SECTOR_ORDER):
        c = fs.SECTOR_COLORS[s]
        yrs, _, dn = normalised_series(jn, s, season)
        yrs_o, _, do = normalised_series(jo, s, season)
        ax.axvspan(POST0 - 0.5, POST1 + 0.5, color=fs.BAND, zorder=0, lw=0)
        ax.axhline(100, color=fs.INK, lw=0.6, zorder=1)
        ax.plot(yrs, dn, color=c, lw=fs.lw(0.9), zorder=3)
        ax.plot(yrs_o, do, color=c, lw=fs.lw(0.8), ls=(0, (1.5, 1.5)), zorder=3)
        ax.set_title(fs.SECTOR_NAME[s], loc="left", color=c,
                     fontproperties=fs.st.bold_font_properties(plt.rcParams["axes.titlesize"]))
        ax.set_xlim(yrs.min() - 0.5, yrs.max() + 0.5)
    h = [Line2D([0], [0], color="#666", lw=fs.lw(0.9), label="NSIDC-0116"),
         Line2D([0], [0], color="#666", lw=fs.lw(0.8), ls=(0, (1.5, 1.5)), label="OSI-455")]
    axes[0].legend(handles=h, loc="upper left", ncol=2)
    if fs.mode() == "manuscript":
        axes[2].set_ylabel(f"JJA drift speed (% of {BASE0}–{BASE1} mean)")
        axes[-1].set_xlabel("Year")
    else:
        axes[0].set_ylabel(f"JJA drift speed\n(% of {BASE0}–{BASE1} mean)")
        for ax in axes:
            ax.set_xlabel("Year")
    fig.subplots_adjust(hspace=0.35, wspace=0.08)
    fs.save(fig, "fig05_product_timeseries")


def _panel_b(ax, title):
    """All-cell vs fixed-mask excess (drift% - wind%) for NSIDC-0116, both with bootstrap 95% CIs."""
    tab = bootstrap_table(MAIN_SEASONS, "main")
    ypos, y = {}, 0.0
    for s in fs.SECTOR_ORDER:
        for se in MASK_SEASONS:
            ypos[(s, se)] = -y
            y += 1
        y += 0.7
    ax.axvline(0, color=fs.INK, lw=0.8, zorder=1)
    for (s, se), yy in ypos.items():
        c = fs.SECTOR_COLORS[s]
        ra = tab[(tab.sector == s) & (tab.season == se) & (tab["product"] == "NSIDC-0116")]
        rf = tab[(tab.sector == s) & (tab.season == se) & (tab["product"] == FIXED_LABEL)]
        if ra.empty or rf.empty:
            continue
        ra, rf = ra.iloc[0], rf.iloc[0]
        ax.plot([ra.excess_pp, rf.excess_pp], [yy, yy], color=c, lw=fs.lw(0.5), alpha=0.5, zorder=2)
        for r, dy, hollow in ((ra, +0.2, True), (rf, -0.2, False)):
            ax.plot([r.excess_lo, r.excess_hi], [yy + dy, yy + dy], color=c, lw=fs.lw(0.7),
                    alpha=1.0 if r.sig else 0.4, solid_capstyle="round", zorder=2)
            ax.plot(r.excess_pp, yy + dy, marker=fs.SEASON_MARKERS[se], ls="none",
                    markerfacecolor="none" if hollow else c, markeredgecolor=c,
                    markeredgewidth=fs.lw(0.9), ms=fs.ms(0.75), alpha=1.0 if r.sig else 0.4, zorder=3)
        ax.text(-0.01, yy, se, ha="right", va="center", color=fs.INK, transform=ax.get_yaxis_transform(),
                fontsize=plt.rcParams["ytick.labelsize"])
    for s in fs.SECTOR_ORDER:
        ys = [ypos[(s, se)] for se in MASK_SEASONS]
        ax.text(-0.09, np.mean(ys), fs.SECTOR_NAME[s], transform=ax.get_yaxis_transform(), ha="right",
                va="center", color=fs.SECTOR_COLORS[s],
                fontproperties=fs.st.bold_font_properties(plt.rcParams["axes.labelsize"]))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Excess, drift % − wind % (pp)\n{POST0}–{POST1} vs {BASE0}–{BASE1}, NSIDC-0116")
    ax.set_title(title, loc="left")
    h = [Line2D([0], [0], marker="o", ls="none", markerfacecolor="none", markeredgecolor="#666",
                markeredgewidth=fs.lw(0.9), ms=fs.ms(0.75), label="all ice-covered cells"),
         Line2D([0], [0], marker="o", ls="none", color="#666", ms=fs.ms(0.75), label="fixed cells"),
         Line2D([0], [0], marker="o", ls="none", color="#666", ms=fs.ms(0.75), alpha=0.4,
                label="faded: CI spans 0")]
    ax.legend(handles=h, loc="lower right")
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.45 * (x1 - x0))


def _panel_a(ax, title):
    tab = bootstrap_table(["DJF"], "djf")
    forest_panel(ax, tab, ["DJF"], title=title)


def fig06():
    """Appendix. Manuscript: one stacked two-panel figure. Slides: two separate PNGs (a, b)."""
    has_djf = "DJF" in set(joined("OSI-455").season)
    if not has_djf:
        print("  NOTE: no DJF months in the OSI-455 CSV; appendix panel (a) skipped. "
              "Download Dec-Feb for 1991-2020 and re-run.")
    if fs.mode() == "manuscript":
        if has_djf:
            fig, (axa, axb) = plt.subplots(2, 1, figsize=fs.size(0.8, 1.9), gridspec_kw={"height_ratios": [0.8, 1.2]})
            _panel_a(axa, "(a) Summer (DJF)")
            _panel_b(axb, "(b) Fixed mask vs all cells")
            fig.subplots_adjust(hspace=0.3)
        else:
            fig, axb = plt.subplots(figsize=fs.size(0.8, 0.9))
            _panel_b(axb, "Fixed mask vs all cells")
        fs.save(fig, "fig06_appendix")
    else:
        if has_djf:
            fig, axa = plt.subplots(figsize=fs.size(0.6, 0.7))
            _panel_a(axa, "Summer (DJF)")
            fs.save(fig, "fig06a_appendix_djf")
        fig, axb = plt.subplots(figsize=fs.size(0.6, 0.75))
        _panel_b(axb, "Fixed mask vs all cells")
        fs.save(fig, "fig06b_appendix_fixedmask")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["manuscript", "slide", "both"], default="both")
    p.add_argument("--only", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6])
    a = p.parse_args()
    modes = ["manuscript", "slide"] if a.mode == "both" else [a.mode]
    funcs = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06}
    for m in modes:
        fs.set_mode(m)
        print(f"== {m} ==")
        for n in a.only:
            print(f" fig{n:02d}")
            try:
                funcs[n]()
            except Exception as e:  # keep going; report at the end
                print(f"  FAILED fig{n:02d}: {type(e).__name__}: {e}")



if __name__ == "__main__":
    main()
