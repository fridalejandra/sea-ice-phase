#!/usr/bin/env python
"""
make_ch4_figures.py -- every Chapter 4 figure, in manuscript AND slide form.

    python make_ch4_figures.py                 # all figures, both modes
    python make_ch4_figures.py --mode slide    # one mode
    python make_ch4_figures.py --only 3 4      # subset by figure number (1,3,4,5,6)

Outputs:  results/ch4/figures/manuscript/*.png|pdf
          results/ch4/figures/slides/*.png            (transparent)

Figure list (6):
  fig01_beta_ratio                 forest plot, post/pre beta ratio, 95% CI          (Ch3 beta CSV)
  fig02_drift_wind_vectors         maps -- NOT made here; see note in main()
  fig03_wind_vs_drift_two_products scatter, MAM/JJA/SON, NSIDC filled, OSI hollow
  fig04_ab_timeseries              AB Sep-Oct: wind, NSIDC, OSI, normalised, 2016-20 band
  fig05_product_timeseries         JJA by sector, small multiples, two products
  fig06_appendix                   (a) DJF scatter  (b) fixed-mask vs all-cell excess

Inputs are the sector-daily CSVs already on the cluster (ERA5 v2, NSIDC-0116,
OSI-455), the Ch3 beta-shift CSV, and the period-season climatology NetCDFs.
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
WIND_NC = f"{ROOT}/results/ERA5/ERA5_windSpeed_period_season_climatology.nc"
DRIFT_NC = f"{ROOT}/results/NSIDC0116/NSIDC0116_driftSpeed_period_season_climatology.nc"
SECTOR_NC = f"{ROOT}/data/canonical_sectors.nc"
TABLE_DIR = f"{ROOT}/results/ch4/tables"

BASE0, BASE1 = 1992, 2015
POST0, POST1 = 2016, 2020
N_BOOT, SEED = 2000, 42
MAIN_SEASONS = ["MAM", "JJA", "SON"]
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


def bootstrap_table(seasons, tag):
    """Pre->post % change in wind and drift with block-bootstrap (years) 95% CIs, both products."""
    cache = f"{TABLE_DIR}/bootstrap_{tag}_{BASE0}-{BASE1}_vs_{POST0}-{POST1}.csv"
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rng = np.random.default_rng(SEED)
    rows = []
    for product in DRIFT_CSV:
        j = joined(product)
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
def legend_handles(seasons, products=True):
    h_sec = [Line2D([0], [0], marker="o", ls="none", markerfacecolor=fs.SECTOR_COLORS[s],
                    markeredgecolor=fs.SECTOR_COLORS[s], markersize=fs.ms(0.8),
                    label=fs.SECTOR_NAME[s]) for s in fs.SECTOR_ORDER]
    h_sea = [Line2D([0], [0], marker=fs.SEASON_MARKERS[s], ls="none", markerfacecolor="#666",
                    markeredgecolor="#666", markersize=fs.ms(0.8), label=s) for s in seasons]
    h_prod = [Line2D([0], [0], marker="o", ls="none", markerfacecolor="#666", markeredgecolor="#666",
                     markersize=fs.ms(0.8), label="NSIDC-0116"),
              Line2D([0], [0], marker="o", ls="none", markerfacecolor="none", markeredgecolor="#666",
                     markeredgewidth=fs.lw(0.9), markersize=fs.ms(0.8), label="OSI-455")]
    return h_sec, h_sea, (h_prod if products else [])


def scatter_panel(ax, tab, seasons, legends=True, title=None):
    lim = max(15, float(np.nanmax(np.abs(tab[["wind_lo", "wind_hi", "drift_lo", "drift_hi"]].values))) * 1.08)
    ax.fill_between([-lim, lim], [-lim, lim], [lim, lim], color=fs.BAND, alpha=0.5, zorder=0, lw=0)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color=fs.GREY, lw=fs.lw(0.6), zorder=1)
    ax.axhline(0, color=fs.INK, lw=0.6, zorder=1)
    ax.axvline(0, color=fs.INK, lw=0.6, zorder=1)
    for _, r in tab.iterrows():
        c = fs.SECTOR_COLORS[r.sector]
        hollow = r["product"] == "OSI-455"
        alpha = 1.0 if r.sig else 0.4
        ax.errorbar(r.wind_pct, r.drift_pct,
                    xerr=[[r.wind_pct - r.wind_lo], [r.wind_hi - r.wind_pct]],
                    yerr=[[r.drift_pct - r.drift_lo], [r.drift_hi - r.drift_pct]],
                    fmt="none", ecolor=c, alpha=alpha * 0.5, lw=fs.lw(0.5), zorder=2)
        ax.scatter(r.wind_pct, r.drift_pct, s=fs.ms(1.0) ** 2, marker=fs.SEASON_MARKERS[r.season],
                   facecolors="none" if hollow else c, edgecolors=c, linewidths=fs.lw(0.9),
                   alpha=alpha, zorder=3)
    ax.text(0.52 * lim, 0.96 * lim, "drift > wind", ha="center", va="top", color=fs.INK,
            fontsize=plt.rcParams["legend.fontsize"])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Wind-speed change, {POST0}–{POST1} vs {BASE0}–{BASE1} (%)")
    ax.set_ylabel(f"Drift-speed change, {POST0}–{POST1} vs {BASE0}–{BASE1} (%)")
    if title:
        ax.set_title(title, loc="left")
    if legends:
        h_sec, h_sea, h_prod = legend_handles(seasons)
        l1 = ax.legend(handles=h_sec, loc="upper left", title="Sector")
        ax.add_artist(l1)
        l2 = ax.legend(handles=h_sea, loc="lower right", title="Season")
        ax.add_artist(l2)
        ax.legend(handles=h_prod, loc="lower left", title="Product (faded: CI spans 0)")


def normalised_series(j, sector, season):
    g = j[(j.sector == sector) & (j.season == season)].sort_values("syear")
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
    qcol = next((c for c in b.columns if c.lower() in ("q", "q_fdr", "qval", "q_value", "fdr_q")), None)
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


def fig03():
    tab = bootstrap_table(MAIN_SEASONS, "main")
    fig, ax = plt.subplots(figsize=fs.size(0.8 if fs.mode() == "manuscript" else 0.48, 1.0))
    scatter_panel(ax, tab, MAIN_SEASONS)
    fs.save(fig, "fig03_wind_vs_drift_two_products")


def fig04():
    fig, ax = plt.subplots(figsize=fs.size(1.0, 0.42 if fs.mode() == "manuscript" else 0.48))
    ax.axvspan(POST0 - 0.5, POST1 + 0.5, color=fs.BAND, zorder=0, lw=0)
    ax.axhline(100, color=fs.INK, lw=0.6, zorder=1)
    c = fs.SECTOR_COLORS["ABS"]
    yrs, wind, nsidc = normalised_series(joined("NSIDC-0116"), "ABS", "SON")
    yrs_o, _, osi = normalised_series(joined("OSI-455"), "ABS", "SON")
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
    ax.set_title("Amundsen–Bellingshausen, September–October", loc="left")
    ax.set_xlim(yrs.min() - 0.5, yrs.max() + 0.5)
    fs.save(fig, "fig04_ab_timeseries")


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


def _mask_vs_allcell():
    """Sector excess (drift% - wind%) from the period-season climatology NetCDFs, two ways:
    all ice-covered cells per period vs the fixed mask (pre drift > 0.03 m/s and finite post)."""
    import xarray as xr
    wind = xr.open_dataset(WIND_NC)["wind_speed_mps"]
    drift = xr.open_dataset(DRIFT_NC)["drift_speed_mps"]
    sector = xr.open_dataset(SECTOR_NC)["sector_id"]
    fixed = (drift.sel(period="pre") > 0.03) & np.isfinite(drift.sel(period="post"))
    rows = []
    for season in ["JJA", "SON"]:
        for sid, code in fs.SECTOR_ID.items():
            for kind, extra in (("all", None), ("fixed", fixed.sel(season=season))):
                m = (sector == sid) if extra is None else ((sector == sid) & extra)
                v = {}
                for var, da in (("d", drift), ("w", wind)):
                    pre = float(da.sel(period="pre", season=season).where(m).mean())
                    post = float(da.sel(period="post", season=season).where(m).mean())
                    v[var] = 100 * (post - pre) / pre
                rows.append(dict(season=season, sector=code, kind=kind, excess_pp=v["d"] - v["w"]))
    t = pd.DataFrame(rows)
    t.to_csv(f"{TABLE_DIR}/excess_fixedmask_vs_allcell.csv", index=False)
    return t


def _panel_b(ax, title):
    t = _mask_vs_allcell()
    ypos, y = {}, 0.0
    for s in fs.SECTOR_ORDER:
        for se in ["JJA", "SON"]:
            ypos[(s, se)] = -y
            y += 1
        y += 0.7
    ax.axvline(0, color=fs.INK, lw=0.8, zorder=1)
    for (s, se), yy in ypos.items():
        c = fs.SECTOR_COLORS[s]
        a = float(t[(t.sector == s) & (t.season == se) & (t.kind == "all")].excess_pp.iloc[0])
        f = float(t[(t.sector == s) & (t.season == se) & (t.kind == "fixed")].excess_pp.iloc[0])
        ax.plot([a, f], [yy, yy], color=c, lw=fs.lw(0.6), alpha=0.6, zorder=2)
        ax.plot(a, yy, marker=fs.SEASON_MARKERS[se], ls="none", markerfacecolor="none", markeredgecolor=c,
                markeredgewidth=fs.lw(0.9), ms=fs.ms(0.75), zorder=3)
        ax.plot(f, yy, marker=fs.SEASON_MARKERS[se], ls="none", color=c, ms=fs.ms(0.75), zorder=3)
        ax.text(-0.01, yy, se, ha="right", va="center", color=fs.INK, transform=ax.get_yaxis_transform(),
                fontsize=plt.rcParams["ytick.labelsize"])
    for s in fs.SECTOR_ORDER:
        ys = [ypos[(s, se)] for se in ["JJA", "SON"]]
        ax.text(-0.09, np.mean(ys), fs.SECTOR_NAME[s], transform=ax.get_yaxis_transform(), ha="right",
                va="center", color=fs.SECTOR_COLORS[s],
                fontproperties=fs.st.bold_font_properties(plt.rcParams["axes.labelsize"]))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Excess, drift % − wind % (pp)\n2016–2024 vs 1979–2015, NSIDC-0116")
    ax.set_title(title, loc="left")
    h = [Line2D([0], [0], marker="o", ls="none", markerfacecolor="none", markeredgecolor="#666",
                markeredgewidth=fs.lw(0.9), ms=fs.ms(0.75), label="all ice-covered cells"),
         Line2D([0], [0], marker="o", ls="none", color="#666", ms=fs.ms(0.75), label="fixed mask")]
    ax.legend(handles=h, loc="upper right")
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.5 * (x1 - x0))


def _panel_a(ax, title):
    tab = bootstrap_table(["DJF"], "djf")
    scatter_panel(ax, tab, ["DJF"], legends=False, title=title)
    h_sec, h_sea, h_prod = legend_handles(["DJF"])
    ax.legend(handles=h_sec + h_prod, loc="upper left")


def fig06():
    """Appendix. Manuscript: one stacked two-panel figure. Slides: two separate PNGs (a, b)."""
    has_djf = "DJF" in set(joined("OSI-455").season)
    if not has_djf:
        print("  NOTE: no DJF months in the OSI-455 CSV; appendix panel (a) skipped. "
              "Download Dec-Feb for 1991-2020 and re-run.")
    if fs.mode() == "manuscript":
        if has_djf:
            fig, (axa, axb) = plt.subplots(2, 1, figsize=fs.size(0.8, 2.0))
            _panel_a(axa, "(a) Summer (DJF)")
            _panel_b(axb, "(b) Fixed mask vs all cells")
            fig.subplots_adjust(hspace=0.3)
        else:
            fig, axb = plt.subplots(figsize=fs.size(0.8, 0.9))
            _panel_b(axb, "Fixed mask vs all cells")
        fs.save(fig, "fig06_appendix")
    else:
        if has_djf:
            fig, axa = plt.subplots(figsize=fs.size(0.48, 1.0))
            _panel_a(axa, "Summer (DJF)")
            fs.save(fig, "fig06a_appendix_djf")
        fig, axb = plt.subplots(figsize=fs.size(0.6, 0.75))
        _panel_b(axb, "Fixed mask vs all cells")
        fs.save(fig, "fig06b_appendix_fixedmask")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["manuscript", "slide", "both"], default="both")
    p.add_argument("--only", nargs="*", type=int, default=[1, 3, 4, 5, 6])
    a = p.parse_args()
    modes = ["manuscript", "slide"] if a.mode == "both" else [a.mode]
    funcs = {1: fig01, 3: fig03, 4: fig04, 5: fig05, 6: fig06}
    for m in modes:
        fs.set_mode(m)
        print(f"== {m} ==")
        for n in a.only:
            print(f" fig{n:02d}")
            try:
                funcs[n]()
            except Exception as e:  # keep going; report at the end
                print(f"  FAILED fig{n:02d}: {type(e).__name__}: {e}")
    print("\nfig02_drift_wind_vectors: not made here. Re-run plot_drift_wind_vectors.py twice,\n"
          "  once with transparent=False / white background into figures/manuscript/ (add a PDF),\n"
          "  once as it is into figures/slides/. Output names: fig02_drift_wind_vectors.{png,pdf}")


if __name__ == "__main__":
    main()
