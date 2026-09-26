#!/usr/bin/env python
"""
25_local_coupling.py -- Fig 4: where and when is daily SIA tendency coupled to wind, and did that
change after 2016? Month x longitude version of the sector beta (Fig 2), after Eabry et al. (2026)
Fig 2e, without sector averaging.

  --build    SIC (Bootstrap v4, 25 km polar stereographic) mapped nearest-neighbour onto the EASE-2
             grid of the wind file; for 36 x 10-degree longitude bins (starting at 20E, so every
             sector is contiguous) writes daily
               SIA  (km^2, cells with SIC >= 0.15, SIC-weighted, 625 km^2 EASE cells)
               tau_y, tau_mag  (mean over ice-covered cells of the bin that day)
             -> tables/lonbin_daily_SIA_wind.csv                                   (~15-30 min)
  --analyze  dSIA(t) = SIA(t) - SIA(t-1) (consecutive days only); anomalies from a 31-day-smoothed
             day-of-year climatology, period-specific (as in 21_build_clean_anomalies.py);
             per bin x month x period: r and beta of dSIA' on wind'; change in beta tested with
             the stratified whole-year bootstrap used for Fig 2; BH-FDR over all cells
             -> tables/lonbin_month_coupling_{wind}.csv
  --plot     (a) r 1988-2015  (b) r 2016-2023  (c) change in r; boxes where the beta change is
             significant (thick: FDR q < 0.05; thin: p < 0.05)
             -> figures/fig4_local_coupling_{wind}.png
  optional 2nd argument: wind variable, tau_y (default; northward = off-ice) or tau_mag
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.environ.get("CH4_ROOT", "/user/geog/falejandraperez/sea-ice-phase")
SIC_NC = f"{ROOT}/data/merged/merged_bootstrap_SH_latest.nc"
SIC_VAR = "N07_ICECON"
WIND_NC = f"{ROOT}/results/ch4/derived_nc/wind_stress_on_ease_sh.nc"
TAB = f"{ROOT}/results/ch4/tables"
FIG = f"{ROOT}/results/ch4/figures"
DAILY_CSV = f"{TAB}/lonbin_daily_SIA_wind.csv"
START, END, BREAK = int(os.environ.get("CH4_START", 1988)), int(os.environ.get("CH4_END", 2023)), 2016
LON0, DLON = 20, 10
NBIN = 360 // DLON
CELL_KM2 = 625.0
N_BOOT = 1000
MIN_YEARS = int(os.environ.get("CH4_MIN_YEARS", 5))
SECTOR_EDGES = {"KH": 20, "EA": 90, "RA": 160, "ABS": 230, "WS": 300}


# ----------------------------------------------------------------------------- build
def grid_map(wind, sic):
    from pyproj import Transformer
    X, Y = np.meshgrid(wind.x.values, wind.y.values)
    lon, lat = Transformer.from_crs("EPSG:6932", "EPSG:4326", always_xy=True).transform(X, Y)
    try:
        tr = Transformer.from_crs("EPSG:6932", "EPSG:3412", always_xy=True)
    except Exception:
        tr = Transformer.from_crs("EPSG:6932", "EPSG:3976", always_xy=True)
    PX, PY = tr.transform(X, Y)
    sx, sy = sic.x.values, sic.y.values
    ix = np.rint((PX - sx[0]) / (sx[1] - sx[0])).astype(int)
    iy = np.rint((PY - sy[0]) / (sy[1] - sy[0])).astype(int)
    ok = (ix >= 0) & (ix < sx.size) & (iy >= 0) & (iy < sy.size) & (lat < -50)
    ey, ex = np.nonzero(ok)
    b = (((lon[ok] - LON0) % 360) // DLON).astype(int)
    B = np.zeros((b.size, NBIN), "float32")
    B[np.arange(b.size), b] = 1
    print(f"  {b.size} EASE cells south of 50S mapped to the SIC grid")
    return ey, ex, iy[ok], ix[ok], B


def read_sic(sic, t0, t1, iy, ix):
    da = sic[SIC_VAR].sel(time=slice(t0, t1))
    v = da.values[:, iy, ix].astype("float32")
    if np.nanmax(v) > 2:          # still packed 0-1000
        v = v / 1000.0
    v[(v > 1.0) | (v < 0)] = np.nan   # land / missing flags
    return pd.DatetimeIndex(da.time.values).normalize(), v


def build():
    sic = xr.open_dataset(SIC_NC)
    wind = xr.open_dataset(WIND_NC)
    wt = pd.DatetimeIndex(pd.to_datetime(wind.valid_time.values)).normalize()
    ey, ex, iy, ix, B = grid_map(wind, sic)
    ref_valid = None
    out = []
    for yr in range(START, END + 1):
        t0 = f"{yr - 1}-12-31" if yr == START else f"{yr}-01-01"
        st, s = read_sic(sic, t0, f"{yr}-12-31", iy, ix)
        if ref_valid is None:
            ref_valid = np.isfinite(s).mean(0) > 0.5          # ocean cells of the SIC grid
        wj = np.nonzero(wt.isin(st))[0]
        wdates = wt[wj]
        tau = {k: wind[k].isel(valid_time=wj).values[:, ey, ex].astype("float32") for k in ("tau_y", "tau_mag")}
        wpos = pd.Series(np.arange(len(wdates)), index=wdates)
        for d in range(len(st)):
            day = st[d]
            si = s[d]
            miss = np.mean(~np.isfinite(si[ref_valid]))
            if miss > 0.05:
                out.append(pd.DataFrame(dict(date=day, lonbin=(np.arange(NBIN) * DLON + LON0) % 360)))
                continue
            ice = np.nan_to_num(si) >= 0.15
            sia = (np.where(ice, np.nan_to_num(si), 0) * CELL_KM2) @ B
            nice = ice.astype("float32") @ B
            row = dict(date=day, lonbin=(np.arange(NBIN) * DLON + LON0) % 360, SIA=sia, n_ice=nice)
            if day in wpos.index:
                k = wpos[day]
                for name, arr in tau.items():
                    a = np.where(ice, np.nan_to_num(arr[k]), 0) @ B
                    row[name] = np.where(nice > 0, a / np.maximum(nice, 1), np.nan)
            out.append(pd.DataFrame(row))
        print(f"  {yr}", flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(DAILY_CSV, index=False, float_format="%.6g")
    print(f"wrote {DAILY_CSV}: {len(res)} rows")


# ----------------------------------------------------------------------------- analyze
def doy_clim(x, years):
    s = x[x.index.year.isin(years)]
    c = s.groupby(np.minimum(s.index.dayofyear.values, 365)).mean().reindex(range(1, 366))
    c = pd.concat([c.iloc[-15:], c, c.iloc[:15]]).rolling(31, center=True, min_periods=10).mean().iloc[15:-15]
    c.index = range(1, 366)
    return pd.Series(c.reindex(np.minimum(x.index.dayofyear.values, 365)).values, index=x.index)


def anom(x, last):
    pre, post = range(START, BREAK), range(BREAK, last + 1)
    return x - pd.Series(np.where(x.index.year >= BREAK, doy_clim(x, post), doy_clim(x, pre)), index=x.index)


def stats_by_year(x, y, years):
    """per-year sufficient statistics n, Sx, Sy, Sxx, Sxy, Syy"""
    ok = np.isfinite(x) & np.isfinite(y)
    df = pd.DataFrame(dict(yr=years[ok], x=x[ok], y=y[ok]))
    df["xx"], df["xy"], df["yy"] = df.x ** 2, df.x * df.y, df.y ** 2
    g = df.groupby("yr").agg(n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"),
                             sxx=("xx", "sum"), sxy=("xy", "sum"), syy=("yy", "sum"))
    return g[g.n >= 10].values


def beta_r(S):
    n, sx, sy, sxx, sxy, syy = (S[..., i] for i in range(6))
    cxx, cxy, cyy = n * sxx - sx ** 2, n * sxy - sx * sy, n * syy - sy ** 2
    return cxy / cxx, cxy / np.sqrt(cxx * cyy)


def analyze(wvar):
    from statsmodels.stats.multitest import multipletests
    d = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    rng = np.random.default_rng(42)
    rows = []
    for lb, g in d.groupby("lonbin"):
        g = g.set_index("date").sort_index().asfreq("D")
        dsia = g.SIA - g.SIA.shift(1)
        g = g[g.index.year >= START]
        dsia = dsia[dsia.index.year >= START]
        last = g.index.year.max()
        ya, xa = anom(dsia, last), anom(g[wvar], last)
        for m in range(1, 13):
            sel = ya.index.month == m
            yrs = ya.index.year.values[sel]
            x, y = xa.values[sel], ya.values[sel]
            pre = stats_by_year(x[yrs < BREAK], y[yrs < BREAK], yrs[yrs < BREAK])
            post = stats_by_year(x[yrs >= BREAK], y[yrs >= BREAK], yrs[yrs >= BREAK])
            rec = dict(lonbin=lb, month=m, n_years_pre=len(pre), n_years_post=len(post))
            if len(pre) >= MIN_YEARS and len(post) >= max(3, MIN_YEARS // 2):
                bp, rp = beta_r(pre.sum(0))
                bq, rq = beta_r(post.sum(0))
                Bp = pre[rng.integers(0, len(pre), (N_BOOT, len(pre)))].sum(1)
                Bq = post[rng.integers(0, len(post), (N_BOOT, len(post)))].sum(1)
                (bp_s, rp_s), (bq_s, rq_s) = beta_r(Bp), beta_r(Bq)
                dd = bq_s - bp_s
                dd = dd[np.isfinite(dd)]
                pv = lambda s: min(1.0, (2 * min((s <= 0).sum(), (s >= 0).sum()) + 1) / (len(s) + 1))
                rec.update(beta_pre=bp, beta_post=bq, r_pre=rp, r_post=rq,
                           p_r_pre=pv(rp_s[np.isfinite(rp_s)]), p_r_post=pv(rq_s[np.isfinite(rq_s)]),
                           dbeta=bq - bp, dbeta_lo=np.percentile(dd, 2.5), dbeta_hi=np.percentile(dd, 97.5),
                           p_dbeta=pv(dd))
            rows.append(rec)
    res = pd.DataFrame(rows)
    ok = res.p_dbeta.notna() if "p_dbeta" in res else pd.Series(False, index=res.index)
    res["q_dbeta"] = np.nan
    if ok.any():
        res.loc[ok, "q_dbeta"] = multipletests(res.loc[ok, "p_dbeta"], method="fdr_bh")[1]
    out = f"{TAB}/lonbin_month_coupling_{wvar}.csv"
    res.to_csv(out, index=False, float_format="%.5g")
    print(f"wrote {out}")
    v = res[ok]
    print(f"{wvar}: {len(v)} bin-months tested; beta change p<0.05: {(v.p_dbeta < 0.05).sum()} "
          f"(expected by chance ~{0.05 * len(v):.0f}); FDR q<0.05: {(v.q_dbeta < 0.05).sum()}")
    print(f"  median r 1988-2015 = {v.r_pre.median():+.2f}; 2016-2023 = {v.r_post.median():+.2f}; "
          f"cells with r significant pre: {(v.p_r_pre < 0.05).mean():.0%}")


# ----------------------------------------------------------------------------- plot
def plot(wvar):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ch4_style as st
    res = pd.read_csv(f"{TAB}/lonbin_month_coupling_{wvar}.csv")
    res["xpos"] = (res.lonbin - LON0) % 360
    grid = lambda col: res.pivot(index="month", columns="xpos", values=col).reindex(
        index=range(1, 13), columns=range(0, 360, DLON)).values
    xe, ye = np.arange(0, 361, DLON), np.arange(0.5, 13.5)
    rmax = np.nanpercentile(np.abs(np.r_[grid("r_pre").ravel(), grid("r_post").ravel()]), 98)
    rmax = max(0.1, np.ceil(rmax * 10) / 10)
    wlab = {"tau_y": "northward wind stress", "tau_mag": "wind-stress magnitude"}[wvar]
    panels = [("r_pre", f"(a) 1988–2015: correlation of daily ΔSIA′ with {wlab}", rmax, "p_r_pre"),
              ("r_post", "(b) 2016–2023", rmax, "p_r_post"),
              (None, "(c) Change, (b) − (a); boxes: change in β  (thick q < 0.05, thin p < 0.05)", rmax, None)]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.2), sharex=True, gridspec_kw=dict(hspace=0.28))
    for ax, (col, title, vmax, pcol) in zip(axes, panels):
        Z = grid("r_post") - grid("r_pre") if col is None else grid(col)
        im = ax.pcolormesh(xe, ye, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
        if pcol is not None:
            P = grid(pcol)
            for (i, j) in zip(*np.nonzero(P < 0.05)):
                ax.plot(xe[j] + DLON / 2, i + 1, ".", color=st.INK, ms=2.2)
        else:
            Q, Pd = grid("q_dbeta"), grid("p_dbeta")
            for (i, j) in zip(*np.nonzero(Pd < 0.05)):
                strong = Q[i, j] < 0.05
                ax.add_patch(Rectangle((xe[j], i + 0.5), DLON, 1, fill=False, ec=st.INK,
                                       lw=1.6 if strong else 0.6, zorder=4))
        for sec, lo in SECTOR_EDGES.items():
            x0 = (lo - LON0) % 360
            ax.axvline(x0, color="0.35", lw=0.7, ls=(0, (3, 2)), zorder=5)
            if ax is axes[0]:
                ax.text(x0 + 35, 12.75, sec, color=st.SECTOR_COLORS[sec], ha="center", va="bottom",
                        fontproperties=st.bold_font_properties(size=9))
        ax.set_yticks(range(1, 13))
        ax.set_yticklabels(list("JFMAMJJASOND"), fontsize=8)
        ax.set_ylim(0.5, 12.5)
        ax.set_title(title, loc="left", fontsize=9, pad=20 if ax is axes[0] else 4)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("Δr" if col is None else "r", fontsize=8)
        cb.outline.set_visible(False)
    ticks = [0, 70, 140, 210, 280, 360]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(["20°E", "90°E", "160°E", "130°W", "60°W", "20°E"])
    axes[-1].set_xlabel("Longitude (10° bins)")
    axes[0].text(1.0, -0.02, "dots: p < 0.05", transform=axes[0].transAxes, ha="right", va="top", fontsize=7)
    out = f"{FIG}/fig4_local_coupling_{wvar}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    wv = sys.argv[2] if len(sys.argv) > 2 else "tau_y"
    if step == "--build":
        build()
    elif step == "--analyze":
        analyze(wv)
    elif step == "--plot":
        plot(wv)
    else:
        print(__doc__)
