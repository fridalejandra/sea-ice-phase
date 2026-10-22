#!/usr/bin/env python
"""
cellwise_residual_map.py -- the wind-independent drift change dC as a MAP.
Per cell and season, from the sums written by compute_cellwise_sums.py, fit
U_ice = A*U_wind + C over pre (1992-2015) and post (2016-2020) years, take
dC = C_post - C_pre, year-block bootstrap + Benjamini-Hochberg FDR across cells.
Also 10-degree longitude bins of dC with CI, and a sanity table against the
sector-mean regression of windfactor_analysis.py.
    python cellwise_residual_map.py [--nboot 2000] [--q 0.10] [--replot]
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from pyproj import CRS, Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_figstyle as fs  # noqa: E402

ROOT = fs.ROOT
SUMS = f"{ROOT}/results/ch4/cellwise"
DRIFT_NC = f"{ROOT}/results/NSIDC0116/NSIDC0116_uv_season_year.nc"
SECTOR_NC = f"{ROOT}/data/canonical_sectors.nc"
LAND_NC = f"{ROOT}/data/landmask_sic.nc"
WF_TAB = f"{ROOT}/results/ch4/tables/windfactor_prepost.csv"
OUT_NC = f"{SUMS}/dc_cellwise.nc"
OUT_LON = f"{ROOT}/results/ch4/tables/dc_lonbins.csv"
OUT_CHK = f"{ROOT}/results/ch4/tables/dc_cellwise_sector_check.csv"
BASE0, BASE1, POST0, POST1 = 1992, 2015, 2016, 2020
SEASONS = ["MAM", "JJA", "SON"]
SEED = 42
MIN_DAYS_PRE, MIN_DAYS_POST, MIN_YEARS_PRE, MIN_YEARS_POST = 1000, 150, 15, 3
LONBIN = 10
SECTOR_EDGES = {"ABS": (-110, -70), "WS": (-70, -14), "KH": (-14, 71), "EA": (71, 162)}   # RA wraps 162 -> -110


def fit(n, sw, si, sww, swi, sii):
    """Closed-form complex least squares from sums. Arrays broadcast; n > 1 assumed."""
    n = n.astype(np.float64)
    den = n * sww - np.abs(sw) ** 2
    A = (n * swi - np.conj(sw) * si) / den
    C = (si - A * sw) / n
    ssres = (sii + np.abs(A) ** 2 * sww + n * np.abs(C) ** 2 - 2 * (np.conj(A) * swi).real
             - 2 * (np.conj(C) * si).real + 2 * (A * np.conj(C) * sw).real)
    sstot = sii - np.abs(si) ** 2 / n
    return A, C, 1 - ssres / sstot


def load_sums(season):
    files = sorted(glob.glob(f"{SUMS}/sums_*.nc"))
    ds = xr.concat([xr.open_dataset(f).sel(season=season) for f in files], dim="syear")
    out = {"year": ds.syear.values.astype(int), "n": ds.n.values.astype(np.float64),
           "sww": ds.sww.values.astype(np.float64), "sii": ds.sii.values.astype(np.float64)}
    for k in ("sw", "si", "swi"):
        out[k] = ds[k + "_re"].values.astype(np.float64) + 1j * ds[k + "_im"].values.astype(np.float64)
    return out, ds.y.values, ds.x.values


def bh(p):
    """Benjamini-Hochberg q-values (NaN-safe)."""
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(m); qq[order] = np.minimum(ranked, 1)
    q[ok] = qq
    return q


def analyse(nboot, qthr):
    rng = np.random.default_rng(SEED)
    sector = xr.open_dataset(SECTOR_NC)["sector_id"].values
    d = xr.open_dataset(DRIFT_NC)["speed"]
    dpre, dpost = d.sel(year=slice(BASE0, BASE1)), d.sel(year=slice(POST0, POST1))
    fixed_all = ((dpre.mean("year") > 0.03) & (np.isfinite(dpre).sum("year") >= 15)
                 & (np.isfinite(dpost).sum("year") >= 3))
    wf = pd.read_csv(WF_TAB)
    fields, lonrows, chkrows, lon = {}, [], [], None
    for sea in SEASONS:
        S, yv, xv = load_sums(sea)
        if lon is None:
            to_wgs = Transformer.from_crs(CRS.from_epsg(3412), CRS.from_epsg(4326), always_xy=True)
            X, Y = np.meshgrid(xv, yv)
            lon, lat = to_wgs.transform(X, Y)
        yrs = S["year"]
        ipre, ipost = np.where((yrs >= BASE0) & (yrs <= BASE1))[0], np.where((yrs >= POST0) & (yrs <= POST1))[0]
        npre, npost = S["n"][ipre].sum(0), S["n"][ipost].sum(0)
        fixed = fixed_all.sel(season=sea).values
        use = (fixed & (npre >= MIN_DAYS_PRE) & (npost >= MIN_DAYS_POST)
               & ((S["n"][ipre] > 20).sum(0) >= MIN_YEARS_PRE) & ((S["n"][ipost] > 20).sum(0) >= MIN_YEARS_POST))
        cells = np.where(use.ravel())[0]
        print(f"{sea}: {len(cells)} cells, {len(ipre)} pre years, {len(ipost)} post years")
        st = {k: S[k].reshape(len(yrs), -1)[:, cells] for k in ("n", "sw", "si", "sww", "swi", "sii")}

        def fit_years(idx):
            return fit(*[st[k][idx].sum(0) for k in ("n", "sw", "si", "sww", "swi", "sii")])

        Apre, Cpre, r2pre = fit_years(ipre)
        Apost, Cpost, r2post = fit_years(ipost)
        dc = Cpost - Cpre
        dhat = dc / np.abs(dc)

        lonc = lon.ravel()[cells]
        bins = np.arange(-180, 181, LONBIN)
        bidx = np.clip(np.digitize(lonc, bins) - 1, 0, len(bins) - 2)
        secc = sector.ravel()[cells]
        sec_ids = {sid: np.where(secc == sid)[0] for sid in fs.SECTOR_ID}

        neg = np.zeros(len(cells)); within45 = np.zeros(len(cells))
        lon_e = np.empty((nboot, len(bins) - 1)); lon_n = np.empty((nboot, len(bins) - 1))
        sec_b = {sid: np.empty(nboot, complex) for sid in sec_ids}
        for b in range(nboot):
            _, cp, _ = fit_years(rng.choice(ipre, len(ipre)))
            _, cq, _ = fit_years(rng.choice(ipost, len(ipost)))
            dcb = cq - cp
            proj = (dcb * np.conj(dhat)).real
            neg += proj <= 0
            within45 += np.abs(np.angle(dcb * np.conj(dhat))) < np.pi / 4
            cnt = np.bincount(bidx, minlength=len(bins) - 1).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                lon_e[b] = np.bincount(bidx, weights=dcb.real, minlength=len(bins) - 1) / cnt
                lon_n[b] = np.bincount(bidx, weights=dcb.imag, minlength=len(bins) - 1) / cnt
            for sid, ii in sec_ids.items():
                sec_b[sid][b] = dcb[ii].mean() if len(ii) else np.nan
        p = np.minimum(1, 2 * neg / nboot)
        q = bh(p)
        sig = q < qthr
        print(f"   {sig.sum()} cells with q < {qthr} ({100 * sig.mean():.0f}%); "
              f"median |dC| {np.median(np.abs(dc)) * 100:.2f} cm/s; median R2 pre {np.median(r2pre):.2f}")

        def full(v, fill=np.nan, dtype=float):
            f = np.full(lon.size, fill, dtype=dtype); f[cells] = v; return f.reshape(lon.shape)
        fields[sea] = dict(c_pre_e=full(Cpre.real), c_pre_n=full(Cpre.imag), c_post_e=full(Cpost.real),
                           c_post_n=full(Cpost.imag), dc_e=full(dc.real), dc_n=full(dc.imag),
                           dc_mag=full(np.abs(dc)), p=full(p), q=full(q), sig=full(sig, False, bool),
                           consistency=full(within45 / nboot), alpha_pre=full(np.abs(Apre)),
                           alpha_post=full(np.abs(Apost)), r2_pre=full(r2pre), r2_post=full(r2post),
                           n_pre=full(npre.ravel()[cells]), n_post=full(npost.ravel()[cells]), used=full(True, False, bool))

        cnt = np.bincount(bidx, minlength=len(bins) - 1)
        for k in range(len(bins) - 1):
            if cnt[k] < 10:
                continue
            lonrows.append(dict(season=sea, lon_lo=bins[k], lon_hi=bins[k + 1], n_cells=int(cnt[k]),
                                dc_e=dc.real[bidx == k].mean(), dc_n=dc.imag[bidx == k].mean(),
                                dc_e_lo=np.nanpercentile(lon_e[:, k], 2.5), dc_e_hi=np.nanpercentile(lon_e[:, k], 97.5),
                                dc_n_lo=np.nanpercentile(lon_n[:, k], 2.5), dc_n_hi=np.nanpercentile(lon_n[:, k], 97.5)))
        for sid, code in fs.SECTOR_ID.items():
            ii = sec_ids[sid]
            w = wf[(wf.sector == code) & (wf.season == sea)]
            if not len(ii) or w.empty:
                continue
            w = w.iloc[0]
            m = dc[ii].mean()
            chkrows.append(dict(sector=code, season=sea, n_cells=len(ii),
                                cellmean_dc_e=m.real, cellmean_dc_n=m.imag, cellmean_dc_dir=np.degrees(np.angle(m)),
                                cellmean_dc_lo=np.percentile(np.abs(sec_b[sid]), 2.5),
                                cellmean_dc_hi=np.percentile(np.abs(sec_b[sid]), 97.5),
                                sectorfit_dc_e=w.dc_e, sectorfit_dc_n=w.dc_n, sectorfit_dc_dir=w.dc_dir,
                                frac_cells_sig=sig[ii].mean(),
                                cellmean_c_pre=abs(Cpre[ii].mean()), sectorfit_c_pre=abs(complex(w.c_e_pre, w.c_n_pre))))

    ds = xr.Dataset({f"{k}": (("season", "y", "x"), np.stack([fields[s][k] for s in SEASONS]))
                     for k in fields[SEASONS[0]]},
                    coords=dict(season=SEASONS, y=yv, x=xv, lon=(("y", "x"), lon), lat=(("y", "x"), lat)))
    ds.attrs.update(pre=f"{BASE0}-{BASE1}", post=f"{POST0}-{POST1}", nboot=nboot, fdr_q=qthr,
                    p="two-sided bootstrap p for dC != 0 along the point-estimate heading")
    ds.to_netcdf(OUT_NC)
    pd.DataFrame(lonrows).to_csv(OUT_LON, index=False)
    chk = pd.DataFrame(chkrows); chk.to_csv(OUT_CHK, index=False)
    print(f"wrote {OUT_NC}\n      {OUT_LON}\n      {OUT_CHK}")
    return ds


def fig10(ds, lonb):
    sector = xr.open_dataset(SECTOR_NC)["sector_id"].values
    try:
        lm = xr.open_dataset(LAND_NC); v = [k for k in lm.data_vars if lm[k].ndim == 2][0]
        land = (lm[v].values != 0) & np.isfinite(lm[v].values)
    except Exception:
        land = None
    x, y = ds.x.values, ds.y.values
    X, Y = np.meshgrid(x, y)
    used_any = ds.used.any("season").values | (land if land is not None else False)
    xs, ys = X[used_any], Y[used_any]
    pad = 0.04
    xlim = (xs.min() - pad * np.ptp(xs), xs.max() + pad * np.ptp(xs))
    ylim = (ys.min() - pad * np.ptp(ys), ys.max() + pad * np.ptp(ys))

    man = fs.mode() == "manuscript"
    fig = plt.figure(figsize=fs.size(1.0, 0.66 if man else 0.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.62], hspace=0.05, wspace=0.06)
    step, key = (5, 0.01)
    sl = (slice(None, None, step), slice(None, None, step))
    scale = key / (step * 25000 * 1.5)
    for j, sea in enumerate(SEASONS):
        ax = fig.add_subplot(gs[0, j])
        f = ds.sel(season=sea)
        ax.pcolormesh(x, y, np.where(f.used.values, 1.0, np.nan), cmap=ListedColormap(["#F1F1F1"]),
                      shading="auto", zorder=1, rasterized=True)
        if land is not None:
            ax.pcolormesh(x, y, np.where(land, 1.0, np.nan), cmap=ListedColormap(["#3b3b3b"]), shading="auto", zorder=2)
        ax.contour(x, y, np.nan_to_num(sector, nan=0), levels=[1.5, 2.5, 3.5, 4.5], colors="#777", linewidths=0.5, zorder=3)
        U, V, S = f.dc_e.values, f.dc_n.values, f.sig.values
        for mask, col, z in ((~S, "#B5B5B5", 4), (S, "#111", 5)):
            Um, Vm = np.where(mask, U, np.nan), np.where(mask, V, np.nan)
            q = ax.quiver(X[sl], Y[sl], Um[sl], Vm[sl], color=col, scale_units="xy", angles="xy", scale=scale,
                          width=0.0045, headwidth=3.5, headlength=4.5, zorder=z)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        nsig, nuse = int(S.sum()), int(f.used.values.sum())
        ax.set_title(sea, loc="left")
        ax.set_anchor("S")
        ax.text(1.0, 1.0, f"{100 * nsig / nuse:.0f}% of cells q < {ds.attrs['fdr_q']}", transform=ax.transAxes, ha="right", va="bottom", color=fs.INK, fontsize=plt.rcParams["legend.fontsize"])
        if j == 2:
            ax.quiverkey(q, 0.5, -0.06, key, f"{key * 100:g} cm s$^{{-1}}$", labelpos="E", coordinates="axes",
                         fontproperties={"size": plt.rcParams["legend.fontsize"]})

        ax2 = fig.add_subplot(gs[1, j]); ax2.set_anchor("N")
        t = lonb[lonb.season == sea]
        lc = (t.lon_lo + t.lon_hi) / 2
        ax2.axhline(0, color=fs.INK, lw=0.6, zorder=1)
        for code, (lo, hi) in SECTOR_EDGES.items():
            ax2.axvspan(lo, hi, color=fs.SECTOR_COLORS[code], alpha=0.10, lw=0, zorder=0)
        for lo, hi in ((162, 180), (-180, -110)):
            ax2.axvspan(lo, hi, color=fs.SECTOR_COLORS["RA"], alpha=0.10, lw=0, zorder=0)
        ax2.fill_between(lc, t.dc_n_lo * 100, t.dc_n_hi * 100, color="#999", alpha=0.35, lw=0, zorder=2)
        ax2.plot(lc, t.dc_n * 100, color=fs.INK, lw=fs.lw(0.8), zorder=3)
        ax2.set_xlim(-180, 180); ax2.set_xticks([-90, 0, 90])
        ax2.set_xticklabels(["90°W", "0°", "90°E"])
        ax2.set_xlabel("Longitude")
        if j == 0:
            ax2.set_ylabel("ΔC northward\n(cm s$^{-1}$)")
        else:
            ax2.tick_params(labelleft=False)
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ylims = [a.get_ylim() for a in fig.axes if a.get_xlabel() == "Longitude"]
    lo, hi = min(l for l, _ in ylims), max(h for _, h in ylims)
    for a in fig.axes:
        if a.get_xlabel() == "Longitude":
            a.set_ylim(lo, hi)
    h = [Line2D([0], [0], color="#111", lw=fs.lw(0.8), label="ΔC, FDR q < %.2f" % ds.attrs["fdr_q"]),
         Line2D([0], [0], color="#B5B5B5", lw=fs.lw(0.8), label="ΔC, not significant"),
         Line2D([0], [0], color="#999", lw=6, alpha=0.35, label="10° bin mean, 95% CI")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fs.save(fig, "fig10_dc_map")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--q", type=float, default=0.10)
    ap.add_argument("--replot", action="store_true", help="skip the analysis, use the saved files")
    a = ap.parse_args()
    if a.replot and os.path.exists(OUT_NC):
        ds = xr.open_dataset(OUT_NC)
    else:
        ds = analyse(a.nboot, a.q)
    lonb = pd.read_csv(OUT_LON)
    pd.set_option("display.width", 220)
    chk = pd.read_csv(OUT_CHK)
    print("\nSanity: fixed-cell mean of the cellwise dC vs the sector-mean regression dC (windfactor_prepost.csv)")
    print(chk[["sector", "season", "n_cells", "cellmean_dc_dir", "sectorfit_dc_dir", "cellmean_dc_lo", "cellmean_dc_hi",
               "frac_cells_sig", "cellmean_c_pre", "sectorfit_c_pre"]].round(3).to_string(index=False))
    for m in ("manuscript", "slide"):
        fs.set_mode(m)
        fig10(ds, lonb)
