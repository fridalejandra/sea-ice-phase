#!/usr/bin/env python
"""
04_variance_prepost.py  --  Ch4, Results point 3b
Did the DAY-TO-DAY VARIABILITY of divergence / divergence-only / convergence-only
change after the break year?

Statistic (per grid cell, per season, per variable)
1. Split record into season-years (DJF of year Y = Dec Y-1 + Jan-Feb Y).
2. Within each season-year, variance of daily values about that season-year's
   own mean (removes mean shift + interannual variability).
     metric "anom": var of daily values about the season-year mean
     metric "diff": var of day-to-day differences (consecutive days only)
3. Pool per period: sum((n_i-1) v_i) / sum(n_i-1). Report log2(var_post/var_pre).
4. Significance: Welch t-test on log(v_i) across SEASON-YEARS (robust to daily
   autocorrelation). Field significance: Benjamini-Hochberg FDR, alpha=0.10.
5. Two baselines: pre_full (first..BREAK_YEAR-1) and pre_recent
   (PRE_RECENT_START..BREAK_YEAR-1) to guard against NSIDC-0116 input changes.

Outputs
derived_nc/variance_prepost_{var}.nc
figures/prepost_variance_ratio_{var}_{metric}.png
tables/variance_prepost_by_sector_season.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# --------------------------------------------------------------------------- config
ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
IN_NC = ROOT / "results/ch4/derived_nc/ease_divergence_with_latlon.nc"
OUT_NC_DIR = ROOT / "results/ch4/derived_nc"
FIG_DIR = ROOT / "results/ch4/figures"
TAB_DIR = ROOT / "results/ch4/tables"

VARS = ["divergence", "div_positive", "div_negative"]
VAR_LABEL = {"divergence": "Net divergence",
             "div_positive": "Divergence (opening)",
             "div_negative": "Convergence (closing)"}
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}

BREAK_YEAR = 2016        # post = season-years >= BREAK_YEAR. MATCH your beta3 script.
PRE_RECENT_START = 2003  # robustness baseline start
MIN_DAYS = 20            # min valid days per cell per season-year
MIN_YEARS_PRE = 10
MIN_YEARS_POST = 5
FDR_ALPHA = 0.10

# !!! CHECK: replace with the exact sector bounds used for your poster/sector table.
# Degrees east in [-180, 180). (lon_min, lon_max); wraps across 180 if min > max.
SECTORS = {
    "WS":  (-60.0,   20.0),   # Weddell Sea
    "KH":  ( 20.0,   90.0),   # King Haakon VII / Indian Ocean
    "EA":  ( 90.0,  160.0),   # East Antarctica / Western Pacific
    "RS":  (160.0, -130.0),   # Ross Sea (wraps dateline)
    "ABS": (-130.0, -60.0),   # Amundsen-Bellingshausen
    "SH":  (-180.0, 180.0),   # whole Southern Ocean
}

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------------------------------------------------------------- helpers
def find_name(ds, candidates):
    for c in candidates:
        if c in ds.variables:
            return c
    raise KeyError(f"None of {candidates} found. Variables: {list(ds.variables)}")


def season_year(times):
    return times.year + (times.month == 12).astype(int)


def season_window(season, Y):
    m = SEASONS[season]
    if season == "DJF":
        start = pd.Timestamp(Y - 1, 12, 1)
    else:
        start = pd.Timestamp(Y, m[0], 1)
    end = pd.Timestamp(Y, m[-1], 1) + pd.offsets.MonthEnd(0)
    return start, end


def sector_mask(lon2d, lo, hi):
    if lo <= hi:
        return (lon2d >= lo) & (lon2d < hi)
    return (lon2d >= lo) | (lon2d < hi)


def nanvar_n(x, axis=0):
    n = np.isfinite(x).sum(axis=axis)
    v = np.nanvar(x, axis=axis, ddof=1)
    v = np.where(n >= MIN_DAYS, v, np.nan)
    return v.astype("float64"), n


def pooled(v, n, sel):
    v, n = v[sel], n[sel].astype("float64")
    ok = np.isfinite(v)
    w = np.where(ok, n - 1.0, 0.0)
    num = np.nansum(np.where(ok, v, 0.0) * w, axis=0)
    den = w.sum(axis=0)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def welch_logvar(v, sel_a, sel_b):
    L = np.where(v > 0, np.log(np.where(v > 0, v, 1.0)), np.nan)
    A, B = L[sel_a], L[sel_b]
    na, nb = np.isfinite(A).sum(0), np.isfinite(B).sum(0)
    ma, mb = np.nanmean(A, 0), np.nanmean(B, 0)
    va, vb = np.nanvar(A, 0, ddof=1), np.nanvar(B, 0, ddof=1)
    sa, sb = va / na, vb / nb
    se = np.sqrt(sa + sb)
    t = (mb - ma) / se
    dof = (sa + sb) ** 2 / (sa ** 2 / (na - 1) + sb ** 2 / (nb - 1))
    p = 2 * stats.t.sf(np.abs(t), dof)
    bad = (na < MIN_YEARS_PRE) | (nb < MIN_YEARS_POST) | ~np.isfinite(p)
    return np.where(bad, np.nan, p), na, nb


def fdr_mask(p, alpha=FDR_ALPHA):
    flat = p.ravel()
    ok = np.isfinite(flat)
    pv = np.sort(flat[ok])
    m = pv.size
    out = np.zeros(flat.shape, bool)
    if m == 0:
        return out.reshape(p.shape)
    passed = pv <= alpha * np.arange(1, m + 1) / m
    if passed.any():
        thresh = pv[np.nonzero(passed)[0].max()]
        out[ok] = flat[ok] <= thresh
    return out.reshape(p.shape)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(IN_NC))
    ap.add_argument("--nc-dir", default=str(OUT_NC_DIR))
    ap.add_argument("--fig-dir", default=str(FIG_DIR))
    ap.add_argument("--tab-dir", default=str(TAB_DIR))
    ap.add_argument("--vars", nargs="+", default=VARS)
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    for d in (args.nc_dir, args.fig_dir, args.tab_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.inp)
    tdim = "time"
    latn = find_name(ds, ["lat", "latitude", "nav_lat", "LAT"])
    lonn = find_name(ds, ["lon", "longitude", "nav_lon", "LON"])
    lat2d = ds[latn].values
    lon2d = ((ds[lonn].values + 180.0) % 360.0) - 180.0
    times = pd.DatetimeIndex(ds[tdim].values)
    if not times.is_monotonic_increasing:
        raise ValueError("time axis not sorted")
    months = times.month.values
    syear = season_year(times).values
    t0, t1 = times[0], times[-1]
    print(f"Input {args.inp}\n  {t0.date()} -> {t1.date()}, {len(times)} steps, grid {lat2d.shape}")
    print(f"  BREAK_YEAR={BREAK_YEAR}  PRE_RECENT_START={PRE_RECENT_START}  MIN_DAYS={MIN_DAYS}")
    print("  Sectors (CHECK these match your poster):", dict(SECTORS))

    smask = {k: sector_mask(lon2d, *b) & np.isfinite(lat2d) for k, b in SECTORS.items()}
    rows = []

    for var in args.vars:
        da = ds[var]
        spatial = [d for d in da.dims if d != tdim]
        da = da.transpose(tdim, *spatial)
        ny, nx = da.shape[1:]
        probe = da.isel({tdim: slice(len(times) // 2, len(times) // 2 + 5)}).values
        fin = probe[np.isfinite(probe)]
        if fin.size:
            print(f"\n[{var}] probe: {np.mean(fin == 0):.1%} exact zeros, "
                  f"{np.mean(fin > 0):.1%} >0, {np.mean(fin < 0):.1%} <0 among finite values")

        out = {}
        for season, mths in SEASONS.items():
            in_s = np.isin(months, mths)
            Ys = []
            for Y in np.unique(syear[in_s]):
                s, e = season_window(season, Y)
                if s >= t0 and e <= t1:
                    Ys.append(int(Y))
            Ys = np.array(Ys)
            nY = len(Ys)
            V = {m: np.full((nY, ny, nx), np.nan) for m in ("anom", "diff")}
            N = {m: np.zeros((nY, ny, nx), np.int16) for m in ("anom", "diff")}
            SV = {m: {k: np.full(nY, np.nan) for k in SECTORS} for m in ("anom", "diff")}
            SN = {m: {k: np.zeros(nY, int) for k in SECTORS} for m in ("anom", "diff")}

            for i, Y in enumerate(Ys):
                idx = np.nonzero(in_s & (syear == Y))[0]
                x = da.isel({tdim: slice(idx[0], idx[-1] + 1)}).values.astype("float64")
                tt = times[idx[0]: idx[-1] + 1]
                consec = np.diff(tt.values).astype("timedelta64[h]") == np.timedelta64(24, "h")
                d = x[1:] - x[:-1]
                d[~consec] = np.nan
                V["anom"][i], N["anom"][i] = nanvar_n(x)
                V["diff"][i], N["diff"][i] = nanvar_n(d)
                for k, msk in smask.items():
                    sx = np.nanmean(x[:, msk], axis=1)
                    sd = sx[1:] - sx[:-1]
                    sd[~consec] = np.nan
                    for m, arr in (("anom", sx), ("diff", sd)):
                        v, n = nanvar_n(arr)
                        SV[m][k][i], SN[m][k][i] = v, n
            print(f"  {var} {season}: {nY} season-years {Ys.min()}-{Ys.max()}")

            post = Ys >= BREAK_YEAR
            pre_full = Ys < BREAK_YEAR
            pre_rec = (Ys >= PRE_RECENT_START) & (Ys < BREAK_YEAR)
            for m in ("anom", "diff"):
                vpost = pooled(V[m], N[m], post)
                res = {"var_post": vpost}
                for tag, pre in (("full", pre_full), ("recent", pre_rec)):
                    vpre = pooled(V[m], N[m], pre)
                    p, na, nb = welch_logvar(V[m], pre, post)
                    sig = fdr_mask(p)
                    res[f"var_pre_{tag}"] = vpre
                    res[f"log2_ratio_{tag}"] = np.log2(vpost / vpre)
                    res[f"p_{tag}"] = p
                    res[f"sig_fdr_{tag}"] = sig.astype("int8")
                    res[f"nyears_pre_{tag}"] = na.astype("int16")
                    for k in SECTORS:
                        sv = SV[m][k]
                        vpo = pooled(sv[:, None], SN[m][k][:, None], post)[0]
                        vpr = pooled(sv[:, None], SN[m][k][:, None], pre)[0]
                        ps, nas, nbs = welch_logvar(sv[:, None], pre, post)
                        cell = smask[k] & np.isfinite(res[f"log2_ratio_{tag}"])
                        lr = res[f"log2_ratio_{tag}"][cell]
                        rows.append(dict(
                            variable=var, season=season, metric=m, baseline=tag,
                            sector=k, pre_years=f"{Ys[pre].min()}-{Ys[pre].max()}",
                            post_years=f"{Ys[post].min()}-{Ys[post].max()}",
                            n_years_pre=int(nas[0]), n_years_post=int(nbs[0]),
                            sd_pre=np.sqrt(vpr), sd_post=np.sqrt(vpo),
                            sd_ratio=np.sqrt(vpo / vpr), log2_var_ratio=np.log2(vpo / vpr),
                            p_welch_logvar=ps[0],
                            cells_median_log2_ratio=np.median(lr) if lr.size else np.nan,
                            cells_frac_increase=np.mean(lr > 0) if lr.size else np.nan,
                            cells_frac_sig_increase=(np.mean((sig[cell] == 1) & (lr > 0))
                                                     if lr.size else np.nan),
                            cells_frac_sig_decrease=(np.mean((sig[cell] == 1) & (lr < 0))
                                                     if lr.size else np.nan),
                            n_cells=int(cell.sum())))
                out[(season, m)] = res

        keys = list(next(iter(out.values())).keys())
        dv = {}
        for m in ("anom", "diff"):
            for key in keys:
                arr = np.stack([out[(s, m)][key] for s in SEASONS])
                dv[f"{key}_{m}"] = (("season", *spatial), arr)
        ods = xr.Dataset(dv, coords={"season": list(SEASONS)})
        for c in spatial:
            if c in ds.coords:
                ods = ods.assign_coords({c: ds[c]})
        ods[latn] = ((spatial[0], spatial[1]), lat2d)
        ods[lonn] = ((spatial[0], spatial[1]), ds[lonn].values)
        ods.attrs.update(source=str(args.inp), variable=var, break_year=BREAK_YEAR,
                         pre_recent_start=PRE_RECENT_START, min_days=MIN_DAYS,
                         fdr_alpha=FDR_ALPHA,
                         note=("var = pooled within-season-year variance; anom = about "
                               "season-year mean; diff = of consecutive-day differences; "
                               "p = Welch t on log(season-year variance)"))
        f_nc = Path(args.nc_dir) / f"variance_prepost_{var}.nc"
        ods.to_netcdf(f_nc)
        print(f"  wrote {f_nc}")

        if not args.no_figs:
            for m in ("anom", "diff"):
                plot_maps(out, var, m, lat2d, lon2d, Path(args.fig_dir))

    df = pd.DataFrame(rows)
    f_csv = Path(args.tab_dir) / "variance_prepost_by_sector_season.csv"
    df.to_csv(f_csv, index=False, float_format="%.5g")
    print(f"\nwrote {f_csv}")
    show = df[(df.metric == "anom")].pivot_table(
        index=["variable", "sector"], columns=["baseline", "season"], values="sd_ratio")
    print("\nSD ratio post/pre (metric=anom):\n", show.round(2).to_string())


def plot_maps(out, var, m, lat2d, lon2d, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.SouthPolarStereo()
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.2), subplot_kw={"projection": proj})
    stride = 3
    sub = (slice(None, None, stride), slice(None, None, stride))
    im = None
    for r, tag in enumerate(("full", "recent")):
        for c, season in enumerate(SEASONS):
            ax = axes[r, c]
            res = out[(season, m)]
            ax.set_extent([-180, 180, -90, -52], ccrs.PlateCarree())
            im = ax.pcolormesh(lon2d, lat2d, res[f"log2_ratio_{tag}"], cmap="RdBu_r",
                               vmin=-1, vmax=1, transform=ccrs.PlateCarree(), shading="auto")
            sig = res[f"sig_fdr_{tag}"][sub].astype(bool)
            ax.scatter(lon2d[sub][sig], lat2d[sub][sig], s=1.5, c="k", lw=0,
                       transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
            ax.coastlines(lw=0.4, zorder=3)
            if r == 0:
                ax.set_title(season, fontsize=12)
    lab = {"full": "vs 1979-" + str(BREAK_YEAR - 1),
           "recent": f"vs {PRE_RECENT_START}-{BREAK_YEAR - 1}"}
    for r, tag in enumerate(("full", "recent")):
        axes[r, 0].text(-0.08, 0.5, lab[tag], transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=11)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04, pad=0.04,
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_xticklabels(["0.5x", "0.71x", "1x", "1.41x", "2x"])
    what = "daily values about season-year mean" if m == "anom" else "day-to-day differences"
    cb.set_label(f"Variance ratio, post ({BREAK_YEAR}-) / pre   [{what}]"
                 f"   dots: FDR-significant (alpha={FDR_ALPHA})")
    fig.suptitle(f"{VAR_LABEL[var]}: change in day-to-day variability", fontsize=14)
    f = fig_dir / f"prepost_variance_ratio_{var}_{m}.png"
    fig.savefig(f, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {f}")


if __name__ == "__main__":
    main()
