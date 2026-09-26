#!/usr/bin/env python
"""
12_fig4_opening_trend_map.py -- Ch4 Figure 4 (option A):
where did opening/closing intensity rise, relative to where the ice is?

- 100 km blocks (4 x 4 native 25 km EASE cells), JJA and SON, 1988-2024
- per block: season-year mean opening (div_positive) and closing (-div_negative)
- trend = OLS of log(season-year mean) on year -> % per decade; BH-FDR (alpha 0.10) stippling
- contours: where the drift product has ice on >= 50% of days,
  1988-2015 (solid) vs 2016-2024 (dashed)  [ice-presence proxy from the drift mask]
Outputs: figures/fig4_opening_closing_trend_map.png, derived_nc/opening_closing_trend_blocks100km.nc
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
DIV_NC = f"{ROOT}/results/ch4/derived_nc/ease_divergence_with_latlon.nc"
OUT_NC = f"{ROOT}/results/ch4/derived_nc/opening_closing_trend_blocks100km.nc"
OUT_FIG = f"{ROOT}/results/ch4/figures/fig4_opening_closing_trend_map.png"
START, END, BREAK = 1988, 2024, 2016
SEASONS = {"JJA": (6, 7, 8), "SON": (9, 10, 11)}
B = 4                 # 4 x 25 km = 100 km blocks
MIN_FRAC = 0.3        # block-season valid if >= 30% of its cell-days have a value
MIN_YEARS = 20
FDR_ALPHA = 0.10
VMAX = 15             # colour range, % per decade


def block_mean(a):
    """nanmean over B x B spatial blocks; a: (t, y, x) -> (t, y//B, x//B); also valid fraction."""
    t, ny, nx = a.shape
    a = a[:, :ny // B * B, :nx // B * B].reshape(t, ny // B, B, nx // B, B)
    return np.nanmean(a, axis=(0, 2, 4)), np.isfinite(a).mean(axis=(0, 2, 4))


def bh(p):
    flat = p.ravel(); ok = np.isfinite(flat); pv = np.sort(flat[ok]); m = pv.size
    out = np.zeros(flat.shape, bool)
    if m:
        passed = pv <= FDR_ALPHA * np.arange(1, m + 1) / m
        if passed.any():
            out[ok] = flat[ok] <= pv[np.nonzero(passed)[0].max()]
    return out.reshape(p.shape)


def main():
    ds = xr.open_dataset(DIV_NC)
    latn = "lat" if "lat" in ds else "latitude"
    lonn = "lon" if "lon" in ds else "longitude"
    lat, lon = ds[latn].values, ds[lonn].values
    pos = ds["div_positive"].transpose("time", ...)
    neg = ds["div_negative"].transpose("time", ...)
    t = pd.DatetimeIndex(ds.time.values)
    ny, nx = lat.shape
    yrs = np.arange(START, END + 1)

    res, presence = {}, {}
    for s, months in SEASONS.items():
        vals = {"opening": np.full((len(yrs), ny // B, nx // B), np.nan),
                "closing": np.full((len(yrs), ny // B, nx // B), np.nan)}
        ice_days = {"pre": np.zeros((ny, nx)), "post": np.zeros((ny, nx))}
        n_days = {"pre": 0, "post": 0}
        for i, y in enumerate(yrs):
            idx = np.nonzero((t.year == y) & np.isin(t.month, months))[0]
            if len(idx) < 60:
                continue
            sl = slice(idx[0], idx[-1] + 1)
            p = pos.isel(time=sl).values.astype("float32")
            n = -neg.isel(time=sl).values.astype("float32")
            for name, arr in (("opening", p), ("closing", n)):
                m, frac = block_mean(arr)
                # a cell is "open" OR "closing" on a given day, so valid frac is ~half the ice frac
                vals[name][i] = np.where(frac >= MIN_FRAC / 2, m, np.nan)
            per = "post" if y >= BREAK else "pre"
            ice_days[per] += (np.isfinite(p) | np.isfinite(n)).sum(0)
            n_days[per] += len(idx)
            print(f"  {s} {y}", flush=True)
        for per in ("pre", "post"):
            presence[(s, per)] = ice_days[per] / max(n_days[per], 1)
        for name, v in vals.items():
            L = np.log(np.where(v > 0, v, np.nan))
            ok = np.isfinite(L)
            nyr = ok.sum(0)
            x = np.where(ok, yrs[:, None, None] - yrs.mean(), np.nan)
            xm, Lm = np.nanmean(x, 0), np.nanmean(L, 0)
            sxx = np.nansum((x - xm) ** 2, 0)
            slope = np.nansum((x - xm) * (L - Lm), 0) / sxx
            resid = L - (Lm + slope * (x - xm))
            se = np.sqrt(np.nansum(resid ** 2, 0) / (nyr - 2) / sxx)
            pval = 2 * stats.t.sf(np.abs(slope / se), nyr - 2)
            good = nyr >= MIN_YEARS
            trend = np.where(good, (np.exp(slope * 10) - 1) * 100, np.nan)
            pval = np.where(good, pval, np.nan)
            res[(s, name)] = (trend, pval, bh(pval))

    # block-centre coordinates
    c = B // 2
    blat = lat[c:ny // B * B:B, c:nx // B * B:B]
    blon = lon[c:ny // B * B:B, c:nx // B * B:B]
    dv = {}
    for (s, name), (tr, pv, sig) in res.items():
        dv[f"{name}_trend_pct_dec_{s}"] = (("by", "bx"), tr)
        dv[f"{name}_p_{s}"] = (("by", "bx"), pv)
        dv[f"{name}_sig_fdr_{s}"] = (("by", "bx"), sig.astype("int8"))
    for (s, per), fr in presence.items():
        dv[f"ice_presence_{s}_{per}"] = (("y", "x"), fr)
    out = xr.Dataset(dv)
    out["block_lat"] = (("by", "bx"), blat); out["block_lon"] = (("by", "bx"), blon)
    out["lat"] = (("y", "x"), lat); out["lon"] = (("y", "x"), lon)
    out.attrs.update(years=f"{START}-{END}", block_km=25 * B, fdr_alpha=FDR_ALPHA,
                     note="trend = OLS on log(season-year mean), % per decade")
    out.to_netcdf(OUT_NC)
    print(f"wrote {OUT_NC}")

    # summary: area-fraction of blocks with significant rise / fall
    for (s, name), (tr, pv, sig) in res.items():
        fin = np.isfinite(tr)
        print(f"  {name:8s} {s}: median {np.nanmedian(tr):+.1f}%/dec; "
              f"rising {np.mean(tr[fin] > 0):.0%}; FDR-sig rise {np.mean(sig[fin] & (tr[fin] > 0)):.0%}, "
              f"sig fall {np.mean(sig[fin] & (tr[fin] < 0)):.0%}  (n={fin.sum()} blocks)")
    plot(res, presence, blat, blon, lat, lon)


def plot(res, presence, blat, blon, lat, lon):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.SouthPolarStereo()
    P = proj.transform_points(ccrs.PlateCarree(), blon, blat)
    BX, BY = P[..., 0], P[..., 1]
    Q = proj.transform_points(ccrs.PlateCarree(), lon, lat)
    QX, QY = Q[..., 0], Q[..., 1]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10.4), subplot_kw={"projection": proj})
    im = None
    for r, name in enumerate(("opening", "closing")):
        for c, s in enumerate(SEASONS):
            ax = axes[r, c]
            tr, pv, sig = res[(s, name)]
            ax.set_extent([-180, 180, -90, -53], ccrs.PlateCarree())
            im = ax.pcolormesh(BX, BY, tr, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, shading="nearest")
            ax.scatter(BX[sig], BY[sig], s=1.2, c="k", lw=0)
            for per, ls in (("pre", "-"), ("post", "--")):
                ax.contour(QX, QY, presence[(s, per)], levels=[0.5], colors="k",
                           linewidths=1.0, linestyles=ls)
            ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=3)
            ax.coastlines(lw=0.4, zorder=4)
            ax.set_title(f"{name.capitalize()} intensity, {s}", fontsize=12)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.035, pad=0.04, extend="both")
    cb.set_label(f"Trend {START}–{END} (% per decade)   ·   dots: FDR-significant (α={FDR_ALPHA})\n"
                 f"contours: ice on ≥50% of days, {START}–{BREAK - 1} (solid) vs {BREAK}–{END} (dashed)",
                 fontsize=10)
    fig.savefig(OUT_FIG, dpi=250, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
