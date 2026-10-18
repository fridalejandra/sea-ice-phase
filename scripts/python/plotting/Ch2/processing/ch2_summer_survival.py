#!/usr/bin/env python3
"""
ch2_summer_survival.py

Summer survival as an SIC-only proxy for ice that becomes multi-year, and its link to the
dynamic method.

  survival[y]  : 5-day-smoothed SIC stays >= 0.15 through the summer minimum period
                 (1 Feb - 10 Mar of year y+1), i.e. the ice of season y did not melt out.
                 Summers y = 1979..2023 (2024 needs Feb 2025; 1986 unavailable).
  dyn_only_MS  : pixel-years where the dynamic method dates a Melt Start and the static
                 method cannot -- a within-pack decline that never reaches 15%.

Outputs
  (a) surviving-ice area by sector, 1979-2015 vs 2016-2023 (year-bootstrap 95% CI)
  (b) the same for survival in two consecutive summers (older-ice proxy)
  (c) dynamic-only Melt Start pixel-years by sector, pre vs post, and the share that are
      also survival years (should be high if dynamic-only = within-pack decline)
  (d) newly seasonal pixels (survived >= 50% of pre summers, <= 20% of post summers):
      count by sector and, if ch2_consolidation_time.nc exists, their post-2016
      consolidation time vs established long-season interior pixels
  (e) map: survival frequency pre, post, change  -> Fig99_summer_survival_pre_post.png

Caveat: survival at a location in consecutive summers is not the same ice (drift), so this
is a persistently-ice-covered-location proxy, not ice age.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
_TOL_SUNSET_R = LinearSegmentedColormap.from_list("tol_sunset_r", [
    "#A50026", "#DD3D2D", "#F67E4B", "#FDB366", "#FEDA8B", "#EAECCC",
    "#C2E4EF", "#98CAE1", "#6EA6CD", "#4A7BB7", "#364B9A"])
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compute_phase_dates_v3 as v3  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.python.plotting.Ch2.utils.ch2_fig_utils import (  # noqa: E402
    set_mpl_defaults, format_fig_name, get_fig_path, save_and_upload)

SENSOR = "SMMR"
ROOT = PROJECT_ROOT
ANOM = ROOT / "data" / "anomalies" / "SMMR"
SECT = ROOT / "data" / "canonical_sectors.nc"
LANDMASK = ROOT / "data" / "landmask_sic.nc"
CONS_NC = ROOT / "results" / "ch2_consolidation_time.nc"
OUT = ROOT / "results"
REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"

SECTORS = {1: "ABS", 2: "WS", 3: "KHV", 4: "EA", 5: "RS"}
TAGS = {("FS", "static"): "FS_static_thr15_k5", ("FS", "dynamic"): "FS_dynamic_k5_q70",
        ("MS", "static"): "MS_static_thr15_k5", ("MS", "dynamic"): "MS_dynamic_k5_q70"}
PRE, POST = (1979, 2015), (2016, 2023)
SURV_WIN = ("-02-01", "-03-10")
THETA, K = 0.15, 5
MIN_DAYS = 10
NEW_PRE, NEW_POST = 0.5, 0.2
MONTH, AUG15_DOY, UB, COAST_KM, PIXEL_KM = 30.4, 227, 8.5, 100, 25.0
NBOOT = 2000
rng = np.random.default_rng(42)

DATA_CRS = ccrs.Stereographic(
    central_latitude=-90, central_longitude=0, true_scale_latitude=-70,
    globe=ccrs.Globe(ellipse=None, semimajor_axis=6378273.0, semiminor_axis=6356889.449))
MAP_CRS = ccrs.SouthPolarStereo()


def _var(path, names):
    with xr.open_dataset(path, decode_times=False) as ds:
        for n in names:
            if n in ds:
                return ds[n].load()
    raise KeyError(f"{names} not in {path}")


def load_dates(phase, method):
    t = TAGS[(phase, method)]
    cs = [f"{t}_clim_dsa", f"{t}_clim"] if phase == "MS" else [f"{t}_clim"]
    an = [f"{t}_anom_dsa", f"{t}_anom"] if phase == "MS" else [f"{t}_anom"]
    clim = _var(ANOM / f"{t}_climatology.nc", cs)
    anom = _var(ANOM / f"{t}_anomalies.nc", an)
    return clim.values[None] + anom.values, anom["year"].values, clim["x"].values, clim["y"].values


def boot_change(pre, post):
    pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
    if len(pre) < 3 or len(post) < 3:
        return (np.nan,) * 5
    bs = (rng.choice(post, (NBOOT, len(post))).mean(1)
          - rng.choice(pre, (NBOOT, len(pre))).mean(1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pre.mean(), post.mean(), post.mean() - pre.mean(), lo, hi


def fmt(b, nd=1):
    if not np.isfinite(b[2]):
        return "nan"
    star = "*" if (b[3] > 0 or b[4] < 0) else " "
    return f"{b[2]:+.{nd}f} [{b[3]:+.{nd}f},{b[4]:+.{nd}f}]{star}"


def main():
    with xr.open_dataset(SECT) as m:
        sid = m["sector_id"].values
        try:
            cell_km2 = float(m["area_m2"].values) / 1e6
        except Exception:
            cell_km2 = np.nan
    if not np.isfinite(cell_km2) or cell_km2 <= 0:
        cell_km2 = PIXEL_KM ** 2
    land = xr.open_dataset(LANDMASK)["land"].values.astype(bool)

    print(f"Loading daily SIC ({SENSOR}) ...", flush=True)
    ice365, _ = v3.load_sic(SENSOR)

    years = np.arange(PRE[0], POST[1] + 1)
    surv = np.full((len(years),) + land.shape, np.nan)
    for i, y in enumerate(years):
        ts = v3.slice_season(ice365, f"{y+1}{SURV_WIN[0]}", f"{y+1}{SURV_WIN[1]}")
        nobs = int(np.isfinite(ts.values).any(axis=(1, 2)).sum()) if ts.time.size else 0
        if nobs < MIN_DAYS:
            print(f"  summer {y}/{y+1}: skipped ({nobs} observed days)", flush=True)
            continue
        s = ts.rolling(time=K, center=True, min_periods=1).mean()
        smin = s.min("time").values
        surv[i] = np.where(land | ~np.isfinite(smin), np.nan, (smin >= THETA).astype(float))
        print(f"  summer {y}/{y+1}: {int(np.nansum(surv[i]))} surviving pixels", flush=True)
    del ice365

    ipre = (years >= PRE[0]) & (years <= PRE[1])
    ipost = (years >= POST[0]) & (years <= POST[1])
    consec = np.full_like(surv, np.nan)
    consec[1:] = np.where(np.isfinite(surv[1:]) & np.isfinite(surv[:-1]), surv[1:] * surv[:-1], np.nan)

    regions = [(SECTORS[s], sid == s) for s in SECTORS] + [("ALL", np.isfinite(sid))]

    def area_series(field, sel):
        return np.array([np.nansum(field[i][sel]) * cell_km2 / 1e3 if np.isfinite(field[i]).any()
                         else np.nan for i in range(len(years))])

    rows = []
    for lab, sel in regions:
        a = area_series(surv, sel)
        c = area_series(consec, sel)
        ba, bc = boot_change(a[ipre], a[ipost]), boot_change(c[ipre], c[ipost])
        rows.append(dict(sector=lab,
                         surv_area_pre=round(ba[0], 1), surv_area_post=round(ba[1], 1),
                         surv_change=fmt(ba), surv_change_pct=round(100 * ba[2] / ba[0], 0) if ba[0] else np.nan,
                         two_summer_pre=round(bc[0], 1), two_summer_post=round(bc[1], 1),
                         two_summer_change=fmt(bc)))
    df_a = pd.DataFrame(rows)
    ts_df = pd.DataFrame({"year": years, **{f"surv_{lab}_1e3km2": area_series(surv, sel) for lab, sel in regions}})

    ms_sta, yrs_d, x, yy = load_dates("MS", "static")
    ms_dyn, _, _, _ = load_dates("MS", "dynamic")
    idx = {int(v): j for j, v in enumerate(yrs_d)}
    dyn_only = np.full_like(surv, np.nan)
    for i, y in enumerate(years):
        if y in idx:
            j = idx[y]
            dyn_only[i] = np.where(land, np.nan, (np.isfinite(ms_dyn[j]) & ~np.isfinite(ms_sta[j])).astype(float))
    rows = []
    for lab, sel in regions:
        n_pre = np.nansum(dyn_only[ipre][:, sel]) / ipre.sum()
        n_post = np.nansum(dyn_only[ipost][:, sel]) / ipost.sum()
        both = (dyn_only == 1) & (surv == 1)
        share = np.nansum(both[:, sel]) / max(np.nansum(dyn_only[:, sel]), 1)
        rows.append(dict(sector=lab, dyn_only_px_per_yr_pre=round(n_pre, 0),
                         dyn_only_px_per_yr_post=round(n_post, 0),
                         change_pct=round(100 * (n_post - n_pre) / n_pre, 0) if n_pre else np.nan,
                         share_also_survival_pct=round(100 * share, 0)))
    df_c = pd.DataFrame(rows)

    f_pre, f_post = np.nanmean(surv[ipre], 0), np.nanmean(surv[ipost], 0)
    newly = (f_pre >= NEW_PRE) & (f_post <= NEW_POST)
    rows = [dict(sector=lab, newly_seasonal_px=int((newly & sel).sum()),
                 area_1e3km2=round((newly & sel).sum() * cell_km2 / 1e3, 1)) for lab, sel in regions]
    df_d = pd.DataFrame(rows)
    cons_txt = "(consolidation comparison skipped: results/ch2_consolidation_time.nc not found)"
    if CONS_NC.exists():
        cons = xr.open_dataset(CONS_NC)
        cy = cons["year"].values
        cd = cons["cons_days"].values[cy >= POST[0]]
        fs_sta, _, _, _ = load_dates("FS", "static")
        ms_pre = ms_sta[(yrs_d >= 1988) & (yrs_d <= 2015)]
        fs_pre = fs_sta[(yrs_d >= 1988) & (yrs_d <= 2015)]
        dur = np.nanmean((ms_pre + AUG15_DOY) - fs_pre, 0) / MONTH
        interior = distance_transform_edt(~land) * PIXEL_KM > COAST_KM
        est = (dur > UB) & interior & ~newly
        med_new = np.nanmedian(cd[:, newly]) if newly.any() else np.nan
        med_est = np.nanmedian(cd[:, est]) if est.any() else np.nan
        n_new = int(np.isfinite(cd[:, newly]).sum())
        cons_txt = (f"Post-2016 consolidation time (median days): newly seasonal pixels {med_new:.1f} "
                    f"(n={n_new} freeze-ups) vs established long-season interior {med_est:.1f}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    for title, df, fname in [
        (f"(a,b) Surviving-ice area (10^3 km^2), summers {PRE[0]}-{PRE[1]} vs {POST[0]}-{POST[1]}; "
         "change [95% CI]", df_a, "ch2_survival_area"),
        ("(c) Dynamic-only Melt Start (static cannot date it): pixel-years per year", df_c,
         "ch2_survival_dynamic_only"),
        (f"(d) Newly seasonal pixels (survived >= {NEW_PRE:.0%} of pre summers, <= {NEW_POST:.0%} of post)",
         df_d, "ch2_survival_newly_seasonal")]:
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))
        df.to_csv(OUT / f"{fname}.csv", index=False)
    print(f"\n{cons_txt}")
    ts_df.to_csv(OUT / "ch2_survival_area_timeseries.csv", index=False)
    xr.Dataset({"survival": (("year", "y", "x"), surv)}, coords={"year": years}).to_netcdf(
        OUT / "ch2_summer_survival.nc")

    set_mpl_defaults()
    _lon_grid = xr.open_dataset(SECT)["lon"].values
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), subplot_kw=dict(projection=MAP_CRS),
                             constrained_layout=True)
    sid_plot = np.where(~land, sid, np.nan)
    for ax, (title, data, cmap, vmin, vmax, ext, clab), letter in zip(axes, [
            (f"{PRE[0]}–{PRE[1]}", np.where(f_pre > 0, f_pre, np.nan), "Blues", 0, 1, "neither", "Survival frequency"),
            (f"{POST[0]}–{POST[1]}", np.where(f_post > 0, f_post, np.nan), "Blues", 0, 1, "neither", "Survival frequency"),
            ("Change (post − pre)", np.where((f_pre > 0) | (f_post > 0), f_post - f_pre, np.nan),
             _TOL_SUNSET_R, -0.6, 0.6, "both", "Δ survival frequency")], "abc"):
        im = ax.pcolormesh(x, yy, data, transform=DATA_CRS, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto", zorder=1)
        ax.set_extent([-180, 180, -90, -55], ccrs.PlateCarree())
        ax.contour(x, yy, sid_plot, levels=[1.5, 2.5, 3.5, 4.5], colors="0.15", linewidths=0.6,
                   transform=DATA_CRS, zorder=3)
        if letter == "a":
            for _s, _lab in SECTORS.items():
                _sel = (sid == _s) & ~land & np.isfinite(_lon_grid)
                if _sel.any():
                    _r = np.deg2rad(_lon_grid[_sel])
                    _lon = np.rad2deg(np.arctan2(np.sin(_r).mean(), np.cos(_r).mean()))
                    ax.text(_lon, -57, _lab, transform=ccrs.PlateCarree(), ha="center",
                            va="center", fontsize=8, fontweight="bold", zorder=8,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="0.6", zorder=4.5)
        ax.coastlines(linewidth=0.4, zorder=4.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold", zorder=7)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.8, extend=ext)
        cb.ax.grid(False)
        cb.outline.set_visible(False)
        cb.set_label(clab, fontsize=9)
    fig_name = "FigS3_summer_survival.png"
    save_and_upload(fig, get_fig_path(project_root=PROJECT_ROOT, subfolder="", fig_name=fig_name),
                    remote_root=REMOTE_ROOT, remote_subdir="")


if __name__ == "__main__":
    main()
