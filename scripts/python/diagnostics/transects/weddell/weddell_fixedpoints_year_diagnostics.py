#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# --------------------------
# Helpers
# --------------------------
def _find_var(ds: xr.Dataset, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    return None


def _find_sector_var(ds: xr.Dataset):
    candidates = [
        "sector", "sectors", "sector_id", "sector_mask", "canonical_sectors",
        "sector_index", "region", "region_id"
    ]
    for v in candidates:
        if v in ds.data_vars and set(ds[v].dims) >= {"y", "x"}:
            return v
    for c in candidates:
        if c in ds.coords and set(ds[c].dims) >= {"y", "x"}:
            return c
    return None


def _wrap_lon_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)


def _month_bounds(year: int, month: int):
    t0 = np.datetime64(f"{year}-{month:02d}-01")
    if month == 12:
        t1 = np.datetime64(f"{year+1}-01-01")
    else:
        t1 = np.datetime64(f"{year}-{month+1:02d}-01")
    t_end = t1 - np.timedelta64(1, "D")
    return t0, t_end


def _require_1d_finite(name: str, arr: np.ndarray):
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise RuntimeError(f"{name} must be 1D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name} contains NaN/Inf; cannot use for plotting.")
    return arr.astype(float)


def _build_lon_transect_indices(lon2d, sector2d, dist2d, sector_id, target_lon, min_rows=80, debug=False):
    lon = lon2d.values
    sec = sector2d.values
    dist = dist2d.values
    ny, _ = lon.shape

    ys, xs = [], []
    for yi in range(ny):
        m = (sec[yi, :] == sector_id) & np.isfinite(lon[yi, :]) & np.isfinite(dist[yi, :])
        if not np.any(m):
            continue
        cand = np.where(m)[0]
        diffs = _wrap_lon_diff(lon[yi, cand], target_lon)
        xi = int(cand[np.argmin(diffs)])
        ys.append(yi)
        xs.append(xi)

    ys = np.asarray(ys, dtype=int)
    xs = np.asarray(xs, dtype=int)

    if ys.size < min_rows:
        raise RuntimeError(f"Transect too short: {ys.size} rows (min_rows={min_rows}).")

    if debug:
        medlon = float(np.nanmedian(lon[ys, xs]))
        print(f"[debug] transect rows={ys.size}, median lon={medlon:.2f}", flush=True)

    return ys, xs


def _cumulative_distance_polyline_km(x_m, y_m):
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    ds = np.sqrt(dx * dx + dy * dy) / 1000.0
    return np.concatenate([[0.0], np.cumsum(ds)])


def _plot_2x2(times, series_dict, labels, title, ylabel, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.ravel()

    for i, lab in enumerate(labels):
        ax = axes[i]
        y = np.asarray(series_dict[lab], dtype=float)
        x = times

        ax.plot(x, y)
        ax.set_title(lab)
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.5, alpha=0.4)

        finite = np.isfinite(y)
        if finite.any():
            ymin = float(np.nanmin(y[finite]))
            ymax = float(np.nanmax(y[finite]))
            pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
            ax.set_ylim(ymin - pad, ymax + pad)

    axes[2].set_xlabel("Date")
    axes[3].set_xlabel("Date")

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def _parse_years(year: int, years: str | None):
    if years is None:
        return [int(year)]
    years = years.strip()
    if "-" in years and "," not in years:
        a, b = years.split("-")
        a = int(a); b = int(b)
        return list(range(a, b + 1))
    out = []
    for tok in years.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _monthly_mean_panel(fig, axes, x_sic, y_sic, land_under, sic, extent_xy, year, pt_xy, pt_colors, labels,
                        ax_proj, data_crs):
    mappable = None
    for mi, month in enumerate(range(1, 13)):
        ax = axes[mi]
        ax.set_extent(extent_xy, crs=data_crs)
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)

        ax.pcolormesh(x_sic, y_sic, land_under, transform=data_crs,
                      shading="auto", alpha=0.25, zorder=2)

        t0, t1 = _month_bounds(year, month)
        sic_m = sic.sel(time=slice(t0, t1)).mean("time", skipna=True)

        m = ax.pcolormesh(x_sic, y_sic, sic_m.values, transform=data_crs,
                          shading="auto", vmin=0.0, vmax=1.0, zorder=3)
        mappable = m

        # points
        px, py = pt_xy
        for i in range(len(labels)):
            ax.scatter(px[i], py[i], s=45, color=pt_colors[i],
                       edgecolor="k", linewidth=0.4, transform=data_crs, zorder=6)

        ax.set_title(f"{year}-{month:02d}", fontsize=10)

    cb = fig.colorbar(mappable, ax=axes.tolist(), fraction=0.03, pad=0.02)
    cb.set_label("Monthly mean SIC (fraction)")
    fig.suptitle(f"Monthly mean SIC ({year}) at fixed Weddell transect points", y=0.98)


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Fixed-point Weddell diagnostics (ref-month fixed point selection)")
    ap.add_argument("--sic", required=True, type=Path, help="Merged daily SIC NetCDF")
    ap.add_argument("--var", default="N07_ICECON", help="SIC variable name")
    ap.add_argument("--sectors", required=True, type=Path, help="canonical_sectors.nc")
    ap.add_argument("--dist-dir", required=True, type=Path, help="Directory containing dist_to_edge_km_monthMM.nc")
    ap.add_argument("--dist-pattern", default="dist_to_edge_km_month{mm:02d}.nc",
                    help="Filename pattern inside dist-dir")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory (base weddell folder)")
    ap.add_argument("--year", type=int, default=2022, help="Single year (ignored if --years is provided)")
    ap.add_argument("--years", type=str, default=None, help="Year list '1980,1981' or range '1980-2023'")
    ap.add_argument("--target-lon", type=float, default=-30.0, help="Target lon (degE, -30=30W)")
    ap.add_argument("--weddell-id", type=int, required=True, help="Weddell sector numeric ID")
    ap.add_argument("--fractions", default="0.3,0.5,0.7,0.9", help="Fractions (exactly 4)")
    ap.add_argument("--ref-month", type=int, default=9, help="Reference month for fixed point selection")
    ap.add_argument("--min-transect-rows", type=int, default=80, help="Minimum rows in transect build")
    ap.add_argument("--edge-eps-km", type=float, default=1e-3, help="Edge threshold for dist_to_edge (km)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--mask-year", type=int, default=2022,
                    help="Year used to define fixed-point geometry/mask (default 2022)")

    args = ap.parse_args()

    years = _parse_years(args.year, args.years)

    fracs = [float(x) for x in args.fractions.split(",") if x.strip() != ""]
    if len(fracs) != 4:
        raise ValueError("Expect exactly 4 fractions, e.g. 0.3,0.5,0.7,0.9")
    if not all(0.0 < f < 1.0 for f in fracs):
        raise ValueError("All fractions must be between 0 and 1 (exclusive).")

    labels = [f"f_{f:.1f}" for f in fracs]
    cmap_pts = cm.get_cmap("tab10", len(fracs))
    pt_colors = [cmap_pts(i) for i in range(len(fracs))]

    # output structure
    base = args.outdir / "fixed_points"
    base.mkdir(parents=True, exist_ok=True)
    maps_panel_dir = base / "maps_monthly_mean_panels"
    maps_panel_dir.mkdir(parents=True, exist_ok=True)
    ts_dir = base / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    # ---- sectors grid
    ds_sec = xr.open_dataset(args.sectors)
    sector_var = _find_sector_var(ds_sec)
    if sector_var is None:
        raise RuntimeError("Could not find sector variable in canonical_sectors.nc")
    lat2d = ds_sec["lat"].transpose("y", "x")
    lon2d = ds_sec["lon"].transpose("y", "x")
    sector2d = ds_sec[sector_var].transpose("y", "x")
    x_coords = ds_sec["x"].values.astype(float)
    y_coords = ds_sec["y"].values.astype(float)

    # ---- SIC
    ds_sic = xr.open_dataset(args.sic)
    sic_raw = ds_sic[args.var]

    if (ds_sic.sizes["y"] != ds_sec.sizes["y"]) or (ds_sic.sizes["x"] != ds_sec.sizes["x"]):
        raise RuntimeError("Sector grid and SIC grid shapes differ. Stop and regrid/match indices first.")

    x_sic = _require_1d_finite("ds_sic['x']", ds_sic["x"].values)
    y_sic = _require_1d_finite("ds_sic['y']", ds_sic["y"].values)
    extent_xy = [float(x_sic.min()), float(x_sic.max()), float(y_sic.min()), float(y_sic.max())]

    land_2d = (sic_raw.isel(time=0) == 1200)
    ocean_2d = ~land_2d
    land_under = land_2d.astype(float).values
    land_under[land_under == 0] = np.nan

    sic = sic_raw.where(sic_raw < 100)
    sic = sic.where((sic >= 0.0) & (sic <= 1.0))

    # ---- dist field for reference month (fixed point selection)
    refm = int(args.ref_month)
    dist_path = args.dist_dir / args.dist_pattern.format(mm=refm)
    if not dist_path.exists():
        raise FileNotFoundError(f"Missing dist file for ref month{refm:02d}: {dist_path}")
    ds_d = xr.open_dataset(dist_path)
    dvar = _find_var(ds_d, ["dist_to_edge_km", "dist_to_edge", "distance_to_edge_km"])
    if dvar is None:
        raise RuntimeError(f"Could not find dist var in {dist_path}")
    dist2d_ref = ds_d[dvar].transpose("y", "x")

    # ---- ocean_valid for reference month (selection gate) USING FIRST YEAR IN LIST
    # This is intentional: fixed points are defined once.
    sel_year = int(args.mask_year)
    t0_ref, t1_ref = _month_bounds(sel_year, refm)
    sic_ref_month = sic.sel(time=slice(t0_ref, t1_ref))
    ocean_valid_ref = ocean_2d & np.isfinite(sic_ref_month).any("time")

    # ---- build transect once using ref month dist field
    ty, tx = _build_lon_transect_indices(
        lon2d=lon2d, sector2d=sector2d, dist2d=dist2d_ref,
        sector_id=args.weddell_id, target_lon=args.target_lon,
        min_rows=args.min_transect_rows, debug=args.debug
    )

    t_lat = lat2d.values[ty, tx]
    t_lon = lon2d.values[ty, tx]
    t_dist = dist2d_ref.values[ty, tx]
    t_ocean = ocean_2d.values[ty, tx].astype(bool)
    t_ok = ocean_valid_ref.values[ty, tx].astype(bool)

    keep = np.isfinite(t_lat) & np.isfinite(t_lon) & np.isfinite(t_dist) & t_ocean & t_ok
    tyk = ty[keep]
    txk = tx[keep]
    latk = t_lat[keep]
    lonk = t_lon[keep]
    distk = t_dist[keep]

    if tyk.size < 30:
        raise RuntimeError(f"Too few valid transect points after ref-month filtering: {tyk.size}")

    # distance along transect
    x_path_m = x_coords[txk]
    y_path_m = y_coords[tyk]
    s_km = _cumulative_distance_polyline_km(x_path_m, y_path_m)

    # anchors
    coast_i = int(np.argmin(latk))
    edge_candidates = np.where(distk <= args.edge_eps_km)[0]
    if edge_candidates.size > 0:
        edge_i = int(edge_candidates[np.argmax(latk[edge_candidates])])
    else:
        dmin = float(np.nanmin(distk))
        close = np.where(np.isclose(distk, dmin, atol=1e-6))[0]
        edge_i = int(close[np.argmax(latk[close])])
        if args.debug:
            print(f"[debug] ref month: no dist<=eps; using min dist={dmin:.3f} km", flush=True)

    s_coast = float(s_km[coast_i])
    s_edge = float(s_km[edge_i])
    if s_edge <= s_coast:
        s_coast, s_edge = s_edge, s_coast
    D = s_edge - s_coast
    if D <= 0:
        raise RuntimeError(f"Non-positive coast->edge distance D={D:.3f} km")

    seg_idx = np.where((s_km >= s_coast) & (s_km <= s_edge))[0]
    if seg_idx.size < 10:
        raise RuntimeError("Too few cells between coast and edge anchors along the transect.")

    # select fixed points ONCE
    points = []
    for f, lab in zip(fracs, labels):
        target_s = s_coast + f * D
        j = int(seg_idx[np.argmin(np.abs(s_km[seg_idx] - target_s))])
        points.append({
            "point": lab,
            "fraction": f,
            "x_idx": int(txk[j]),
            "y_idx": int(tyk[j]),
            "lat": float(latk[j]),
            "lon": float(lonk[j]),
            "dist_along_km_ref": float(s_km[j] - s_coast),
            "dist_to_edge_km_ref": float(distk[j]),
            "ref_month": refm,
            "ref_year_mask": sel_year,
            "target_lon": args.target_lon,
            "weddell_id": args.weddell_id,
        })

    df = pd.DataFrame(points).sort_values("fraction").reset_index(drop=True)
    points_csv = base / f"points_fixed_lon{args.target_lon:.0f}_refmonth{refm:02d}_maskyear{sel_year}.csv"
    df.to_csv(points_csv, index=False)
    print(f"[info] wrote fixed points CSV: {points_csv}", flush=True)

    # precompute point xy
    px = x_sic[df["x_idx"].astype(int).values]
    py = y_sic[df["y_idx"].astype(int).values]
    pt_xy = (px, py)

    # plotting CRS
    ax_proj = ccrs.SouthPolarStereo()
    data_crs = ccrs.epsg(3412)

    # ---- per-year outputs
    for Y in years:
        # (1) 3x4 monthly mean maps
        fig, axes = plt.subplots(3, 4, figsize=(16, 12), subplot_kw={"projection": ax_proj})
        axes = axes.ravel()
        _monthly_mean_panel(fig, axes, x_sic, y_sic, land_under, sic, extent_xy, Y,
                            pt_xy, pt_colors, labels, ax_proj, data_crs)
        outpng = maps_panel_dir / f"monthly_mean_SIC_3x4_{Y}.png"
        fig.savefig(outpng, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[info] wrote monthly panel: {outpng}", flush=True)

        # (2) full-year SIC and ΔSIC at fixed points
        t0y = np.datetime64(f"{Y}-01-01")
        t1y = np.datetime64(f"{Y+1}-01-01")
        sic_y = sic.sel(time=slice(t0y, t1y - np.timedelta64(1, "D")))
        times = sic_y["time"].values

        ts = {}
        for lab, yi, xi in zip(df["point"].values, df["y_idx"].astype(int).values, df["x_idx"].astype(int).values):
            ts[lab] = sic_y.isel(y=int(yi), x=int(xi)).values.astype(float)

        # ΔSIC with NaN safety: if either day is NaN → NaN
        dts = {}
        for lab in labels:
            v = ts[lab]
            dv = np.full_like(v, np.nan, dtype=float)
            good = np.isfinite(v[1:]) & np.isfinite(v[:-1])
            dv[1:][good] = v[1:][good] - v[:-1][good]
            dts[lab] = dv

        sic_ts_path = ts_dir / f"sic_2x2_{Y}.png"
        _plot_2x2(times, ts, labels, f"Daily SIC at fixed points ({Y})", "SIC", sic_ts_path)

        dsic_ts_path = ts_dir / f"dsic_2x2_{Y}.png"
        _plot_2x2(times, dts, labels, f"Daily ΔSIC at fixed points ({Y})", "ΔSIC", dsic_ts_path)

        print(f"[info] wrote timeseries: {sic_ts_path}", flush=True)
        print(f"[info] wrote delta:      {dsic_ts_path}", flush=True)


if __name__ == "__main__":
    main()
