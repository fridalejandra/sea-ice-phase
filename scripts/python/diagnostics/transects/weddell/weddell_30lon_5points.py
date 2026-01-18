#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weddell_30lon_points.py

Pick fractional-distance points along a Weddell meridional transect near 30°W
using a monthly climatological distance-to-edge field (e.g., month11).

Key behavior:
  1) SIC is fractional (0–1) but Bootstrap-style files include flag codes:
       1100 = missing, 1200 = land (and generally anything >=100 is non-physical)
     We mask SIC >= 100 to NaN, then enforce 0<=SIC<=1.
  2) Land mask is taken explicitly from the SIC flags (==1200). Points are forced to ocean.
  3) Ocean validity mask requires (ocean) AND (any finite SIC in target month).
  4) Plotting uses Cartopy with a SouthPolarStereo display projection, and treats the
     gridded x/y meters as being in EPSG:3412 (data CRS).
  5) Single-panel reference map (NO overview), auto-zoomed to Weddell + finite SIC.
  6) Transect is a true ~constant-lon path: for each y-row, choose the x-cell in Weddell
     whose lon is closest to target_lon → diagonal/curved in projection.

Outputs (in --outdir):
  - weddell_lon30_fracpoints_monthMM_YYYY.csv
  - weddell_lon30_fracpoints_monthMM_YYYY_map.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
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
    """Try common names for the sector ID variable (must be 2D y,x)."""
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
    """Smallest absolute difference between two longitudes (degrees), handling wrap."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)


def _month_bounds(year: int, month: int):
    """Return numpy datetime64 start (inclusive) and end (inclusive) for a month."""
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


def _build_lon_transect_indices(
    lon2d: xr.DataArray,
    sector2d: xr.DataArray,
    dist2d: xr.DataArray,
    sector_id: int,
    target_lon: float,
    min_rows: int = 100,
    debug: bool = False,
):
    """
    For each y-row, choose the x-cell within sector_id with lon closest to target_lon.
    Returns arrays y_idx, x_idx (same length, ordered by y).
    """
    lon = lon2d.values
    sec = sector2d.values
    dist = dist2d.values
    ny, _ = lon.shape

    ys = []
    xs = []
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
        raise RuntimeError(
            f"Transect too short: only {ys.size} rows intersect sector {sector_id} with finite lon/dist. "
            f"(min_rows={min_rows})"
        )

    if debug:
        medlon = float(np.nanmedian(lon[ys, xs]))
        print(f"[debug] built transect with {ys.size} points; median lon={medlon:.2f}", flush=True)

    return ys, xs


def _cumulative_distance_polyline_km(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Cumulative distance in km along a polyline (x_m,y_m) in meters."""
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    ds = np.sqrt(dx * dx + dy * dy) / 1000.0
    return np.concatenate([[0.0], np.cumsum(ds)])


def _extent_from_mask(lon2d: xr.DataArray, lat2d: xr.DataArray, mask: np.ndarray, pad_deg: float = 2.0):
    """Return [lon_min, lon_max, lat_min, lat_max] from a boolean mask in lon/lat space."""
    if mask.sum() < 10:
        raise RuntimeError("Extent mask too small to compute extent.")
    lonv = lon2d.values[mask]
    latv = lat2d.values[mask]
    lon_min = float(np.nanmin(lonv)) - pad_deg
    lon_max = float(np.nanmax(lonv)) + pad_deg
    lat_min = float(np.nanmin(latv)) - pad_deg
    lat_max = float(np.nanmax(latv)) + pad_deg
    return [lon_min, lon_max, lat_min, lat_max]


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Weddell 30°W: pick fractional points + single reference map")
    ap.add_argument("--sic", required=True, type=Path, help="Merged daily SIC NetCDF")
    ap.add_argument("--var", default="N07_ICECON", help="SIC variable name (default: N07_ICECON)")
    ap.add_argument("--sectors", required=True, type=Path, help="canonical_sectors.nc")
    ap.add_argument("--dist-month", required=True, type=Path, help="monthly dist_to_edge_km_monthMM.nc")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--target-lon", type=float, default=-30.0, help="Target lon (degE, -30=30W)")
    ap.add_argument("--fractions", default="0.3,0.5,0.7,0.9",
                    help="Comma-separated fractions 0<f<1 (default: 0.3,0.5,0.7,0.9)")
    ap.add_argument("--year", type=int, default=2022, help="Year for SIC background/ocean mask")
    ap.add_argument("--month", type=int, default=11, help="Month for SIC background/ocean mask")
    ap.add_argument("--weddell-id", type=int, required=True, help="Weddell sector numeric ID (e.g., 2)")
    ap.add_argument("--min-transect-rows", type=int, default=100, help="Minimum rows required in transect build")
    ap.add_argument("--edge-eps-km", type=float, default=1e-3, help="Edge threshold for dist_to_edge (km)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fracs = [float(x) for x in args.fractions.split(",") if x.strip() != ""]
    if len(fracs) < 2:
        raise ValueError("Provide at least 2 fractions (e.g., 0.3,0.5,0.7,0.9).")
    if not all(0.0 < f < 1.0 for f in fracs):
        raise ValueError("All fractions must be between 0 and 1 (exclusive).")

    weddell_id = int(args.weddell_id)

    # ---- Load canonical sectors
    ds_sec = xr.open_dataset(args.sectors)
    if "lat" not in ds_sec or "lon" not in ds_sec:
        raise RuntimeError("canonical_sectors.nc must contain lat(y,x) and lon(y,x).")
    if "x" not in ds_sec.coords or "y" not in ds_sec.coords:
        raise RuntimeError("canonical_sectors.nc must contain x(x) and y(y) coordinates (meters).")

    sector_var = _find_sector_var(ds_sec)
    if sector_var is None:
        raise RuntimeError(
            f"Could not find a sector ID variable in {args.sectors}. "
            f"Data vars: {list(ds_sec.data_vars)} coords: {list(ds_sec.coords)}"
        )

    lat2d = ds_sec["lat"].transpose("y", "x")
    lon2d = ds_sec["lon"].transpose("y", "x")
    sector2d = ds_sec[sector_var].transpose("y", "x")

    # ---- Load monthly climatological dist-to-edge
    ds_d = xr.open_dataset(args.dist_month)
    dvar = _find_var(ds_d, ["dist_to_edge_km", "dist_to_edge", "distance_to_edge_km"])
    if dvar is None:
        raise RuntimeError(f"Could not find dist variable in {args.dist_month}. Found: {list(ds_d.data_vars)}")
    dist2d = ds_d[dvar].transpose("y", "x")

    if args.debug:
        print(f"[debug] sector_var={sector_var} weddell_id={weddell_id}", flush=True)
        print(f"[debug] sector grid: y={ds_sec.sizes.get('y')} x={ds_sec.sizes.get('x')}", flush=True)
        print(f"[debug] dist grid:   y={ds_d.sizes.get('y')} x={ds_d.sizes.get('x')}", flush=True)

    # ---- Load SIC (flag-aware masking)
    ds_sic = xr.open_dataset(args.sic)
    if args.var not in ds_sic.data_vars:
        raise RuntimeError(f"SIC var '{args.var}' not found in {args.sic}. Available: {list(ds_sic.data_vars)}")

    sic_raw = ds_sic[args.var]
    if not {"time", "y", "x"}.issubset(set(sic_raw.dims)):
        raise RuntimeError(f"SIC dims are {sic_raw.dims}; expected to include ('time','y','x').")

    # Hard sanity check: grids must match
    if (ds_sec.sizes["y"] != ds_sic.sizes["y"]) or (ds_sec.sizes["x"] != ds_sic.sizes["x"]):
        raise RuntimeError(
            "Sector grid and SIC grid shapes differ; regrid or build an index mapping first.\n"
            f"sectors: (y,x)=({ds_sec.sizes['y']},{ds_sec.sizes['x']})  "
            f"sic: (y,x)=({ds_sic.sizes['y']},{ds_sic.sizes['x']})"
        )

    x_sic = _require_1d_finite("ds_sic['x']", ds_sic["x"].values)
    y_sic = _require_1d_finite("ds_sic['y']", ds_sic["y"].values)

    # Flags: 1100=missing, 1200=land
    land_2d = (sic_raw.isel(time=0) == 1200)
    ocean_2d = ~land_2d

    # Clean SIC: flags -> NaN, then enforce physical bounds
    sic = sic_raw.where(sic_raw < 100)
    sic = sic.where((sic >= 0.0) & (sic <= 1.0))

    # Land underlay for plotting (1=land, NaN=ocean)
    land_under = land_2d.astype(float).values
    land_under[land_under == 0] = np.nan

    # ---- Time slice for chosen month/year
    t0, t_end = _month_bounds(args.year, args.month)
    sic_month = sic.sel(time=slice(t0, t_end))

    # Ocean validity mask: ocean AND any finite SIC in month
    ocean_valid_2d = ocean_2d & np.isfinite(sic_month).any("time")

    # Mean SIC background
    sic_m = sic_month.mean("time", skipna=True)

    if args.debug:
        finite = int(np.isfinite(sic_m.values).sum())
        print(f"[debug] mean SIC finite cells: {finite} / {sic_m.size}", flush=True)
        if finite > 0:
            print(
                f"[debug] mean SIC min/max: {float(np.nanmin(sic_m.values)):.3f} / {float(np.nanmax(sic_m.values)):.3f}",
                flush=True,
            )

    # ---- Build ~constant lon transect
    ty, tx = _build_lon_transect_indices(
        lon2d=lon2d,
        sector2d=sector2d,
        dist2d=dist2d,
        sector_id=weddell_id,
        target_lon=args.target_lon,
        min_rows=args.min_transect_rows,
        debug=args.debug,
    )

    t_lat = lat2d.values[ty, tx]
    t_lon = lon2d.values[ty, tx]
    t_dist = dist2d.values[ty, tx]

    t_ocean = ocean_2d.values[ty, tx].astype(bool)
    t_ocean_valid = ocean_valid_2d.values[ty, tx].astype(bool)

    t_keep = np.isfinite(t_lat) & np.isfinite(t_lon) & np.isfinite(t_dist) & t_ocean & t_ocean_valid
    tyk = ty[t_keep]
    txk = tx[t_keep]
    latk = t_lat[t_keep]
    lonk = t_lon[t_keep]
    distk = t_dist[t_keep]

    if tyk.size < 30:
        raise RuntimeError(f"Too few valid transect points after ocean filtering: {tyk.size}")

    # Along-transect cumulative distance
    x_coords = ds_sec["x"].values.astype(float)
    y_coords = ds_sec["y"].values.astype(float)
    x_path_m = x_coords[txk]
    y_path_m = y_coords[tyk]
    s_km = _cumulative_distance_polyline_km(x_path_m, y_path_m)

    # ---- Coast and edge anchors
    coast_i = int(np.argmin(latk))

    edge_candidates = np.where(distk <= args.edge_eps_km)[0]
    if edge_candidates.size > 0:
        edge_i = int(edge_candidates[np.argmax(latk[edge_candidates])])
    else:
        dmin = float(np.nanmin(distk))
        close = np.where(np.isclose(distk, dmin, atol=1e-6))[0]
        edge_i = int(close[np.argmax(latk[close])])
        if args.debug:
            print(f"[debug] no dist<=edge_eps found; using min dist={dmin:.3f} km", flush=True)

    s_coast = float(s_km[coast_i])
    s_edge = float(s_km[edge_i])
    if s_edge <= s_coast:
        s_coast, s_edge = s_edge, s_coast

    D = s_edge - s_coast
    if D <= 0:
        raise RuntimeError(f"Non-positive coast→edge distance: D={D:.3f} km")

    in_segment = (s_km >= s_coast) & (s_km <= s_edge)
    seg_idx = np.where(in_segment)[0]
    if seg_idx.size < 10:
        raise RuntimeError("Too few cells between coast and edge anchors along the chosen transect.")

    # ---- Select points at fractions
    points = []
    for f in fracs:
        target_s = s_coast + f * D
        j = int(seg_idx[np.argmin(np.abs(s_km[seg_idx] - target_s))])

        if not bool(ocean_2d.values[tyk[j], txk[j]]):
            raise RuntimeError(f"Selected point is on land! x_idx={int(txk[j])} y_idx={int(tyk[j])} f={f}")

        points.append({
            "point": f"f_{f:.1f}",
            "fraction": f,
            "x_idx": int(txk[j]),
            "y_idx": int(tyk[j]),
            "lat": float(latk[j]),
            "lon": float(lonk[j]),
            "dist_along_km": float(s_km[j] - s_coast),
            "dist_to_edge_km": float(distk[j]),
        })

    df = pd.DataFrame(points).sort_values("fraction").reset_index(drop=True)

    # ---- Write CSV
    csv_path = outdir / f"weddell_lon30_fracpoints_month{args.month:02d}_{args.year}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[info] wrote points CSV: {csv_path}", flush=True)

    # --------------------------
    # Single reference map (no overview)
    # --------------------------
    ax_proj = ccrs.SouthPolarStereo()
    data_crs = ccrs.epsg(3412)
    pc = ccrs.PlateCarree()

    # auto extent: Weddell sector + finite SIC (reduces empty white)
    finite_mask = np.isfinite(sic_m.values)
    zoom_mask = finite_mask & (sector2d.values == weddell_id)
    zoom_extent = _extent_from_mask(lon2d, lat2d, zoom_mask, pad_deg=2.0)

    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(1, 1, 1, projection=ax_proj)
    ax.set_extent(zoom_extent, crs=pc)

    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)

    # land underlay from SIC flags
    ax.pcolormesh(x_sic, y_sic, land_under, transform=data_crs,
                  shading="auto", alpha=0.35, zorder=2)

    # SIC background
    pm = ax.pcolormesh(x_sic, y_sic, sic_m.values, transform=data_crs,
                       shading="auto", vmin=0.0, vmax=1.0, zorder=3)

    # edge contour
    try:
        ax.contour(x_sic, y_sic, dist2d.values, levels=[0.0],
                   linewidths=1.2, transform=data_crs, zorder=5)
    except Exception as e:
        if args.debug:
            print(f"[debug] edge contour failed: {e}", flush=True)

    # transect line
    ax.plot(x_coords[txk], y_coords[tyk], transform=data_crs, linewidth=1.2, zorder=6)

    # points, colored
    px = x_sic[df["x_idx"].astype(int).values]
    py = y_sic[df["y_idx"].astype(int).values]
    cvals = np.arange(len(df))
    cmap = cm.get_cmap("tab10", len(df))

    ax.scatter(px, py, s=90, c=cvals, cmap=cmap,
               edgecolor="k", linewidth=0.6,
               transform=data_crs, zorder=7)

    for i, r in enumerate(df.itertuples(index=False)):
        ax.text(
            float(x_sic[int(r.x_idx)]) + 25_000,
            float(y_sic[int(r.y_idx)]) + 25_000,
            f"{r.point}\n{r.lat:.2f}, {r.lon:.2f}",
            fontsize=8,
            color=cmap(i),
            transform=data_crs,
            zorder=8
        )

    ax.set_title(
        f"Weddell (~30°W) fractional points (month{args.month:02d} edge)\n"
        f"Mean SIC {args.year}-{args.month:02d} | Edge: climatological month{args.month:02d}"
    )

    cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"Mean SIC (fraction) ({args.year}-{args.month:02d})")

    png_path = outdir / f"weddell_lon30_fracpoints_month{args.month:02d}_{args.year}_map.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote map PNG: {png_path}", flush=True)


if __name__ == "__main__":
    main()
