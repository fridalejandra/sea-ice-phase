#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weddell_30lon_5points.py

Pick 5 fractional-distance points along a Weddell meridional transect near 30°W
using a *monthly climatological* distance-to-edge field (e.g., month11).

Key fixes:
  1) SIC is fractional (units=1), but files include flag codes (e.g., 1100 missing, 1200 land).
     We hard-mask SIC >= 100 to NaN, then enforce [0,1].
  2) Explicit land/ocean mask from SIC flags (1200=land) ensures points never land on land.
     Ocean validity mask requires (ocean) AND (any finite SIC in target month).
  3) Proper map projection: EPSG:3412 (Antarctic polar stereographic). Plot on native x/y grid.

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

# Cartopy is used for correct polar-stereo plotting.
# If not installed in your env, install it or switch back to the lon/lat diagnostic.
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


def _cumulative_distance_km(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Cumulative along-path distance in km for a 1D (x,y) path in meters."""
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    ds = np.sqrt(dx * dx + dy * dy) / 1000.0
    return np.concatenate([[0.0], np.cumsum(ds)])


def _choose_xidx_for_lon_in_sector(
    lon2d: xr.DataArray,
    sector2d: xr.DataArray,
    dist2d: xr.DataArray,
    sector_id: int,
    target_lon: float,
    min_keep: int = 50,
    debug: bool = False,
) -> int:
    """
    Choose x column *within sector_id* whose sector-only median lon is closest to target_lon,
    requiring at least min_keep valid cells (sector & finite lon & finite dist).
    """
    lon = lon2d.values
    sec = sector2d.values
    dist = dist2d.values
    ny, nx = lon.shape

    best = None  # (diff, x, med_lon, keep)

    for x in range(nx):
        m = (sec[:, x] == sector_id) & np.isfinite(lon[:, x]) & np.isfinite(dist[:, x])
        keep = int(m.sum())
        if keep < min_keep:
            continue
        med_lon = float(np.nanmedian(lon[:, x][m]))
        diff = float(_wrap_lon_diff(med_lon, target_lon))
        if (best is None) or (diff < best[0]):
            best = (diff, x, med_lon, keep)

    if best is None:
        if debug:
            print(f"[debug] no x column met min_keep={min_keep}; relaxing to min_keep=10", flush=True)
        for x in range(nx):
            m = (sec[:, x] == sector_id) & np.isfinite(lon[:, x]) & np.isfinite(dist[:, x])
            keep = int(m.sum())
            if keep < 10:
                continue
            med_lon = float(np.nanmedian(lon[:, x][m]))
            diff = float(_wrap_lon_diff(med_lon, target_lon))
            if (best is None) or (diff < best[0]):
                best = (diff, x, med_lon, keep)

    if best is None:
        raise RuntimeError(
            f"Could not find any x column intersecting sector_id={sector_id} with valid lon/dist."
        )

    diff, x, med_lon, keep = best
    if debug:
        print(
            f"[debug] chose x_idx={x} (sector-only median lon={med_lon:.2f}, "
            f"diff={diff:.2f}°, keep={keep})",
            flush=True,
        )
    return int(x)


def _month_slice_inclusive_safe(year: int, month: int):
    """Return (t0, t1_exclusive) strings and also a safe inclusive end date."""
    t0 = np.datetime64(f"{year}-{month:02d}-01")
    if month == 12:
        t1 = np.datetime64(f"{year+1}-01-01")
    else:
        t1 = np.datetime64(f"{year}-{month+1:02d}-01")
    # xarray slice is inclusive; use end = t1 - 1 day
    t_end = t1 - np.timedelta64(1, "D")
    return str(t0), str(t1), t0, t_end


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Weddell 30°W: pick 5 fractional points + month map")
    ap.add_argument("--sic", required=True, type=Path, help="Merged daily SIC NetCDF")
    ap.add_argument("--var", default="N07_ICECON", help="SIC variable name (default: N07_ICECON)")
    ap.add_argument("--sectors", required=True, type=Path, help="canonical_sectors.nc")
    ap.add_argument("--dist-month", required=True, type=Path, help="monthly dist_to_edge_km_monthMM.nc")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--target-lon", type=float, default=-30.0, help="Target lon for transect (degE, -30=30W)")
    ap.add_argument("--fractions", default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated fractions 0<f<1 (expect 5)")
    ap.add_argument("--year", type=int, default=2022, help="Year for SIC background/ocean mask (default: 2022)")
    ap.add_argument("--month", type=int, default=11, help="Month for SIC background/ocean mask (default: 11)")
    ap.add_argument("--weddell-id", type=int, required=True, help="Weddell sector numeric ID (you said: 2)")
    ap.add_argument("--min-keep", type=int, default=50, help="Min valid sector cells required in x-column selection")
    ap.add_argument("--edge-eps-km", type=float, default=1e-3, help="Edge candidate threshold for dist_to_edge (km)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fracs = [float(x) for x in args.fractions.split(",") if x.strip() != ""]
    if len(fracs) != 5:
        raise ValueError("This script expects exactly 5 fractions (e.g., 0.1,0.3,0.5,0.7,0.9).")
    if not all(0.0 < f < 1.0 for f in fracs):
        raise ValueError("All fractions must be between 0 and 1 (exclusive).")

    weddell_id = int(args.weddell_id)

    # ---- Load canonical sectors (lat/lon + sector id + x/y)
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

    lat2d = ds_sec["lat"]
    lon2d = ds_sec["lon"]
    sector2d = ds_sec[sector_var]

    if args.debug:
        print(f"[debug] sector_var={sector_var} weddell_id={weddell_id}", flush=True)

    # ---- Load monthly climatological dist-to-edge
    ds_d = xr.open_dataset(args.dist_month)
    dvar = _find_var(ds_d, ["dist_to_edge_km", "dist_to_edge", "distance_to_edge_km"])
    if dvar is None:
        raise RuntimeError(f"Could not find dist variable in {args.dist_month}. Found: {list(ds_d.data_vars)}")
    dist2d = ds_d[dvar]

    # ---- Choose x index within Weddell near target lon
    x_idx = _choose_xidx_for_lon_in_sector(
        lon2d=lon2d,
        sector2d=sector2d,
        dist2d=dist2d,
        sector_id=weddell_id,
        target_lon=args.target_lon,
        min_keep=args.min_keep,
        debug=args.debug,
    )

    # ---- Load SIC (flag-aware masking)
    ds_sic = xr.open_dataset(args.sic)
    if args.var not in ds_sic.data_vars:
        raise RuntimeError(f"SIC var '{args.var}' not found in {args.sic}. Available: {list(ds_sic.data_vars)}")

    sic_raw = ds_sic[args.var]
    if not {"time", "y", "x"}.issubset(set(sic_raw.dims)):
        raise RuntimeError(f"SIC dims are {sic_raw.dims}; expected to include ('time','y','x').")

    # Flags: 1100=missing, 1200=land (and anything >=100 is non-physical for SIC)
    land_2d = (sic_raw.isel(time=0) == 1200)
    ocean_2d = ~land_2d

    # Clean SIC: flags -> NaN, then enforce physical bounds
    sic = sic_raw.where(sic_raw < 100)  # kills 1100/1200 (and any other >=100 codes)
    sic = sic.where((sic >= 0.0) & (sic <= 1.0))

    # ---- Time slice for the chosen month/year (safe inclusive end)
    _, _, t0_dt64, t_end_dt64 = _month_slice_inclusive_safe(args.year, args.month)
    sic_month = sic.sel(time=slice(t0_dt64, t_end_dt64))

    # Ocean validity mask: must be ocean AND have any finite SIC within month window
    ocean_valid_2d = ocean_2d & np.isfinite(sic_month).any("time")

    # Mean SIC background for the map
    sic_m = sic_month.mean("time", skipna=True)

    if args.debug:
        mx = float(sic_m.max(skipna=True))
        print(f"[debug] mean SIC month {args.year}-{args.month:02d} max(after mask)={mx:.3f}", flush=True)
        land_n = int(land_2d.sum())
        print(f"[debug] land cells (flag==1200) = {land_n}", flush=True)

    # ---- Slice chosen column for selection
    sec_col = sector2d.isel(x=x_idx)
    lat_col = lat2d.isel(x=x_idx)
    lon_col = lon2d.isel(x=x_idx)
    d_col = dist2d.isel(x=x_idx)

    ocean_col = ocean_2d.isel(x=x_idx).values.astype(bool)
    ocean_valid_col = ocean_valid_2d.isel(x=x_idx).values.astype(bool)

    # Along-column path coordinates (meters)
    x_m = float(ds_sec["x"].values[x_idx])
    y_m_arr = ds_sec["y"].values.astype(float)  # len y
    x_path_m = np.full_like(y_m_arr, x_m, dtype=float)
    s_km = _cumulative_distance_km(x_path_m, y_m_arr)

    # ---- Keep mask: Weddell + valid lat/lon/dist + ocean + ocean_valid
    weddell_mask = (sec_col.values == weddell_id)
    valid_mask = np.isfinite(d_col.values) & np.isfinite(lat_col.values) & np.isfinite(lon_col.values)
    keep = weddell_mask & valid_mask & ocean_col & ocean_valid_col

    keep_n = int(keep.sum())
    if args.debug:
        med_lon_keep = float(np.nanmedian(lon_col.values[keep])) if keep_n > 0 else np.nan
        print(f"[debug] x_idx={x_idx} keep.sum()={keep_n} median lon (keep)={med_lon_keep:.2f}", flush=True)

    if keep_n < 30:
        raise RuntimeError(
            f"Too few valid-ocean cells along x_idx={x_idx} for Weddell in {args.year}-{args.month:02d}. "
            f"keep.sum()={keep_n}. Try a different month/year or relax the concept of 'ocean_valid'."
        )

    idxs = np.where(keep)[0]
    latv = lat_col.values
    dv = d_col.values

    # Coast (poleward) anchor: most negative latitude among keep
    coast_i = int(idxs[np.nanargmin(latv[idxs])])

    # Edge (northward) anchor: prefer dist<=eps among keep, then most northward
    edge_candidates = np.where(keep & (dv <= args.edge_eps_km))[0]
    if edge_candidates.size > 0:
        edge_i = int(edge_candidates[np.nanargmax(latv[edge_candidates])])
    else:
        # fallback: closest-to-edge, tie-break by most northward
        d_keep = dv[idxs]
        dmin = np.nanmin(d_keep)
        close = idxs[np.where(np.isclose(d_keep, dmin, atol=1e-6))[0]]
        edge_i = int(close[np.nanargmax(latv[close])])
        if args.debug:
            print(f"[debug] no dist<=edge_eps found; using min dist={dmin:.3f} km (northward tie-break)", flush=True)

    s_coast = float(s_km[coast_i])
    s_edge = float(s_km[edge_i])

    # If cumulative distance ordering is reversed, swap for fraction math
    if s_edge <= s_coast:
        if args.debug:
            print("[debug] s_km ordering opposite of latitude ordering; swapping s_coast/s_edge for fraction calc", flush=True)
        s_coast, s_edge = s_edge, s_coast

    D = s_edge - s_coast
    if D <= 0:
        raise RuntimeError(f"Non-positive coast→edge distance: D={D:.3f} (s_coast={s_coast:.3f}, s_edge={s_edge:.3f})")

    # Segment mask between anchors (in s_km space)
    in_segment = keep & (s_km >= s_coast) & (s_km <= s_edge)
    seg_idx = np.where(in_segment)[0]
    if seg_idx.size < 10:
        raise RuntimeError("Too few cells between coast and edge anchors along the chosen transect.")

    # ---- Select points at fractions of D (nearest along-path keep cell)
    points = []
    for f in fracs:
        target_s = s_coast + f * D
        nearest_i = int(seg_idx[np.argmin(np.abs(s_km[seg_idx] - target_s))])

        # Hard assertion: never land
        if not bool(ocean_2d.isel(y=nearest_i, x=x_idx).values):
            raise RuntimeError(f"Selected point is on land! x_idx={x_idx} y_idx={nearest_i} f={f}")

        points.append({
            "point": f"f_{f:.1f}",
            "fraction": f,
            "x_idx": int(x_idx),
            "y_idx": int(nearest_i),
            "lat": float(lat_col.values[nearest_i]),
            "lon": float(lon_col.values[nearest_i]),
            "dist_along_km": float(s_km[nearest_i] - s_coast),
            "dist_to_edge_km": float(d_col.values[nearest_i]),
        })

    df = pd.DataFrame(points).sort_values("fraction").reset_index(drop=True)

    # ---- Write CSV
    csv_path = outdir / f"weddell_lon30_fracpoints_month{args.month:02d}_{args.year}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[info] wrote points CSV: {csv_path}", flush=True)

    # ---- Build two-panel map: overview + zoom (proper projection EPSG:3412)
    # Data are on native x/y meters; use EPSG:3412 as per metadata.
    proj = ccrs.epsg(3412)
    pc = ccrs.PlateCarree()

    # Use x/y from SIC (should match canonical grid). Fall back to ds_sec if needed.
    if "x" in ds_sic.coords and "y" in ds_sic.coords:
        x = ds_sic["x"].values
        y = ds_sic["y"].values
    else:
        x = ds_sec["x"].values
        y = ds_sec["y"].values

    # Land underlay as a mask on the same grid (1=land, nan=ocean)
    land_under = land_2d.astype(float).values
    land_under[land_under == 0] = np.nan

    # Weddell mask for plotting
    weddell_2d = (sector2d.values == weddell_id)
    plot_mask_zoom = weddell_2d & np.isfinite(sic_m.values)

    fig = plt.figure(figsize=(13, 6))

    ax0 = fig.add_subplot(1, 2, 1, projection=proj)  # overview
    ax1 = fig.add_subplot(1, 2, 2, projection=proj)  # zoom

    # ---- Robust extents in native EPSG:3412 meters (avoid lon/lat set_extent NaNs)
    # Overview: use full grid bounds (from coords)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    ax0.set_extent([xmin, xmax, ymin, ymax], crs=proj)

    # Zoom: compute bounds of the Weddell sector in x/y (with padding)
    weddell_2d = (sector2d.values == weddell_id)
    yy, xx = np.where(weddell_2d)
    if yy.size == 0 or xx.size == 0:
        raise RuntimeError("Weddell sector mask is empty; check weddell_id / sector file.")

    pad = 200_000.0  # 200 km padding; tweak if you want
    zxmin = float(np.min(x[xx]) - pad)
    zxmax = float(np.max(x[xx]) + pad)
    zymin = float(np.min(y[yy]) - pad)
    zymax = float(np.max(y[yy]) + pad)
    ax1.set_extent([zxmin, zxmax, zymin, zymax], crs=proj)

    # Coastline/land for context (cosmetic)
    for ax in (ax0, ax1):
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)

    # Land mask underlay from your grid (this makes “land” unambiguous)
    ax0.pcolormesh(x, y, land_under, transform=proj, shading="auto", alpha=0.35, zorder=2)
    ax1.pcolormesh(x, y, land_under, transform=proj, shading="auto", alpha=0.35, zorder=2)

    # SIC background
    pm0 = ax0.pcolormesh(
        x, y, sic_m.values,
        transform=proj, shading="auto", vmin=0.0, vmax=1.0, zorder=3
    )
    pm1 = ax1.pcolormesh(
        x, y, np.where(plot_mask_zoom, sic_m.values, np.nan),
        transform=proj, shading="auto", vmin=0.0, vmax=1.0, zorder=3
    )

    ax0.set_title(f"Antarctic overview: mean SIC {args.year}-{args.month:02d}")
    ax1.set_title("Weddell zoom + selected points")

    # Draw zoom box on overview in projected coordinates
    ax0.plot(
        [zxmin, zxmax, zxmax, zxmin, zxmin],
        [zymin, zymin, zymax, zymax, zymin],
        transform=proj, linewidth=1.5, zorder=6
    )

    # Edge contour from climatological dist field (assumed same grid y,x)
    drew = False
    try:
        ax1.contour(
            x, y, dist2d.values,
            levels=[0.0], linewidths=1.2,
            transform=proj, zorder=5
        )
        drew = True
    except Exception:
        pass
    if not drew:
        try:
            ax1.contour(
                x, y, dist2d.values,
                levels=[25.0], linewidths=1.2,
                transform=proj, zorder=5
            )
        except Exception:
            if args.debug:
                print("[debug] could not draw edge contour at 0 or 25 km", flush=True)

    # Plot selected points using x/y indices (most robust)
    px = ds_sec["x"].isel(x=df["x_idx"].astype(int)).values
    py = ds_sec["y"].isel(y=df["y_idx"].astype(int)).values
    ax1.scatter(px, py, s=60, transform=proj, zorder=7)

    # Labels: keep lon/lat in text for sanity checks
    for _, r in df.iterrows():
        ax1.text(
            ds_sec["x"].isel(x=int(r["x_idx"])).item() + 25_000,  # small offset in meters
            ds_sec["y"].isel(y=int(r["y_idx"])).item() + 25_000,
            f"{r['point']}\n{r['lat']:.2f}, {r['lon']:.2f}",
            fontsize=8,
            transform=proj,
            zorder=8
        )

    fig.suptitle(
        f"Weddell (~30°W) fractional points (month{args.month:02d} edge)\n"
        f"Background: mean SIC {args.year}-{args.month:02d} | Edge: climatological month{args.month:02d}",
        y=1.02
    )

    # Shared colorbar
    cb = fig.colorbar(pm1, ax=[ax0, ax1], fraction=0.046, pad=0.04)
    cb.set_label(f"Mean SIC (fraction) ({args.year}-{args.month:02d})")

    png_path = outdir / f"weddell_lon30_fracpoints_month{args.month:02d}_{args.year}_map.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote map PNG: {png_path}", flush=True)


if __name__ == "__main__":
    main()
