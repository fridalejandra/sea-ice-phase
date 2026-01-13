#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weddell_30lon_5points.py

Select 5 fractional-distance points (f=[0.1,0.3,0.5,0.7,0.9]) along a Weddell meridional
transect near 30°W, using the *monthly climatological* distance-to-edge field for November
(dist_to_edge_km_month11.nc). Then plot a sanity-check map using mean SIC for Nov 2022,
overlay the climatological Nov ice edge, and label the 5 points with f + lat/lon.

Inputs (you already have these):
  --sic         merged daily SIC file (e.g., merged_bootstrap_SH_latest.nc)
  --sectors     canonical_sectors.nc (must include lat(y,x), lon(y,x), and a 2D sector id field)
  --dist-month11 dist_to_edge_km_month11.nc (climatological Nov edge distance field)

Outputs:
  outdir/
    weddell_lon30_fracpoints_month11_2022.csv
    weddell_lon30_fracpoints_month11_2022_map.png

Important choices:
  - We choose the x-column *within Weddell* whose Weddell-only median lon is closest to -30,
    and that has enough valid Weddell cells (min_keep).
  - Coast anchor: first valid Weddell cell along that column in array order (smallest y-index).
  - Edge anchor: northernmost cell along that column where dist_to_edge <= edge_eps_km;
    if none, we use the minimum dist_to_edge cell (and warn in debug).

This is meant for point-selection sanity checks, not publication cartography.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# --------------------------
# Helpers
# --------------------------
def _find_var(ds: xr.Dataset, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    return None


def _find_sector_var(ds: xr.Dataset):
    """Try common names for the sector ID variable."""
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
        # Relax rather than die silently; caller will still have downstream checks.
        if debug:
            print(f"[debug] no x column met min_keep={min_keep}; relaxing to min_keep=10", flush=True)
        return _choose_xidx_for_lon_in_sector(
            lon2d, sector2d, dist2d, sector_id, target_lon, min_keep=10, debug=debug
        )

    diff, x, med_lon, keep = best
    if debug:
        print(
            f"[debug] chose x_idx={x} (Weddell-only median lon={med_lon:.2f}, "
            f"diff={diff:.2f}°, keep={keep})",
            flush=True,
        )
    return int(x)


def _cumulative_distance_km(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Cumulative along-path distance in km for a 1D (x,y) path in meters."""
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    ds = np.sqrt(dx * dx + dy * dy) / 1000.0
    return np.concatenate([[0.0], np.cumsum(ds)])


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Weddell 30°W: pick 5 fractional points + Nov 2022 map")
    ap.add_argument("--sic", required=True, type=Path, help="Merged daily SIC NetCDF")
    ap.add_argument("--var", default="N07_ICECON", help="SIC variable name (default: N07_ICECON)")
    ap.add_argument("--sectors", required=True, type=Path, help="canonical_sectors.nc")
    ap.add_argument("--dist-month11", required=True, type=Path, help="dist_to_edge_km_month11.nc")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--target-lon", type=float, default=-30.0, help="Target lon for transect (degE, -30=30W)")
    ap.add_argument("--fractions", default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated fractions 0<f<1")
    ap.add_argument("--year", type=int, default=2022, help="Year for SIC background (default: 2022)")
    ap.add_argument("--month", type=int, default=11, help="Month for SIC background (default: 11)")
    ap.add_argument("--edge-eps-km", type=float, default=1e-3, help="Edge threshold for dist_to_edge (km)")
    ap.add_argument("--min-keep", type=int, default=50, help="Minimum Weddell-valid cells required in chosen x-column")
    ap.add_argument("--weddell-id", type=int, required=True, help="Weddell sector numeric ID (you said: 2)")
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

    # ---- Load monthly dist-to-edge for Nov (climatology)
    ds_d = xr.open_dataset(args.dist_month11)
    dvar = _find_var(ds_d, ["dist_to_edge_km", "dist_to_edge", "distance_to_edge_km"])
    if dvar is None:
        raise RuntimeError(f"Could not find dist variable in {args.dist_month11}. Found: {list(ds_d.data_vars)}")
    d2d = ds_d[dvar]

    # ---- Choose x index within Weddell near target lon
    x_idx = _choose_xidx_for_lon_in_sector(
        lon2d=lon2d,
        sector2d=sector2d,
        dist2d=d2d,
        sector_id=weddell_id,
        target_lon=args.target_lon,
        min_keep=args.min_keep,
        debug=args.debug,
    )

    # ---- Slice the chosen column
    sec_col = sector2d.isel(x=x_idx)
    lat_col = lat2d.isel(x=x_idx)
    lon_col = lon2d.isel(x=x_idx)
    d_col = d2d.isel(x=x_idx)

    # Along-column path coordinates (meters)
    x_m = float(ds_sec["x"].values[x_idx])
    y_m_arr = ds_sec["y"].values.astype(float)  # len y
    x_path_m = np.full_like(y_m_arr, x_m, dtype=float)
    s_km = _cumulative_distance_km(x_path_m, y_m_arr)

    # ---- Keep mask: Weddell + valid
    weddell_mask = (sec_col.values == weddell_id)
    valid_mask = np.isfinite(d_col.values) & np.isfinite(lat_col.values) & np.isfinite(lon_col.values)
    keep = weddell_mask & valid_mask
    keep_n = int(keep.sum())

    if args.debug:
        med_lon_keep = float(np.nanmedian(lon_col.values[keep])) if keep_n > 0 else np.nan
        print(f"[debug] x_idx={x_idx} keep.sum()={keep_n} median lon (keep)={med_lon_keep:.2f}", flush=True)

    if keep_n < max(20, args.min_keep // 2):
        raise RuntimeError(
            f"Too few valid cells along x_idx={x_idx} for Weddell. keep.sum()={keep_n}. "
            "This means the chosen x-column still barely intersects Weddell or masks differ."
        )

    idxs = np.where(keep)[0]

    # Coast anchor: first valid keep cell in array order (smallest y-index)
    coast_i = int(idxs.min())

    # Edge anchor: northernmost keep cell where dist_to_edge <= eps
    edge_candidates = np.where(keep & (d_col.values <= args.edge_eps_km))[0]
    if edge_candidates.size == 0:
        edge_i = int(idxs[np.nanargmin(d_col.values[keep])])
        if args.debug:
            min_d = float(np.nanmin(d_col.values[keep]))
            print(f"[debug] no dist<=edge_eps found; using min dist={min_d:.3f} km as edge anchor", flush=True)
    else:
        edge_i = int(edge_candidates[np.nanargmax(s_km[edge_candidates])])

    s_coast = float(s_km[coast_i])
    s_edge = float(s_km[edge_i])
    if not (s_edge > s_coast):
        raise RuntimeError(
            f"Edge anchor is not 'northward' of coast anchor: s_edge={s_edge:.2f} s_coast={s_coast:.2f}. "
            "If this happens, the y-ordering may be opposite what we assumed. We can flip the path."
        )

    D = s_edge - s_coast

    if args.debug:
        print(f"[debug] coast_i={coast_i} edge_i={edge_i} D={D:.1f} km", flush=True)

    # ---- Select points at fractions of D (nearest along-path keep cell)
    points = []
    in_segment = keep & (s_km >= s_coast) & (s_km <= s_edge)
    seg_idx = np.where(in_segment)[0]
    if seg_idx.size < 10:
        raise RuntimeError("Too few cells between coast and edge anchors along the chosen transect.")

    for f in fracs:
        target_s = s_coast + f * D
        nearest_i = int(seg_idx[np.argmin(np.abs(s_km[seg_idx] - target_s))])
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

    # ---- SIC background: mean SIC for specified month/year (Nov 2022)
    ds_sic = xr.open_dataset(args.sic)
    if args.var not in ds_sic.data_vars:
        raise RuntimeError(f"SIC var '{args.var}' not found in {args.sic}. Available: {list(ds_sic.data_vars)}")

    sic = ds_sic[args.var]
    if not {"time", "y", "x"}.issubset(set(sic.dims)):
        raise RuntimeError(f"SIC dims are {sic.dims}; expected to include ('time','y','x').")

    t0 = f"{args.year}-{args.month:02d}-01"
    t1 = f"{args.year + (1 if args.month == 12 else 0)}-{(1 if args.month == 12 else args.month + 1):02d}-01"

    sic_m = sic.sel(time=slice(t0, t1)).mean("time", skipna=True)

    # ---- Plot quick sanity-check "lon/lat pcolormesh" (not true projection)
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    lon = lon2d.values
    lat = lat2d.values
    sic_bg = sic_m.values

    # Mask for plotting (optional: limit to Weddell for clarity)
    weddell_2d = (sector2d.values == weddell_id)
    plot_mask = np.isfinite(sic_bg) & weddell_2d

    pm = ax.pcolormesh(lon, lat, np.where(plot_mask, sic_bg, np.nan), shading="auto")
    cb = plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"Mean SIC ({args.year}-{args.month:02d})")

    # Edge contour from climatological Nov dist field
    d_bg = d2d.values
    # Try 0 km; if empty/throws, also try 25 km
    drew = False
    try:
        ax.contour(lon, lat, d_bg, levels=[0.0], linewidths=1.2)
        drew = True
    except Exception:
        pass
    if not drew:
        try:
            ax.contour(lon, lat, d_bg, levels=[25.0], linewidths=1.2)
        except Exception:
            if args.debug:
                print("[debug] could not draw edge contour at 0 or 25 km", flush=True)

    # Plot points + labels
    for _, r in df.iterrows():
        ax.plot(r["lon"], r["lat"], marker="o", markersize=8)
        ax.text(
            r["lon"] + 1.0, r["lat"] + 0.5,
            f"{r['point']}\n{r['lat']:.2f}, {r['lon']:.2f}",
            fontsize=8
        )

    ax.set_title(
        f"Weddell (~30°W) fractional points (month{args.month:02d} edge)\n"
        f"Background: mean SIC {args.year}-{args.month:02d} | Edge: climatological month{args.month:02d}"
    )
    ax.set_xlabel("Longitude (degE)")
    ax.set_ylabel("Latitude (degN)")

    # Zoom to a reasonable Weddell-ish window
    ax.set_xlim(-80, 20)
    ax.set_ylim(-85, -45)

    png_path = outdir / f"weddell_lon30_fracpoints_month{args.month:02d}_{args.year}_map.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote map PNG: {png_path}", flush=True)


if __name__ == "__main__":
    main()
