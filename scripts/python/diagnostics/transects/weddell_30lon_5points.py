#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weddell (30°W) fractional point selection + Nov 2022 map.

What it does:
  1) Loads:
     - daily SIC (merged bootstrap) for Nov 2022 mean background
     - canonical_sectors.nc (contains lat/lon + sector IDs)
     - monthly dist_to_edge_km_month11.nc (climatological Nov edge-distance)
  2) Defines a "meridional transect at ~30°W" by selecting the x-column whose
     median lon is closest to -30.
  3) Restricts to Weddell sector along that column.
  4) Defines:
       coast_anchor = southernmost valid ocean cell along the column within Weddell
       edge_anchor  = northernmost cell along the column with dist_to_edge ~ 0
     Then total tape length D = distance(coast->edge) along the column.
  5) Selects 5 points at fractions f=[0.1,0.3,0.5,0.7,0.9] of D.
  6) Writes:
       - CSV with x_idx,y_idx,lat,lon,fraction,dist_along_km,dist_to_edge_km
       - PNG map: Nov 2022 mean SIC + Nov climatological edge contour + labeled points

Notes / assumptions (be skeptical):
  - "Coast anchor" here is the most poleward (largest y index? actually depends on y values)
    valid Weddell-ocean point along the chosen lon column. It's a pragmatic anchor, not a
    true shoreline-adjacent cell. If you want true coast adjacency, we'd use a land mask
    and pick the first ocean cell next to land along the column.
  - "Edge anchor" uses dist_to_edge_km <= edge_eps_km (default 1e-3 km). If your dist field
    is not exactly zero at the edge, increase edge_eps_km (e.g., 1.0 km).
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
    # also allow coord
    for c in candidates:
        if c in ds.coords and set(ds[c].dims) >= {"y", "x"}:
            return c
    return None


def _infer_weddell_id(ds_sector: xr.Dataset, sector_var: str):
    """
    Attempt to infer the numeric sector ID for Weddell.
    This is intentionally cautious: we look for a coord/attr mapping if present.
    If not found, we fail loudly and ask you to provide --weddell-id.
    """
    # Common pattern: a 1D coord mapping id->name, or attrs like flag_meanings
    v = ds_sector[sector_var]
    # Try CF-style flags
    flag_meanings = v.attrs.get("flag_meanings", None)
    flag_values = v.attrs.get("flag_values", None) or v.attrs.get("flag_value", None)
    if flag_meanings is not None and flag_values is not None:
        meanings = str(flag_meanings).split()
        vals = np.array(flag_values).astype(int)
        for m, val in zip(meanings, vals):
            if "weddell" in m.lower():
                return int(val)

    # Try global attrs that store names
    for k, val in ds_sector.attrs.items():
        if isinstance(val, str) and "weddell" in val.lower():
            # too ambiguous, skip
            pass

    # Try a coordinate called "sector_name" etc.
    for cname in ["sector_name", "sector_names", "name", "names", "region_name", "region_names"]:
        if cname in ds_sector.coords:
            names = [str(x).lower() for x in ds_sector.coords[cname].values]
            if any("weddell" in n for n in names):
                # Need matching id coord
                for idname in ["sector_id", "sector", "region_id", "id"]:
                    if idname in ds_sector.coords:
                        ids = ds_sector.coords[idname].values
                        for n, _id in zip(names, ids):
                            if "weddell" in n:
                                return int(_id)

    return None


def _choose_xidx_for_lon(lon2d: xr.DataArray, target_lon: float) -> int:
    """
    Choose the x-index whose column median lon is closest to target_lon.
    Works because lon(y,x) varies mostly with x for this grid.
    """
    # lon2d dims are (y,x)
    lon_col_med = np.nanmedian(lon2d.values, axis=0)  # median over y
    # Handle wrap-around: lon in [-180,180] already per your file
    diffs = np.abs(lon_col_med - target_lon)
    x_idx = int(np.nanargmin(diffs))
    return x_idx


def _cumulative_distance_km(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """
    Given x,y coordinates (meters) along a 1D path (ordered), return cumulative distance in km.
    """
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    ds = np.sqrt(dx * dx + dy * dy) / 1000.0
    cum = np.concatenate([[0.0], np.cumsum(ds)])
    return cum


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Select fractional points along Weddell 30°W transect and plot Nov 2022 map.")
    ap.add_argument("--sic", required=True, type=Path, help="Merged daily SIC NetCDF (e.g., merged_bootstrap_SH_latest.nc)")
    ap.add_argument("--var", default="N07_ICECON", help="SIC variable name (default: N07_ICECON)")
    ap.add_argument("--sectors", required=True, type=Path, help="canonical_sectors.nc (must include lat(y,x), lon(y,x), sector IDs)")
    ap.add_argument("--dist-month11", required=True, type=Path, help="dist_to_edge_km_month11.nc (climatological Nov distance-to-edge)")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--target-lon", type=float, default=-30.0, help="Target longitude for transect (default: -30)")
    ap.add_argument("--fractions", default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated fractions along coast->edge")
    ap.add_argument("--year", type=int, default=2022, help="Year for SIC background (default: 2022)")
    ap.add_argument("--month", type=int, default=11, help="Month for SIC background (default: 11 = Nov)")
    ap.add_argument("--edge-eps-km", type=float, default=1e-3, help="Threshold for dist_to_edge considered 'edge' (km)")
    ap.add_argument("--weddell-id", type=int, default=None, help="Numeric sector ID for Weddell (override auto-detect)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fracs = [float(x) for x in args.fractions.split(",") if x.strip() != ""]
    if len(fracs) < 3:
        raise ValueError("Provide at least 3 fractions (e.g., 0.1,0.3,0.5,0.7,0.9).")
    if not all(0.0 < f < 1.0 for f in fracs):
        raise ValueError("All fractions must be between 0 and 1 (exclusive).")

    # ---- Load sector file (lat/lon + sector IDs + x/y)
    ds_sec = xr.open_dataset(args.sectors)
    if "lat" not in ds_sec or "lon" not in ds_sec:
        raise RuntimeError("canonical_sectors.nc must include lat(y,x) and lon(y,x).")

    lat2d = ds_sec["lat"]
    lon2d = ds_sec["lon"]

    sector_var = _find_sector_var(ds_sec)
    if sector_var is None:
        raise RuntimeError(f"Could not find a sector ID variable in {args.sectors}. "
                           f"Data vars: {list(ds_sec.data_vars)} coords: {list(ds_sec.coords)}")

    if args.weddell_id is None:
        weddell_id = _infer_weddell_id(ds_sec, sector_var)
        if weddell_id is None:
            raise RuntimeError(
                "Could not auto-detect Weddell sector ID from canonical_sectors.nc. "
                "Rerun with --weddell-id <INT>. (You can inspect unique IDs with a quick python one-liner.)"
            )
    else:
        weddell_id = int(args.weddell_id)

    if args.debug:
        print(f"[debug] sector_var={sector_var} weddell_id={weddell_id}", flush=True)

    # Choose x index closest to target lon
    x_idx = _choose_xidx_for_lon(lon2d, args.target_lon)

    # ---- Load dist_to_edge month11 and slice column
    ds_d = xr.open_dataset(args.dist_month11)
    dvar = _find_var(ds_d, ["dist_to_edge_km", "dist_to_edge", "distance_to_edge_km"])
    if dvar is None:
        raise RuntimeError(f"Could not find distance variable in {args.dist_month11}. Found: {list(ds_d.data_vars)}")
    d2d = ds_d[dvar]

    # ---- Ensure x/y grids align by coordinate values
    # We'll rely on x/y coords; if missing, assume same index ordering.
    # Slice column in all relevant arrays.
    # IMPORTANT: x_idx is an integer index in canonical grid; we use .isel for all arrays.
    sec_col = ds_sec[sector_var].isel(x=x_idx)
    lat_col = lat2d.isel(x=x_idx)
    lon_col = lon2d.isel(x=x_idx)
    d_col = d2d.isel(x=x_idx)

    # x/y coordinates (meters) for cumulative distance
    # canonical_sectors has x(x), y(y) coords.
    x_m = ds_sec["x"].values[x_idx]
    y_m_arr = ds_sec["y"].values  # length y
    # Along the column, x is constant, y varies. Distance between adjacent y centers is constant 25 km.
    # Still compute it generally:
    x_path_m = np.full_like(y_m_arr, fill_value=float(x_m), dtype=float)
    y_path_m = y_m_arr.astype(float)
    s_km = _cumulative_distance_km(x_path_m, y_path_m)

    # ---- Restrict to Weddell sector & valid distance values
    weddell_mask = (sec_col.values == weddell_id)
    valid_mask = np.isfinite(d_col.values) & np.isfinite(lat_col.values) & np.isfinite(lon_col.values)
    keep = weddell_mask & valid_mask

    if keep.sum() < 20:
        raise RuntimeError(f"Too few valid cells along x_idx={x_idx} for Weddell. keep.sum()={keep.sum()}. "
                           "This could mean the chosen x-column isn't actually in the Weddell sector or masks differ.")

    # Determine coast anchor: southernmost valid cell along this column within Weddell.
    # In EPSG:3412, y increases northward? Actually y coordinate is meters; for Antarctic polar stereo,
    # more negative y is generally "toward one direction". We avoid assumptions and define "coast anchor"
    # as the MINIMUM s_km (start of the path) among keep cells (i.e., first along our y ordering).
    idxs = np.where(keep)[0]
    coast_i = int(idxs.min())  # first in array order (smallest y-index)
    # Determine edge anchor: nearest-to-edge cell (dist<=edge_eps) along keep cells.
    edge_candidates = np.where(keep & (d_col.values <= args.edge_eps_km))[0]
    if edge_candidates.size == 0:
        # If dist isn't exactly 0 at edge, use smallest dist as "edge" anchor.
        edge_i = int(idxs[np.nanargmin(d_col.values[keep])])
        if args.debug:
            print("[debug] no dist<=edge_eps found; using min dist as edge anchor", flush=True)
    else:
        # Choose the northernmost (largest s) edge candidate so we get the outer edge, not a coastal polynya edge.
        edge_i = int(edge_candidates[np.nanargmax(s_km[edge_candidates])])

    s_coast = float(s_km[coast_i])
    s_edge = float(s_km[edge_i])
    if s_edge <= s_coast:
        raise RuntimeError(f"Edge anchor is not northward of coast anchor: s_edge={s_edge:.2f} s_coast={s_coast:.2f}. "
                           "You may need to flip the ordering or refine coast/edge definitions.")

    D = s_edge - s_coast

    if args.debug:
        print(f"[debug] chosen x_idx={x_idx} lon_median≈{np.nanmedian(lon2d.values[:, x_idx]):.2f}", flush=True)
        print(f"[debug] coast_i={coast_i} edge_i={edge_i} D={D:.1f} km", flush=True)

    # Select points at fractions of D
    points = []
    for f in fracs:
        target_s = s_coast + f * D
        # among keep cells between coast and edge, choose nearest along-path distance
        in_segment = keep & (s_km >= s_coast) & (s_km <= s_edge)
        seg_idx = np.where(in_segment)[0]
        nearest_i = int(seg_idx[np.argmin(np.abs(s_km[seg_idx] - target_s))])

        points.append({
            "fraction": f,
            "x_idx": int(x_idx),
            "y_idx": int(nearest_i),
            "lat": float(lat_col.values[nearest_i]),
            "lon": float(lon_col.values[nearest_i]),
            "dist_along_km": float(s_km[nearest_i] - s_coast),
            "dist_to_edge_km": float(d_col.values[nearest_i]),
        })

    df = pd.DataFrame(points).sort_values("fraction")
    csv_path = outdir / f"weddell_lon{int(abs(args.target_lon))}_fracpoints_month{args.month:02d}_{args.year}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[info] wrote points CSV: {csv_path}", flush=True)

    # ---- Build SIC background: Nov 2022 mean SIC
    ds_sic = xr.open_dataset(args.sic)
    if args.var not in ds_sic.data_vars:
        raise RuntimeError(f"SIC var '{args.var}' not found in {args.sic}. Available: {list(ds_sic.data_vars)}")
    sic = ds_sic[args.var]
    if not {"time", "y", "x"}.issubset(set(sic.dims)):
        raise RuntimeError(f"SIC dims are {sic.dims}; expected to include ('time','y','x').")

    t0 = f"{args.year}-{args.month:02d}-01"
    # crude month end: slice through next month start
    if args.month == 12:
        t1 = f"{args.year+1}-01-01"
    else:
        t1 = f"{args.year}-{args.month+1:02d}-01"

    sic_m = sic.sel(time=slice(t0, t1)).mean("time", skipna=True)

    # ---- Plot map
    # We'll plot in lon/lat space using canonical lat/lon arrays to avoid projection libraries.
    # This produces a "warped" polar view but is fine for sanity-checking point locations.
    # If you want a true polar-stereo map later, we can add cartopy, but keep it simple now.

    # Mask to Weddell sector for plotting context (optional)
    weddell_2d = (ds_sec[sector_var].values == weddell_id)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Background: SIC mean in Nov 2022
    # Use pcolormesh on lon/lat (2D) with SIC on same y/x indices
    lon = lon2d.values
    lat = lat2d.values
    sic_bg = sic_m.values

    # Mask background outside ocean (optional): keep only where SIC not nan
    m = np.isfinite(sic_bg)
    # and optionally within Weddell sector for clarity
    # m = m & weddell_2d

    # pcolormesh wants 2D grids; this works but is not "projected". Fine for quick check.
    pm = ax.pcolormesh(lon, lat, np.where(m, sic_bg, np.nan), shading="auto")

    cb = plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Mean SIC (Nov 2022)")

    # Overlay climatological Nov edge as contour dist_to_edge==0 (or close)
    d_bg = d2d.values
    # Contour at 0 km (may be sparse if exact zeros); also contour at 25 km as a fallback
    try:
        ax.contour(lon, lat, d_bg, levels=[0.0], linewidths=1.2)
    except Exception:
        # fallback
        ax.contour(lon, lat, d_bg, levels=[25.0], linewidths=1.2)

    # Plot points
    for _, row in df.iterrows():
        ax.plot(row["lon"], row["lat"], marker="o", markersize=8)
        ax.text(
            row["lon"] + 1.0, row["lat"] + 0.5,
            f"f={row['fraction']:.1f}\n{row['lat']:.2f},{row['lon']:.2f}",
            fontsize=8
        )

    # Title
    ax.set_title(f"Weddell sector: ~{abs(args.target_lon):.0f}°W transect fraction points\n"
                 f"Background: mean SIC Nov {args.year} | Edge: climatological Nov")
    ax.set_xlabel("Longitude (degE)")
    ax.set_ylabel("Latitude (degN)")

    # Reasonable view window focused on Weddell-ish region
    # (keeps plot readable; adjust if needed)
    ax.set_xlim(-80, 20)
    ax.set_ylim(-85, -45)

    png_path = outdir / f"weddell_lon{int(abs(args.target_lon))}_fracpoints_month{args.month:02d}_{args.year}_map.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[info] wrote map PNG: {png_path}", flush=True)


if __name__ == "__main__":
    main()
