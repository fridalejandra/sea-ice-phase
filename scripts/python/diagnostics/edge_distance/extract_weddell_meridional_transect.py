#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import xarray as xr

def wrap_lon(lon_deg: np.ndarray) -> np.ndarray:
    """Wrap to [-180, 180]."""
    return ((lon_deg + 180.0) % 360.0) - 180.0

def main():
    ap = argparse.ArgumentParser(description="Extract Weddell meridional transect from seasonal WS edge-distance products.")
    ap.add_argument("--seasonal-dir", required=True, type=Path, help="Directory containing seasonal .nc products")
    ap.add_argument("--lon0", type=float, default=-30.0, help="Target longitude in degrees (e.g., -30 for 30W)")
    ap.add_argument("--lat-min", type=float, default=-80.0)
    ap.add_argument("--lat-max", type=float, default=-55.0)
    ap.add_argument("--weddell-lon-min", type=float, default=-60.0, help="Longitude gate to keep selection in Weddell")
    ap.add_argument("--weddell-lon-max", type=float, default=20.0)
    ap.add_argument("--seasons", nargs="*", default=["GROWTH_MAMJ", "WINTER_JASO", "DECAY_NDJF"])
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    seasonal_dir: Path = args.seasonal_dir
    lon0 = wrap_lon(np.array([args.lon0], dtype=float))[0]
    lat_min, lat_max = args.lat_min, args.lat_max

    print(f"[info] seasonal_dir={seasonal_dir}", flush=True)
    print(f"[info] lon0={lon0:+.2f} deg, lat_range=[{lat_min:.1f},{lat_max:.1f}]", flush=True)
    print(f"[info] Weddell lon gate=[{args.weddell_lon_min:.1f},{args.weddell_lon_max:.1f}]", flush=True)
    print(f"[info] seasons={args.seasons}", flush=True)

    # --- Open one file to get x/y grid (meters) ---
    sample_file = seasonal_dir / f"dist_to_edge_km_{args.seasons[0]}.nc"
    if not sample_file.exists():
        raise FileNotFoundError(f"Missing sample file: {sample_file}")

    dist0 = xr.open_dataarray(sample_file)
    x = dist0["x"].values.astype(float)  # 1D
    y = dist0["y"].values.astype(float)  # 1D
    ny, nx = len(y), len(x)
    print(f"[info] grid: nx={nx} ny={ny} | x[m]=[{x.min():.0f},{x.max():.0f}] y[m]=[{y.min():.0f},{y.max():.0f}]", flush=True)

    # --- Build lon/lat from EPSG:3412 ---
    # This requires pyproj installed.
    try:
        from pyproj import Transformer
    except Exception as e:
        raise RuntimeError("pyproj is required (pip/conda install pyproj).") from e

    print("[info] deriving lon/lat grid via EPSG:3412 -> EPSG:4326 (this is a one-time cost)...", flush=True)
    transformer = Transformer.from_crs("EPSG:3412", "EPSG:4326", always_xy=True)

    # meshgrid in (x,y) order
    xx, yy = np.meshgrid(x, y)  # shapes (ny,nx)
    lon, lat = transformer.transform(xx, yy)  # degrees
    lon = wrap_lon(np.array(lon, dtype=float))
    lat = np.array(lat, dtype=float)

    if args.debug:
        print(f"[debug] lon range={np.nanmin(lon):.1f}..{np.nanmax(lon):.1f} lat range={np.nanmin(lat):.1f}..{np.nanmax(lat):.1f}", flush=True)

    # --- Weddell + latitude mask ---
    mask = (
        (lat >= lat_min) & (lat <= lat_max) &
        (lon >= args.weddell_lon_min) & (lon <= args.weddell_lon_max)
    )

    # precompute x index for each y: closest lon to lon0 within mask
    print("[info] computing per-row nearest-x indices to target longitude...", flush=True)
    absdiff = np.abs(lon - lon0)
    absdiff = np.where(mask, absdiff, np.inf)
    x_idx = np.argmin(absdiff, axis=1)  # (ny,)
    row_has_valid = np.isfinite(np.min(absdiff, axis=1))  # (ny,)

    n_valid_rows = int(np.sum(row_has_valid))
    print(f"[info] valid y-rows in mask: {n_valid_rows} / {ny}", flush=True)
    if n_valid_rows == 0:
        raise RuntimeError("No valid rows found. Your lon/lat window is too strict or lon0 is outside Weddell gate.")

    # Along-transect coordinates (1D arrays of length ny)
    tran_lon = lon[np.arange(ny), x_idx]
    tran_lat = lat[np.arange(ny), x_idx]
    tran_xm = xx[np.arange(ny), x_idx]
    tran_ym = yy[np.arange(ny), x_idx]

    # Drop invalid rows now
    keep = row_has_valid
    keep_idx = np.where(keep)[0]

    # Along-transect cumulative distance (km) using geodesic
    try:
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        # cumulative distance along successive points
        lons = tran_lon[keep]
        lats = tran_lat[keep]
        s_km = np.zeros_like(lons, dtype=float)
        for i in range(1, len(lons)):
            _, _, dist_m = geod.inv(lons[i-1], lats[i-1], lons[i], lats[i])
            s_km[i] = s_km[i-1] + dist_m / 1000.0
    except Exception:
        # fallback: use projected distance (still fine for a quick look)
        dx = np.diff(tran_xm[keep])
        dy = np.diff(tran_ym[keep])
        s_km = np.concatenate([[0.0], np.cumsum(np.sqrt(dx*dx + dy*dy) / 1000.0)])

    # --- Extract variables season-by-season ---
    for season in args.seasons:
        print(f"\n[info] ===== extracting season {season} =====", flush=True)

        dist_path = seasonal_dir / f"dist_to_edge_km_{season}.nc"
        prob_path = seasonal_dir / f"ice_edge_prob_{season}.nc"
        mask_path = seasonal_dir / f"ice_edge_mask_{season}.nc"

        for p in [dist_path, prob_path, mask_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required file: {p}")

        dist_da = xr.open_dataarray(dist_path)
        prob_da = xr.open_dataarray(prob_path)
        edge_da = xr.open_dataarray(mask_path)

        # index arrays for x/y selection
        xi = xr.DataArray(x_idx[keep], dims="s")
        yi = xr.DataArray(keep_idx, dims="s")

        # sample along transect (nearest: we already chose exact grid cells)
        tran_dist = dist_da.isel(y=yi, x=xi)
        tran_prob = prob_da.isel(y=yi, x=xi)
        tran_edge = edge_da.isel(y=yi, x=xi)

        out = xr.Dataset(
            data_vars=dict(
                dist_to_edge_km=tran_dist,
                ice_edge_prob=tran_prob,
                ice_edge_mask=tran_edge,
            ),
            coords=dict(
                s_km=("s", s_km),
                lon=("s", tran_lon[keep]),
                lat=("s", tran_lat[keep]),
                x_m=("s", tran_xm[keep]),
                y_m=("s", tran_ym[keep]),
                x_idx=("s", x_idx[keep]),
                y_idx=("s", keep_idx),
            ),
            attrs=dict(
                description="Meridional transect sampled from seasonal climatological fields",
                crs_source="EPSG:3412 (NSIDC_SH_PolarStereo_25km)",
                lon0_deg=float(lon0),
                lat_min_deg=float(lat_min),
                lat_max_deg=float(lat_max),
                weddell_lon_gate=f"[{args.weddell_lon_min},{args.weddell_lon_max}]",
                season=season,
            ),
        )

        out_nc = seasonal_dir / f"transect_Weddell_meridional_lon{lon0:+05.1f}_{season}.nc"
        out_csv = seasonal_dir / f"transect_Weddell_meridional_lon{lon0:+05.1f}_{season}.csv"

        print(f"[info] writing {out_nc.name}", flush=True)
        out.to_netcdf(out_nc)

        # lightweight CSV for quick eyeballing
        df = out[["s_km", "lat", "lon", "dist_to_edge_km", "ice_edge_prob", "ice_edge_mask"]].to_dataframe()
        df.to_csv(out_csv)
        print(f"[info] wrote {out_csv.name} (rows={len(df)})", flush=True)

    print("\n[info] done.", flush=True)

if __name__ == "__main__":
    main()
