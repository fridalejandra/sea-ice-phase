#!/usr/bin/env python3
"""
Compute Watkins–Simmonds (WS) seasonal climatological sea-ice edge and
distance-to-edge fields from sea-ice concentration (SIC).

This is a *WS-seasons-only* version:
- NO monthly outputs
- NO DJF/MAM/JJA/SON standard seasons
- Only the three SH-oriented WS windows are computed:
    * GROWTH_MAMJ  = [3, 4, 5, 6]
    * WINTER_JASO  = [7, 8, 9, 10]
    * DECAY_NDJF   = [11, 12, 1, 2]

Outputs (NetCDF) written under:  <outdir>/seasonal/
- ice_edge_prob_<TAG>.nc
- ice_mask_clim_<TAG>.nc
- ice_edge_mask_<TAG>.nc
- dist_to_edge_km_<TAG>.nc

Notes
-----
- Distance is computed in grid-cells and converted to km by multiplying by --grid-km
  (default 25). This is appropriate for diagnostics/stratification on the Bootstrap
  25 km grid, not precise geodesy.
- This script is instrumented with verbose progress prints (flush=True) so you can
  run it under screen/nohup and see where time is being spent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import xarray as xr

try:
    from scipy.ndimage import distance_transform_edt, binary_dilation
except ImportError as e:
    raise SystemExit(
        "This script requires scipy. Install it (e.g., conda install scipy) and rerun."
    ) from e


# Watkins–Simmonds SH windows
SEASONS_CUSTOM = {
    "GROWTH_MAMJ": [3, 4, 5, 6],
    "WINTER_JASO": [7, 8, 9, 10],
    "DECAY_NDJF": [11, 12, 1, 2],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to SIC NetCDF or Zarr store.")
    p.add_argument("--var", required=True, help="Variable name for SIC (0..1).")
    p.add_argument("--time", default="time", help="Time dimension name.")
    p.add_argument("--y", default=None, help="Y dimension name (default: infer).")
    p.add_argument("--x", default=None, help="X dimension name (default: infer).")
    p.add_argument("--threshold", type=float, default=0.15, help="SIC threshold (fraction).")
    p.add_argument("--prob-cut", type=float, default=0.5, help="Probability cutoff for climatological ice region.")
    p.add_argument("--grid-km", type=float, default=25.0, help="Grid spacing in km (Bootstrap ~25 km).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--chunks", default=None, help="Optional dask chunks, e.g. 'time:30,y:200,x:200'.")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Extra debug prints (counts, timings). Recommended.",
    )
    return p.parse_args()


def parse_chunks(chunks: str) -> dict[str, int]:
    # Example: "time:30,y:200,x:200"
    out: dict[str, int] = {}
    for part in chunks.split(","):
        k, v = part.split(":")
        out[k.strip()] = int(v.strip())
    return out


def open_dataset(path: str, chunks: str | None) -> xr.Dataset:
    p = Path(path)
    if p.suffix == ".zarr" or p.is_dir():
        return xr.open_zarr(p, consolidated=False)
    if chunks:
        return xr.open_dataset(p, chunks=parse_chunks(chunks))
    return xr.open_dataset(p)


def infer_xy(da: xr.DataArray, x_name: str | None, y_name: str | None, time_dim: str) -> tuple[str, str]:
    if x_name and y_name:
        return y_name, x_name

    dims = list(da.dims)
    if time_dim in dims:
        dims.remove(time_dim)

    # Try common NSIDC names first
    y = next((d for d in ["y", "ygrid", "Y", "nj"] if d in da.dims), None)
    x = next((d for d in ["x", "xgrid", "X", "ni"] if d in da.dims), None)

    if y is None or x is None:
        # Fallback: assume last two dims are y,x
        if len(dims) < 2:
            raise ValueError(f"Cannot infer spatial dims from {da.dims}")
        y = dims[-2]
        x = dims[-1]

    return y, x


def ensure_365_calendar(da: xr.DataArray, time_dim: str) -> xr.DataArray:
    """Drop Feb 29 if present (datetime64)."""
    t = da[time_dim]
    if np.issubdtype(t.dtype, np.datetime64):
        is_feb29 = (t.dt.month == 2) & (t.dt.day == 29)
        if bool(is_feb29.any()):
            da = da.sel({time_dim: ~is_feb29})
    return da


def ice_probability_season(
    da: xr.DataArray,
    time_dim: str,
    thr: float,
    months: list[int],
    *,
    debug: bool = False,
) -> xr.DataArray:
    """p(y, x) = mean over all selected months across all years of I(SIC>=thr)."""
    t0 = perf_counter()
    if debug:
        print(f"[debug] ice_probability_season: start months={months} thr={thr}", flush=True)

    # Boolean indicator of ice presence at threshold
    ice = (da >= thr).where(da.notnull())
    if debug:
        dt = perf_counter() - t0
        print(f"[debug] built indicator (da>=thr) in {dt:.2f}s; indicator dtype={ice.dtype}", flush=True)

    # Month selection
    t1 = perf_counter()
    month_mask = ice[time_dim].dt.month.isin(months)
    # Count selected timesteps without loading full array
    try:
        n_sel = int(month_mask.sum().compute()) if hasattr(month_mask.data, "compute") else int(month_mask.sum().values)
    except Exception:
        n_sel = -1
    if debug:
        dt = perf_counter() - t1
        print(f"[debug] built month mask in {dt:.2f}s; n_time_selected={n_sel}", flush=True)

    t2 = perf_counter()
    sel = ice.sel({time_dim: month_mask})
    if debug:
        dt = perf_counter() - t2
        print(f"[debug] applied month selection in {dt:.2f}s; sel dims={sel.dims}", flush=True)

    # Mean over time (this is often the expensive compute)
    t3 = perf_counter()
    p = sel.mean(time_dim)
    if debug:
        dt = perf_counter() - t3
        print(f"[debug] computed mean over time in {dt:.2f}s (may still be lazy)", flush=True)

    p.name = "ice_edge_prob"
    p.attrs.update({"threshold": float(thr), "description": "Probability that SIC >= threshold"})
    if debug:
        dt = perf_counter() - t0
        print(f"[debug] ice_probability_season: done (total {dt:.2f}s)", flush=True)
    return p


def edge_from_mask(ice_mask: np.ndarray) -> np.ndarray:
    """Edge pixels = ice pixels adjacent to water pixels (8-neighbor dilation of water)."""
    ice = ice_mask.astype(bool)
    water = ~ice
    water_dil = binary_dilation(water)
    return ice & water_dil


def dist_to_edge_km(edge_mask: np.ndarray, ice_mask: np.ndarray, grid_km: float) -> np.ndarray:
    """
    Distance (km) to nearest edge pixel, computed for all pixels then masked outside ice.
    distance_transform_edt computes distance to the nearest zero in the input array.
    We pass ~edge_mask so that edge pixels are zeros.
    """
    edge = edge_mask.astype(bool)
    ice = ice_mask.astype(bool)
    dist_cells = distance_transform_edt(~edge)  # distance to nearest edge pixel
    dist_km = dist_cells * float(grid_km)
    return np.where(ice, dist_km, np.nan)


def write_products(
    outdir: Path,
    tag: str,
    p_ice: xr.DataArray,
    threshold: float,
    prob_cut: float,
    grid_km: float,
    *,
    debug: bool = False,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if debug:
        print(f"[debug] write_products: tag={tag} outdir={outdir}", flush=True)

    # Climatological ice region
    t0 = perf_counter()
    ice_clim = (p_ice >= prob_cut).astype("i1").rename("ice_mask_clim")
    ice_clim.attrs.update(
        {
            "prob_cut": float(prob_cut),
            "threshold": float(threshold),
            "description": "Climatological ice mask derived from ice-edge probability",
        }
    )
    if debug:
        print(f"[debug] built ice_clim mask in {perf_counter()-t0:.2f}s", flush=True)

    # Edge + distance (numpy for speed)
    t1 = perf_counter()
    ice_np = ice_clim.values.astype(bool)
    if debug:
        print(f"[debug] pulled ice_clim.values to numpy in {perf_counter()-t1:.2f}s; shape={ice_np.shape}", flush=True)

    t2 = perf_counter()
    edge_np = edge_from_mask(ice_np)
    if debug:
        print(f"[debug] computed edge mask in {perf_counter()-t2:.2f}s", flush=True)

    t3 = perf_counter()
    dist_np = dist_to_edge_km(edge_np, ice_np, grid_km)
    if debug:
        print(f"[debug] computed distance transform in {perf_counter()-t3:.2f}s", flush=True)

    edge_da = xr.DataArray(
        edge_np.astype("i1"),
        dims=ice_clim.dims,
        coords=ice_clim.coords,
        name="ice_edge_mask",
        attrs={
            "description": "Edge pixels: ice adjacent to water in climatological mask",
            "prob_cut": float(prob_cut),
            "threshold": float(threshold),
        },
    )

    dist_da = xr.DataArray(
        dist_np.astype("f4"),
        dims=ice_clim.dims,
        coords=ice_clim.coords,
        name="dist_to_edge_km",
        attrs={
            "units": "km",
            "grid_km": float(grid_km),
            "prob_cut": float(prob_cut),
            "threshold": float(threshold),
            "description": "Distance to nearest climatological edge pixel, inside-ice only",
        },
    )

    # Save individual products
    f_prob = outdir / f"ice_edge_prob_{tag}.nc"
    f_mask = outdir / f"ice_mask_clim_{tag}.nc"
    f_edge = outdir / f"ice_edge_mask_{tag}.nc"
    f_dist = outdir / f"dist_to_edge_km_{tag}.nc"

    if debug:
        print(f"[debug] writing: {f_prob.name}", flush=True)
    xr.Dataset({"ice_edge_prob": p_ice}).to_netcdf(f_prob)

    if debug:
        print(f"[debug] writing: {f_mask.name}", flush=True)
    xr.Dataset({"ice_mask_clim": ice_clim}).to_netcdf(f_mask)

    if debug:
        print(f"[debug] writing: {f_edge.name}", flush=True)
    xr.Dataset({"ice_edge_mask": edge_da}).to_netcdf(f_edge)

    if debug:
        print(f"[debug] writing: {f_dist.name}", flush=True)
    xr.Dataset({"dist_to_edge_km": dist_da}).to_netcdf(f_dist)

    if debug:
        print(f"[debug] write_products: done tag={tag}", flush=True)


def main() -> None:
    args = parse_args()

    print(f"[info] input={args.input} var={args.var} time={args.time} outdir={args.outdir}", flush=True)
    print(f"[info] threshold={args.threshold} prob_cut={args.prob_cut} grid_km={args.grid_km} chunks={args.chunks}", flush=True)

    t0 = perf_counter()
    ds = open_dataset(args.input, args.chunks)
    print(f"[info] opened dataset in {perf_counter()-t0:.2f}s", flush=True)

    if args.var not in ds:
        raise SystemExit(f"Variable '{args.var}' not found. Available: {list(ds.data_vars)}")

    print(f"[info] data_vars={list(ds.data_vars)}", flush=True)
    print(f"[info] dims={dict(ds.dims)}", flush=True)

    sic = ds[args.var]
    t1 = perf_counter()
    sic = ensure_365_calendar(sic, args.time)
    print(f"[info] calendar-normalized SIC in {perf_counter()-t1:.2f}s; shape={sic.shape} dims={sic.dims}", flush=True)

    y_dim, x_dim = infer_xy(sic, args.x, args.y, args.time)
    print(f"[info] inferred spatial dims: y={y_dim} x={x_dim}", flush=True)

    outroot = Path(args.outdir)
    seasonal_dir = outroot / "seasonal"
    seasonal_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] running WS seasons: {list(SEASONS_CUSTOM.keys())}", flush=True)
    print(f"[info] seasonal outdir: {seasonal_dir}", flush=True)

    for name, months in SEASONS_CUSTOM.items():
        print(f"\n[info] ===== SEASON {name} months={months} =====", flush=True)

        t_season = perf_counter()
        p = ice_probability_season(sic, args.time, args.threshold, months, debug=args.debug)

        # Force computation of p before moving on (so we know where time goes)
        # This is useful for debugging 'silent stalls' under dask/lazy evaluation.
        t_comp = perf_counter()
        try:
            p_vals = p.values  # triggers compute if dask-backed
            _ = np.nanmean(p_vals)  # touch values
            print(f"[info] materialized probability field in {perf_counter()-t_comp:.2f}s", flush=True)
        except Exception as e:
            print(f"[warn] could not materialize p.values cleanly: {e}", flush=True)

        write_products(seasonal_dir, name, p, args.threshold, args.prob_cut, args.grid_km, debug=args.debug)
        print(f"[info] season {name} complete in {perf_counter()-t_season:.2f}s", flush=True)

    print(f"\n[info] DONE. Wrote WS seasonal edge/distance products to: {seasonal_dir}", flush=True)


if __name__ == "__main__":
    main()
