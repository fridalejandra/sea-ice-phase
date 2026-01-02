#!/usr/bin/env python3
"""
Compute climatological sea-ice edge "mean location" and distance-to-edge fields
using only SIC.

Bootstrap NSIDC 25 km grid:
- Distance is computed in grid-cells and converted to km by multiplying by `--grid-km`
  (default 25). This is intended for diagnostics, not precise geodesy.

Outputs (NetCDF) per tag:
- ice_edge_prob_<tag>.nc : probability of SIC >= threshold (default 0.15)
- ice_mask_clim_<tag>.nc : climatological ice region (prob >= prob_cut)
- ice_edge_mask_<tag>.nc : edge pixels (ice adjacent to water)
- dist_to_edge_km_<tag>.nc : distance to nearest edge pixel, inside ice only

Notes:
- "Mean location" here is operationalized as a climatological edge mask.
- If you later want a polyline contour (for plotting), you can vectorize the edge mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import xarray as xr

try:
    from scipy.ndimage import distance_transform_edt, binary_dilation
except ImportError as e:
    raise SystemExit(
        "This script requires scipy. Install it (e.g., conda install scipy) and rerun."
    ) from e


# Watkins & Simmonds-style process seasons (3 blocks)
SEASONS_WS = {
    "GROWTH_MAMJ": [3, 4, 5, 6],
    "WINTER_JASO": [7, 8, 9, 10],
    "DECAY_NDJF":  [11, 12, 1, 2],
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
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite/recompute outputs even if files already exist.",
    )
    p.add_argument("--chunks", default=None, help="Optional dask chunks, e.g. 'time:30,y:200,x:200'.")
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


def ice_probability_monthly(da: xr.DataArray, time_dim: str, thr: float) -> xr.DataArray:
    """p(month, y, x) = mean over all years of I(SIC>=thr) for each calendar month."""
    ice = (da >= thr).where(da.notnull())
    p = ice.groupby(f"{time_dim}.month").mean(time_dim)
    p.name = "ice_edge_prob"
    p.attrs.update({"threshold": float(thr), "description": "Probability that SIC >= threshold"})
    return p


def ice_probability_season(da: xr.DataArray, time_dim: str, thr: float, months: list[int]) -> xr.DataArray:
    """p(y, x) = mean over all selected months across all years of I(SIC>=thr)."""
    ice = (da >= thr).where(da.notnull())
    sel = ice.sel({time_dim: ice[time_dim].dt.month.isin(months)})
    p = sel.mean(time_dim)
    p.name = "ice_edge_prob"
    p.attrs.update({"threshold": float(thr), "description": "Probability that SIC >= threshold"})
    return p


def edge_from_mask(ice_mask: np.ndarray) -> np.ndarray:
    """
    Edge pixels = ice pixels adjacent to water pixels.
    Uses an 8-neighbor dilation of water (fine for diagnostics).
    """
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


def outputs_exist(outdir: Path, tag: str) -> bool:
    """True if all expected files exist for this tag in this folder."""
    return all(
        (outdir / f"{stem}_{tag}.nc").exists()
        for stem in ["ice_edge_prob", "ice_mask_clim", "ice_edge_mask", "dist_to_edge_km"]
    )


def write_products(
    outdir: Path,
    tag: str,
    p_ice: xr.DataArray,
    threshold: float,
    prob_cut: float,
    grid_km: float,
    force: bool = False,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[debug] write_products tag={tag} outdir={outdir}", flush=True)
    print(f"[debug] outputs_exist={outputs_exist(outdir, tag)} force={force}", flush=True)

    # Skip if already computed
    if (not force) and outputs_exist(outdir, tag):
        print(f"[skip] {tag} already exists in {outdir}")
        return

    # Climatological ice region
    ice_clim = (p_ice >= prob_cut).astype("i1").rename("ice_mask_clim")
    ice_clim.attrs.update(
        {
            "prob_cut": float(prob_cut),
            "threshold": float(threshold),
            "description": "Climatological ice mask derived from ice-edge probability",
        }
    )

    # Edge + distance (numpy for speed)
    ice_np = ice_clim.values.astype(bool)
    edge_np = edge_from_mask(ice_np)
    dist_np = dist_to_edge_km(edge_np, ice_np, grid_km)

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

    # Save explicit files
    print('[debug] writing NetCDF files...', flush=True)
    xr.Dataset({"ice_edge_prob": p_ice}).to_netcdf(outdir / f"ice_edge_prob_{tag}.nc")
    print(f"[debug] wrote: {outdir / f'ice_edge_prob_{tag}.nc'}", flush=True)
    xr.Dataset({"ice_mask_clim": ice_clim}).to_netcdf(outdir / f"ice_mask_clim_{tag}.nc")
    print(f"[debug] wrote: {outdir / f'ice_mask_clim_{tag}.nc'}", flush=True)
    xr.Dataset({"ice_edge_mask": edge_da}).to_netcdf(outdir / f"ice_edge_mask_{tag}.nc")
    print(f"[debug] wrote: {outdir / f'ice_edge_mask_{tag}.nc'}", flush=True)
    xr.Dataset({"dist_to_edge_km": dist_da}).to_netcdf(outdir / f"dist_to_edge_km_{tag}.nc")
    print(f"[debug] wrote: {outdir / f'dist_to_edge_km_{tag}.nc'}", flush=True)


def main() -> None:
    args = parse_args()
    print('[debug] starting wat_simm_compute_climatological_ice_edge_distance', flush=True)
    print(f"[debug] input={args.input} var={args.var} time={args.time} outdir={args.outdir}", flush=True)
    print(f"[debug] threshold={args.threshold} prob_cut={args.prob_cut} grid_km={args.grid_km} force={args.force}", flush=True)

    ds = open_dataset(args.input, args.chunks)
    print(f"[debug] opened dataset: dims={dict(ds.dims)}", flush=True)
    print(f"[debug] data_vars={list(ds.data_vars)}", flush=True)
    if args.var not in ds:
        raise SystemExit(f"Variable '{args.var}' not found. Available: {list(ds.data_vars)}")

    sic = ds[args.var]
    sic = ensure_365_calendar(sic, args.time)
    print(f"[debug] SIC after calendar normalize: shape={sic.shape} dims={sic.dims}", flush=True)

    # sanity / inference
    infer_xy(sic, args.x, args.y, args.time)

    outroot = Path(args.outdir)

    # WATKINS & SIMMONDS SEASONS (always)
    print(f"[debug] running seasons: {list(SEASONS_WS.keys())}", flush=True)
    print(f"[debug] seasonal outdir will be: {outroot / 'seasonal'}", flush=True)
    for name, months in SEASONS_WS.items():
        p = ice_probability_season(sic, args.time, args.threshold, months)
        write_products(
            outroot / "seasonal",
            name,
            p,
            args.threshold,
            args.prob_cut,
            args.grid_km,
            force=args.force,
        )

    print("Done. Wrote edge/distance products to:", outroot)


if __name__ == "__main__":
    main()
