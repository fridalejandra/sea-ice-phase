#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def main():
    ap = argparse.ArgumentParser(description="Extract daily SIC time series at 3 selected transect points and write NetCDF.")
    ap.add_argument("--sic-file", required=True, type=Path, help="Merged daily SIC NetCDF (e.g., merged_bootstrap_SH_latest.nc)")
    ap.add_argument("--var", default="N07_ICECON", help="SIC variable name in the SIC file")
    ap.add_argument("--points-csv", required=True, type=Path, help="CSV with selected points (lat/lon/x_idx/y_idx/dist_to_edge_km)")
    ap.add_argument("--out", required=True, type=Path, help="Output NetCDF path")
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    print(f"[info] sic_file={args.sic_file}", flush=True)
    print(f"[info] var={args.var}", flush=True)
    print(f"[info] points_csv={args.points_csv}", flush=True)
    print(f"[info] window={args.start}..{args.end}", flush=True)
    print(f"[info] out={args.out}", flush=True)

    if not args.sic_file.exists():
        raise FileNotFoundError(f"SIC file not found: {args.sic_file}")
    if not args.points_csv.exists():
        raise FileNotFoundError(f"Points CSV not found: {args.points_csv}")

    pts = pd.read_csv(args.points_csv)
    # Expect an index column or a name column; handle both
    if "Unnamed: 0" in pts.columns and "point" not in pts.columns:
        pts = pts.rename(columns={"Unnamed: 0": "point"})
    if "point" not in pts.columns:
        # If saved with index=True, pandas will write an empty first column; handle that
        # Try to treat first column as point names
        pts = pts.rename(columns={pts.columns[0]: "point"})

    required = {"point", "lat", "lon", "x_idx", "y_idx", "dist_to_edge_km"}
    missing = required - set(pts.columns)
    if missing:
        raise RuntimeError(f"Points CSV missing columns: {missing}. Found: {list(pts.columns)}")

    # Enforce exactly 3 points
    pts = pts.copy()
    pts["x_idx"] = pts["x_idx"].astype(int)
    pts["y_idx"] = pts["y_idx"].astype(int)

    # Preferred order
    preferred = ["MIZ_50km", "INNER_PACK_500km", "SOUTHERN_TRANSECT"]
    if set(preferred).issubset(set(pts["point"].astype(str))):
        pts = pts.set_index("point").loc[preferred].reset_index()
    else:
        # fall back: keep file order
        if args.debug:
            print("[debug] preferred point names not all found; using file order:", pts["point"].tolist(), flush=True)

    print("[info] selected points:", flush=True)
    print(pts[["point", "lat", "lon", "dist_to_edge_km", "x_idx", "y_idx"]].to_string(index=False), flush=True)

    print("[info] opening SIC dataset (this can take a moment)...", flush=True)
    ds = xr.open_dataset(args.sic_file)

    if args.var not in ds.data_vars:
        raise RuntimeError(f"Variable {args.var} not found. Available vars: {list(ds.data_vars)}")

    sic = ds[args.var]
    # Confirm dims
    if not {"time", "y", "x"}.issubset(set(sic.dims)):
        raise RuntimeError(f"{args.var} dims are {sic.dims}; expected to include ('time','y','x').")

    # Subset time window
    print("[info] subsetting time window...", flush=True)
    sic = sic.sel(time=slice(args.start, args.end))

    print(f"[info] time steps in window: {sic.sizes['time']}", flush=True)

    # Extract time series for each point
    print("[info] extracting point time series...", flush=True)
    series = []
    for _, row in pts.iterrows():
        name = str(row["point"])
        xi = int(row["x_idx"])
        yi = int(row["y_idx"])
        ts = sic.isel(y=yi, x=xi).rename(name)
        series.append(ts)

    sic_pts = xr.concat(series, dim="point")
    sic_pts = sic_pts.assign_coords(point=("point", pts["point"].astype(str).tolist()))

    # Compute day-to-day change (aligned to same time axis)
    dsic_pts = sic_pts.diff("time")
    dsic_pts = dsic_pts.reindex(time=sic_pts.time)  # first time becomes NaN
    dsic_pts = dsic_pts.rename("dsic")

    # Build output dataset
    out = xr.Dataset(
        data_vars=dict(
            sic=sic_pts.rename("sic"),
            dsic=dsic_pts,
        ),
        coords=dict(
            time=sic_pts.time,
            point=sic_pts.point,
            lat=("point", pts["lat"].to_numpy(dtype=float)),
            lon=("point", pts["lon"].to_numpy(dtype=float)),
            x_idx=("point", pts["x_idx"].to_numpy(dtype=int)),
            y_idx=("point", pts["y_idx"].to_numpy(dtype=int)),
            dist_to_edge_km_clim=("point", pts["dist_to_edge_km"].to_numpy(dtype=float)),
        ),
        attrs=dict(
            source_sic_file=str(args.sic_file),
            source_var=args.var,
            points_csv=str(args.points_csv),
            window_start=str(args.start),
            window_end=str(args.end),
            description="Daily SIC and day-to-day change extracted at 3 objective transect points",
        ),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("[info] writing NetCDF...", flush=True)
    out.to_netcdf(args.out)
    print(f"[info] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
