#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make a single 3x4 panel figure of monthly mean SIC maps for a given year,
overlaying the same representative points (from a CSV) on every month.

- Headless-safe (uses Agg)
- Extent is the full SIC grid footprint in x/y (so no extra map outside the pixel tile)
- Optional: overlay monthly climatological edge (dist_to_edge=0 contour) if you provide --edge-dir
"""

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


def month_bounds(year: int, month: int):
    t0 = np.datetime64(f"{year}-{month:02d}-01")
    if month == 12:
        t1 = np.datetime64(f"{year+1}-01-01")
    else:
        t1 = np.datetime64(f"{year}-{month+1:02d}-01")
    return t0, t1 - np.timedelta64(1, "D")


def find_var(ds: xr.Dataset, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    return None


def main():
    ap = argparse.ArgumentParser(description="3x4 monthly mean SIC maps with fixed representative points")
    ap.add_argument("--sic", required=True, type=Path, help="Merged daily SIC NetCDF")
    ap.add_argument("--var", default="N07_ICECON", help="SIC variable name")
    ap.add_argument("--points-csv", required=True, type=Path,
                    help="CSV with representative points (must include x_idx,y_idx,point,fraction)")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--year", type=int, default=2022, help="Year to plot")
    ap.add_argument("--edge-dir", type=Path, default=None,
                    help="Optional directory containing dist_to_edge_km_monthMM.nc for MM=01..12")
    ap.add_argument("--edge-pattern", type=str, default="dist_to_edge_km_month{mm:02d}.nc",
                    help="Filename pattern inside --edge-dir (default: dist_to_edge_km_month{mm:02d}.nc)")
    ap.add_argument("--drop-f01", action="store_true", help="Drop f_0.1 from the CSV if present")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- points
    df = pd.read_csv(args.points_csv).sort_values("fraction").reset_index(drop=True)
    if args.drop_f01:
        df = df[df["fraction"] > 0.1].copy()
        df = df.sort_values("fraction").reset_index(drop=True)

    labels = df["point"].tolist()
    x_idx = df["x_idx"].astype(int).to_numpy()
    y_idx = df["y_idx"].astype(int).to_numpy()

    if len(labels) < 2:
        raise RuntimeError("Need at least 2 points in the CSV after filtering.")

    # consistent point colors
    cmap_pts = cm.get_cmap("tab10", len(df))
    pt_colors = [cmap_pts(i) for i in range(len(df))]

    # ---- SIC
    ds = xr.open_dataset(args.sic)
    if args.var not in ds.data_vars:
        raise RuntimeError(f"Variable {args.var} not found. Available: {list(ds.data_vars)}")

    sic_raw = ds[args.var]
    sic = sic_raw.where(sic_raw < 100)
    sic = sic.where((sic >= 0.0) & (sic <= 1.0))

    x = ds["x"].values.astype(float)
    y = ds["y"].values.astype(float)

    # land mask from flags (use time=0)
    land_2d = (sic_raw.isel(time=0) == 1200)
    land_under = land_2d.astype(float).values
    land_under[land_under == 0] = np.nan

    # ---- projections
    ax_proj = ccrs.SouthPolarStereo()
    data_crs = ccrs.epsg(3412)

    # ---- extent = full grid footprint (no map beyond the pixel tile)
    extent_xy = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    # ---- figure
    fig, axes = plt.subplots(
        3, 4, figsize=(16, 12),
        subplot_kw={"projection": ax_proj}
    )
    axes = axes.ravel()

    mappable = None

    for mi, month in enumerate(range(1, 13)):
        ax = axes[mi]
        ax.set_extent(extent_xy, crs=data_crs)

        # month slice
        t0, t1 = month_bounds(args.year, month)
        sic_m = sic.sel(time=slice(t0, t1)).mean("time", skipna=True)

        if args.debug:
            finite = int(np.isfinite(sic_m.values).sum())
            print(f"[debug] month={month:02d} finite={finite}", flush=True)

        # base features
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)

        # land underlay from SIC flags (subtle)
        ax.pcolormesh(
            x, y, land_under,
            transform=data_crs, shading="auto",
            alpha=0.25, zorder=2
        )

        # SIC
        m = ax.pcolormesh(
            x, y, sic_m.values,
            transform=data_crs, shading="auto",
            vmin=0.0, vmax=1.0, zorder=3
        )
        mappable = m

        # optional edge contour per month
        if args.edge_dir is not None:
            f = args.edge_dir / args.edge_pattern.format(mm=month)
            if f.exists():
                ds_e = xr.open_dataset(f)
                dv = find_var(ds_e, ["dist_to_edge_km", "dist_to_edge", "distance_to_edge_km"])
                if dv is not None:
                    d = ds_e[dv].transpose("y", "x").values
                    try:
                        ax.contour(
                            x, y, d,
                            levels=[0.0], linewidths=0.8,
                            transform=data_crs, zorder=5
                        )
                    except Exception as e:
                        if args.debug:
                            print(f"[debug] edge contour failed month={month:02d}: {e}", flush=True)

        # points (same on every panel)
        px = x[x_idx]
        py = y[y_idx]
        for i, lab in enumerate(labels):
            ax.scatter(
                px[i], py[i],
                s=45, color=pt_colors[i],
                edgecolor="k", linewidth=0.4,
                transform=data_crs, zorder=6
            )

        ax.set_title(f"{args.year}-{month:02d}", fontsize=11)

    # shared colorbar
    cb = fig.colorbar(mappable, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cb.set_label("Monthly mean SIC (fraction)")

    fig.suptitle(
        f"Monthly mean sea ice concentration ({args.year}) at fixed Weddell transect points",
        y=0.98
    )

    outpng = args.outdir / f"weddell_monthly_mean_SIC_maps_{args.year}.png"
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {outpng}", flush=True)


if __name__ == "__main__":
    main()
