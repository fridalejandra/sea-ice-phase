#!/usr/bin/env python3
"""
Plot a 2x3 grid:
Rows: 3 points (MIZ_50km, INNER_PACK_500km, SOUTHERN_TRANSECT)
Cols: SIC(t) and dSIC(t) over a chosen time window.

Expected NetCDF contents:
- sic(time, point) OR N07_ICECON(time, point)
- optional dsic(time, point) (if missing, computed as diff)
- point coordinate (string labels) and optional lat/lon/dist coords per point
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def _find_var(ds: xr.Dataset, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="Input NetCDF with 3-point time series")
    ap.add_argument("--output", required=True, type=Path, help="Output PNG path")
    ap.add_argument("--start", default="2014-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2023-12-31", help="End date (YYYY-MM-DD)")
    ap.add_argument("--thr", type=float, default=0.15, help="Primary static threshold line")
    ap.add_argument("--thr2", type=float, default=0.30, help="Secondary threshold line")
    ap.add_argument("--no-thr2", action="store_true", help="Disable secondary threshold line")
    ap.add_argument("--q", type=float, default=0.99, help="Quantile for symmetric dSIC y-limits")
    ap.add_argument("--title", default=None, help="Figure title override")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    ds = xr.open_dataset(args.input)

    # Identify variables
    sic_name = _find_var(ds, ["sic", "SIC", "N07_ICECON"])
    if sic_name is None:
        raise RuntimeError(f"Could not find SIC variable. Found: {list(ds.data_vars)}")

    dsic_name = _find_var(ds, ["dsic", "dSIC", "delta_sic"])
    sic = ds[sic_name]

    # Identify dims
    if "time" not in sic.dims:
        raise RuntimeError(f"SIC var {sic_name} has dims {sic.dims}, expected a 'time' dim.")
    # point dim could be named 'point' or something else; assume the non-time dim is the point dim
    point_dim = [d for d in sic.dims if d != "time"]
    if len(point_dim) != 1:
        raise RuntimeError(f"Expected SIC to have exactly 2 dims (time, point). Got dims: {sic.dims}")
    point_dim = point_dim[0]

    # Subset time window
    sic = sic.sel(time=slice(args.start, args.end))

    # Ensure dsic exists
    if dsic_name is None:
        dsic = sic.diff("time")
        dsic = dsic.reindex(time=sic.time)  # align to same time axis (first becomes NaN)
        dsic.name = "dsic"
    else:
        dsic = ds[dsic_name].sel(time=slice(args.start, args.end))
        # Align if needed
        dsic = dsic.reindex(time=sic.time)

    # Pull point labels
    if point_dim in ds.coords:
        point_labels = ds.coords[point_dim].values
    else:
        # fallback: numeric index
        point_labels = np.array([f"p{i}" for i in range(sic.sizes[point_dim])], dtype=object)

    # Expected order; if missing, we’ll just use file order
    preferred = ["MIZ_50km", "INNER_PACK_500km", "SOUTHERN_TRANSECT"]
    labels_str = [str(x) for x in point_labels]

    if all(p in labels_str for p in preferred):
        order = [labels_str.index(p) for p in preferred]
        ordered_labels = preferred
    else:
        order = list(range(len(labels_str)))
        ordered_labels = labels_str
        if args.debug:
            print("[debug] point labels not matching preferred names; using file order:", ordered_labels, flush=True)

    # Optional per-point metadata for titles
    def _get_coord(name):
        if name in ds.coords and point_dim in ds.coords[name].dims:
            return ds.coords[name].values
        if name in ds.data_vars and point_dim in ds[name].dims and "time" not in ds[name].dims:
            return ds[name].values
        return None

    lat = _get_coord("lat")
    lon = _get_coord("lon")
    dist = _get_coord("dist_to_edge_km_clim")
    if dist is None:
        dist = _get_coord("dist_to_edge_km")  # sometimes stored as coord/var in point-only files

    # dSIC robust symmetric limits
    abs_all = np.abs(dsic.values[np.isfinite(dsic.values)])
    if abs_all.size == 0:
        ylim = 0.1
    else:
        ylim = float(np.quantile(abs_all, args.q))
        # avoid degenerate
        ylim = max(ylim, 0.02)

    # Build figure: 3 rows x 2 cols
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 9), sharex=True)
    fig.subplots_adjust(hspace=0.25, wspace=0.18)

    # Title
    if args.title is None:
        title = f"Weddell meridional transect (30°W): daily SIC and day-to-day change, {args.start[:4]}–{args.end[:4]}"
    else:
        title = args.title
    fig.suptitle(title)

    for r, idx in enumerate(order[:3]):  # assume 3 points
        label = ordered_labels[r] if r < len(ordered_labels) else f"p{idx}"

        sic_r = sic.isel({point_dim: idx})
        dsic_r = dsic.isel({point_dim: idx})

        # Row title with metadata if available
        meta = []
        if lat is not None and idx < len(lat):
            meta.append(f"lat={float(lat[idx]):.2f}")
        if lon is not None and idx < len(lon):
            meta.append(f"lon={float(lon[idx]):.2f}")
        if dist is not None and idx < len(dist):
            meta.append(f"d_edge={float(dist[idx]):.0f} km")
        row_title = f"{label}" + (" | " + ", ".join(meta) if meta else "")

        # Left: SIC(t)
        axL = axes[r, 0]
        axL.plot(sic_r["time"].values, sic_r.values)
        axL.set_ylim(0, 1)
        axL.set_ylabel("SIC")
        axL.set_title(row_title, loc="left", fontsize=10)
        axL.axhline(args.thr, linestyle="--", linewidth=1)
        if not args.no_thr2:
            axL.axhline(args.thr2, linestyle="--", linewidth=1)
        axL.grid(True, linewidth=0.5, alpha=0.5)

        # Right: dSIC(t)
        axR = axes[r, 1]
        axR.plot(dsic_r["time"].values, dsic_r.values)
        axR.axhline(0.0, linewidth=1)
        axR.set_ylim(-ylim, ylim)
        axR.set_ylabel("ΔSIC")
        axR.grid(True, linewidth=0.5, alpha=0.5)

        if r == 0:
            axL.set_title(axL.get_title() + "   (SIC)", loc="left", fontsize=10)
            axR.set_title("Day-to-day change (ΔSIC)", loc="left", fontsize=10)

    # Bottom x-labels
    axes[2, 0].set_xlabel("Time")
    axes[2, 1].set_xlabel("Time")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[info] wrote figure: {args.output}", flush=True)


if __name__ == "__main__":
    main()
c