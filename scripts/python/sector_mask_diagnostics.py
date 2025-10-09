#!/usr/bin/env python3
"""
build_canonical_sectors.py
Author: Frida A. Perez (2025)

Creates a reusable NetCDF with:
  - lat(y,x), lon(y,x), lonE(y,x)
  - area_m2(y,x)
  - valid_ocean(y,x)
  - sector_id(y,x) : int8 (exclusive)
  - w_AB, w_WE, w_KH, w_EA, w_RA : fractional weights (sum≈1 over ocean)

Also produces a PNG map with the sector wedges and
uploads it to Google Drive via rclone.
"""

from pathlib import Path
import subprocess
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

# ================================================================
# CONFIGURATION
# ================================================================
CFG = {
    # ---- INPUT FILES ----
    "latlon_file": "/user/geog/falejandraperez/sea-ice-phase/data/NSIDC0771_LatLon_PS_S25km_v1.1.nc",
    "lat_var": "latitude",      # variable name in the file (check with xarray.open_dataset)
    "lon_var": "longitude",     # variable name in the file
    "area_file": "/user/geog/falejandraperez/sea-ice-phase/data/NSIDC0771_CellArea_PS_S25km_v1.0.nc",
    "valid_file": None,         # optional; will infer from lat/lon if None

    # ---- OUTPUTS ----
    "out_netcdf": "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc",
    "out_png": "/user/geog/falejandraperez/sea-ice-phase/results/canonical_sectors.png",

    # ---- SECTOR BOUNDS ----
    "sector_bounds": {
        1: {"name": "Amundsen–Bellingshausen", "intervals": [(250.0, 290.0)]},
        2: {"name": "Weddell",                  "intervals": [(290.0, 346.0)]},
        3: {"name": "King Haakon VII",          "intervals": [(346.0, 360.0), (0.0, 71.0)]},
        4: {"name": "East Antarctica",          "intervals": [(71.0, 162.0)]},
        5: {"name": "Ross–Amundsen",            "intervals": [(162.0, 250.0)]},
    },

    "boundary_eps_deg": 1.0,   # tolerance for splitting boundary pixels 50/50

    # ---- RCLONE SETTINGS ----
    "rclone": {
        "enabled": True,
        "remote": "gdrive",                   # rclone remote alias (e.g., gdrive)
        "dst_dir": "sea-ice-phase/results/",  # remote folder path
        "dry_run": False,
        "extra_flags": ["--transfers=8", "--checkers=8", "--fast-list"]
    },

    "dpi": 180
}

# ================================================================
# HELPERS
# ================================================================
def to_lonE(lon):
    return (lon % 360 + 360) % 360

def in_half_open(x, a, b):
    return (x >= a) & (x < b)

def belongs_to_sector(lonE, sector_def):
    mask = np.zeros(lonE.shape, dtype=bool)
    for (a, b) in sector_def["intervals"]:
        if a <= b:
            mask |= in_half_open(lonE, a, b)
        else:
            mask |= in_half_open(lonE, a, 360.0) | in_half_open(lonE, 0.0, b)
    return mask

def nearest_boundary_distance(lonE, boundaries):
    dmin = np.full(lonE.shape, 9999.0)
    for b in boundaries:
        d = np.abs((lonE - b + 540.0) % 360.0 - 180.0)
        dmin = np.minimum(dmin, d)
    return dmin

def push_png(path_png, cfg):
    rc = cfg["rclone"]
    if not rc.get("enabled", False):
        print("[rclone] disabled; skipping upload.")
        return
    dst = f"{rc['remote']}:{rc['dst_dir']}"
    cmd = ["rclone", "copy", str(path_png), dst] + rc.get("extra_flags", [])
    if rc.get("dry_run"): cmd.insert(1, "--dry-run")
    print("[rclone]", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[rclone] ERROR: {res.stderr.strip()}")
        raise SystemExit(1)

# ================================================================
# MAIN
# ================================================================
def main(cfg=CFG):

    out_nc  = Path(cfg["out_netcdf"])
    out_png = Path(cfg["out_png"])
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # --- Load lat/lon from single file ---
    ds_grid = xr.open_dataset(cfg["latlon_file"])
    lat = ds_grid[cfg["lat_var"]]
    lon = ds_grid[cfg["lon_var"]]
    ds_grid.close()

    lonE = xr.apply_ufunc(to_lonE, lon)

    # --- Load area and valid mask ---
    area = None
    if cfg["area_file"]:
        ds_area = xr.open_dataset(cfg["area_file"])
        area = ds_area[list(ds_area.data_vars)[0]].astype(np.float32)
        ds_area.close()

    if cfg["valid_file"]:
        ds_valid = xr.open_dataset(cfg["valid_file"])
        valid = ds_valid[list(ds_valid.data_vars)[0]].astype(bool)
        ds_valid.close()
    else:
        valid = xr.DataArray(
            np.isfinite(lat.values) & np.isfinite(lon.values),
            dims=lat.dims, coords=lat.coords, name="valid_ocean"
        )

    # --- Build exclusive sector_id ---
    sid = xr.zeros_like(lonE, dtype=np.int8)
    for k, info in cfg["sector_bounds"].items():
        sid.values[belongs_to_sector(lonE.values, info)] = np.int8(k)

    # --- Identify near-boundary pixels and split 50/50 ---
    boundaries = sorted(set([b for s in cfg["sector_bounds"].values() for inter in s["intervals"] for b in inter]))
    dmin = nearest_boundary_distance(lonE.values, boundaries)
    near = dmin <= cfg["boundary_eps_deg"]

    w = {k: xr.zeros_like(lonE, dtype=np.float32) for k in cfg["sector_bounds"].keys()}
    for k in w.keys():
        w[k].values[sid.values == k] = 1.0

    eps = 0.5
    lonE_left  = (lonE.values - eps) % 360.0
    lonE_right = (lonE.values + eps) % 360.0
    left_id  = np.zeros(sid.shape, dtype=np.int8)
    right_id = np.zeros(sid.shape, dtype=np.int8)
    for k in cfg["sector_bounds"].keys():
        left_id[belongs_to_sector(lonE_left,  cfg["sector_bounds"][k])] = np.int8(k)
        right_id[belongs_to_sector(lonE_right, cfg["sector_bounds"][k])] = np.int8(k)

    mask_split = near & (left_id > 0) & (right_id > 0) & valid.values
    for k in w.keys():
        arr = w[k].values
        arr[mask_split] = 0.0
        w[k].values = arr
    for k in w.keys():
        arr = w[k].values
        arr[mask_split & (left_id == k)]  += 0.5
        arr[mask_split & (right_id == k)] += 0.5
        arr[~valid.values] = 0.0
        w[k].values = arr

    w_sum = xr.zeros_like(lonE, dtype=np.float32)
    for k in w.keys():
        w_sum = w_sum + w[k]
    max_dev = np.nanmax(np.abs(w_sum.where(valid).values - 1.0))
    print(f"[check] max |sum(weights)-1| over valid ocean: {max_dev:.3e}")

    if area is None:
        area = xr.full_like(lonE, 1.0, dtype=np.float32).rename("area_m2")

    # --- Build dataset ---
    ds = xr.Dataset()
    ds["lat"] = lat.astype(np.float32)
    ds["lon"] = lon.astype(np.float32)
    ds["lonE"] = lonE.astype(np.float32)
    ds["area_m2"] = area
    ds["valid_ocean"] = valid.astype(bool)
    ds["sector_id"] = sid
    name_map = {1:"AB", 2:"WE", 3:"KH", 4:"EA", 5:"RA"}
    for k, tag in name_map.items():
        ds[f"w_{tag}"] = w[k].astype(np.float32)

    ds.attrs.update({
        "title": "Canonical Antarctic Sectors (exclusive IDs + fractional weights)",
        "created": datetime.utcnow().isoformat() + "Z",
        "lon_convention": "lon in [-180,180], lonE in [0,360)",
        "sector_bounds_degE": "AB:[250,290), WE:[290,346), KH:[346,360)∪[0,71), EA:[71,162), RA:[162,250)",
        "rule": "Half-open; meridian belongs to sector on its right",
    })
    for k, info in cfg["sector_bounds"].items():
        ds[f"sector_{k}_name"] = info["name"]

    ds.to_netcdf(out_nc)
    print(f"[OK] wrote NetCDF: {out_nc}")

    # --- Plot PNG ---
    plt.figure(figsize=(7.6, 6.6))
    plt.imshow(ds["valid_ocean"], origin="lower", cmap="gray", vmin=0, vmax=1)
    overlay = ds["sector_id"].where(ds["sector_id"] > 0).values
    cmap = plt.get_cmap("tab10")
    plt.imshow(overlay, origin="lower", alpha=0.6, cmap=cmap, vmin=1, vmax=5)
    for b in boundaries:
        line = np.abs((ds["lonE"].values - b + 540) % 360 - 180) < 0.25
        yy, xx = np.where(line & ds["valid_ocean"].values)
        if yy.size > 0:
            plt.scatter(xx, yy, s=1, c="k")
    plt.title("Canonical Antarctic Sectors (IDs + meridian boundaries)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=cfg["dpi"])
    plt.close()
    print(f"[OK] wrote PNG: {out_png}")

    # --- Push PNG to rclone remote ---
    push_png(out_png, cfg)

if __name__ == "__main__":
    main()
