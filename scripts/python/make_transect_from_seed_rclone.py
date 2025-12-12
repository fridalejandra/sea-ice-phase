#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a coast→offshore transect automatically from a known-good seed pixel,
and push outputs (JSON) to Google Drive via rclone.

Workflow:
- Open merged SIC file, standardize calendar, select a year
- Find valid pixels (any non-NaN SIC during YEAR)
- Starting from seed (y,x), step offshore in y and find nearest valid pixel
- Write JSON with pixel list
- rclone copy JSON to Drive (optional)

NOTE:
- DY_STEP sign controls offshore direction. If you stepped the wrong way (toward land),
  flip DY_STEP from negative to positive (or vice versa).
"""

import os
import json
import subprocess
from pathlib import Path
import numpy as np
import xarray as xr


# ---------------- CONFIG ---------------- #
INPUT_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
CONC_VAR   = "N07_ICECON"

YEAR = 2005

# seed pixel that you know is valid
SEED_Y, SEED_X = 260, 180

# transect design
N_PIXELS     = 5
DY_STEP      = -10      # offshore step in y (flip sign if it marches into land)
DX_STEP      = 0
SEARCH_RAD_Y = 10
SEARCH_RAD_X = 10

OUT_JSON = "/user/geog/falejandraperez/sea-ice-phase/results/diagnostics/ABS_transects/abs_transect_pixels.json"

# ---- rclone config (override via env vars) ----
RCLONE_PUSH   = True
RCLONE_REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive")  # remote name WITHOUT colon
RCLONE_DEST   = os.environ.get("RCLONE_DEST", "sea-ice-phase/results/diagnostics/ABS_transects")


# -------------- helpers -------------- #
def standardize_calendar(da: xr.DataArray):
    return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))

def select_year(da: xr.DataArray, year: int):
    return da.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))

def prep_ice(ds: xr.Dataset) -> xr.DataArray:
    ice = ds[CONC_VAR].astype("float32")
    if float(ice.max()) > 1.5:
        ice = ice / 100.0
    ice = ice.where(ice < 1.1)
    return ice

def find_valid_near(valid_any: np.ndarray, y0: int, x0: int, ry: int, rx: int):
    """
    Return (y,x) nearest valid pixel to (y0,x0) within +/- ry/rx, else None.
    """
    ny, nx = valid_any.shape
    ylo, yhi = max(0, y0 - ry), min(ny - 1, y0 + ry)
    xlo, xhi = max(0, x0 - rx), min(nx - 1, x0 + rx)

    ys, xs = np.where(valid_any[ylo:yhi+1, xlo:xhi+1])
    if ys.size == 0:
        return None

    ys = ys + ylo
    xs = xs + xlo
    d2 = (ys - y0)**2 + (xs - x0)**2
    k = int(np.argmin(d2))
    return int(ys[k]), int(xs[k])

def rclone_push(local_dir: str, remote: str, remote_dir: str, include_ext=(".json",)):
    """
    Copy selected outputs from local_dir to remote:remote_dir using rclone.
    Uses --update so it won’t re-copy unchanged files.
    """
    local_dir = str(local_dir)

    filters = []
    for ext in include_ext:
        filters += ["--include", f"*{ext}"]
    filters += ["--exclude", "*"]

    cmd = [
        "rclone", "copy",
        local_dir,
        f"{remote}:{remote_dir}",
        "--update",
        "--create-empty-src-dirs",
        "--transfers", "8",
        "--checkers", "16",
        "--stats", "15s",
    ] + filters

    print("\n[RCLOUD] Pushing outputs to Google Drive via rclone:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    out_path = Path(OUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(INPUT_FILE)[[CONC_VAR, "x", "y", "time"]]
    ice = prep_ice(ds)
    ice365 = standardize_calendar(ice)
    ice_year = select_year(ice365, YEAR)

    valid_any = (~ice_year.isnull()).any("time").values  # (y,x) bool

    if not bool(valid_any[SEED_Y, SEED_X]):
        raise RuntimeError(f"Seed pixel (y={SEED_Y}, x={SEED_X}) is not valid in YEAR={YEAR}.")

    pixels = [{"name": "ABS_seed_coast", "y": SEED_Y, "x": SEED_X}]
    y_cur, x_cur = SEED_Y, SEED_X

    for n in range(1, N_PIXELS):
        y_tgt = y_cur + DY_STEP
        x_tgt = x_cur + DX_STEP

        found = find_valid_near(valid_any, y_tgt, x_tgt, SEARCH_RAD_Y, SEARCH_RAD_X)
        if found is None:
            found = find_valid_near(valid_any, y_tgt, x_tgt, SEARCH_RAD_Y * 2, SEARCH_RAD_X * 2)

        if found is None:
            print(f"Could not find valid pixel near target (y={y_tgt}, x={x_tgt}). Stopping at n={n}.")
            break

        y_cur, x_cur = found
        pixels.append({"name": f"ABS_transect_{n}", "y": y_cur, "x": x_cur})

    payload = {"year": YEAR, "seed": {"y": SEED_Y, "x": SEED_X}, "pixels": pixels}
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {out_path}")
    for p in pixels:
        print(f"  {p['name']}: (y={p['y']}, x={p['x']})")

    if RCLONE_PUSH:
        try:
            rclone_push(str(out_path.parent), RCLONE_REMOTE, RCLONE_DEST, include_ext=(".json",))
        except Exception as e:
            print(f"[RCLOUD] rclone push failed (continuing): {e}")


if __name__ == "__main__":
    main()
