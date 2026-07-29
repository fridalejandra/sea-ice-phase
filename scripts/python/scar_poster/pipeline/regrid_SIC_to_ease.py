"""
regrid_sic_to_ease.py  (updated: inlined corner-bounds computation)

Regrid Bootstrap SIC (NSIDC polar stereographic, 332 x 316) onto the
EASE-Grid 2.0 South grid (321 x 321) used by the NSIDC-0116 drift/divergence
fields, so the two can be combined in the area budget.

WHY CONSERVATIVE, NOT BILINEAR
Sea ice concentration is area-intensive and the budget takes its spatial
derivative inside a flux. Conservative regridding preserves the area integral;
bilinear smears the ice edge, which is exactly where the residual signal lives.

INPUTS (both must already have 2D lat/lon centres):
  SIC_PATH      = bootstrap_sic_with_latlon.nc   (from add_latlon_to_bootstrap_sic.py)
  EASE_REF_PATH = ease_divergence_with_latlon.nc (from add_latlon_to_ease_divergence.py)

Conservative regridding also needs cell CORNER bounds (lon_b/lat_b), not just
centres. Those are computed inline here (midpoint + edge extrapolation), so no
separate helper file is required.
"""

import os
import sys

import numpy as np
import xarray as xr

# ---------------- CONFIG ----------------
SIC_PATH = "bootstrap_sic_masked_latlon.nc"
EASE_REF_PATH = "ease_divergence_with_latlon.nc"

SIC_VAR = "N07_ICECON"     # VERIFY against Checkpoint 1 output; use the merged/
                            # consistent concentration var that compute_sia.py uses

OUT_PATH = "sic_bootstrap_on_ease_sh.nc"
REGRID_METHOD = "conservative"
WEIGHTS_PATH = "regrid_weights_bootstrap_to_ease.nc"   # cached after first build
# -----------------------------------------


def checkpoint(n, msg):
    print(f"\n{'='*60}\n[CHECKPOINT {n}] {msg}\n{'='*60}")


# ---------- corner-bounds computation (inlined) ----------
def _corners_from_centres(c):
    """(ny, nx) centres -> (ny+1, nx+1) corners via midpoint + edge extrapolation."""
    ny, nx = c.shape
    corners = np.full((ny + 1, nx + 1), np.nan)
    corners[1:-1, 1:-1] = 0.25 * (
        c[:-1, :-1] + c[:-1, 1:] + c[1:, :-1] + c[1:, 1:]
    )
    corners[0, 1:-1] = 0.5 * (c[0, :-1] + c[0, 1:])
    corners[-1, 1:-1] = 0.5 * (c[-1, :-1] + c[-1, 1:])
    corners[1:-1, 0] = 0.5 * (c[:-1, 0] + c[1:, 0])
    corners[1:-1, -1] = 0.5 * (c[:-1, -1] + c[1:, -1])
    corners[0, 0] = c[0, 0]
    corners[0, -1] = c[0, -1]
    corners[-1, 0] = c[-1, 0]
    corners[-1, -1] = c[-1, -1]
    return corners


def add_corner_bounds(grid, lat_name="lat", lon_name="lon"):
    """Return a copy of `grid` with lon_b/lat_b (ny+1, nx+1) corner arrays."""
    lat = np.asarray(grid[lat_name].values)
    lon = np.asarray(grid[lon_name].values)
    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError(
            f"Expected 2D lat/lon centres; got lat {lat.shape}, lon {lon.shape}."
        )
    lat_b = _corners_from_centres(lat)
    lon_b = _corners_from_centres(lon)
    out = grid.copy()
    out["lat_b"] = (("y_b", "x_b"), lat_b)
    out["lon_b"] = (("y_b", "x_b"), lon_b)
    if not (np.nanmin(lat_b) <= np.nanmin(lat) and
            np.nanmax(lat_b) >= np.nanmax(lat)):
        print("[warn] lat bounds don't fully bracket centres -- edge "
              "extrapolation may be slightly off. Usually harmless for a "
              "masked budget.")
    return out


def has_latlon(ds):
    names = set(ds.coords) | set(ds.data_vars)
    lat = next((n for n in ("lat", "latitude", "TLAT") if n in names), None)
    lon = next((n for n in ("lon", "longitude", "TLON") if n in names), None)
    return lat, lon


def main():
    # -- Checkpoint 0: xesmf importable? --
    checkpoint(0, "Import xesmf")
    try:
        import xesmf as xe
    except ImportError:
        print("xesmf not installed. conda install -c conda-forge xesmf")
        sys.exit(1)
    print("xesmf imported OK.")

    # -- Checkpoint 1: open, confirm variable + coordinates --
    checkpoint(1, "Open files, inspect variables and coordinates")
    sic = xr.open_dataset(SIC_PATH)
    ease = xr.open_dataset(EASE_REF_PATH, decode_times=False)

    print("SIC data_vars:", list(sic.data_vars))
    if SIC_VAR not in sic:
        print(f"\n[STOP] SIC_VAR {SIC_VAR!r} not found. Pick the right "
              f"concentration variable from the list above and re-run.")
        sys.exit(1)

    sic_lat, sic_lon = has_latlon(sic)
    ease_lat, ease_lon = has_latlon(ease)
    print(f"SIC lat/lon:  {sic_lat}, {sic_lon}")
    print(f"EASE lat/lon: {ease_lat}, {ease_lon}")
    if not (sic_lat and sic_lon and ease_lat and ease_lon):
        print("\n[STOP] One grid is missing lat/lon. Run the add_latlon_* "
              "scripts first.")
        sys.exit(1)

    # -- Checkpoint 2: assemble grids + corner bounds --
    checkpoint(2, "Assemble source/target grids + cell bounds")
    sic_grid = xr.Dataset({"lat": sic[sic_lat], "lon": sic[sic_lon]})
    ease_grid = xr.Dataset({"lat": ease[ease_lat], "lon": ease[ease_lon]})
    sic_grid = add_corner_bounds(sic_grid)
    ease_grid = add_corner_bounds(ease_grid)
    print("Added lon_b/lat_b corner bounds to both grids.")

    # -- Checkpoint 3: build (or load cached) regridder --
    checkpoint(3, "Build regridder (weights)")
    reuse = os.path.exists(WEIGHTS_PATH)
    regridder = xe.Regridder(
        sic_grid, ease_grid, REGRID_METHOD,
        filename=WEIGHTS_PATH, reuse_weights=reuse,
    )
    print(f"Regridder ready (weights {'reused' if reuse else 'built'}).")

    # -- Checkpoint 4: apply to one timestep --
    checkpoint(4, "Apply to one timestep, sanity check")
    A = sic[SIC_VAR]
    if float(A.max()) > 1.5:
        print("Converting percent -> fraction.")
        A = A / 100.0

    first = A.isel(time=0)
    first_rg = regridder(first)
    print(f"One-timestep regrid: {tuple(first.shape)} -> {tuple(first_rg.shape)}")

    # -- Checkpoint 5: AREA CONSERVATION --
    checkpoint(5, "Area conservation check")
    src_sum = float(np.nansum(first.values))
    dst_sum = float(np.nansum(first_rg.values))
    rel_err = abs(dst_sum - src_sum) / src_sum if src_sum else np.nan
    print(f"Sum(SIC) source={src_sum:.1f}, target={dst_sum:.1f}, "
          f"relative diff={rel_err:.3%}")
    print("NOTE: source and target are DIFFERENT grids (polar-stereo 25 km vs "
          "EASE2 ~25 km), so a raw cell-sum comparison is only approximate -- "
          "the cell areas aren't identical. A few % difference is expected and "
          "fine. A LARGE difference (tens of %) means the grids or bounds are "
          "wrong -- stop and diagnose before trusting the regrid.")
    if np.isfinite(rel_err) and rel_err > 0.25:
        print("\n[WARN] >25% difference. Inspect the one-timestep regrid "
              "visually before proceeding; something may be misaligned.")

    # -- Checkpoint 6: full apply + write --
    checkpoint(6, "Regrid full record and write")
    A_rg = regridder(A)
    A_rg.name = "sic"
    A_rg.attrs["note"] = (
        f"Bootstrap SIC ({SIC_VAR}) conservatively regridded from NSIDC "
        f"polar-stereo (EPSG:3976) onto EASE-Grid 2.0 South (EPSG:6932) to "
        f"match NSIDC-0116 drift. Method={REGRID_METHOD}. Corner bounds "
        f"estimated by midpoint extrapolation."
    )
    A_rg.to_netcdf(OUT_PATH)
    print(f"-> {OUT_PATH}")
    print("\nNEXT: point test_budget_feasibility.py SIC_PATH at this file. "
          "Q1 (common grid) should now pass; proceed to the one-winter-month "
          "sanity test, watching the winter pack-interior residual.")


if __name__ == "__main__":
    main()