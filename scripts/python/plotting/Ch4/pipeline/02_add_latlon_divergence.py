"""
add_latlon_to_ease_divergence.py

Reconstruct 2D lat/lon for the EASE-Grid 2.0 South grid used by the
divergence/drift fields, so it can serve as the TARGET grid for the
conservative regrid. Same idea as add_latlon_to_bootstrap_sic.py, but for
the EASE grid, which uses a DIFFERENT projection:

    EASE-Grid 2.0 South = EPSG:6932
    (Lambert Azimuthal Equal-Area, lat_0=-90, lon_0=0, WGS84)

    NOT EPSG:3976 (that was the Bootstrap polar-stereo grid). Equal-area,
    which is one reason it's a sensible target for an area budget.

Writes ease_divergence_with_latlon.nc -- point regrid_sic_to_ease.py's
EASE_REF_PATH at this.
"""

import sys

import numpy as np
import xarray as xr

# ---------------- CONFIG ----------------
EASE_PATH = ("/user/geog/falejandraperez/sea-ice-phase/scripts/python/"
             "scar_poster/ice_divergence_daily_sh.nc")
OUT_PATH = "ease_divergence_with_latlon.nc"

X_COORD = "x"
Y_COORD = "y"

EASE2_SOUTH_EPSG = 6932
# -----------------------------------------


def main():
    ds = xr.open_dataset(EASE_PATH, decode_times=False)

    if X_COORD not in ds or Y_COORD not in ds:
        print(f"[STOP] {X_COORD}/{Y_COORD} not in file. Coords: {list(ds.coords)}")
        sys.exit(1)

    x = ds[X_COORD].values
    y = ds[Y_COORD].values
    print(f"x: {x[0]:.1f} .. {x[-1]:.1f} m, n={x.size}, "
          f"spacing {np.diff(x)[0]:.3f} m")
    print(f"y: {y[0]:.1f} .. {y[-1]:.1f} m, n={y.size}, "
          f"spacing {np.diff(y)[0]:.3f} m")

    from pyproj import CRS, Transformer
    crs = CRS.from_epsg(EASE2_SOUTH_EPSG)
    print(f"\nUsing CRS: {crs.name}  (EPSG:{EASE2_SOUTH_EPSG})")
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)

    xx, yy = np.meshgrid(x, y)
    lon, lat = transformer.transform(xx, yy)

    # ---- same physical validation as the Bootstrap script ----
    print("\n=== VALIDATION ===")
    print(f"lat range: {np.nanmin(lat):.2f} .. {np.nanmax(lat):.2f}")
    print(f"lon range: {np.nanmin(lon):.2f} .. {np.nanmax(lon):.2f}")

    ok = True
    if np.nanmin(lat) < -90.1 or np.nanmax(lat) > -30:
        print("[FAIL] latitudes outside plausible SH sea-ice range.")
        ok = False
    else:
        print("[ok] latitudes within plausible SH range.")
    if (np.nanmax(lon) - np.nanmin(lon)) < 300:
        print("[WARN] longitude span < 300 deg; possible projection error.")
        ok = False
    else:
        print("[ok] longitude spans the pole.")
    ci, cj = lat.shape[0] // 2, lat.shape[1] // 2
    print(f"centre cell lat={lat[ci, cj]:.2f} (near -90 if pole centred).")

    if not ok:
        print("\n[STOP] Validation flagged a problem -- do not use for "
              "regridding until resolved.")
        sys.exit(1)

    print("\n[PASS] EASE grid lat/lon reconstructed and sensible.")

    ds = ds.assign_coords(
        lat=((Y_COORD, X_COORD), lat),
        lon=((Y_COORD, X_COORD), lon),
    )
    ds.to_netcdf(OUT_PATH)
    print(f"\n-> {OUT_PATH}")
    print("NEXT: set EASE_REF_PATH = this file in regrid_sic_to_ease.py, "
          "and SIC_PATH = bootstrap_sic_with_latlon.nc. Both grids now have "
          "lat/lon; Checkpoint 1 should clear.")


if __name__ == "__main__":
    main()