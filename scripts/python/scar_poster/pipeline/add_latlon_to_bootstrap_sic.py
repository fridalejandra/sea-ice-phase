"""
add_latlon_to_bootstrap_sic.py

Reconstruct 2D lat/lon for the Bootstrap SIC grid from its polar-stereographic
x/y projection coordinates, so it can be conservatively regridded onto EASE.

This is the bolt-on that unblocks regrid_sic_to_ease.py past Checkpoint 1.

THE ONE THING THAT MATTERS: correct projection parameters.
Wrong params -> plausible-but-wrong lat/lon -> a regrid that "works" and a
budget that is silently garbage. So this script does NOT just trust a hardcoded
CRS -- it reads whatever CRS metadata the file carries, reconstructs lat/lon,
and VALIDATES against an independent check before writing anything.

NSIDC SH polar stereographic (the standard "Sea Ice Polar Stereographic South",
EPSG:3976 / NSIDC-0051 family) is:
    proj=stere lat_0=-90 lat_ts=-70 lon_0=0 (NOT -45; that's the NH grid)
    ellipsoid: Hughes 1980  (a=6378273, b=6356889.449)
BUT: verify against the file's own crs variable. If the file already carries a
proj4 string or EPSG code, USE THAT, not this default.
"""

import sys

import numpy as np
import xarray as xr

# ---------------- CONFIG ----------------
SIC_PATH = ("/user/geog/falejandraperez/sea-ice-phase/data/merged/"
            "SMMR_merged_19781101_20251231_complete.nc")
OUT_PATH = "bootstrap_sic_with_latlon.nc"

X_COORD = "x"
Y_COORD = "y"

# Fallback CRS if the file carries no usable projection metadata.
# EPSG:3976 = NSIDC Sea Ice Polar Stereographic South.
FALLBACK_EPSG = 3976
# -----------------------------------------


def find_crs(ds):
    """Prefer the file's own projection metadata over any hardcoded default."""
    # look for a grid_mapping variable (commonly named 'crs')
    for name in ("crs", "polar_stereographic", "projection", "Polar_Stereographic"):
        if name in ds.variables:
            attrs = dict(ds[name].attrs)
            print(f"Found grid_mapping variable {name!r} with attrs:")
            for k, v in attrs.items():
                print(f"    {k}: {v}")
            # common ways the CRS is encoded
            if "proj4text" in attrs:
                return ("proj4", attrs["proj4text"])
            if "spatial_ref" in attrs:
                return ("wkt", attrs["spatial_ref"])
            if "crs_wkt" in attrs:
                return ("wkt", attrs["crs_wkt"])
            if "epsg_code" in attrs:
                return ("epsg", int(str(attrs["epsg_code"]).split(":")[-1]))
            # reconstruct from CF grid_mapping attributes
            if attrs.get("grid_mapping_name") == "polar_stereographic":
                return ("cf", attrs)
    print("No usable CRS metadata found in file.")
    return (None, None)


def build_transformer(crs_kind, crs_val):
    from pyproj import CRS, Transformer

    if crs_kind == "proj4":
        crs = CRS.from_proj4(crs_val)
    elif crs_kind == "wkt":
        crs = CRS.from_wkt(crs_val)
    elif crs_kind == "epsg":
        crs = CRS.from_epsg(crs_val)
    elif crs_kind == "cf":
        # reconstruct proj4 from CF attributes
        a = crs_val
        lat_ts = a.get("standard_parallel",
                       a.get("latitude_of_projection_origin", -70))
        lon_0 = a.get("straight_vertical_longitude_from_pole", 0)
        proj4 = (f"+proj=stere +lat_0=-90 +lat_ts={lat_ts} +lon_0={lon_0} "
                 f"+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs")
        print(f"Reconstructed proj4 from CF attrs: {proj4}")
        crs = CRS.from_proj4(proj4)
    else:
        print(f"Falling back to EPSG:{FALLBACK_EPSG}. VERIFY this is correct "
              f"for your file before trusting the output.")
        crs = CRS.from_epsg(FALLBACK_EPSG)

    print(f"\nUsing CRS: {crs.name}")
    return Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)


def main():
    ds = xr.open_dataset(SIC_PATH)

    if X_COORD not in ds or Y_COORD not in ds:
        print(f"[STOP] {X_COORD}/{Y_COORD} not in file. Coords: {list(ds.coords)}")
        sys.exit(1)

    x = ds[X_COORD].values
    y = ds[Y_COORD].values
    print(f"x: {x[0]:.1f} .. {x[-1]:.1f} m, n={x.size}, "
          f"spacing {np.diff(x)[0]:.1f} m")
    print(f"y: {y[0]:.1f} .. {y[-1]:.1f} m, n={y.size}, "
          f"spacing {np.diff(y)[0]:.1f} m")

    # sanity: NSIDC polar-stereo x/y are in metres, order 1e6. If these are
    # tiny (degrees) or huge, something is already wrong.
    if abs(x).max() < 1e5 or abs(x).max() > 1e7:
        print(f"[WARN] x magnitude ~{abs(x).max():.1e} m looks wrong for a "
              f"polar-stereo grid in metres. Check units before proceeding.")

    crs_kind, crs_val = find_crs(ds)
    transformer = build_transformer(crs_kind, crs_val)

    xx, yy = np.meshgrid(x, y)
    lon, lat = transformer.transform(xx, yy)

    # ---- VALIDATION: does the reconstructed grid make physical sense? ----
    print("\n=== VALIDATION ===")
    print(f"lat range: {np.nanmin(lat):.2f} .. {np.nanmax(lat):.2f}")
    print(f"lon range: {np.nanmin(lon):.2f} .. {np.nanmax(lon):.2f}")

    ok = True
    # (1) SH sea ice grid: all lats should be southern, roughly -40 to -90
    if np.nanmin(lat) < -90.1 or np.nanmax(lat) > -30:
        print("[FAIL] latitudes outside plausible SH sea-ice range (-40..-90).")
        ok = False
    else:
        print("[ok] latitudes within plausible SH range.")
    # (2) lon should span the full -180..180
    if (np.nanmax(lon) - np.nanmin(lon)) < 300:
        print("[WARN] longitude span < 300 deg; a full polar grid should wrap "
              "nearly all longitudes. Possible projection error.")
        ok = False
    else:
        print("[ok] longitude spans the pole.")
    # (3) the grid centre cell should be very near the South Pole
    ci, cj = lat.shape[0] // 2, lat.shape[1] // 2
    print(f"centre cell lat={lat[ci, cj]:.2f} (should be near -90 if the pole "
          f"is centred).")
    if lat[ci, cj] > -80:
        print("[WARN] grid centre is not near the South Pole. Either the pole "
              "isn't centred in this grid, or the projection is off.")

    if not ok:
        print("\n[STOP] Validation flagged a problem. Do NOT use this output "
              "for regridding until resolved -- a wrong grid gives a "
              "silently corrupt budget. Most likely cause: wrong lat_ts, "
              "wrong ellipsoid, or NH params on an SH grid.")
        sys.exit(1)

    print("\n[PASS] Reconstructed grid looks physically sensible. Still worth "
          "a visual check: plot lat/lon as pcolormesh and confirm smooth, "
          "concentric structure around the pole.")

    ds = ds.assign_coords(
        lat=((Y_COORD, X_COORD), lat),
        lon=((Y_COORD, X_COORD), lon),
    )
    ds.to_netcdf(OUT_PATH)
    print(f"\n-> {OUT_PATH}")
    print("NEXT: point regrid_sic_to_ease.py SIC_PATH at this file. It now has "
          "lat/lon and should clear Checkpoint 1.")
    print("Do the same lat/lon check for the EASE target grid if it also "
          "lacked lat/lon (the regrid needs BOTH sides to have them).")


if __name__ == "__main__":
    main()