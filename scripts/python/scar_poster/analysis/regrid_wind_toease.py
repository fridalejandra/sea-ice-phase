"""
regrid_wind_to_ease.py

Regrid ERA5 wind stress (tau_x, tau_y) from native 1-degree regular lat/lon
onto the EASE-Grid 2.0 South (321x321) used by the divergence and SIC fields.

Then compute:
  - magnitude: |tau| = sqrt(tau_x^2 + tau_y^2)
  - curl: d(tau_y)/dx - d(tau_x)/dy  (Ekman pumping driver)

DEACCUMULATION
ERA5 ewss/nsss are ACCUMULATED stress (J/m^2 = Pa·s over the accumulation
period). To get instantaneous stress (Pa), difference consecutive timesteps
and divide by the accumulation period (usually 24h for daily means from
hourly accumulations, but verify — the step attribute matters).

REGRID METHOD
Bilinear is fine here (not conservative). Wind stress is a smooth atmospheric
field, not an area-intensive quantity like concentration. Conservative would
also work but is slower and unnecessary.

CURL COMPUTATION
Computed on the EASE grid AFTER regridding, using the EASE grid spacing
(~25 km). Computing on the native 1-degree grid and then regridding the curl
would also work, but the EASE grid is where everything else lives and where
the spatial correlations will be computed.
"""

import glob
import os
import sys

import numpy as np
import xarray as xr

# ---------------- CONFIG ----------------
WIND_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/wind_stress"
TAU_X_PATTERN = "era5_windstress_tau_x_{year}.nc"
TAU_Y_PATTERN = "era5_windstress_tau_y_{year}.nc"
TAU_X_VAR = "ewss"
TAU_Y_VAR = "nsss"           # VERIFY -- might be 'nsss' or check ncdump
TIME_COORD = "valid_time"
LAT_COORD = "latitude"
LON_COORD = "longitude"

EASE_REF_PATH = "ease_divergence_with_latlon.nc"

YEARS = range(1979, 2024)
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]

# deaccumulation: ERA5 accumulated stress needs differencing
# set to True if the data is accumulated (GRIB_stepType = "accum")
DEACCUMULATE = True
ACCUM_SECONDS = 86400.0       # 24h accumulation period for daily data

REGRID_METHOD = "bilinear"
WEIGHTS_PATH = "regrid_weights_era5wind_to_ease.nc"

OUT_REGRIDDED = "wind_stress_on_ease_sh.nc"
OUT_CURL = "wind_stress_curl_on_ease_sh.nc"
# -----------------------------------------


def load_year(year):
    tx_path = os.path.join(WIND_DIR, str(year), TAU_X_PATTERN.format(year=year))
    ty_path = os.path.join(WIND_DIR, str(year), TAU_Y_PATTERN.format(year=year))

    if not os.path.exists(tx_path):
        print(f"  [skip] {tx_path} not found")
        return None
    if not os.path.exists(ty_path):
        print(f"  [skip] {ty_path} not found")
        return None

    tx = xr.open_dataset(tx_path)
    ty = xr.open_dataset(ty_path)

    tau_x = tx[TAU_X_VAR]
    tau_y = ty[TAU_Y_VAR]

    if DEACCUMULATE:
        # difference consecutive timesteps to get instantaneous stress
        tau_x = tau_x.diff(dim=TIME_COORD) / ACCUM_SECONDS
        tau_y = tau_y.diff(dim=TIME_COORD) / ACCUM_SECONDS

    # align time coordinates
    common_times = np.intersect1d(tau_x[TIME_COORD].values, tau_y[TIME_COORD].values)
    tau_x = tau_x.sel({TIME_COORD: common_times})
    tau_y = tau_y.sel({TIME_COORD: common_times})

    return xr.Dataset({"tau_x": tau_x, "tau_y": tau_y})


def build_regridder(src_ds, ease_ds):
    import xesmf as xe

    src_grid = xr.Dataset({
        "lat": src_ds[LAT_COORD],
        "lon": src_ds[LON_COORD],
    })
    ease_grid = xr.Dataset({
        "lat": ease_ds["lat"],
        "lon": ease_ds["lon"],
    })

    reuse = os.path.exists(WEIGHTS_PATH)
    regridder = xe.Regridder(
        src_grid, ease_grid, REGRID_METHOD,
        filename=WEIGHTS_PATH, reuse_weights=reuse,
    )
    print(f"Regridder ready (weights {'reused' if reuse else 'built'}).")
    return regridder


def compute_curl(tau_x, tau_y, dx, dy):
    """Wind stress curl: d(tau_y)/dx - d(tau_x)/dy on the EASE grid."""
    dtau_y_dx = (tau_y.shift(x=-1) - tau_y.shift(x=1)) / (2 * dx)
    dtau_x_dy = (tau_x.shift(y=-1) - tau_x.shift(y=1)) / (2 * dy)
    return dtau_y_dx - dtau_x_dy


def main():
    print("Loading EASE reference grid...")
    ease = xr.open_dataset(EASE_REF_PATH, decode_times=False)
    if "lat" not in ease.coords:
        print("[STOP] EASE reference has no lat/lon. Run add_latlon_to_ease_divergence.py first.")
        sys.exit(1)

    # get EASE grid spacing
    dx = float(np.abs(np.diff(ease["x"].values)[0]))
    dy = float(np.abs(np.diff(ease["y"].values)[0]))
    print(f"EASE grid spacing: dx={dx:.1f} m, dy={dy:.1f} m")

    regridder = None
    all_years = []

    for year in YEARS:
        if year in EXCLUDE_YEARS:
            continue
        print(f"Processing {year}...")
        ds = load_year(year)
        if ds is None:
            continue

        if regridder is None:
            regridder = build_regridder(ds, ease)

        tx_rg = regridder(ds["tau_x"])
        ty_rg = regridder(ds["tau_y"])

        yearly = xr.Dataset({
            "tau_x": tx_rg,
            "tau_y": ty_rg,
        })
        all_years.append(yearly)

    if not all_years:
        print("[STOP] No years loaded.")
        sys.exit(1)

    print("Concatenating...")
    full = xr.concat(all_years, dim=TIME_COORD)

    # compute derived fields
    print("Computing magnitude and curl...")
    full["tau_mag"] = np.sqrt(full["tau_x"]**2 + full["tau_y"]**2)
    full["tau_curl"] = compute_curl(full["tau_x"], full["tau_y"], dx, dy)

    # save
    print(f"Writing {OUT_REGRIDDED}...")
    full[["tau_x", "tau_y", "tau_mag"]].to_netcdf(OUT_REGRIDDED)
    print(f"-> {OUT_REGRIDDED}")

    print(f"Writing {OUT_CURL}...")
    full[["tau_curl"]].to_netcdf(OUT_CURL)
    print(f"-> {OUT_CURL}")

    # quick sanity
    print(f"\nSanity check (one timestep):")
    t0 = full.isel({TIME_COORD: 100})
    print(f"  tau_mag: min={float(t0.tau_mag.min()):.6f}, "
          f"max={float(t0.tau_mag.max()):.6f} Pa")
    print(f"  tau_curl: min={float(t0.tau_curl.min()):.2e}, "
          f"max={float(t0.tau_curl.max()):.2e} Pa/m")
    print(f"  Expected: tau_mag ~ 0.01-0.3 Pa, curl ~ 1e-7 to 1e-6 Pa/m")

    print("\nNEXT: add 'wind_stress' and 'wind_curl' to plot_monthly_maps.py's "
          "VARIABLES dict, pointing at these files.")


if __name__ == "__main__":
    main()