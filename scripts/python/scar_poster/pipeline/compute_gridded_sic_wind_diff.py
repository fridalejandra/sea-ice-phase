"""
compute_gridded_sic_wind_diff.py

Produces a gridded NetCDF for a spatial map figure (styled after Feba et
al. 2026 Figure 2: SIC difference between periods, with wind-stress
difference vectors overlaid as arrows) - NOT an "SIA netcdf", since SIA
is inherently a spatial integral (sum of SIC x cell_area over many
pixels) and doesn't have a meaningful per-pixel value on its own. SIC
(concentration, naturally defined per grid cell) is the right variable
for a map; mapping "SIA per pixel" would just be a rescaled copy of the
SIC map, since cell area barely varies across the polar-stereo grid.

For consistency with everything else in this pipeline, "difference" here
means: mean SIC in the post-2016 period MINUS mean SIC in the pre-2016
period, using the SAME 2016 split (REGIME_SHIFT_YEAR) used throughout -
not an arbitrary pre-loss/post-loss window like Feba's 1976-1980 vs
1990-1994 (they're studying a different, older event; you're studying
the 2016 shift specifically).

Two fields are produced, both on the native polar-stereo grid:
  1. SIC_diff: per-pixel mean SIC(post) - mean SIC(pre)
  2. tau_x_diff, tau_y_diff: per-pixel mean wind stress vector components,
     post minus pre (regridded from ERA5 lat/lon onto the polar-stereo
     grid, reusing the exact regridding logic from
     build_forcing_sector_table.py, including the longitude-convention
     fix already found necessary there for sectors spanning >179degE)

Output is a NetCDF with x, y (polar-stereo) coordinates plus lat/lon for
reference, ready to plot with cartopy (pcolormesh for SIC_diff, quiver
for the wind vector difference) - the same visual grammar as Feba
Figure 2 and Figure 6.

Uses the SAME *_ICECON dynamic-detection + combine_first approach
already validated in the sector-map mean-SIA-outline contour code, not
a hardcoded single-sensor variable - your merged file has no single
unified "SIC" variable, only per-sensor-era ones.
"""

import numpy as np
import pandas as pd
import xarray as xr

MERGED_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
SECTORS_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"
WIND_STRESS_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/wind_stress"
OUT_NC = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_results/gridded_sic_wind_diff.nc"

REGIME_SHIFT_YEAR = 2016
START_YEAR = 1979
END_YEAR = 2024
ACCUM_SECONDS = 86400


def load_and_combine_sic(merged_file):
    """Dynamically find and combine all *_ICECON sensor-era variables,
    same approach as the sector-map mean-SIA-outline contour code."""
    ds = xr.open_dataset(merged_file)
    icecon_vars = [v for v in ds.data_vars if v.endswith("_ICECON")]
    if not icecon_vars:
        raise ValueError(f"No *_ICECON variable found. Available: {list(ds.data_vars)}")

    sic = ds[icecon_vars[0]]
    for v in icecon_vars[1:]:
        sic = sic.combine_first(ds[v])

    sic = sic.where(sic <= 1.0)  # mask land (1.2 flag) and other invalid values
    return sic, ds


def compute_sic_diff(sic):
    years = sic["time"].dt.year
    pre_mean = sic.where(years < REGIME_SHIFT_YEAR).mean(dim="time", skipna=True)
    post_mean = sic.where(years >= REGIME_SHIFT_YEAR).mean(dim="time", skipna=True)
    return post_mean - pre_mean, pre_mean, post_mean


def load_and_regrid_wind_stress(sectors_ds):
    """Same regridding logic as build_forcing_sector_table.py - reused
    directly so the wind field lines up on the exact same grid as SIC."""
    lat_ps = sectors_ds["lat"]
    lon_ps = sectors_ds["lonE"]

    tau_x_years, tau_y_years = [], []
    for year in range(START_YEAR, END_YEAR + 1):
        fx = f"{WIND_STRESS_DIR}/{year}/era5_windstress_tau_x_{year}.nc"
        fy = f"{WIND_STRESS_DIR}/{year}/era5_windstress_tau_y_{year}.nc"
        try:
            tx = xr.open_dataset(fx)["ewss"]
            ty = xr.open_dataset(fy)["nsss"]
        except FileNotFoundError:
            print(f"  {year}: missing wind stress file(s), skipping")
            continue
        tau_x_years.append(tx / ACCUM_SECONDS)
        tau_y_years.append(ty / ACCUM_SECONDS)

    tau_x_all = xr.concat(tau_x_years, dim="valid_time").rename({"valid_time": "time"})
    tau_y_all = xr.concat(tau_y_years, dim="valid_time").rename({"valid_time": "time"})

    # longitude convention fix (identical to build_forcing_sector_table.py) -
    # without this, sectors spanning >179degE come back all-NaN
    tau_x_all = tau_x_all.assign_coords(longitude=(tau_x_all.longitude % 360)).sortby("longitude")
    tau_y_all = tau_y_all.assign_coords(longitude=(tau_y_all.longitude % 360)).sortby("longitude")

    lat_da = xr.DataArray(lat_ps.values, dims=["y", "x"])
    lon_da = xr.DataArray(lon_ps.values, dims=["y", "x"])
    tau_x_ps = tau_x_all.interp(latitude=lat_da, longitude=lon_da, method="nearest")
    tau_y_ps = tau_y_all.interp(latitude=lat_da, longitude=lon_da, method="nearest")

    return tau_x_ps, tau_y_ps


def compute_wind_diff(tau_x_ps, tau_y_ps):
    years_x = tau_x_ps["time"].dt.year
    years_y = tau_y_ps["time"].dt.year

    tau_x_pre = tau_x_ps.where(years_x < REGIME_SHIFT_YEAR).mean(dim="time", skipna=True)
    tau_x_post = tau_x_ps.where(years_x >= REGIME_SHIFT_YEAR).mean(dim="time", skipna=True)
    tau_y_pre = tau_y_ps.where(years_y < REGIME_SHIFT_YEAR).mean(dim="time", skipna=True)
    tau_y_post = tau_y_ps.where(years_y >= REGIME_SHIFT_YEAR).mean(dim="time", skipna=True)

    return (tau_x_post - tau_x_pre), (tau_y_post - tau_y_pre)


if __name__ == "__main__":
    print("Loading and combining SIC (dynamic *_ICECON detection)...")
    sic, sic_ds = load_and_combine_sic(MERGED_FILE)

    print("Computing SIC difference (post-2016 mean - pre-2016 mean)...")
    sic_diff, sic_pre, sic_post = compute_sic_diff(sic)

    print("\nLoading sector grid for wind regridding reference...")
    sectors_ds = xr.open_dataset(SECTORS_FILE)

    print("Loading and regridding wind stress onto the polar-stereo grid...")
    tau_x_ps, tau_y_ps = load_and_regrid_wind_stress(sectors_ds)

    print("Computing wind stress vector difference (post-2016 mean - pre-2016 mean)...")
    tau_x_diff, tau_y_diff = compute_wind_diff(tau_x_ps, tau_y_ps)

    print("\nAssembling output NetCDF...")
    out = xr.Dataset(
        {
            "SIC_diff": sic_diff,
            "SIC_pre_mean": sic_pre,
            "SIC_post_mean": sic_post,
            "tau_x_diff": tau_x_diff,
            "tau_y_diff": tau_y_diff,
        },
        coords={
            "x": sic_ds["x"], "y": sic_ds["y"],
            "lat": sectors_ds["lat"], "lon": sectors_ds["lonE"],
        },
        attrs={
            "description": "Gridded SIC and wind stress vector difference, "
                            f"post-{REGIME_SHIFT_YEAR} mean minus pre-{REGIME_SHIFT_YEAR} mean",
            "units_SIC_diff": "fractional (0-1)",
            "units_tau_diff": "N/m^2",
        },
    )
    out.to_netcdf(OUT_NC)
    print(f"\nSaved to: {OUT_NC}")
    print("\nColumns: SIC_diff, SIC_pre_mean, SIC_post_mean, tau_x_diff, tau_y_diff")
    print("Ready for: pcolormesh(SIC_diff) + quiver(tau_x_diff, tau_y_diff) on the "
          "polar-stereo projection - same visual grammar as Feba et al. Figure 2.")