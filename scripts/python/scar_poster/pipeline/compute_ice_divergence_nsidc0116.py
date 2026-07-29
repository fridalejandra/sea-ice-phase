"""
compute_ice_divergence_nsidc0116.py

Computes sea ice divergence (del . u) from NSIDC-0116 Polar Pathfinder daily
25 km EASE-Grid Sea Ice Motion Vectors v4.1, Southern Hemisphere, and
aggregates to sector/season means for the Ch4 Layer 2 analysis.

FILE STRUCTURE (verified against v4.1 SH header)
  dims:  x=321, y=321, time=UNLIMITED (365/366 per annual file)
  u,v:   float, cm/s, _FillValue=-9999, along-x / along-y (GRID-RELATIVE,
         explicitly *not* eastward/northward -- so no rotation is needed
         before differencing on this grid)
  crs:   lambert_azimuthal_equal_area, EPSG:3409, EASE-Grid South
  time:  days since 1970-01-01, julian calendar
  icemotion_error_estimate: short, cm/s, _FillValue=-9999

GRID SPACING
  x spans -4023337.7625 .. 4023337.7625 over 321 cells
  => 8046675.525 / 321 = 25067.525 m, NOT 25000 m.
  "25 km" is nominal. The script reads dx from the coordinate array rather
  than hardcoding, so this stays correct if the grid ever changes.

QUALITY MASKING VIA icemotion_error_estimate
  Per the v4 user guide:
    - NEGATIVE values indicate the vector is near the coast.
    - 1000 is ADDED to the value when the closest input vector is more than
      1250 km away.
  The second flag matters a great deal here. NSIDC-0116 blends NCEP/NCAR
  reanalysis winds into its optimal interpolation, and the Southern Hemisphere
  has no IABP buoy constraint (IABP is Arctic-only), so SH vectors lean harder
  on satellite feature tracking and reanalysis winds than Arctic ones do.
  Cells flagged >= 1000 are precisely those with no nearby observational
  constraint -- i.e. the most reanalysis-dependent, and the most likely to
  produce circular results when divergence is later regressed against ERA5
  wind stress. Excluding them targets the contamination directly rather than
  approximating it with a fixed coastal buffer.

  The script reports what fraction of cells each flag removes, which is worth
  quoting in the methods.
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

# ---------------- CONFIG ----------------
DRIFT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/drift_nsidc/"                                  # dir of annual *_sh_*.nc files
DRIFT_GLOB = "icemotion_daily_sh_25km_*_v4.1.nc"

U_VAR = "u"
V_VAR = "v"
ERR_VAR = "icemotion_error_estimate"
X_COORD = "x"
Y_COORD = "y"
TIME_COORD = "time"

CMS_TO_MS = 0.01                                 # u,v are cm/s

# Quality masking
EXCLUDE_COASTAL = True        # drop cells with negative error estimate
EXCLUDE_FAR_FROM_INPUT = True # drop cells with >= 1000 (nearest input > 1250 km)
FAR_FLAG_OFFSET = 1000.0
MAX_ERROR_CMS = None          # optional extra cap on error magnitude, e.g. 5.0

# Central differences need both x- and both y-neighbours; cells adjacent to a
# missing neighbour are set to NaN rather than one-sided-differenced. At the
# ice edge a one-sided difference silently compares ice against no-ice and
# manufactures large spurious divergence exactly where the signal matters most.
REQUIRE_ALL_NEIGHBOURS = True

# Sector definitions (Raphael & Hobbs 2014), longitude bounds on 0-360.
# Weddell wraps the prime meridian -- handled generically.
SECTORS = {
    "WED": (300.0, 20.0),      # wraps
    "KHV": (20.0, 90.0),
    "EA":  (90.0, 160.0),
    "RA":  (160.0, 230.0),
    "ABS": (230.0, 300.0),
}

OUTPUT_GRIDDED = "ice_divergence_daily_sh.nc"
OUTPUT_SECTOR_TABLE = "ice_divergence_by_sector_season.csv"
WRITE_GRIDDED = True          # set False if disk/memory is tight
# -----------------------------------------


def grid_spacing(ds):
    """Read dx from the coordinate array instead of assuming 25000 m.

    EASE-Grid South 25 km is actually 25067.525 m; the nominal '25 km' in the
    product name is rounded. Verifies x and y spacing agree and are uniform.
    """
    x = ds[X_COORD].values
    y = ds[Y_COORD].values
    dx = np.diff(x)
    dy = np.diff(y)

    if not np.allclose(dx, dx[0], rtol=1e-6):
        raise ValueError("x spacing is not uniform; central differences invalid")
    if not np.allclose(np.abs(dy), np.abs(dy[0]), rtol=1e-6):
        raise ValueError("y spacing is not uniform; central differences invalid")
    if not np.isclose(abs(dx[0]), abs(dy[0]), rtol=1e-6):
        raise ValueError(f"dx ({dx[0]}) != dy ({dy[0]}); grid is not square")

    return abs(float(dx[0])), float(np.sign(dy[0]))


def decode_time(ds):
    """Build a proper datetime64 time coordinate.

    The files declare calendar="julian" with units "days since 1970-01-01".
    Taken literally that is the Julian calendar, which xarray decodes into
    cftime.DatetimeJulian objects (not convertible to pandas datetimes) and
    which would place every date ~13 days off Gregorian in the modern era.
    NSIDC almost certainly means ordinary dates -- verify by checking that the
    first value of the 1978 file is 3226 (= days from 1970-01-01 to
    1978-11-01), not 3213.

    This decodes against the proleptic Gregorian calendar, which is what the
    filenames imply.
    """
    t = ds[TIME_COORD]
    units = t.attrs.get("units", "days since 1970-01-01")
    if "since" not in units:
        raise ValueError(f"Unrecognised time units: {units!r}")

    interval, epoch_str = [s.strip() for s in units.split("since", 1)]
    epoch = pd.Timestamp(epoch_str)

    unit_map = {"days": "D", "hours": "h", "minutes": "m", "seconds": "s"}
    if interval not in unit_map:
        raise ValueError(f"Unsupported time interval: {interval!r}")

    return epoch + pd.to_timedelta(t.values, unit=unit_map[interval])


def load_drift(path):
    """Open one annual file, apply quality masks, convert to m/s."""
    # decode_times=False: the julian calendar attribute would otherwise
    # produce cftime objects. See decode_time() for why.
    ds = xr.open_dataset(path, mask_and_scale=True, decode_times=False)
    ds = ds.assign_coords({TIME_COORD: decode_time(ds)})

    u = ds[U_VAR]
    v = ds[V_VAR]

    for name, da in (("u", u), ("v", v)):
        units = da.attrs.get("units", "")
        if "cm" not in units.replace(" ", "").lower():
            warnings.warn(
                f"{name} declares units='{units}'; script assumes cm/s. "
                f"Check the conversion factor."
            )

    n_total = int(u.notnull().sum())

    # --- quality masking on the error estimate ---
    if ERR_VAR in ds and (EXCLUDE_COASTAL or EXCLUDE_FAR_FROM_INPUT
                          or MAX_ERROR_CMS is not None):
        err = ds[ERR_VAR]
        keep = xr.ones_like(err, dtype=bool)
        stats = {}

        if EXCLUDE_COASTAL:
            coastal = err < 0
            stats["coastal"] = int((coastal & u.notnull()).sum())
            keep = keep & ~coastal

        if EXCLUDE_FAR_FROM_INPUT:
            far = err >= FAR_FLAG_OFFSET
            stats["far_from_input"] = int((far & u.notnull()).sum())
            keep = keep & ~far

        if MAX_ERROR_CMS is not None:
            # only meaningful on the non-offset, non-negative population
            noisy = (err >= 0) & (err < FAR_FLAG_OFFSET) & (err > MAX_ERROR_CMS)
            stats["high_error"] = int((noisy & u.notnull()).sum())
            keep = keep & ~noisy

        u = u.where(keep)
        v = v.where(keep)

        n_kept = int(u.notnull().sum())
        frac = 100.0 * (1 - n_kept / n_total) if n_total else 0.0
        detail = ", ".join(f"{k}={v_:,}" for k, v_ in stats.items())
        print(f"      quality mask removed {frac:.1f}% of valid cells ({detail})")
    else:
        n_kept = n_total
        if ERR_VAR not in ds:
            warnings.warn(f"{ERR_VAR} not present; no quality masking applied")

    u = u * CMS_TO_MS
    v = v * CMS_TO_MS
    return ds, u, v, n_total, n_kept


def divergence(u, v, dx, y_sign):
    """Central-difference divergence on a uniform equal-area grid.

    Returns divergence in s^-1. Positive = divergence (opening),
    negative = convergence (closing).

    y_sign handles the case where the y coordinate decreases with index, which
    would otherwise flip the sign of dv/dy.
    """
    du_dx = (u.shift({X_COORD: -1}) - u.shift({X_COORD: 1})) / (2.0 * dx)
    dv_dy = (v.shift({Y_COORD: -1}) - v.shift({Y_COORD: 1})) / (2.0 * dx) * y_sign

    div = du_dx + dv_dy

    if REQUIRE_ALL_NEIGHBOURS:
        neighbours_valid = (
            u.shift({X_COORD: -1}).notnull()
            & u.shift({X_COORD: 1}).notnull()
            & v.shift({Y_COORD: -1}).notnull()
            & v.shift({Y_COORD: 1}).notnull()
        )
        div = div.where(neighbours_valid)

    div.attrs.update({
        "units": "s-1",
        "long_name": "sea ice divergence",
        "note": ("Central differences on EASE-Grid South (dx read from "
                 "coordinate array). NaN where any neighbour is missing or "
                 "quality-masked. Positive = opening, negative = closing."),
    })
    return div


def sector_mask_from(ds):
    """Assign each grid cell to a sector using its longitude field."""
    if "longitude" not in ds:
        raise KeyError("No 'longitude' field found in file.")
    lon360 = ds["longitude"] % 360.0

    sector = xr.full_like(lon360, fill_value="", dtype=object)
    for name, (lo, hi) in SECTORS.items():
        lo360, hi360 = lo % 360.0, hi % 360.0
        if lo360 <= hi360:
            mask = (lon360 >= lo360) & (lon360 < hi360)
        else:                                   # wraps prime meridian (Weddell)
            mask = (lon360 >= lo360) | (lon360 < hi360)
        sector = xr.where(mask, name, sector)
    return sector


def season_of(month):
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def aggregate_by_sector(div, sector_mask):
    """Mean divergence per (date, sector).

    Divergence and convergence are reported separately as well as net: the net
    mean cancels opening against closing, hiding the asymmetry that matters for
    the area budget, since convergence goes partly into ridging rather than
    proportional area loss.
    """
    frames = []
    for name in SECTORS:
        d = div.where(sector_mask == name)
        frames.append(pd.DataFrame({
            "date": pd.DatetimeIndex(div[TIME_COORD].values),
            "sector": name,
            "div_net": d.mean(dim=[X_COORD, Y_COORD], skipna=True).values,
            "div_positive": d.where(d > 0).mean(dim=[X_COORD, Y_COORD],
                                                skipna=True).values,
            "div_negative": d.where(d < 0).mean(dim=[X_COORD, Y_COORD],
                                                skipna=True).values,
            "n_valid_cells": d.notnull().sum(dim=[X_COORD, Y_COORD]).values,
        }))

    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["season"] = out["month"].map(season_of)
    # DJF spans the calendar boundary: assign December to the following year's
    # DJF so each season is contiguous in time.
    out.loc[out["month"] == 12, "year"] += 1
    return out


def run():
    files = sorted(glob.glob(os.path.join(DRIFT_DIR, DRIFT_GLOB)))
    if not files:
        raise FileNotFoundError(f"No files matched {DRIFT_GLOB} in {DRIFT_DIR}")
    print(f"Found {len(files)} annual drift files "
          f"({os.path.basename(files[0])} .. {os.path.basename(files[-1])})")

    gridded_parts, sector_parts = [], []
    tot_all = tot_kept = 0

    for path in files:
        print(f"  {os.path.basename(path)}")
        ds, u, v, n_tot, n_keep = load_drift(path)
        tot_all += n_tot
        tot_kept += n_keep

        dx, y_sign = grid_spacing(ds)
        div = divergence(u, v, dx, y_sign)
        sec = sector_mask_from(ds)

        if WRITE_GRIDDED:
            gridded_parts.append(div)
        sector_parts.append(aggregate_by_sector(div, sec))
        ds.close()

    print(f"\nGrid spacing used: {dx:.3f} m "
          f"(nominal 25 km; EASE-Grid South actual is 25067.525 m)")
    print(f"Quality masking retained {100.0 * tot_kept / tot_all:.1f}% "
          f"of originally-valid vectors across all files")

    if WRITE_GRIDDED:
        gridded = xr.concat(gridded_parts, dim=TIME_COORD).sortby(TIME_COORD)
        gridded.name = "divergence"
        gridded.to_netcdf(OUTPUT_GRIDDED)
        print(f"Wrote gridded divergence -> {OUTPUT_GRIDDED}")

    table = pd.concat(sector_parts, ignore_index=True).sort_values(
        ["sector", "date"])
    table = table[table["n_valid_cells"] > 0]
    table.to_csv(OUTPUT_SECTOR_TABLE, index=False)
    print(f"Wrote sector table -> {OUTPUT_SECTOR_TABLE}")

    print("\nSanity check -- net divergence by sector (s^-1):")
    print(table.groupby("sector")["div_net"].describe()[
        ["mean", "std", "min", "max"]])
    print("\nExpected magnitudes are O(1e-7) to O(1e-6) s^-1. Values far "
          "outside that range indicate a unit-conversion or grid-spacing "
          "problem rather than a physical signal.")

    return table


if __name__ == "__main__":
    run()