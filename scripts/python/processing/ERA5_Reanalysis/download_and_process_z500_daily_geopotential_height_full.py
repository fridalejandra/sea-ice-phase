# download_and_process_z500_daily_geopotential_height_full.py

import calendar
import tempfile
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start-year", type=int, required=True)
parser.add_argument("--end-year", type=int, required=True)
args = parser.parse_args()

YEARS = range(args.start_year, args.end_year + 1)

# =========================
# User settings
# =========================
BASE = Path("/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/z500_daily_geopotential_height")
RAW_DIR = BASE / "raw_zip"
FINAL_DIR = BASE / "daily_nc"

YEARS = range(2008, 2018)      # 1979-2023 inclusive
MONTHS = range(1, 13)

AREA = [-40, -180, -90, 180]   # North, West, South, East
G = 9.80665                    # standard gravity for z -> geopotential height

DELETE_ZIP_AFTER_SUCCESS = False
COMPRESS_OUTPUT = True

RAW_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

c = cdsapi.Client()

# =========================
# Main loop
# =========================
for yr in YEARS:
    for mo in MONTHS:
        month_str = f"{mo:02d}"
        ndays = calendar.monthrange(yr, mo)[1]
        days = [f"{d:02d}" for d in range(1, ndays + 1)]

        zip_path = RAW_DIR / f"era5_z500_daily_mean_{yr}{month_str}.zip"
        final_nc = FINAL_DIR / f"era5_z500_geopotential_height_daily_{yr}{month_str}.nc"

        if final_nc.exists():
            print(f"Skipping existing final file: {final_nc}")
            continue

        try:
            # -------------------------
            # Step 1: Download zip
            # -------------------------
            if not zip_path.exists():
                print(f"Downloading {yr}-{month_str} to {zip_path}")
                c.retrieve(
                    "derived-era5-pressure-levels-daily-statistics",
                    {
                        "product_type": "reanalysis",
                        "variable": "geopotential",
                        "pressure_level": "500",
                        "year": str(yr),
                        "month": month_str,
                        "day": days,
                        "daily_statistic": "daily_mean",
                        "frequency": "1_hourly",
                        "time_zone": "utc+00:00",
                        "area": AREA,
                        "data_format": "netcdf",
                        "download_format": "zip",
                    },
                    str(zip_path),
                )
            else:
                print(f"Using existing zip: {zip_path}")

            # -------------------------
            # Step 2: Unzip and process
            # -------------------------
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmpdir)

                nc_files = sorted(tmpdir.glob("*.nc"))
                if len(nc_files) == 0:
                    raise FileNotFoundError(f"No .nc file found inside {zip_path}")
                if len(nc_files) > 1:
                    print(f"Warning: multiple .nc files found in {zip_path}, using first one")

                raw_nc = nc_files[0]
                print(f"Opened extracted file: {raw_nc.name}")

                ds = xr.open_dataset(raw_nc)

                # Rename valid_time -> time if needed
                if "valid_time" in ds.dims or "valid_time" in ds.coords:
                    ds = ds.rename({"valid_time": "time"})

                if "z" not in ds.data_vars:
                    raise KeyError(f"Variable 'z' not found in {raw_nc}")

                # Drop singular pressure dimension if present
                for dim in ["pressure_level", "level", "number"]:
                    if dim in ds.dims and ds.sizes[dim] == 1:
                        ds = ds.squeeze(dim, drop=True)

                # Sort coordinates
                if "time" in ds.coords:
                    ds = ds.sortby("time")
                if "latitude" in ds.coords:
                    ds = ds.sortby("latitude")
                if "longitude" in ds.coords:
                    ds = ds.sortby("longitude")

                # -------------------------
                # Step 3: Convert z -> zg
                # -------------------------
                zg = ds["z"] / G
                zg.name = "zg"
                zg.attrs["long_name"] = "daily mean geopotential height at 500 hPa"
                zg.attrs["units"] = "m"

                out_ds = zg.to_dataset()
                out_ds.attrs["source"] = "ERA5 derived daily statistics, pressure levels"
                out_ds.attrs["variable_original"] = "geopotential"
                out_ds.attrs["conversion"] = "zg = z / 9.80665"
                out_ds.attrs["pressure_level_hPa"] = 500
                out_ds.attrs["time_statistic"] = "daily_mean"
                out_ds.attrs["time_zone"] = "UTC"

                # -------------------------
                # Step 4: Save final NetCDF
                # -------------------------
                if COMPRESS_OUTPUT:
                    encoding = {"zg": {"zlib": True, "complevel": 4}}
                    out_ds.to_netcdf(final_nc, encoding=encoding)
                else:
                    out_ds.to_netcdf(final_nc)

                print(f"Saved final daily geopotential height file: {final_nc}")

                ds.close()
                out_ds.close()

            # -------------------------
            # Step 5: Optional cleanup
            # -------------------------
            if DELETE_ZIP_AFTER_SUCCESS and zip_path.exists():
                zip_path.unlink()
                print(f"Deleted zip: {zip_path}")

            print(f"Finished {yr}-{month_str}")

        except Exception as e:
            print(f"ERROR for {yr}-{month_str}: {e}")
            continue

print("Done.")