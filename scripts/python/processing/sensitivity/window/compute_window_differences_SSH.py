import os
import xarray as xr
import numpy as np

# === USER CONFIGURATION === #
DATASET = "SMMR"  # Change to "AMSRE"
PHASE = "retreat"  # or "advance"
YEARS = range(1979, 2024) if DATASET == "SMMR" else range(2012, 2024)
WINDOWS = [3, 5, 7]

# === PATH SETUP === #
BASEDIR = f"/user/geog/falejandraperez/sea-ice-phase/results/sensitivity/{DATASET}_phase"
INPUT_DIRS = {
    w: f"{BASEDIR}/seaice_phases_{DATASET}_{w}day" for w in WINDOWS
}
OUTDIR = f"{BASEDIR}/{DATASET}_window_comparison"
os.makedirs(OUTDIR, exist_ok=True)

# === BLOCK 2: Loop through years and compute window differences === #
diff_3_minus_5 = {}
diff_7_minus_5 = {}

for year in YEARS:
    try:
        # Construct file paths
        files = {
            w: os.path.join(INPUT_DIRS[w], f"seaice_phases_{DATASET}_{year}_{w}day.nc")
            for w in WINDOWS
        }

        # Load datasets
        ds3 = xr.open_dataset(files[3])
        ds5 = xr.open_dataset(files[5])
        ds7 = xr.open_dataset(files[7])

        varname = f"{PHASE}_{year}"
        da3 = ds3[varname].load()
        da5 = ds5[varname].load()
        da7 = ds7[varname].load()

        # Compute pixelwise differences
        diff_3 = da3 - da5
        diff_7 = da7 - da5

        diff_3_minus_5[year] = diff_3
        diff_7_minus_5[year] = diff_7

    except FileNotFoundError as e:
        print(f"Missing file for year {year}: {e}")
    except Exception as e:
        print(f"Error in year {year}: {e}")

# === BLOCK 3: Save yearly difference maps as NetCDF === #
for year in diff_3_minus_5.keys():
    try:
        diff3 = diff_3_minus_5[year]
        diff7 = diff_7_minus_5[year]

        out3_path = os.path.join(OUTDIR, f"diff_{PHASE}_3minus5_{year}.nc")
        out7_path = os.path.join(OUTDIR, f"diff_{PHASE}_7minus5_{year}.nc")

        diff3.to_dataset(name=f"diff_{PHASE}_3minus5").to_netcdf(out3_path)
        diff7.to_dataset(name=f"diff_{PHASE}_7minus5").to_netcdf(out7_path)

    except Exception as e:
        print(f"Error saving files for year {year}: {e}")
