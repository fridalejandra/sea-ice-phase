"""
fetch_oras5_mld.py

Pull ORAS5 mixed layer depth (+ SST for cross-check) from CDS, Southern Ocean,
1979-2023. Fire this off and let it queue while you work on other things --
the download is decoupled from the analysis.

WHY MLD RATHER THAN SST
SST is a surface skin measurement of a subsurface process, and ERA5 masks it
under its own sea ice -- so it is blank in winter exactly where the ice is.
MLD is a direct measure of upper-ocean stratification, is defined under ice,
and is the closer analogue to the mechanism the low-ice-state literature
invokes.

*** READ BEFORE USING THE OUTPUT ***

1. PRODUCT SPLIT AT YOUR REGIME BOUNDARY.
   ORAS5 "consolidated" (to 2014) uses ERA-Interim atmospheric forcing;
   "operational" (2015 on) uses ECMWF operational forcing and near-real-time
   observations. That system change sits ONE YEAR before your 2016 split, so
   any pre/post difference is partly confounded with a change in the
   reanalysis system.
   TEST FOR IT: plot the sector-mean MLD time series and look for a step at
   2014/15. Also check a low-latitude region far from the ice, where no
   physical step should exist. If a step is visible, say so, and consider
   restricting the pre-period to 2015 onward is impossible (too short) --
   more realistically, report the artifact and treat MLD results as
   indicative rather than definitive.

2. MONTHLY, NOT DAILY.
   Your regression is daily. Broadcasting a monthly MLD across that month's
   days is defensible -- ocean state is a slowly varying background
   condition, not a daily forcing -- but say so explicitly rather than
   letting it look like an oversight.

3. CURVILINEAR GRID.
   ORAS5 comes on the NEMO ORCA tripolar grid with 2D nav_lat/nav_lon, not a
   regular lat/lon mesh. For gridded spatial correlation you will need to
   regrid onto EASE -- use the same conservative-regrid machinery already
   built for the Bootstrap SIC (it handles curvilinear grids; that is exactly
   what the corner-bounds step was for).

4. VARIABLE NAMES.
   CDS form variable names change occasionally. If the request errors on an
   invalid variable, open the dataset's Download form, tick the variable, and
   hit "Show API request" to get the current name. Expected NetCDF variable
   inside the files: somxl010 (MLD, 0.01 kg/m3 criterion), sosstsst (SST).
"""

import os
import cdsapi

OUT_DIR = "oras5_raw"
os.makedirs(OUT_DIR, exist_ok=True)

# CDS form names -- verify via "Show API request" if these error
VARIABLES = [
    "mixed_layer_depth_0.01",
    "sea_surface_temperature",     # cross-check against your ERA5 SST
]

CONSOLIDATED_YEARS = [str(y) for y in range(1979, 2015)]   # ERA-Interim forced
OPERATIONAL_YEARS = [str(y) for y in range(2015, 2024)]    # operational forced
MONTHS = [f"{m:02d}" for m in range(1, 13)]

client = cdsapi.Client()


def request(product_type, years, tag):
    for var in VARIABLES:
        target = os.path.join(OUT_DIR, f"oras5_{var}_{tag}.zip")
        if os.path.exists(target):
            print(f"[skip] {target} exists")
            continue
        print(f"\nRequesting {var} ({product_type}, {years[0]}-{years[-1]}) ...")
        client.retrieve(
            "reanalysis-oras5",
            {
                "product_type": product_type,
                "vertical_resolution": "single_level",
                "variable": var,
                "year": years,
                "month": MONTHS,
                "format": "zip",
            },
            target,
        )
        print(f"  -> {target}")


if __name__ == "__main__":
    # NOTE: no 'area' subsetting -- ORAS5 is on a curvilinear grid and CDS
    # area subsetting is unreliable for it. Download global, subset locally
    # by nav_lat after unzipping. Larger download, fewer surprises.
    request("consolidated", CONSOLIDATED_YEARS, "1979_2014")
    request("operational", OPERATIONAL_YEARS, "2015_2023")

    print("\nDone. Files arrive as ZIPs of one-month NetCDFs (~540 per")
    print("variable-period). Unzip, then concatenate along time, then subset")
    print("to the Southern Ocean by nav_lat before doing anything else --")
    print("global monthly fields for 45 years are large.")
    print("\nNEXT: sector-average (or regrid to EASE for gridded work), then")
    print("deseasonalize against period-specific climatologies, matching")
    print("what process_era5_sst.py already does for SST.")