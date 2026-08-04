"""
fetchoras5mld.py

Pull ORAS5 mixed layer depth (+ SST for cross-check) from CDS, 1979-2023.
Chunked ONE YEAR PER REQUEST to stay under CDS cost limits.

Product split: consolidated (to 2014, ERA-Interim forced), operational
(2015 on, operational forced). The system change sits one year before the
2016 split -- check for a step at 2014/15 in the MLD time series before
trusting any pre/post difference (see notes at bottom).
"""

import os
import cdsapi

OUT_DIR = "oras5_raw"
os.makedirs(OUT_DIR, exist_ok=True)

# CDS API variable names (underscores, not periods)
VARIABLES = [
    "mixed_layer_depth_0_01",
    "sea_surface_temperature",
]

MONTHS = [f"{m:02d}" for m in range(1, 13)]

client = cdsapi.Client()


def request_year(product_type, year, var):
    """One request per (year, variable) to stay under cost limits."""
    target = os.path.join(OUT_DIR, f"oras5_{var}_{year}.zip")
    if os.path.exists(target):
        print(f"[skip] {target} exists")
        return
    print(f"Requesting {var} ({product_type}, {year}) ...")
    try:
        client.retrieve(
            "reanalysis-oras5",
            {
                "product_type": product_type,
                "vertical_resolution": "single_level",
                "variable": var,
                "year": year,
                "month": MONTHS,
            },
            target,
        )
        print(f"  -> {target}")
    except Exception as e:
        print(f"  [FAIL] {var} {year}: {e}")


if __name__ == "__main__":
    # consolidated: 1979-2014
    for year in range(1979, 2015):
        for var in VARIABLES:
            request_year("consolidated", str(year), var)

    # operational: 2015-2023
    for year in range(2015, 2024):
        for var in VARIABLES:
            request_year("operational", str(year), var)

    print("\nDone. One ZIP per (variable, year). Unzip, concatenate along")
    print("time, subset to Southern Ocean by nav_lat, then deseasonalize")
    print("against period-specific climatologies.")
    print("\nCHECK FIRST: plot sector-mean MLD and look for a step at 2014/15")
    print("(the consolidated->operational product change). If present, treat")
    print("MLD pre/post results as indicative, not definitive.")