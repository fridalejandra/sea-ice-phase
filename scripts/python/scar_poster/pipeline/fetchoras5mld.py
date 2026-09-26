import os
import cdsapi

OUT_DIR = "oras5_raw"
os.makedirs(OUT_DIR, exist_ok=True)

VARIABLES = [
    "mixed_layer_depth_0_01",
    "sea_surface_temperature",
]

MONTHS = [f"{m:02d}" for m in range(1, 13)]

client = cdsapi.Client()


def request_year(product_type, year, var):
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
    for year in range(1979, 2015):
        for var in VARIABLES:
            request_year("consolidated", str(year), var)
    for year in range(2015, 2024):
        for var in VARIABLES:
            request_year("operational", str(year), var)
    print("Done.")
