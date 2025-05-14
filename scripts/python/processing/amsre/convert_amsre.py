import os
import h5py
import xarray as xr
import numpy as np
from tqdm import tqdm

# ---- CONFIG ----
RAW_DIR = "data/amsre/raw"
OUT_DIR = "data/amsre/daily_nc"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_date(filename):
    # e.g., AMSR_E_L3_SeaIce12km_B04_20230101.he5
    parts = filename.split("_")
    for part in parts:
        if part.isdigit() and len(part) == 8:
            return part
    return None

# ---- CONVERSION ----
files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".he5")])

for fname in tqdm(files, desc="Converting HE5 to NetCDF"):
    date = extract_date(fname)
    if not date:
        print(f"⚠️ Skipping {fname}: date not found")
        continue

    in_path = os.path.join(RAW_DIR, fname)
    out_path = os.path.join(OUT_DIR, f"SIC_{date}.nc")

    try:
        with h5py.File(in_path, "r") as f:
            # Adjust this path if needed (this is common for AMSRE)
            sic = f["HDFEOS/GRIDS/PolarGrid/Data Fields/Sea_Ice_Concentration"][:]

            # Convert to proper units (often scaled by 10)
            sic = sic.astype(np.float32) / 10.0
            sic[sic > 100] = np.nan  # Mask fill values or bad data

            # Wrap in xarray
            da = xr.DataArray(sic, dims=("y", "x"), name="sic")
            da.attrs["units"] = "percent"
            da.attrs["long_name"] = "Sea Ice Concentration"
            ds = xr.Dataset({"sic": da})
            ds.attrs["source_file"] = fname

            ds.to_netcdf(out_path)
    except Exception as e:
        print(f"❌ Failed to convert {fname}: {e}")
