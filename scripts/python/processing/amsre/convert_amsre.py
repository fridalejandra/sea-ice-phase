import os
import h5py
import xarray as xr
import numpy as np
from tqdm import tqdm

RAW_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/amsre/raw"
OUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/amsre/daily_nc"
os.makedirs(OUT_DIR, exist_ok=True)

SIC_PATH = "HDFEOS/GRIDS/SpPolarGrid12km/Data Fields/SI_12km_SH_ICECON_DAY"
LAT_PATH = "HDFEOS/GRIDS/SpPolarGrid12km/lat"
LON_PATH = "HDFEOS/GRIDS/SpPolarGrid12km/lon"

def extract_date(filename):
    parts = filename.split("_")
    for part in parts:
        cleaned = part.replace(".he5", "")
        if cleaned.isdigit() and len(cleaned) == 8:
            return cleaned
    return None

files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".he5")])
print(f"Found {len(files)} .he5 files")

# remove old converted files to start clean
for f in os.listdir(OUT_DIR):
    if f.endswith(".nc"):
        os.remove(os.path.join(OUT_DIR, f))

for fname in tqdm(files, desc="Converting HE5 to NetCDF"):
    date = extract_date(fname)
    if not date:
        print(f"Skipping {fname}: date not found")
        continue

    out_path = os.path.join(OUT_DIR, f"SIC_{date}.nc")
    in_path  = os.path.join(RAW_DIR, fname)

    try:
        with h5py.File(in_path, "r") as f:
            sic = f[SIC_PATH][:].astype(np.int32)   # keep as integer 0-120
            lat = f[LAT_PATH][:].astype(np.float64)
            lon = f[LON_PATH][:].astype(np.float64)

        t = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}")
        ds = xr.Dataset(
            {"SI_12km_SH_ICECON_DAY_SpPolarGrid12km": (["y", "x"], sic)},
            coords={
                "GridLat_SpPolarGrid12km": (["y", "x"], lat),
                "GridLon_SpPolarGrid12km": (["y", "x"], lon),
                "time": t,
            },
        )
        ds["SI_12km_SH_ICECON_DAY_SpPolarGrid12km"].attrs = {
            "long_name": "Sea ice concentration daily average",
            "units":     "percent",
            "comment":   "0=Open Water, 110=missing, 120=Land",
        }
        ds.to_netcdf(out_path)
    except Exception as e:
        print(f"Failed {fname}: {e}")
