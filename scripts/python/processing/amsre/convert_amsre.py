import os
import h5py
import xarray as xr
import numpy as np
from datetime import datetime
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

for fname in tqdm(files, desc="Converting HE5 to NetCDF"):
    date = extract_date(fname)
    if not date:
        print(f"Skipping {fname}: date not found")
        continue

    out_path = os.path.join(OUT_DIR, f"SIC_{date}.nc")
    if os.path.exists(out_path):
        continue

    in_path = os.path.join(RAW_DIR, fname)
    try:
        with h5py.File(in_path, "r") as f:
            sic  = f[SIC_PATH][:].astype(np.float32)
            lat  = f[LAT_PATH][:]
            lon  = f[LON_PATH][:]

        sic[sic > 100] = np.nan  # mask fill values

        t = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}")
        ds = xr.Dataset(
            {"SI_12km_SH_ICECON_DAY": (["y", "x"], sic)},
            coords={"lat": (["y", "x"], lat),
                    "lon": (["y", "x"], lon),
                    "time": t},
        )
        ds["SI_12km_SH_ICECON_DAY"].attrs["units"] = "percent"
        ds.to_netcdf(out_path)
    except Exception as e:
        print(f"Failed {fname}: {e}")
