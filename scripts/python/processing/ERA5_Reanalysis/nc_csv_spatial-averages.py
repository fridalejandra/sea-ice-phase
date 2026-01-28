import xarray as xr
import numpy as np
from pathlib import Path
import glob

# =====================================================
# User settings (mirror SIE script)
# =====================================================
era5_dir = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/winds"
mask_file = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"

out_dir = Path("/user/geog/falejandraperez/sea-ice-phase/results/ERA5")
out_dir.mkdir(parents=True, exist_ok=True)

out_file = out_dir / "ERA5_winds_daily_sector.csv"

sectors = {
    1: "Weddell",
    2: "Amundsen_Bellingshausen",
    3: "Ross",
    4: "East_Antarctica",
    5: "King_Haakon"
}

# =====================================================
# load ERA5 (all years)
# =====================================================
files = sorted(glob.glob(f"{era5_dir}/**/*.nc", recursive=True))

ds = xr.open_mfdataset(
    files,
    combine="nested",
    concat_dim="valid_time",
    decode_times=True
)

# =====================================================
# rename time to match SIE convention
# =====================================================
ds = ds.rename({"valid_time": "time"})

# =====================================================
# derive wind speed
# =====================================================
ds["wind_speed"] = np.sqrt(ds.u10**2 + ds.v10**2)

# =====================================================
# load sector mask
# =====================================================
mask = xr.open_dataset(mask_file)["sector_id"]

# =====================================================
# area weights (latitude)
# =====================================================
weights = np.cos(np.deg2rad(ds.latitude))
weights.name = "weights"

# =====================================================
# circumpolar mean (independent, like SIE)
# =====================================================
antarctic_ocean = mask.notnull()

circ = (
    ds[["u10", "v10", "wind_speed"]]
    .where(antarctic_ocean)
    .weighted(weights)
    .mean(dim=("latitude", "longitude"))
)

circ = circ.rename({
    "u10": "u10_circumpolar",
    "v10": "v10_circumpolar",
    "wind_speed": "wind_circumpolar"
})

# =====================================================
# sector means
# =====================================================
vars_out = [circ]

for code, name in sectors.items():
    ds_sec = ds.where(mask == code)

    sec = (
        ds_sec[["u10", "v10", "wind_speed"]]
        .weighted(weights)
        .mean(dim=("latitude", "longitude"))
    )

    sec = sec.rename({
        "u10": f"u10_{name}",
        "v10": f"v10_{name}",
        "wind_speed": f"wind_{name}"
    })

    vars_out.append(sec)

# =====================================================
# merge + daily aggregation
# =====================================================
era_ds = xr.merge(vars_out)

era_ds = era_ds.resample(time="1D").mean()

# =====================================================
# to CSV
# =====================================================
df = era_ds.to_dataframe().reset_index()
df.to_csv(out_file, index=False)

print(f"Saved: {out_file}")
