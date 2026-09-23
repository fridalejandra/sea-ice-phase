#!/usr/bin/env python
# =====================================================
# ERA5 10 m winds -> SIE (x,y) grid -> daily sector means
# u, v (signed) and speed, per sector and circumpolar.
# Resumable: one CSV per year in OUT_DIR/by_year/, skipped
# if already complete, then concatenated.
# =====================================================

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import glob
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from tqdm import tqdm
from pathlib import Path
from pyproj import CRS, Transformer

# =====================================================
# PATHS
# =====================================================
ERA5_BASE = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/winds"

SIE_GRID_FILE = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/"
    "merged_bootstrap_SH_latest.nc"
)

SECTOR_MASK_FILE = (
    "/user/geog/falejandraperez/sea-ice-phase/data/"
    "canonical_sectors.nc"
)

OUT_DIR  = "/user/geog/falejandraperez/sea-ice-phase/results/ERA5"
OUT_FILE = "ERA5_winds_daily_sector.csv"      # u_, v_, wind_ columns
BY_YEAR  = os.path.join(OUT_DIR, "by_year")

START_YEAR = int(os.environ.get("START_YEAR", 1979))
END_YEAR   = int(os.environ.get("END_YEAR", 2025))

SECTORS = {
    1: "Weddell",
    2: "Amundsen_Bellingshausen",
    3: "Ross",
    4: "East_Antarctica",
    5: "King_Haakon"
}

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(BY_YEAR).mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD SIE GRID + SECTOR MASK (ONCE)
# =====================================================
sie = xr.open_dataset(SIE_GRID_FILE)
sector_mask = xr.open_dataset(SECTOR_MASK_FILE)["sector_id"]
assert "x" in sie.dims and "y" in sie.dims

crs = CRS.from_epsg(3412)
transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
xx, yy = np.meshgrid(sie.x.values, sie.y.values)
lon, lat = transformer.transform(xx, yy)
sie = sie.assign_coords(lon=(("y", "x"), lon), lat=(("y", "x"), lat))

# =====================================================
# REGRIDDER (ONCE; weights reused if the file exists)
# =====================================================
sample_file = sorted(glob.glob(f"{ERA5_BASE}/{START_YEAR}/*.nc"))[0]
era5_sample = xr.open_dataset(sample_file)
wfile = "era5_to_sie_bilinear_weights.nc"
regridder = xe.Regridder(era5_sample, sie, method="bilinear",
                         reuse_weights=os.path.exists(wfile), filename=wfile)
era5_sample.close()

# Masks. The circumpolar mask is the UNION of the five sectors, not every
# non-missing cell of the grid (sector_id may be 0, not NaN, outside them).
masks = {}
for code, name in SECTORS.items():
    masks[name] = (sector_mask == code)
masks["circumpolar"] = sector_mask.isin(list(SECTORS.keys()))
n_cells = {name: int(m.sum()) for name, m in masks.items()}
print("mask cells:", n_cells, "  sector_id dtype:", sector_mask.dtype)
assert n_cells["circumpolar"] == sum(n_cells[n] for n in SECTORS.values()), \
    "circumpolar mask is not the union of the sectors"


def sector_means(field):
    """field on the SIE grid -> unweighted mean over each mask, as float.

    BUG FIXED 2026-09-21: this used to be
        field.where(m).weighted(xr.ones_like(sector_mask)).mean(...)
    ones_like keeps sector_id's small-integer dtype, and xarray's weighted
    mean sums the weights with an integer dot product, which WRAPS AROUND for
    a few thousand cells. The result was the field's sum divided by a wrapped
    (often negative) integer: values in the hundreds to thousands, signs that
    differed by sector, and negative 'speeds'. A plain mean over the masked
    cells is what was intended. The mask cell counts are asserted above and
    the first day's means are sanity-checked below."""
    out = {}
    for name, m in masks.items():
        out[name] = float(field.where(m).mean(dim=("y", "x"), skipna=True))
    return out


# =====================================================
# MAIN LOOP, one year at a time, resumable
# =====================================================
EXPECTED_COLS = {f"{lab}_{name}" for lab in ("u", "v", "wind") for name in masks}


def by_year_is_current(path, n_files):
    """A cached by-year file is reused only if THIS script could have written
    it: same column names (u_/v_/wind_, not u10_/v10_), speed never negative,
    and one row per input file. Anything else is stale and is regenerated."""
    try:
        old = pd.read_csv(path)
    except Exception:
        return False
    if not EXPECTED_COLS.issubset(old.columns):
        return False
    if (old[[c for c in old.columns if c.startswith("wind_")]] < 0).any().any():
        return False
    return len(old) >= n_files


for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Years"):
    files = sorted(glob.glob(f"{ERA5_BASE}/{year}/*.nc"))
    if not files:
        continue
    ypath = os.path.join(BY_YEAR, f"winds_{year}.csv")
    if os.path.exists(ypath):
        if by_year_is_current(ypath, len(files)):
            continue          # this year is complete and from this script
        print(f"  {year}: cached by-year file is stale (old columns or negative speed) -- regenerating")
        os.remove(ypath)

    rows = []
    for f in tqdm(files, desc=str(year), leave=False):
        ds = xr.open_dataset(f)
        if "valid_time" in ds.dims or "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        time_val = pd.Timestamp(ds.time.values.item())

        u = regridder(ds.u10)
        v = regridder(ds.v10)
        spd = np.sqrt(u ** 2 + v ** 2)

        row = {"time": time_val}
        for label, field in (("u", u), ("v", v), ("wind", spd)):
            for name, val in sector_means(field).items():
                row[f"{label}_{name}"] = val
        rows.append(row)
        ds.close()

        # first day of the run: is this a 10 m wind in m/s?
        if not globals().get("_checked_first_day"):
            _checked_first_day = True
            globals()["_checked_first_day"] = True
            print(f"\nfirst day {time_val.date()}: ERA5 u10 range {float(ds.u10.min()):+.1f}..{float(ds.u10.max()):+.1f}"
                  f"  regridded u range {float(u.min()):+.1f}..{float(u.max()):+.1f}"
                  f"  NaN cells in regridded u: {int(u.isnull().sum())}")
            for name in masks:
                print(f"   {name:26s} u {row[f'u_{name}']:+6.2f}  v {row[f'v_{name}']:+6.2f}  "
                      f"speed {row[f'wind_{name}']:5.2f}  (cells {n_cells[name]})")
            bad = [n for n in masks if not (0 <= row[f"wind_{n}"] < 40)
                   or not (abs(row[f"u_{n}"]) < 40 and abs(row[f"v_{n}"]) < 40)]
            if bad:
                raise SystemExit(f"first-day sector means are not plausible 10 m winds in m/s for {bad}; "
                                 "check the regridder weights file and the mask alignment before continuing")

    pd.DataFrame(rows).sort_values("time").to_csv(ypath, index=False)

# =====================================================
# CONCATENATE
# =====================================================
parts = sorted(glob.glob(os.path.join(BY_YEAR, "winds_*.csv")))
df = pd.concat([pd.read_csv(p) for p in parts]).sort_values("time")
df = df.drop_duplicates("time")
# =====================================================
# VALIDATE before writing: a file that fails here is not written
# =====================================================
problems = []
spd_cols = [c for c in df.columns if c.startswith("wind_")]
if (df[spd_cols] < 0).any().any():
    problems.append("negative wind speed")
print("\nmean 10 m wind by sector, 1979-onward (m/s; unweighted grid-cell mean over the mask):")
for name in masks:
    mu, mv, ms = df[f"u_{name}"].mean(), df[f"v_{name}"].mean(), df[f"wind_{name}"].mean()
    flag = ""
    if mu <= 0:
        flag = "   <-- mean zonal wind easterly: check the mask extent and the u10 field"
    if ms < np.hypot(mu, mv) - 1e-6:
        flag += "   <-- speed below |mean vector|: impossible"
        problems.append(f"{name}: speed < |mean vector|")
    print(f"  {name:26s} u {mu:+6.2f}   v {mv:+6.2f}   speed {ms:5.2f}{flag}")
if problems:
    raise SystemExit("NOT WRITTEN -- validation failed: " + "; ".join(problems))
out_path = os.path.join(OUT_DIR, OUT_FILE)
df.to_csv(out_path, index=False)
print(f"\nSaved ERA5 sector winds (u, v, speed) to:\n{out_path}\n"
      f"{df['time'].min()} to {df['time'].max()}, {len(df)} days")
print("columns are u_<sector>, v_<sector>, wind_<sector>: geographic east/north 10 m wind and its speed,\n"
      "unweighted mean over the sector mask, m/s. No vector rotation is applied.")
