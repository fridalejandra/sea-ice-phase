"""
fetch_era5_sst.py

Submits an ERA5 sea surface temperature request via the CDS API, Southern
Ocean subset only, daily. Matching ERA5 as the SST source (rather than OISST)
means the ocean-state moderator comes from the same reanalysis as the wind
stress data already in use -- no cross-product reconciliation needed in the
methods section.

SETUP (one-time)
  pip install cdsapi --break-system-packages
  Create ~/.cdsapirc with:
    url: https://cds.climate.copernicus.eu/api
    key: <your-personal-access-token>
  (Get the token from your CDS profile page after logging in.)

QUEUE TIME
  CDS request queues are variable -- anywhere from minutes to many hours
  depending on load and how the request is structured. Submit this NOW and
  let it run in the background; do not wait on it interactively. The script
  polls and downloads once ready, but you can also just re-run it later --
  cdsapi.Client().retrieve() will reconnect to an in-progress or completed
  request rather than resubmitting a duplicate if called again with the same
  request dict, depending on API version behaviour -- if in doubt, check your
  request status at https://cds.climate.copernicus.eu/requests before
  resubmitting.

REQUEST SIZE
  Full-record (1979-2024) daily SST at 0.25 degree, Southern Ocean only
  (south of ~40S), single level. This is small enough for one request; if it
  times out or the queue rejects it as too large, split by decade and
  concatenate (see SPLIT_BY_DECADE below).
"""

import os

import cdsapi

# ---------------- CONFIG ----------------
OUTPUT_DIR = "era5_sst_raw"
YEAR_START = 1979
YEAR_END = 2024

# Southern Ocean bounding box: North/West/South/East, ERA5 convention
AREA = [-40, -180, -90, 180]

SPLIT_BY_DECADE = False   # set True if a single full-record request is rejected
# -----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_request(year_start, year_end):
    return {
        "product_type": ["reanalysis"],
        "variable": ["sea_surface_temperature"],
        "year": [str(y) for y in range(year_start, year_end + 1)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00"],          # one snapshot/day is enough for a daily proxy
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def submit(year_start, year_end, tag):
    client = cdsapi.Client()
    target = os.path.join(OUTPUT_DIR, f"era5_sst_{tag}.nc")

    if os.path.exists(target):
        print(f"{target} already exists, skipping")
        return target

    print(f"Submitting request for {tag} ({year_start}-{year_end})...")
    print("This will queue on the CDS side -- may take minutes to hours. "
          "Safe to leave running in the background (e.g. under nohup/tmux).")

    client.retrieve("reanalysis-era5-single-levels", build_request(year_start, year_end)) \
          .download(target)

    print(f"Downloaded -> {target}")
    return target


def main():
    if SPLIT_BY_DECADE:
        decade_starts = range((YEAR_START // 10) * 10, YEAR_END + 1, 10)
        for ds in decade_starts:
            de = min(ds + 9, YEAR_END)
            if de < YEAR_START:
                continue
            submit(max(ds, YEAR_START), de, tag=f"{max(ds, YEAR_START)}_{de}")
    else:
        submit(YEAR_START, YEAR_END, tag=f"{YEAR_START}_{YEAR_END}")


if __name__ == "__main__":
    main()