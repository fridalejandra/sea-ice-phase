import cdsapi
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, time

# ---------------- CONFIG ----------------
OUT_ROOT = "/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5/mslp"
AREA = [-40, -180, -90, 180]   # [North, West, South, East] -> south of 50°S
GRID = [0.5, 0.5]              # coarsen to 0.5°; comment out to keep native
TIMES = ["12:00"]               # or ["00:00","06:00","12:00","18:00"] for 6-hourly
MAX_WORKERS = 4                    # be gentle with CDS (2–4 is sane)
RETRIES = 5                       # simple retry count
SLEEP0 = 3                       # initial backoff seconds

START = datetime(1979, 1, 1)
END   = datetime(2024, 12, 31)
# ---------------------------------------

def day_iter(start_dt, end_dt):
    d = start_dt
    while d <= end_dt:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=1)

def out_path(date_str):
    year = date_str[:4]
    times_tag = "12UTC" if TIMES == ["12:00"] else "6hourly"
    out_dir = os.path.join(OUT_ROOT, year)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"era5_msl_{date_str}_{times_tag}.nc")

def download_day(date_str):
    client = cdsapi.Client()   # grabs creds from ~/.cdsapirc

    year  = date_str[:4]
    month = date_str[4:6]
    day   = date_str[6:8]

    target = out_path(date_str)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return f"SKIP  {target}"

    req = {
        "product_type": "reanalysis",
        "variable": ["mean_sea_level_pressure"],
        "year": year,
        "month": month,
        "day": day,
        "time": TIMES,
        "data_format": "netcdf",
        "area": AREA,           # N, W, S, E (CDS convention)
    }
    # Add grid only if you want coarsened output
    if GRID:
        req["grid"] = GRID

    tmp = target + ".part"

    # retry with exponential backoff
    for k in range(RETRIES):
        try:
            print(f"→ Downloading {target}")
            client.retrieve("reanalysis-era5-single-levels", req, tmp)
            os.replace(tmp, target)
            return f"  ✔ Done: {target}"
        except Exception as e:
            # clean temp if it exists (partial/corrupt)
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if k == RETRIES - 1:
                return f"  ✖ FAIL: {target} :: {repr(e)}"
            sleep = min(60, SLEEP0 * (2 ** k))
            print(f"    Retry {k+1}/{RETRIES} after {sleep}s due to {repr(e)}")
            time.sleep(sleep)

def main():
    dates = list(day_iter(START, END))
    # Parallel, but not too aggressive
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(download_day, d) for d in dates]
        for f in as_completed(futs):
            msg = f.result()
            print(msg)
            results.append(msg)

if __name__ == "__main__":
    main()
