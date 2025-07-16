import cdsapi
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import os

# Function to download ERA5 data for a specific day
def download_day(date_str):
    client = cdsapi.Client()

    year = date_str[:4] # breaking up the string
    month = date_str[4:6]
    day = date_str[6:8]

    # Define the folder and filename
    output_dir = os.path.join("/user/geog/falejandraperez/sea-ice-phase/data/Reanalysis_ERA5", year)
    os.makedirs(output_dir, exist_ok=True)   # Create folder if it doesn't exist

    filename = os.path.join(output_dir, f"era5_wind_{date_str}_12UTC.nc")

    print(f"→ Downloading {filename}")

    # Submit request to CDS API
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": year,
            "month": month,
            "day": day,
            "time": ["12:00"],
            "data_format": "netcdf",
            "area": [-50, -180, -90, 180]
        },
        filename
    )

    print(f"  ✔ Done: {filename}")

# Generate list of all days from Jan 1, 1979 to Dec 31, 2024
start_date = datetime(1979, 1, 1)
end_date = datetime(2024, 12, 31)

date_list = []
current_date = start_date
while current_date <= end_date:
    date_list.append(current_date.strftime("%Y%m%d"))
    current_date += timedelta(days=1)

# Use a thread pool to download in parallel (adjust max_workers if needed)
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(download_day, date_list)
