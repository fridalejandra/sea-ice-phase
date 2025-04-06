import numpy as np
import xarray as xr
import os

THRESHOLD = 0.15
WINDOW = 5

# Load your dataset
DATA_DIR = '/home/falejandraperez/sea-ice-phase/data/bootstrap_smmr/'
RESULTS_DIR = '/home/falejandraperez/sea-ice-phase/results/bootstrap_smmr/'
DATA_FILE = DATA_DIR + 'merged_bootsstrap.nc'

data = xr.open_dataset(DATA_FILE)

# Preprocess data
data = data.sortby('time')
data = data.sel(time=~data.get_index("time").duplicated())

sic_xr = data.N07_ICECON

# Reverse latitude if necessary
if sic_xr.y[0] > sic_xr.y[-1]:
    sic_xr = sic_xr.sel(y=slice(None, None, -1))

def compute_phase_metric(da, threshold, window_size, condition='above'):
    cond_met = da >= threshold if condition == 'above' else da <= threshold
    cond_met_rolling = cond_met.rolling(time=window_size, center=True).sum() >= window_size
    indices = xr.where(cond_met_rolling, da['time'].dt.dayofyear, np.nan)
    first_occurrence = indices.min(dim='time', skipna=True)
    return first_occurrence

def compute_melt_end(da, threshold, window_size):
    cond_met = da <= threshold
    cond_met_reversed = cond_met.isel(time=slice(None, None, -1))
    cond_met_rolling = cond_met_reversed.rolling(time=window_size, center=True).sum() >= window_size
    indices = xr.where(cond_met_rolling, cond_met_reversed['time'].dt.dayofyear, np.nan)
    last_occurrence = indices.max(dim='time', skipna=True)
    return last_occurrence

def apply_masks(phase_da, yearly_da, threshold):
    annual_min = yearly_da.min(dim='time')
    annual_max = yearly_da.max(dim='time')
    open_ocean_mask = annual_max < threshold
    continent_mask = annual_min > threshold
    return phase_da.where(~open_ocean_mask & ~continent_mask)

def save_phases_to_netcdf(phases, year):
    ds = xr.Dataset(
        phases,
        attrs={"description": f"Sea Ice Phases (threshold: {THRESHOLD}, window: {WINDOW} days) for year {year}"}
    )
    filename = f"seaice_phases_SMMR_{year}.nc"
    ds.to_netcdf(os.path.join(RESULTS_DIR, filename))
    print(f"Phases saved for year {year}")

# Loop through all years in the dataset
years = np.unique(sic_xr.time.dt.year.values)

for year in years:
    print(f"Processing year {year}...")
    yearly_da = sic_xr.sel(time=sic_xr.time.dt.year == year)

    # Compute phases explicitly without month filtering
    freeze_start = compute_phase_metric(yearly_da, THRESHOLD, WINDOW, 'above')
    melt_start = compute_phase_metric(yearly_da, THRESHOLD, WINDOW, 'below')
    melt_end = compute_melt_end(yearly_da, THRESHOLD, WINDOW)

    # Apply masks explicitly
    freeze_start_masked = apply_masks(freeze_start, yearly_da, THRESHOLD)
    melt_start_masked = apply_masks(melt_start, yearly_da, THRESHOLD)
    melt_end_masked = apply_masks(melt_end, yearly_da, THRESHOLD)

    # Save all phases in a single NetCDF file
    phases = {
        'freeze_start': freeze_start_masked,
        'melt_start': melt_start_masked,
        'melt_end': melt_end_masked
    }
    save_phases_to_netcdf(phases, year)
