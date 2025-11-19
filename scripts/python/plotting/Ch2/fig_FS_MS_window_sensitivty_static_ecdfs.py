from pathlib import Path
import xarray as xr
import numpy as np

import sys
from pathlib import Path

# This script lives in .../plotting/Ch2
# ch2_fig_utils.py lives one level up: .../plotting
HERE = Path(__file__).resolve().parent
PLOTTING_ROOT = HERE.parent  # directory that actually contains ch2_fig_utils.py

if str(PLOTTING_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTTING_ROOT))

from ch2_fig_utils import (
    set_mpl_defaults,
    load_static_phase_year,
    flatten_field,
    plot_window_sensitivity_ecdf,
    get_fig_path,
    save_and_upload,
    PROJECT_ROOT_CLUSTER,
)



set_mpl_defaults()

# Example: build climatology for SMMR period for 3,5,7-day windows
years = np.arange(1979, 2024)  # whatever range you’re using

def climatology_for_window(phase: str, window_days: int) -> xr.DataArray:
    das = []
    for y in years:
        da = load_static_phase_year(phase=phase, year=y, window_days=window_days)
        das.append(da.expand_dims(year=[y]))
    da_all = xr.concat(das, dim="year")
    return da_all.mean("year", skipna=True)

adv_3 = climatology_for_window("advance", 3)
adv_5 = climatology_for_window("advance", 5)
adv_7 = climatology_for_window("advance", 7)

ret_3 = climatology_for_window("retreat", 3)
ret_5 = climatology_for_window("retreat", 5)
ret_7 = climatology_for_window("retreat", 7)

# Build a simple valid mask: finite advance & retreat across all windows
mask = np.isfinite(adv_3.values) & np.isfinite(adv_5.values) & np.isfinite(adv_7.values)
mask &= np.isfinite(ret_3.values) & np.isfinite(ret_5.values) & np.isfinite(ret_7.values)

fig, axes = plot_window_sensitivity_ecdf(
    adv_3=adv_3,
    adv_5=adv_5,
    adv_7=adv_7,
    ret_3=ret_3,
    ret_5=ret_5,
    ret_7=ret_7,
    mask=mask,
    phase_label_advance="FS",
    phase_label_retreat="MS",
)


out_path = get_fig_path(
    PROJECT_ROOT_CLUSTER,
    subfolder="sensitivity/window",
    fig_name="FigXX_window_sensitivity_ecdf.png",
)
save_and_upload(
    fig,
    out_path,
    remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
    remote_subdir="sensitivity/window",
)
