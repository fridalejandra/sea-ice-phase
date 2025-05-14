import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from glob import glob

# === CONFIG === #
INPUT_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/"
SAVE_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/trends/"
os.makedirs(SAVE_DIR, exist_ok=True)

YEAR_START = 1979
YEAR_END = 2024
YEAR_SPLIT = 2016

periods = {
    "full": (1979, 2024),
    "pre2016": (1979, 2015),
    "post2016": (2016, 2024),
}

# === LOAD STACKS === #
advance_stack, retreat_stack = [], []
years = []

files = sorted(glob(f"{INPUT_DIR}/seaice_phases_SMMR_*.nc"))
for f in files:
    year = int(os.path.basename(f).split("_")[-1].split(".")[0])
    if YEAR_START <= year <= YEAR_END:
        ds = xr.open_dataset(f)
        adv_var = f"advance_{year}"
        ret_var = f"retreat_{year}"
        if adv_var in ds and ret_var in ds:
            advance_stack.append(ds[adv_var])
            retreat_stack.append(ds[ret_var])
            years.append(year)
        else:
            print(f"⚠️ Missing {adv_var} or {ret_var} in {f}")

advance = xr.concat(advance_stack, dim="year")
retreat = xr.concat(retreat_stack, dim="year")
advance["year"] = years
retreat["year"] = years

# === TREND FUNCTION === #
def compute_trend(da, start, end):
    da_sel = da.sel(year=slice(start, end))
    x = da_sel.year.values
    y = da_sel.values

    mask_valid = np.isfinite(y).sum(axis=0) > int(0.8 * len(x))  # At least 80% valid
    slope = np.full(y.shape[1:], np.nan)

    for j in range(y.shape[1]):
        for i in range(y.shape[2]):
            if mask_valid[j, i]:
                coeffs = np.polyfit(x, y[:, j, i], 1)
                slope[j, i] = coeffs[0]  # slope = days per year

    return slope

# === PLOTTING FUNCTION === #
def plot_trend(trend, title, save_path, vmin=-2, vmax=2):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})

    im = ax.pcolormesh(
        advance.x, advance.y, trend,
        transform=ccrs.SouthPolarStereo(),
        cmap="coolwarm", vmin=vmin, vmax=vmax, shading="auto"
    )
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, zorder=100, facecolor="gray")
    ax.coastlines(linewidth=0.4)

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_label("Trend (days/year)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

# === RUN FOR EACH PERIOD === #
for label, (start, end) in periods.items():
    print(f"🔹 Processing {label}: {start}-{end}")
    adv_trend = compute_trend(advance, start, end)
    ret_trend = compute_trend(retreat, start, end)

    plot_trend(adv_trend, f"Advance Trend ({start}-{end})", f"{SAVE_DIR}/advance_trend_{label}.png")
    plot_trend(ret_trend, f"Retreat Trend ({start}-{end})", f"{SAVE_DIR}/retreat_trend_{label}.png")

print("✅ All trend maps done!")
