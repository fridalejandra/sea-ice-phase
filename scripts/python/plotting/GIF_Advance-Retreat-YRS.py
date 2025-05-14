import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
from glob import glob

# === CONFIG ===
PHASE_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/"
OUT_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/yearly_phase_maps/"
os.makedirs(OUT_DIR, exist_ok=True)

# === Plotting function ===
def plot_map(data, year, phase_type, cmap, vmin, vmax, label):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor='gray', zorder=100)
    ax.spines['geo'].set_visible(False)

    mesh = ax.pcolormesh(data.x, data.y, data, transform=ccrs.SouthPolarStereo(),
                         cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    ax.set_title(f"{phase_type.capitalize()} {year}", fontsize=14, fontweight='bold')

    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    filename = os.path.join(OUT_DIR, f"{phase_type}_{year}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

# === Loop through all per-year files ===
files = sorted(glob(os.path.join(PHASE_DIR, "seaice_phases_SMMR_*.nc")))
for fpath in files:
    year_str = os.path.basename(fpath).split("_")[-1].split(".")[0]
    if not year_str.isdigit():
        continue
    year = int(year_str)

    ds = xr.open_dataset(fpath)
    adv_var = f"advance_{year}"
    ret_var = f"retreat_{year}"

    if adv_var not in ds or ret_var not in ds:
        print(f"⚠️ Missing data for {year}")
        continue

    # === Load and wrap ===
    advance = ds[adv_var].where((ds[adv_var] >= 32) & (ds[adv_var] <= 274))
    retreat = ds[ret_var]
    retreat_wrapped = retreat.where(retreat >= 100, retreat + 365)

    duration = retreat_wrapped - advance
    duration = duration.where(duration >= 0)

    # === Plot three maps ===
    plot_map(advance, year, "advance", cmap=plt.cm.viridis, vmin=32, vmax=274, label="Advance (Day of Year)")
    plot_map(retreat_wrapped, year, "retreat", cmap=plt.cm.viridis, vmin=274, vmax=424, label="Retreat (Day of Year)")
    plot_map(duration, year, "duration", cmap=plt.cm.viridis, vmin=0, vmax=300, label="Ice Season Duration (days)")

    print(f"✅ Saved maps for {year}")
