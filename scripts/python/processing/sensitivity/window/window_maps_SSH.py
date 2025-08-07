import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# === CONFIGURATION === #
DATASET = "SMMR"     # or "AMSRE"
PHASE = "retreat"    # or "advance"
YEARS = range(1979, 2024) if DATASET == "SMMR" else range(2012, 2024)

# === PATHS === #
BASEDIR = f"/user/geog/falejandraperez/sea-ice-phase/results/sensitivity/{DATASET}_phase"
DIFFDIR = f"{BASEDIR}/{DATASET}_window_comparison/{PHASE}"
FIGDIR = "/mnt/gdrive/sea-ice-figures/window_facet_maps"
os.makedirs(FIGDIR, exist_ok=True)

# === LOAD + STACK === #
diff3_list = []
diff7_list = []

for year in YEARS:
    try:
        f3 = os.path.join(DIFFDIR, f"diff_{PHASE}_3minus5_{year}.nc")
        f7 = os.path.join(DIFFDIR, f"diff_{PHASE}_7minus5_{year}.nc")

        ds3 = xr.open_dataset(f3)
        ds7 = xr.open_dataset(f7)

        diff3 = ds3[f"diff_{PHASE}_3minus5"]
        diff7 = ds7[f"diff_{PHASE}_7minus5"]

        diff3.load()
        diff7.load()

        diff3_list.append(diff3)
        diff7_list.append(diff7)

    except FileNotFoundError:
        print(f"Skipping {year} — missing file.")
    except Exception as e:
        print(f"Error in {year}: {e}")

# === COMPUTE MEAN DIFFERENCE === #
mean_diff3 = xr.concat(diff3_list, dim="year").mean(dim="year")
mean_diff7 = xr.concat(diff7_list, dim="year").mean(dim="year")

# === PLOT === #
fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})
vmin, vmax = -10, 10

for ax, data, title, cmap in zip(
    axes,
    [mean_diff3, mean_diff7],
    ["3-day minus 5-day", "7-day minus 5-day"],
    ["Blues", "Reds"]
):
    im = ax.pcolormesh(
        data.x, data.y, data,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
    )
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, zorder=100, facecolor="gray")
    ax.coastlines(linewidth=0.4)
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cbar.set_label("Mean Timing Difference (days)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold")

fig.suptitle(f"{PHASE.capitalize()} Window Sensitivity — {DATASET}", fontsize=14)
plt.tight_layout()
figname = f"facet_window_diff_{PHASE}_{DATASET}.png"
plt.savefig(os.path.join(FIGDIR, figname), dpi=400)
plt.close()

