import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
from glob import glob
from matplotlib.colors import TwoSlopeNorm
from matplotlib.path import Path

# === CONFIG === #
YEAR_TARGET = 2022  # 🔁 Change this year as needed
YEAR_START, YEAR_END = 1979, 2024

PHASE_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/"
OUT_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/anomaly/"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUT_DIR, f"retreat_anomaly_{YEAR_TARGET}.png")
TARGET_FILE = os.path.join(PHASE_DIR, f"seaice_phases_SMMR_{YEAR_TARGET}.nc")

# === Load retreat climatology from all years === #
retreat_stack = []
for path in sorted(glob(f"{PHASE_DIR}/seaice_phases_SMMR_*.nc")):
    year = int(os.path.basename(path).split("_")[-1].split(".")[0])
    if not (YEAR_START <= year <= YEAR_END):
        continue
    ds = xr.open_dataset(path)
    var = f"retreat_{year}"
    if var in ds:
        da = ds[var]
        da_wrapped = da.where(da >= 100, da + 365)
        retreat_stack.append(da_wrapped)
    else:
        print(f"⚠️ Skipping {path}: Missing {var}")

if not retreat_stack:
    raise RuntimeError("❌ No retreat data found to build climatology.")

climatology = xr.concat(retreat_stack, dim="year").mean(dim="year", skipna=True)

# === Load target year and wrap === #
ds_target = xr.open_dataset(TARGET_FILE)
da_target = ds_target[f"retreat_{YEAR_TARGET}"]
da_target_wrapped = da_target.where(da_target >= 100, da_target + 365)

# === Anomaly calculation === #
anomaly = da_target_wrapped - climatology

# === Plotting === #
fig = plt.figure(figsize=(6.5, 6.5))
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
circle_verts = np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5
circle_path = Path(circle_verts)
ax.set_boundary(circle_path, transform=ax.transAxes)

# Color settings
norm = TwoSlopeNorm(vcenter=0, vmin=-40, vmax=40)
cmap = plt.cm.coolwarm.copy()
cmap.set_bad("white")

# Map
mesh = ax.pcolormesh(anomaly.x, anomaly.y, anomaly,
                     transform=ccrs.SouthPolarStereo(), cmap=cmap, norm=norm, shading="auto")
ax.add_feature(cfeature.LAND, facecolor='gray', zorder=100)
ax.coastlines(linewidth=0.4)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f"Sea Ice Retreat Anomaly: {YEAR_TARGET}", fontsize=13, fontweight='bold')

# Colorbar
cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.07)
cbar.set_label("Anomaly (days)", fontsize=10)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved anomaly map for {YEAR_TARGET}: {OUTPUT_PATH}")
