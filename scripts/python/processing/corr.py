import os
import numpy as np
import xarray as xr
import scipy.stats as stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from glob import glob
from sklearn.linear_model import LinearRegression
from scipy.signal import detrend

# === CONFIG === #
PHASE_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/"
SAVE_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/retreat_vs_advance/"
os.makedirs(SAVE_DIR, exist_ok=True)

# === LOAD STACKS === #
advance_stack, retreat_stack = [], []
years = []

files = sorted(glob(f"{PHASE_DIR}/seaice_phases_SMMR_*.nc"))
for f in files:
    year = int(f.split("_")[-1].split(".")[0])
    ds = xr.open_dataset(f)
    adv_var = f"advance_{year}"
    ret_var = f"retreat_{year}"
    if adv_var in ds and ret_var in ds:
        advance_stack.append(ds[adv_var])
        retreat_stack.append(ds[ret_var])
        years.append(year)

advance = xr.concat(advance_stack, dim="year")
retreat = xr.concat(retreat_stack, dim="year")
advance["year"] = years
retreat["year"] = years

# === PREPARE ARRAYS === #
adv_vals = advance.transpose("year", "y", "x").values  # (year, y, x)
ret_vals = retreat.transpose("year", "y", "x").values  # (year, y, x)

ny, nx = advance.shape[1], advance.shape[2]

# Initialize empty output arrays
r_map = np.full((ny, nx), np.nan)
r2_map = np.full((ny, nx), np.nan)
resid_map = np.full((ny, nx), np.nan)

# === PIXEL-BY-PIXEL REGRESSION === #
for j in range(ny):
    for i in range(nx):
        y_adv = adv_vals[:, j, i]
        x_ret = ret_vals[:, j, i]

        # Only work if both series are fully valid
        if np.all(np.isfinite(y_adv)) and np.all(np.isfinite(x_ret)):
            # (Optional) Detrend both series
            x_ret_detrend = detrend(x_ret)
            y_adv_detrend = detrend(y_adv)

            # Reshape for sklearn
            x = x_ret_detrend.reshape(-1, 1)
            y = y_adv_detrend

            # Regression
            model = LinearRegression()
            model.fit(x, y)
            y_pred = model.predict(x)

            # Calculate metrics
            r, _ = stats.pearsonr(x_ret_detrend, y_adv_detrend)
            r2 = model.score(x, y)
            resid_std = np.std(y - y_pred)

            # Save results
            r_map[j, i] = r
            r2_map[j, i] = r2
            resid_map[j, i] = resid_std

print("✅ Regression processing complete.")

# === SAVE OUTPUTS === #
np.save(os.path.join(SAVE_DIR, "r_map.npy"), r_map)
np.save(os.path.join(SAVE_DIR, "r2_map.npy"), r2_map)
np.save(os.path.join(SAVE_DIR, "resid_map.npy"), resid_map)

# === LOAD OUTPUTS === #
r_map = np.load(os.path.join(SAVE_DIR, "r_map.npy"))
r2_map = np.load(os.path.join(SAVE_DIR, "r2_map.npy"))
resid_map = np.load(os.path.join(SAVE_DIR, "resid_map.npy"))


# === PLOT FUNCTION === #
def plot_map(data, title, cmap, vmin, vmax, filename):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(
        advance.x, advance.y, data,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap, vmin=vmin, vmax=vmax
    )

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5, pad=0.05)
    cbar.set_label(title)

    plt.title(title, fontsize=14)
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()


# === GENERATE PLOTS === #
plot_map(r_map, "Correlation (r) between Retreat and Advance", cmap="coolwarm", vmin=-1, vmax=1,
         filename="correlation_r.png")
plot_map(r2_map, "Predictive Skill (R²) Retreat → Advance", cmap="viridis", vmin=0, vmax=1,
         filename="predictive_skill_r2.png")
plot_map(resid_map, "Residual Standard Deviation", cmap="magma", vmin=0, vmax=np.nanmax(resid_map),
         filename="residual_std.png")

print("✅ Plotting complete.")
