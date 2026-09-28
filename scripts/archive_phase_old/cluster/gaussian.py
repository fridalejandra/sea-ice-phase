import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from glob import glob
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import BoundaryNorm

# === CONFIG === #
PHASE_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/"
SAVE_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/clusters_fulltimeseries/"
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

# === PREPARE FEATURES === #
def prepare_features(data):
    flat = data.transpose("year", "y", "x").values.reshape(len(data.year), -1).T  # (pixels, years)
    valid = np.all(np.isfinite(flat), axis=1)
    return flat[valid], valid

X_adv, valid_adv = prepare_features(advance)
X_ret, valid_ret = prepare_features(retreat)

# === BIC-BASED CLUSTER SELECTION === #
def gmm_bic(X, feature_name):
    bics = []
    Ks = range(2, 12)
    for k in Ks:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
        gmm.fit(X)
        bics.append(gmm.bic(X))

    plt.figure(figsize=(6, 4))
    plt.plot(Ks, bics, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('BIC')
    plt.title(f'GMM BIC Selection ({feature_name})')
    plt.grid()
    plt.tight_layout()
    outname = f"gmm_bic_{feature_name.lower()}.png"
    plt.savefig(os.path.join(SAVE_DIR, outname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {outname}")

# === CLUSTERING + PLOTTING FUNCTION === #
def cluster_and_plot(X, valid, base_data, feature_name, n_clusters):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    label_map = np.full(base_data.shape[1] * base_data.shape[2], np.nan)
    label_map[valid] = labels
    label_map = label_map.reshape((base_data.shape[1], base_data.shape[2]))

    # Mask out-of-bound values
    label_map = np.where((label_map >= 0) & (label_map < n_clusters), label_map, np.nan)

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    cmap = plt.get_cmap('tab10', n_clusters)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, n_clusters + 0.5, 1), ncolors=n_clusters)

    im = ax.pcolormesh(base_data.x, base_data.y, label_map, cmap=cmap, norm=norm, transform=ccrs.SouthPolarStereo())

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5, pad=0.05)
    cbar.set_label("Cluster ID")
    cbar.set_ticks(np.arange(n_clusters))

    plt.title(f"Sea Ice Phase Clusters ({feature_name} - GMM, Full Time Series)", fontsize=14)
    outname = f"phase_clusters_{feature_name.lower()}_gmm_fulltimeseries.png"
    plt.savefig(os.path.join(SAVE_DIR, outname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {outname}")

# === RUN EVERYTHING === #
print("🔹 Advance: BIC Selection")
gmm_bic(X_adv, feature_name="Advance")

print("🔹 Retreat: BIC Selection")
gmm_bic(X_ret, feature_name="Retreat")

# Set manually after reviewing BIC plots
n_clusters_adv = 6
n_clusters_ret = 8

print("🔹 Clustering Advance...")
cluster_and_plot(X_adv, valid_adv, advance, feature_name="Advance", n_clusters=n_clusters_adv)

print("🔹 Clustering Retreat...")
cluster_and_plot(X_ret, valid_ret, retreat, feature_name="Retreat", n_clusters=n_clusters_ret)

print("✅ GMM clustering complete for full time series.")
