import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from glob import glob
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

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

# === PREPARE COMBINED FEATURES === #
def prepare_combined_features(adv, ret):
    flat_adv = adv.transpose("year", "y", "x").values.reshape(len(adv.year), -1).T
    flat_ret = ret.transpose("year", "y", "x").values.reshape(len(ret.year), -1).T
    valid = np.all(np.isfinite(flat_adv), axis=1) & np.all(np.isfinite(flat_ret), axis=1)
    X_combined = np.concatenate([flat_adv[valid], flat_ret[valid]], axis=1)
    return X_combined, valid

X_comb, valid_comb = prepare_combined_features(advance, retreat)

# === BIC PLOT FOR COMBINED === #
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

# === CLUSTER + PLOT === #
def cluster_and_plot(X, valid, base_data, feature_name, n_clusters=6):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    label_map = np.full(base_data.shape[1] * base_data.shape[2], np.nan)
    label_map[valid] = labels
    label_map = label_map.reshape((base_data.shape[1], base_data.shape[2]))

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(base_data.x, base_data.y, label_map, cmap="tab10", transform=ccrs.SouthPolarStereo())
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5, pad=0.05)
    cbar.set_label("Cluster ID")

    plt.title(f"Sea Ice Phase Clusters (Advance + Retreat - GMM, Full Time Series)", fontsize=14)
    outname = f"phase_clusters_combined_gmm_fulltimeseries.png"
    plt.savefig(os.path.join(SAVE_DIR, outname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {outname}")

# === RUN EVERYTHING === #
print("🔹 Combined Advance + Retreat: BIC Selection")
gmm_bic(X_comb, feature_name="Advance+Retreat")

# Set manually after BIC review
n_clusters_combined = 6

print("🔹 Clustering Combined Advance + Retreat...")
cluster_and_plot(X_comb, valid_comb, advance, feature_name="Advance+Retreat", n_clusters=n_clusters_combined)

print("✅ GMM clustering complete for combined phase features.")

