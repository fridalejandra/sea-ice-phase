import os, re, glob, subprocess
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Optional: cartopy for polar map
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm

# ---------------- EDIT THESE ----------------
DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase"
SAVE_DIR   = "/user/geog/falejandraperez/sea-ice-phase/results/GMM_8feat"
YEARS      = list(range(1979, 2025))             # adjust to what you actually have
GLOB_PAT   = os.path.join(DATA_DIR, "*.nc")
K_MIN, K_MAX = 2, 10                              # BIC sweep range
MIN_YEARS  = 25                                   # require >= this many valid years per pixel
RCLONE_DST = os.environ.get("RCLONE_DST", "")     # e.g. "gdrive:sea-ice/figs" (optional)
# --------------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- helpers ----------
def find_latlon(ds):
    for la in ("lat", "latitude", "nav_lat", "yc"):
        if la in ds.variables:
            lat = ds[la]
            break
    else:
        raise ValueError("No latitude variable found")

    for lo in ("lon", "longitude", "nav_lon", "xc"):
        if lo in ds.variables:
            lon = ds[lo]
            break
    else:
        raise ValueError("No longitude variable found")
    return lat, lon

def doy_to_angle(d):
    return (2*np.pi / 365.0) * (d % 365.0)

def circ_mean_cos_sin(theta):  # theta: (time, y, x)
    c = np.cos(theta).mean("time", skipna=True)
    s = np.sin(theta).mean("time", skipna=True)
    return c, s

def resultant_length(cbar, sbar):
    return np.hypot(cbar, sbar)  # in [0,1]

def circ_diff_days(ret, adv):
    return (ret - adv) % 365.0

def grid_from_vector(vec, mask_boolean):
    out = np.full(mask_boolean.shape, np.nan, dtype=float)
    out[mask_boolean] = vec
    return out

def rclone_copy(src_path, dst_root):
    if not dst_root:
        return
    try:
        subprocess.run(["rclone", "copy", src_path, dst_root, "-P"], check=True)
    except Exception as e:
        print(f"rclone copy failed for {src_path}: {e}")

# ---------- 1) discover files ----------
all_files = sorted(glob.glob(GLOB_PAT))
year_re = re.compile(r"(19|20)\d{2}")
year_to_path = {}
for f in all_files:
    m = year_re.search(os.path.basename(f))
    if not m:
        continue
    y = int(m.group(0))
    if y in YEARS and y not in year_to_path:
        year_to_path[y] = f

print("Discovered files:")
missing = []
for y in YEARS:
    p = year_to_path.get(y)
    print(f"  {y}: {p if p else 'MISSING'}")
    if p is None: missing.append(y)

# ---------- 2) stack advance/retreat ----------
adv_list, ret_list = [], []
lat2d = lon2d = None
for y in YEARS:
    path = year_to_path.get(y)
    if path is None:
        print(f"Skipping {y} (no file)")
        continue
    with xr.open_dataset(path) as ds:
        # typical names: advance_YYYY, retreat_YYYY
        adv_var = f"advance_{y}"
        ret_var = f"retreat_{y}"
        if adv_var not in ds or ret_var not in ds:
            print(f"Warning: {os.path.basename(path)} missing {adv_var} or {ret_var}; skipping")
            continue
        if lat2d is None:
            lat2d, lon2d = find_latlon(ds)
        adv_list.append(ds[adv_var].expand_dims(time=[np.datetime64(f"{y}-07-01")]))
        ret_list.append(ds[ret_var].expand_dims(time=[np.datetime64(f"{y}-07-01")]))

if not adv_list or not ret_list:
    raise RuntimeError("No advance/retreat stacks were built. Check file naming & variable names.")

adv_all = xr.concat(adv_list, dim="time")  # (time,y,x)
ret_all = xr.concat(ret_list, dim="time")

print("Advance stacked:", adv_all.dims, adv_all.shape)
print("Retreat  stacked:", ret_all.dims, ret_all.shape)

# ---------- 3) build 8 features ----------
Aθ = doy_to_angle(adv_all)
Rθ = doy_to_angle(ret_all)
cosA_bar, sinA_bar = circ_mean_cos_sin(Aθ)
cosR_bar, sinR_bar = circ_mean_cos_sin(Rθ)
RA = resultant_length(cosA_bar, sinA_bar)
RR = resultant_length(cosR_bar, sinR_bar)
dur_yearly = circ_diff_days(ret_all, adv_all)
mu_dur = dur_yearly.mean("time", skipna=True)
sd_dur = dur_yearly.std("time",  skipna=True)

# validity mask
valid_A = adv_all.notnull().sum("time") >= MIN_YEARS
valid_R = ret_all.notnull().sum("time") >= MIN_YEARS
valid_D = dur_yearly.notnull().sum("time") >= MIN_YEARS
valid_mask = (valid_A & valid_R & valid_D)

feat_da = xr.concat(
    [cosA_bar, sinA_bar, cosR_bar, sinR_bar, RA, RR, mu_dur, sd_dur],
    dim="feat"
).transpose(..., "feat")
feat_da = feat_da.assign_coords(feat=["cosμA","sinμA","cosμR","sinμR","R_A","R_R","μ_dur","σ_dur"])

finite_mask = np.isfinite(feat_da).all("feat")
mask_grid   = (valid_mask & finite_mask)

vals    = feat_da.values       # (y,x,8)
ny, nx, nf = vals.shape
flat    = vals.reshape(ny*nx, nf)
maskvec = mask_grid.values.reshape(ny*nx)
X8      = flat[maskvec]

# clip extremes (robustness)
X8_clip = X8.copy()
X8_clip[:, 6] = np.clip(X8_clip[:, 6], 30, 330)  # μ_dur
X8_clip[:, 7] = np.clip(X8_clip[:, 7], 0, 180)   # σ_dur

# standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X8_clip)
Xz = scaler.transform(X8_clip)

# ---------- 4) GMM + BIC ----------
from sklearn.mixture import GaussianMixture
Ks, BICs = [], []
best_gmm, best_bic = None, np.inf

for K in range(K_MIN, K_MAX+1):
    gmm = GaussianMixture(
        n_components=K, covariance_type="full",
        n_init=5, random_state=42
    ).fit(Xz)
    bic = gmm.bic(Xz)
    Ks.append(K); BICs.append(bic)
    if bic < best_bic:
        best_bic, best_gmm = bic, gmm

labels = best_gmm.predict(Xz)
resp   = best_gmm.predict_proba(Xz)
uncert = 1.0 - resp.max(axis=1)
n_clusters = best_gmm.n_components
print(f"Best K by BIC: {n_clusters}")

# map back to grid
labels_grid = grid_from_vector(labels, mask_grid.values)
uncert_grid = grid_from_vector(uncert, mask_grid.values)

# ---------- 5) save NetCDF ----------
ds_out = xr.Dataset(
    data_vars=dict(
        cluster=(("y","x"), labels_grid),
        uncert=(("y","x"), uncert_grid),
    ),
    coords=dict(
        y=np.arange(ny), x=np.arange(nx),
        lat=(("y","x"), lat2d.values), lon=(("y","x"), lon2d.values),
    ),
    attrs=dict(
        description="GMM clusters and uncertainty (1 - max responsibility)",
        features="cosμA, sinμA, cosμR, sinμR, R_A, R_R, μ_dur, σ_dur",
        best_K=n_clusters
    )
)
nc_path = os.path.join(SAVE_DIR, f"gmm8_clusters_K{n_clusters}.nc")
ds_out.to_netcdf(nc_path)
print("Saved:", nc_path)

# ---------- 6) plots ----------
def plot_cluster_map(label_grid, lat, lon, outname):
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.coastlines("110m", linewidth=0.5, zorder=2)

    cmap = plt.get_cmap("tab20", n_clusters)
    norm = BoundaryNorm(np.arange(-0.5, n_clusters+0.5, 1), n_clusters)

    im = ax.pcolormesh(lon, lat, label_grid, cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), zorder=0)
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.6, pad=0.05)
    cbar.set_label("Cluster ID"); cbar.set_ticks(range(n_clusters))
    ax.set_title(f"GMM clusters (K={n_clusters})")
    out = os.path.join(SAVE_DIR, outname)
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print("Saved:", out); rclone_copy(out, RCLONE_DST)

def plot_uncertainty_map(u_grid, lat, lon, outname):
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.coastlines("110m", linewidth=0.5, zorder=2)

    im = ax.pcolormesh(lon, lat, u_grid, cmap="viridis", vmin=0, vmax=0.5,
                       transform=ccrs.PlateCarree(), zorder=0)
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.6, pad=0.05)
    cbar.set_label("Uncertainty (1 − max γ)")
    ax.set_title("Cluster uncertainty")
    out = os.path.join(SAVE_DIR, outname)
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print("Saved:", out); rclone_copy(out, RCLONE_DST)

plot_cluster_map(labels_grid, lat2d, lon2d, f"gmm8_clusters_K{n_clusters}.png")
plot_uncertainty_map(uncert_grid, lat2d, lon2d, f"gmm8_uncertainty_K{n_clusters}.png")

# ---------- 7) BIC curve (quick) ----------
plt.figure(figsize=(4.5,3.2))
plt.plot(Ks, BICs, marker="o")
plt.axvline(n_clusters, ls="--", color="k", lw=1)
plt.xlabel("K"); plt.ylabel("BIC")
plt.title("Model selection")
bic_path = os.path.join(SAVE_DIR, "bic_curve.png")
plt.savefig(bic_path, dpi=300, bbox_inches="tight"); plt.close()
print("Saved:", bic_path); rclone_copy(bic_path, RCLONE_DST)
