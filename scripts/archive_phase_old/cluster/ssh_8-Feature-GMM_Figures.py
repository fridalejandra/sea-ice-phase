#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8-feature GMM clustering (advance+retreat) with publishable plots.
- Works on projected x/y (South Polar Stereo) OR lat/lon grids.
- Outputs: NetCDF (labels, uncertainty), cluster map, uncertainty map, BIC curve.
- Optional rclone upload if env var RCLONE_DST is set (e.g., "gdrive:sea-ice/figs").
"""

import os, re, glob, subprocess
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# Cartopy (map plotting)
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Sklearn (GMM)
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# ----------------------- EDIT THESE -----------------------
DATA_DIR = "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase"
SAVE_DIR = "/user/geog/falejandraperez/sea-ice-phrase/results/GMM_8feat"  # <- check typo 'phrase' vs 'phase'
GLOB_PAT = os.path.join(DATA_DIR, "*.nc")
YEARS = list(range(1979, 2025))
K_MIN, K_MAX = 2, 10
MIN_YEARS = 25       # require >= this many valid years per pixel
FIG_DPI = 300
RCLONE_DST = os.environ.get("RCLONE_DST", "")  # e.g., "gdrive:sea-ice/GMM"

# Plot extent (geographic) for polar map
MAP_EXTENT = [-180, 180, -90, -50]  # lon_min, lon_max, lat_min, lat_max
# ----------------------------------------------------------


# ==================== helpers ====================

def get_coords_and_transform(ds):
    """
    Return (X2D, Y2D, transform_crs, coord_names) for pcolormesh.

    - If 2D lat/lon exist -> returns lon2D, lat2D, PlateCarree()
    - Else if 1D lat/lon exist -> meshgrid -> PlateCarree()
    - Else if projected x/y exist -> returns X2D, Y2D, SouthPolarStereo()
    """
    # 2D lat/lon
    for lat_name in ("lat", "latitude", "grid_latitude", "nav_lat", "yc"):
        if lat_name in ds.variables and ds[lat_name].ndim == 2:
            for lon_name in ("lon", "longitude", "grid_longitude", "nav_lon", "xc"):
                if lon_name in ds.variables and ds[lon_name].ndim == 2:
                    return (ds[lon_name].values, ds[lat_name].values,
                            ccrs.PlateCarree(), (lon_name, lat_name))

    # 1D lat/lon
    for lat_name in ("lat", "latitude"):
        for lon_name in ("lon", "longitude"):
            if (lat_name in ds.variables and lon_name in ds.variables and
                ds[lat_name].ndim == 1 and ds[lon_name].ndim == 1):
                lon1d = ds[lon_name].values
                lat1d = ds[lat_name].values
                LON2D, LAT2D = np.meshgrid(lon1d, lat1d)
                return (LON2D, LAT2D, ccrs.PlateCarree(), (lon_name, lat_name))

    # projected x/y (South Polar Stereo typical)
    for y_name in ("y", "yc", "Y", "northing"):
        for x_name in ("x", "xc", "X", "easting"):
            if y_name in ds.variables and x_name in ds.variables:
                yv = ds[y_name].values
                xv = ds[x_name].values
                if ds[y_name].ndim == 1 and ds[x_name].ndim == 1:
                    X2D, Y2D = np.meshgrid(xv, yv)
                else:
                    X2D, Y2D = xv, yv
                return (X2D, Y2D, ccrs.SouthPolarStereo(), (x_name, y_name))

    raise ValueError("No usable coordinate pair found (lat/lon or x/y).")


def doy_to_angle(d):
    return (2*np.pi / 365.0) * (d % 365.0)

def circ_mean_cos_sin(theta):  # theta: (time, y, x)
    c = np.cos(theta).mean("time", skipna=True)
    s = np.sin(theta).mean("time", skipna=True)
    return c, s

def resultant_length(cbar, sbar):
    return np.hypot(cbar, sbar)  # [0,1]

def circ_diff_days(ret, adv):
    return (ret - adv) % 365.0

def grid_from_vector(vec, mask_boolean):
    out = np.full(mask_boolean.shape, np.nan, dtype=float)
    out[mask_boolean] = vec
    return out

def rclone_copy(path, dst_root):
    if not dst_root:
        return
    try:
        subprocess.run(["rclone", "copy", path, dst_root, "-P"], check=True)
    except Exception as e:
        print(f"rclone copy failed for {path}: {e}")


# ==================== main ====================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1) discover files
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
        if p is None:
            missing.append(y)

    # 2) stack advance/retreat
    adv_list, ret_list = [], []
    coord = None
    data_crs = None
    coord_names = None

    for y in YEARS:
        path = year_to_path.get(y)
        if path is None:
            print(f"Skipping {y} (no file)")
            continue
        with xr.open_dataset(path) as ds:
            adv_var = f"advance_{y}"
            ret_var = f"retreat_{y}"
            if adv_var not in ds or ret_var not in ds:
                print(f"Warning: {os.path.basename(path)} missing {adv_var} or {ret_var}; skipping")
                continue

            if coord is None:
                X2D, Y2D, DATA_CRS, coord_names = get_coords_and_transform(ds)
                coord = (X2D, Y2D)
                data_crs = DATA_CRS
                print("Using coords:", coord_names,
                      "| shapes:", np.array(X2D).shape, np.array(Y2D).shape,
                      "| transform:", type(data_crs).__name__)

            # Use a mid-year anchor so concat has a proper datetime index
            adv_list.append(ds[adv_var].expand_dims(time=[np.datetime64(f"{y}-07-01")]))
            ret_list.append(ds[ret_var].expand_dims(time=[np.datetime64(f"{y}-07-01")]))

    if not adv_list or not ret_list:
        raise RuntimeError("No advance/retreat stacks were built. Check file/variable names.")

    adv_all = xr.concat(adv_list, dim="time")  # (time,y,x)
    ret_all = xr.concat(ret_list, dim="time")
    print("Advance stacked:", adv_all.dims, adv_all.shape)
    print("Retreat  stacked:", ret_all.dims, ret_all.shape)

    # 3) build 8 features
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
    mask_grid = (valid_mask & finite_mask)

    vals = feat_da.values       # (y,x,8)
    ny, nx, nf = vals.shape
    flat = vals.reshape(ny*nx, nf)
    maskvec = mask_grid.values.reshape(ny*nx)
    X8 = flat[maskvec]

    print("Feature names:", list(feat_da.feat.values))
    print("Valid pixel fraction:", float(mask_grid.mean().values))
    if X8.shape[0] < 1000:
        print(f"Warning: only {X8.shape[0]} valid pixels—GMM may be unstable.")

    # clip extremes (robustness), then standardize
    X8_clip = X8.copy()
    X8_clip[:, 6] = np.clip(X8_clip[:, 6], 30, 330)  # μ_dur
    X8_clip[:, 7] = np.clip(X8_clip[:, 7], 0, 180)   # σ_dur
    scaler = StandardScaler().fit(X8_clip)
    Xz = scaler.transform(X8_clip)

    # 4) GMM + BIC sweep
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

    # 5) save NetCDF
    coord_x_name, coord_y_name = coord_names  # e.g., ("x","y") or ("lon","lat")
    X2D, Y2D = coord

    ds_out = xr.Dataset(
        data_vars=dict(
            cluster=(("y","x"), labels_grid),
            uncert =(("y","x"), uncert_grid),
        ),
        coords=dict(
            y=np.arange(labels_grid.shape[0]),
            x=np.arange(labels_grid.shape[1]),
            **{coord_x_name: (("y","x"), X2D),
               coord_y_name: (("y","x"), Y2D)}
        ),
        attrs=dict(
            description="GMM clusters and uncertainty (1 - max responsibility)",
            features="cosμA, sinμA, cosμR, sinμR, R_A, R_R, μ_dur, σ_dur",
            best_K=n_clusters,
            crs="EPSG:3031 (assumed) for x/y; PlateCarree for lon/lat"
        )
    )
    nc_path = os.path.join(SAVE_DIR, f"gmm8_clusters_K{n_clusters}.nc")
    ds_out.to_netcdf(nc_path)
    print("Saved:", nc_path)
    rclone_copy(nc_path, RCLONE_DST)

    # 6) plots
    def plot_cluster_map(label_grid, coord, data_crs, outpath, n_clusters):
        X2D, Y2D = coord
        fig = plt.figure(figsize=(6,6))
        ax = plt.axes(projection=ccrs.SouthPolarStereo())
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
        ax.coastlines("110m", linewidth=0.5, zorder=2)

        cmap = plt.get_cmap("tab20", n_clusters)
        norm = BoundaryNorm(np.arange(-0.5, n_clusters+0.5, 1), n_clusters)

        im = ax.pcolormesh(X2D, Y2D, label_grid, cmap=cmap, norm=norm,
                           transform=data_crs, zorder=0)
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.6, pad=0.05)
        cbar.set_label("Cluster ID"); cbar.set_ticks(range(n_clusters))
        ax.set_title(f"GMM clusters (K={n_clusters})")
        plt.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight"); plt.close()
        print("Saved:", outpath)
        rclone_copy(outpath, RCLONE_DST)

    def plot_uncertainty_map(u_grid, coord, data_crs, outpath):
        X2D, Y2D = coord
        fig = plt.figure(figsize=(6,6))
        ax = plt.axes(projection=ccrs.SouthPolarStereo())
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
        ax.coastlines("110m", linewidth=0.5, zorder=2)

        im = ax.pcolormesh(X2D, Y2D, u_grid, cmap="viridis", vmin=0, vmax=0.5,
                           transform=data_crs, zorder=0)
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.6, pad=0.05)
        cbar.set_label("Uncertainty (1 − max γ)")
        ax.set_title("Cluster uncertainty")
        plt.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight"); plt.close()
        print("Saved:", outpath)
        rclone_copy(outpath, RCLONE_DST)

    def plot_bic_curve(Ks, BICs, chosen_k, outpath):
        plt.figure(figsize=(4.8,3.2))
        plt.plot(Ks, BICs, marker="o")
        plt.axvline(chosen_k, ls="--", color="k", lw=1)
        plt.xlabel("Number of components (K)")
        plt.ylabel("BIC")
        plt.title("Model selection")
        plt.tight_layout()
        plt.savefig(outpath, dpi=FIG_DPI)
        plt.close()
        print("Saved:", outpath)
        rclone_copy(outpath, RCLONE_DST)

    # draw
    clusters_png = os.path.join(SAVE_DIR, f"gmm8_clusters_K{n_clusters}.png")
    uncert_png = os.path.join(SAVE_DIR, f"gmm8_uncertainty_K{n_clusters}.png")
    bic_png = os.path.join(SAVE_DIR, "bic_curve.png")

    plot_cluster_map(labels_grid, coord, data_crs, clusters_png, n_clusters)
    plot_uncertainty_map(uncert_grid, coord, data_crs, uncert_png)
    plot_bic_curve(Ks, BICs, n_clusters, bic_png)

    print("All done.")

if __name__ == "__main__":
    main()
