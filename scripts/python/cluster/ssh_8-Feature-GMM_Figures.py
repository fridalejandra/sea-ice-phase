# --- Imports ---
import os      # operating system tools: paths, directories
import re      # regular expressions for pattern matching (extract year from filename)
import glob    # file globbing: matches wildcards like "*.nc"

import xarray as xr   # to open NetCDF datasets
import numpy as np    # numerical array library (not yet used in this cell)


# ---- EDIT THIS: folder containing SMMR NetCDFs ----
# On your local machine this points to your repo.
# On the cluster, you'll want to change this path (or better: set via an environment variable).
DATA_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/"

# File pattern to find all NetCDFs in that folder
GLOB_PAT = os.path.join(DATA_DIR, "*.nc")
# ----------------------------------------------


# Define the mission years you expect to find
# Here: 1979 through 2024 (Python range is end-exclusive, so 2025 means up to 2024)
YEARS = list(range(1979, 2025))


# Find all files matching the pattern (alphabetically sorted)
all_files = sorted(glob.glob(GLOB_PAT))


# --- Build a mapping from year -> file path ---
# Regex explanation:
#   r"(19|20)\d{2}"
#   - (19|20): matches either '19' or '20' (so years beginning with 19xx or 20xx)
#   - \d{2}: exactly two digits (so together, 1900–2099)
# Example: "advance_1985.nc" → match = "1985"
year_re = re.compile(r"(19|20)\d{2}")

year_to_path = {}
for f in all_files:
    # Extract just the filename, not the full path
    base = os.path.basename(f)

    # Search for the first 4-digit year pattern
    m = year_re.search(base)
    if not m:
        # If no year-like string is found, skip this file
        continue

    y = int(m.group(0))  # Convert the regex match (string) into an integer year

    # Only keep if the year is in our expected list and not already mapped
    if y in YEARS and y not in year_to_path:
        # Note: if multiple files exist for a given year, the first one wins.
        # Change this logic if you want the last one to override instead.
        year_to_path[y] = f


# --- Diagnostics ---
print("Discovered files for years:")
for y in YEARS:
    # Print either the path found or 'MISSING'
    print(f"  {y}: {year_to_path.get(y, 'MISSING')}")


# --- Peek inside the first available file ---
def list_phase_vars(nc_path):
    """
    Open a NetCDF file and return the list of variable names.
    Uses a context manager (`with`) so the file closes immediately after.
    """
    with xr.open_dataset(nc_path) as ds:
        return list(ds.data_vars)


# Find the first year in our range that we successfully mapped
first_path = next((year_to_path[y] for y in YEARS if y in year_to_path), None)

if first_path:
    print("\nExample variables in:", os.path.basename(first_path))
    print(list_phase_vars(first_path))
else:
    print("\nNo matching files found. Check DATA_DIR / pattern / YEARS.")

# Cell 2 — Load advance/retreat across 1979–2023 into ('time','y','x')

YEARS = list(range(1979, 2024))  # inclusive 1979..2023

adv_list = []  # will hold per-year DataArrays for advance timing
ret_list = []  # will hold per-year DataArrays for retreat timing

for y in YEARS:
    path = year_to_path.get(y)   # from Cell 1: map year -> file path
    if path is None:
        print(f"Skipping {y} (no file found)")
        continue

    # Open once per year; context manager closes file promptly (good for cluster I/O)
    with xr.open_dataset(path) as ds:
        adv_var = f"advance_{y}"  # expected variable name pattern
        ret_var = f"retreat_{y}"

        # Defensive check: make sure both variables are present
        if adv_var not in ds or ret_var not in ds:
            print(f"Warning: {os.path.basename(path)} missing {adv_var} or {ret_var}")
            continue

        # Attach a 'time' coordinate equal to the integer year and ensure consistent (y,x) dims
        # expand_dims creates a new 'time' axis of length 1 with value [y]
        adv_list.append(ds[adv_var].expand_dims(time=[y]))
        ret_list.append(ds[ret_var].expand_dims(time=[y]))

# Concatenate all years along 'time' into a single DataArray each
adv_all = xr.concat(adv_list, dim="time")
ret_all = xr.concat(ret_list, dim="time")

print("Advance stacked:", adv_all.dims, adv_all.shape)
print("Retreat stacked:", ret_all.dims, ret_all.shape)
print("Years loaded:", adv_all.time.values)


# Cell 3 — Build the 8-feature matrix (no plotting, no modeling)

import numpy as np
import xarray as xr

# --- Preconditions from Cell 2 ---
# adv_all: (time, y, x) DataArray with advance DOY for each year
# ret_all: (time, y, x) DataArray with retreat DOY for each year
assert 'adv_all' in globals() and 'ret_all' in globals(), "Run Cell 2 first."

# --- 0) Basic sanity (prevents silent misalignment) ---
assert adv_all.dims == ret_all.dims == ("time", "y", "x"), "Expect dims ('time','y','x')."
assert adv_all.sizes["y"] == ret_all.sizes["y"] and adv_all.sizes["x"] == ret_all.sizes["x"], "Grid mismatch."

# --- 1) Validity mask: require enough years on BOTH series (tune MIN_YEARS as needed) ---
MIN_YEARS = 25
n_adv = xr.apply_ufunc(np.isfinite, adv_all).sum("time")   # count of non-NaN per pixel
n_ret = xr.apply_ufunc(np.isfinite, ret_all).sum("time")
mask_grid = (n_adv >= MIN_YEARS) & (n_ret >= MIN_YEARS)    # (y, x) boolean

# --- 2) Purist leap-year handling for all angle/duration computations ---
def _is_leap_year(years_da):
    y = years_da.astype(int)
    return ((y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0)))

year_len = xr.where(_is_leap_year(adv_all["time"]), 366.0, 365.0)   # (time,)

# --- 3) Circular means for advance/retreat (encode timing correctly on the circle) ---
TAU = 2.0 * np.pi
theta_adv = TAU * ((adv_all % year_len) / year_len)     # (time, y, x) angles in radians
theta_ret = TAU * ((ret_all % year_len) / year_len)

# mean unit-vector components (dimensionless, wrap-safe)
cbar_adv = np.cos(theta_adv).mean("time", skipna=True)   # (y, x)
sbar_adv = np.sin(theta_adv).mean("time", skipna=True)
cbar_ret = np.cos(theta_ret).mean("time", skipna=True)
sbar_ret = np.sin(theta_ret).mean("time", skipna=True)

# resultant lengths (concentration ∈ [0,1]): how stable the timing is across years
R_adv = np.hypot(cbar_adv, sbar_adv)                    # (y, x)
R_ret = np.hypot(cbar_ret, sbar_ret)

# --- 4) Season duration (days) per year, then interannual mean & std ---
# Definition used here: "ice-season length" = (retreat - advance) wrapped by that year's length
duration = (ret_all - adv_all) % year_len               # (time, y, x) in days
dur_mean_days = duration.mean("time", skipna=True)       # (y, x)
dur_std_days  = duration.std("time",  skipna=True)       # (y, x)

# Optional sanity screen (do NOT delete silently; keep it explicit if you choose to apply)
# plausible = (dur_mean_days >= 20) & (dur_mean_days <= 360)
# mask_grid = mask_grid & plausible

# --- 5) Assemble features and flatten to (N, 8) using the mask ---
feat_names = [
    "adv_cos", "adv_sin",
    "ret_cos", "ret_sin",
    "R_adv", "R_ret",
    "dur_mean_days", "dur_std_days",
]

# Stack into ('feat','y','x') for clean reshape → then flatten valid pixels only
feat_da = xr.concat(
    [cbar_adv, sbar_adv, cbar_ret, sbar_ret, R_adv, R_ret, dur_mean_days, dur_std_days],
    dim=xr.IndexVariable("feat", feat_names)
)  # dims: ('feat','y','x')

F, Y, X = feat_da.sizes["feat"], feat_da.sizes["y"], feat_da.sizes["x"]
vals = feat_da.values.reshape(F, Y * X).T          # (Y*X, 8)
mask_vec = mask_grid.values.reshape(Y * X)         # (Y*X,)
X8_raw = vals[mask_vec]                            # (N, 8) — unstandardized features

# --- 6) Leave everything in well-named variables for later cells ---
# For interpretation/plots later:
#   cbar_adv, sbar_adv, cbar_ret, sbar_ret, R_adv, R_ret, dur_mean_days, dur_std_days
# For modeling in Cell 4:
#   X8_raw (N,8), feat_names, mask_grid, grid shape
grid_shape = (Y, X)

print("Built features:")
print("  X8_raw shape (N,8):", X8_raw.shape)
print("  Valid pixel fraction:", float(mask_grid.mean().values))
print("  Feature order:", feat_names)

# Cell 4 — Standardize features, select K by BIC, fit GMM, map results to the grid
#
# Preconditions from Cell 3:
#   X8_raw      : (N, 8) unstandardized feature matrix (only valid pixels)
#   feat_names  : list of 8 feature names in column order
#   mask_grid   : (y, x) boolean (True where pixel is valid)
#   grid_shape  : (Y, X) tuple
#
# Outputs created here (kept in memory for later cells; nothing saved to disk yet):
#   X8_z            : (N, 8) z-scored features
#   scaler_mu       : (8,) column means used for standardization
#   scaler_sigma    : (8,) column stds used for standardization (zeros replaced with 1.0)
#   best_model      : fitted GaussianMixture with best BIC
#   labels          : (N,) cluster labels for valid pixels (ints 0..K-1)
#   probs           : (N, K) posterior responsibilities
#   uncert          : (N,) = 1 - max_k probs[:,k]
#   labels_grid     : (Y, X) float array with labels at valid pixels, NaN elsewhere
#   uncert_grid     : (Y, X) float array with uncertainty at valid pixels, NaN elsewhere
#   bic_results     : list of dicts with {'K','cov','BIC'} for later plotting if desired

import numpy as np
from sklearn.mixture import GaussianMixture

# --- 0) Defensive checks (fail early with a clear message)
assert 'X8_raw' in globals() and isinstance(X8_raw, np.ndarray) and X8_raw.ndim == 2 and X8_raw.shape[1] == 8, \
    "X8_raw must be an (N, 8) NumPy array from Cell 3."
assert 'mask_grid' in globals() and 'grid_shape' in globals(), \
    "mask_grid (y,x) and grid_shape (Y,X) must exist from Cell 3."

# --- 1) Standardize (z-score) each column manually (clear, no hidden state)
# Rationale: GMM uses covariances; without scaling, 'days' features dominate unitless ones.
scaler_mu = np.nanmean(X8_raw, axis=0)             # (8,)
scaler_sigma = np.nanstd(X8_raw, axis=0, ddof=0)   # (8,)
scaler_sigma = np.where(scaler_sigma == 0.0, 1.0, scaler_sigma)  # avoid divide-by-zero if a column is constant

X8_z = (X8_raw - scaler_mu) / scaler_sigma         # (N, 8)

# --- 2) Small, defensible model sweep and BIC selection
K_RANGE   = range(2, 9)              # K = 2..8
COV_TYPES = ["full", "diag"]         # simple, common choices
SEED      = 42
MAX_ITER  = 500
REG_COVAR = 1e-6

bic_results = []   # store results for inspection/plotting later
best_model  = None
best_score  = np.inf
best_K, best_cov = None, None

for K in K_RANGE:
    for cov in COV_TYPES:
        gm = GaussianMixture(
            n_components=K,
            covariance_type=cov,
            random_state=SEED,
            max_iter=MAX_ITER,
            reg_covar=REG_COVAR
        )
        gm.fit(X8_z)
        bic = gm.bic(X8_z)
        bic_results.append({'K': K, 'cov': cov, 'BIC': float(bic)})
        if bic < best_score:
            best_score = bic
            best_model = gm
            best_K, best_cov = K, cov

print(f"[GMM] Best by BIC: K={best_K}, cov={best_cov}, BIC={best_score:.1f}, converged={best_model.converged_}")

# --- 3) Cluster assignments and a simple uncertainty diagnostic
labels = best_model.predict(X8_z)          # (N,)
probs  = best_model.predict_proba(X8_z)    # (N, K_best)
uncert = 1.0 - probs.max(axis=1)           # (N,)  0=confident, 1=uncertain

# Quick counts (helps catch degenerate solutions where a cluster is empty)
unique, counts = np.unique(labels, return_counts=True)
print("Cluster sizes:", dict(zip(unique.tolist(), counts.tolist())))

# --- 4) Map vector outputs back to (Y, X) grid
def _grid_from_vector(vec_1d, mask_bool, shape):
    """
    vec_1d : (N,) values for the valid pixels
    mask_bool : (Y,X) boolean mask, True where pixel is valid
    shape : (Y, X) grid shape
    -> (Y, X) array with vec_1d filled at True locations, NaN elsewhere
    """
    Y, X = shape
    out = np.full((Y, X), np.nan, dtype=float)
    flat = out.reshape(-1)
    mflat = np.asarray(mask_bool).reshape(-1)
    flat[mflat] = vec_1d.astype(float)
    return out

labels_grid = _grid_from_vector(labels, mask_grid, grid_shape)
uncert_grid = _grid_from_vector(uncert, mask_grid, grid_shape)

print("labels_grid:", labels_grid.shape, "| uncert_grid:", uncert_grid.shape)
