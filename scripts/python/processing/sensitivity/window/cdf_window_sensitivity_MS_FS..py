# --- imports ---
import os
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

# --- user config ---
SENSOR = "SMMR"            # or "SMMR"
THRESH_PCT = 15                 # e.g., 10 / 15 / 20
INPUT_DIR = f"/user/geog/falejandraperez/sea-ice-phase/results/sensitivity/{SENSOR}_phase"
RCLONE_DEST = f"gdrive:sea-ice-phase/results/figures/cdf_ms_fs/{SENSOR}_thr{THRESH_PCT}"

# file names expected inside INPUT_DIR (no years; 1 file per window)
# e.g., seaice_phases_MS_thr15_k3.nc with variable "MS" inside, likewise "FS".
FN = {
    "MS": {
        "k3":  f"seaice_phases_MS_thr{THRESH_PCT}_k3.nc",
        "k5":  f"seaice_phases_MS_thr{THRESH_PCT}_k5.nc",
        "k7":  f"seaice_phases_MS_thr{THRESH_PCT}_k7.nc",
    },
    "FS": {
        "k3":  f"seaice_phases_FS_thr{THRESH_PCT}_k3.nc",
        "k5":  f"seaice_phases_FS_thr{THRESH_PCT}_k5.nc",
        "k7":  f"seaice_phases_FS_thr{THRESH_PCT}_k7.nc",
    }
}

# plotting choices
MAX_X_DAYS = 30     # show CDF up to this absolute difference
MARKS = [2, 5, 10]
sns.set_context("talk")
sns.set_style("whitegrid")


def load_da(path, varname):
    """Open a NetCDF, return the DataArray for varname (e.g., 'MS' or 'FS')."""
    ds = xr.open_dataset(path)
    da = ds[varname].load()
    ds.close()
    return da

def wrapped_difference(a, b, period=366):
    """Wrap a-b into [-period/2, +period/2]. Works for DOY-style timing."""
    raw = a - b
    return ((raw + period//2) % period) - (period//2)

def valid_flatten(*arrays):
    """Mask to common valid pixels across all arrays, then flatten to 1-D."""
    mask = np.ones_like(arrays[0].values, dtype=bool)
    for arr in arrays:
        mask &= ~np.isnan(arr.values)
    # return flattened views
    return [arr.values[mask].ravel() for arr in arrays]

### Compute Differences ###

def diffs_for_metric(metric):
    """
    metric: 'MS' or 'FS'
    returns: |Δ(3-5)|, |Δ(7-5)| as 1-D arrays (masked, wrapped, absolute)
    """
    f3 = os.path.join(INPUT_DIR, FN[metric]["k3"])
    f5 = os.path.join(INPUT_DIR, FN[metric]["k5"])
    f7 = os.path.join(INPUT_DIR, FN[metric]["k7"])

    a3 = load_da(f3, metric)
    a5 = load_da(f5, metric)
    a7 = load_da(f7, metric)

    # common valid mask across windows
    flat3, flat5, flat7 = valid_flatten(a3, a5, a7)

    # wrapped diffs (3-5 and 7-5), then absolute value for CDF of |Δ|
    d35 = np.abs(wrapped_difference(flat3, flat5))
    d75 = np.abs(wrapped_difference(flat7, flat5))

    return d35, d75

### Empirical CDF with thresholds ##
def plot_cdf(metric, max_x=MAX_X_DAYS, marks=MARKS):
    """
    metric: 'MS' or 'FS'
    Saves figure to /tmp/ and uploads to Google Drive via rclone.
    """
    d35, d75 = diffs_for_metric(metric)

    # Restrict x-range for plotting
    d35_clip = d35[d35 <= max_x]
    d75_clip = d75[d75 <= max_x]

    # Figure
    plt.figure(figsize=(7.5, 6))
    # seaborn ECDFs
    sns.ecdfplot(d35_clip, label="|Δ| (k3 − k5)", lw=2)
    sns.ecdfplot(d75_clip, label="|Δ| (k7 − k5)", lw=2)

    # Vertical reference lines
    for m in marks:
        plt.axvline(m, ls="--", c="k", lw=1)
        # annotate fraction ≤ m
        p35 = (d35 <= m).mean() * 100
        p75 = (d75 <= m).mean() * 100
        plt.text(m+0.2, 0.05, f"{m}d:\n{kfmt(p35)}% (3–5)\n{kfmt(p75)}% (7–5)",
                 fontsize=9, va="bottom")

    # Titles and axes
    plt.xlim(0, max_x)
    plt.ylim(0, 1.0)
    plt.xlabel("Absolute timing difference |Δ| (days)")
    plt.ylabel("Fraction of pixels (CDF)")
    plt.title(f"{SENSOR}  •  {metric}  •  threshold={THRESH_PCT}%  (k3/k7 vs k5)")
    plt.legend(frameon=True, title="Window comparison")
    plt.tight_layout()

    # Save locally and upload
    fname = f"CDF_{metric}_{SENSOR}_thr{THRESH_PCT}_k3k7_vs_k5.png"
    local_path = f"/tmp/{fname}"
    plt.savefig(local_path, dpi=300)
    plt.close()

    # rclone upload
    os.system(f"rclone copy '{local_path}' '{RCLONE_DEST}'")
    print(f"✓ Uploaded: {fname}")

def kfmt(x):
    """Compact percentage formatting to 1 decimal."""
    return f"{x:.1f}"


if __name__ == "__main__":
    # Melt Start
    plot_cdf("MS", max_x=MAX_X_DAYS, marks=MARKS)
    # Freeze Start
    plot_cdf("FS", max_x=MAX_X_DAYS, marks=MARKS)
