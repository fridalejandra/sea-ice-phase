#!/usr/bin/env python3
import os, re, glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from math import ceil

# ----------------- USER CONFIG -----------------
SENSOR       = "SMMR"     # "SMMR" or "AMSRE"
THRESH_PCT   = 15         # e.g., 10/15/20  (this script: fixed thr, compare k)
INPUT_ROOT   = f"/user/geog/falejandraperez/sea-ice-phase/results/{SENSOR}_phase"
OUT_DIR      = f"/user/geog/falejandraperez/sea-ice-phase/results/window_sensitivity_plots/{SENSOR}_thr{THRESH_PCT}"

PERIOD       = 366        # DOY wrap
MAX_X        = 30         # x-limit in days for |Δ|

os.makedirs(OUT_DIR, exist_ok=True)

RCLONE = {
    "enabled": True,  # set False if running locally
    "remote": "gdrive",
    "dst_dir": "sea-ice-phase/results/window_sensitivity_plots/",
    "extra_flags": ["--transfers=8", "--checkers=8", "--fast-list"],
    "dry_run": False
}


# ----------------- HELPERS -----------------
year_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    m = year_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def load_window_dict(metric, kdays):
    """Return {year: DataArray} for a given metric ('MS'/'FS') and window length (3/5/7)."""
    subdir = f"{metric}_thr{THRESH_PCT}_k{kdays}"
    folder = os.path.join(INPUT_ROOT, subdir)
    files  = sorted(glob.glob(os.path.join(folder, f"{metric}_*.nc")))
    d = {}
    for f in files:
        y = parse_year(f)
        if y is None:
            continue
        with xr.open_dataset(f) as ds:
            d[y] = ds[metric].load()     # var is exactly FS or MS
    if not d:
        raise FileNotFoundError(f"No {metric} files in {folder}")
    return d

def align_years(dicts):
    common = set.intersection(*[set(d.keys()) for d in dicts])
    years = sorted(common)
    if not years:
        raise ValueError("No overlapping years across windows.")
    return years

def stack_years(d, years):
    arrs = [d[y].expand_dims(year=[y]) for y in years]
    return xr.concat(arrs, dim="year")  # (year,y,x)

def wrapped_abs_diff_np(a, b, period=PERIOD):
    return np.abs(((a - b + period//2) % period) - (period//2))

def ecdf(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    x.sort()
    y = np.linspace(0, 1, x.size, endpoint=False)
    return x, y

# ----------------- CORE -----------------
def diffs_for_metric(metric):
    """Flattened |Δ(3-5)| and |Δ(7-5)| over valid pixels."""
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    years = align_years([d3, d5, d7])

    A3 = stack_years(d3, years)
    A5 = stack_years(d5, years)
    A7 = stack_years(d7, years)

    valid = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7))
    d35 = wrapped_abs_diff_np(A3.values, A5.values)
    d75 = wrapped_abs_diff_np(A7.values, A5.values)

    v35 = d35[valid.values].ravel()
    v75 = d75[valid.values].ravel()
    return v35, v75

def plot_cdf(metric, v35, v75, max_x=MAX_X):
    fig, ax = plt.subplots(figsize=(7,5))
    for vals, lab in [(v35, "3 vs 5-day window"), (v75, "7 vs 5-day window")]:
        vals = vals[np.isfinite(vals)]
        vals = vals[vals <= max_x]
        x, y = ecdf(vals)
        ax.plot(x, y, lw=2, label=lab)
    ax.set_xlim(0, max_x); ax.set_ylim(0,1)
    ax.set_xlabel("Absolute timing difference |Δ| (days)")
    ax.set_ylabel("Cumulative Fraction of Pixels")
    ax.set_title(f"Window CDF • {metric} • thr={THRESH_PCT}%")
    ax.grid(True, ls=":", lw=0.7)
    ax.legend(frameon=True)
    fn = os.path.join(OUT_DIR, f"CDF_{metric}_thr{THRESH_PCT}.png")
    fig.tight_layout(); fig.savefig(fn, dpi=250); plt.close(fig);rclone_copy(fn)

    print("wrote", fn)

import subprocess

def rclone_copy(local_path, cfg=RCLONE):
    """Copy local file to remote using rclone."""
    if not cfg.get("enabled", False):
        return
    dst = f"{cfg['remote']}:{cfg['dst_dir']}"
    cmd = ["rclone", "copy", str(local_path), dst] + cfg.get("extra_flags", [])
    if cfg.get("dry_run"):
        cmd.insert(1, "--dry-run")
    print("[rclone]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!! rclone failed: {e}")


if __name__ == "__main__":
    for metric in ["MS","FS"]:
        v35, v75 = diffs_for_metric(metric)
        plot_cdf(metric, v35, v75)
