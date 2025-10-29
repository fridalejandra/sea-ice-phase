#!/usr/bin/env python3
import os, re, glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# ============== CONFIG ==============
SENSOR       = "SMMR"
INPUT_ROOT   = f"/user/geog/falejandraperez/sea-ice-phase/results/{SENSOR}_phase"
OUT_DIR      = f"/user/geog/falejandraperez/sea-ice-phase/results/threshold_sensitivity"

METRICS      = ["MS","FS"]         # retreat, advance
K_FIXED      = 5                   # compare thresholds at fixed k
THRESHOLDS   = [10, 15, 30]        # percent values (directory convention)
PERIOD       = 366                 # wrap for DOY
MAX_X        = 30                  # CDF/box x-limits in days
CANONICAL    = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"  # optional

os.makedirs(OUT_DIR, exist_ok=True)
sns.set_context("talk"); sns.set_style("whitegrid")

RCLONE = {
    "enabled": True,  # set False if running locally
    "remote": "gdrive",
    "dst_dir": "sea-ice-phase/results/threshold_sensitivity_plots/",
    "extra_flags": ["--transfers=8", "--checkers=8", "--fast-list"],
    "dry_run": False
}


# ============== HELPERS ==============
year_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    m = year_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def load_thr_dict(metric, thr_pct, k=K_FIXED):
    folder = os.path.join(INPUT_ROOT, f"{metric}_thr{thr_pct}_k{k}")
    files  = sorted(glob.glob(os.path.join(folder, f"{metric}_*.nc")))
    d = {}
    for f in files:
        y = parse_year(f)
        if y is None:
            continue
        with xr.open_dataset(f) as ds:
            d[y] = ds[metric].load()
    if not d:
        raise FileNotFoundError(f"No files for {metric} thr{thr_pct} k{k} in {folder}")
    return d

def align_years(dicts):
    common = set.intersection(*[set(d.keys()) for d in dicts])
    years = sorted(common)
    if not years:
        raise ValueError("No overlapping years across thresholds.")
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

def get_canonical():
    if CANONICAL and os.path.exists(CANONICAL):
        cano = xr.open_dataset(CANONICAL).load()
        vo   = cano["valid_ocean"].astype(bool).values
        sid  = cano["sector_id"].astype(np.int16).values
        sector_map = {
            1: "Amundsen–Bellingshausen",
            2: "Weddell",
            3: "King Haakon VII",
            4: "East Antarctic",
            5: "Ross–Amundsen",
        }
        return vo, sid, sector_map, cano
    return None, None, None, None

VO, SID, SECTORS, _CANO = get_canonical()

# ============== CORE: diffs per pair ==============
def diffs_for_metric_pair(metric, thrA, thrB):
    dA = load_thr_dict(metric, thrA)
    dB = load_thr_dict(metric, thrB)
    years = align_years([dA, dB])
    A = stack_years(dA, years)  # (year,y,x)
    B = stack_years(dB, years)
    valid = (~np.isnan(A)) & (~np.isnan(B))
    D = wrapped_abs_diff_np(A.values, B.values)         # numpy
    D = xr.DataArray(D, dims=("year","y","x")).where(valid)
    # active ocean mask (optional)
    if VO is not None and VO.shape == D.isel(year=0).shape:
        D = D.where(VO)
    return years, D

# ============== PLOTS ==============
def plot_cdf(metric, pair_label, D, max_x=MAX_X):
    vals = D.values[np.isfinite(D.values)]
    vals = vals[vals <= max_x]
    x, y = ecdf(vals)
    fig, ax = plt.subplots(figsize=(6.2,4.6))
    ax.plot(x, y, lw=2)
    ax.set_xlim(0, max_x); ax.set_ylim(0,1)
    ax.set_xlabel("|Δ| (days)")
    ax.set_ylabel("Cumulative Fraction of Pixels")
    ax.set_title(f"CDF • {metric} • {pair_label} • k={K_FIXED}")
    ax.grid(True, ls=":", lw=0.7)
    fn = os.path.join(OUT_DIR, f"CDF_{metric}_{pair_label}_k{K_FIXED}.png")
    fig.tight_layout(); fig.savefig(fn, dpi=250); plt.close(fig)
    print("wrote", fn)

def plot_box(metric, pair_label, D):
    """Box plot by sector + circumpolar."""
    data = []
    arr = D
    # circumpolar
    v = arr.values[np.isfinite(arr.values)]
    if v.size:
        data.append(("Circumpolar", v))
    # by sector if available
    if SECTORS is not None:
        for k, name in SECTORS.items():
            m = (SID == k)
            vv = arr.values[:, m]  # (year, n_pix_in_sector)
            vv = vv[np.isfinite(vv)]
            if vv.size:
                data.append((name, vv))

    labels = [d[0] for d in data]
    series = [d[1] for d in data]
    fig, ax = plt.subplots(figsize=(10,4.8))
    ax.boxplot(series, labels=labels, showfliers=False)
    ax.set_ylabel("|Δ| (days)"); ax.set_title(f"Box • {metric} • {pair_label} • k={K_FIXED}")
    ax.set_ylim(0, MAX_X)
    ax.grid(True, axis="y", ls=":", lw=0.7)
    fig.tight_layout()
    fn = os.path.join(OUT_DIR, f"BOX_{metric}_{pair_label}_k{K_FIXED}.png")
    fig.savefig(fn, dpi=250); plt.close(fig)
    print("wrote", fn)

def plot_joint_hist(metric, pair_low_mid, pair_mid_high, D_lm, D_mh):
    """
    Joint histogram: x = mean |Δ(low–mid)| per pixel, y = mean |Δ(mid–high)|.
    Requires 3 thresholds (low < mid < high).
    """
    # reduce across years for each pixel
    X = D_lm.mean("year", skipna=True).values
    Y = D_mh.mean("year", skipna=True).values
    m = np.isfinite(X) & np.isfinite(Y)
    x = X[m].ravel(); y = Y[m].ravel()

    # correlation & fit
    if x.size >= 2:
        r = np.corrcoef(x, y)[0,1]
        coef = np.polyfit(x, y, 1)
    else:
        r = np.nan; coef = [np.nan, np.nan]

    fig, ax = plt.subplots(figsize=(6.8,6.2))
    hb = ax.hexbin(x, y, gridsize=80, bins="log", cmap="magma")
    cb = fig.colorbar(hb, ax=ax, label="pixel count (log)")
    ax.plot([0, MAX_X], [0, MAX_X], ls="--", lw=1, c="0.6", label="1:1")
    if np.isfinite(coef).all():
        xx = np.array([0, MAX_X])
        ax.plot(xx, coef[0]*xx + coef[1], lw=2, label=f"fit: y={coef[0]:.2f}x+{coef[1]:.2f}")
    ax.set_xlim(0, MAX_X); ax.set_ylim(0, MAX_X)
    ax.set_xlabel(f"mean |Δ| ({pair_low_mid}) (days)")
    ax.set_ylabel(f"mean |Δ| ({pair_mid_high}) (days)")
    ax.set_title(f"Joint histogram • {metric} • r={r:.2f}")
    ax.legend(frameon=True, loc="lower right")
    fn = os.path.join(OUT_DIR, f"JOINT_{metric}_{pair_low_mid}_vs_{pair_mid_high}_k{K_FIXED}.png")
    fig.tight_layout(); fig.savefig(fn, dpi=260); plt.close(fig); rclone_copy(fn)

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


# ============== MAIN ==============
if __name__ == "__main__":
    # All threshold pairings for CDF/Box:
    PAIRS = list(combinations(THRESHOLDS, 2))  # e.g. (10,15), (10,30), (15,30)

    for metric in METRICS:
        # Compute and plot for each pair
        pair_maps = {}
        for a, b in PAIRS:
            years, D = diffs_for_metric_pair(metric, a, b)   # (year,y,x)
            label = f"thr{a}-thr{b}"
            pair_maps[(a,b)] = D
            plot_cdf(metric, label, D)
            plot_box(metric, label, D)

        # If we have exactly 3 thresholds, make the joint histogram of (low–mid) vs (mid–high)
        if len(THRESHOLDS) >= 3:
            thr_sorted = sorted(THRESHOLDS)
            lm = (thr_sorted[0], thr_sorted[1])
            mh = (thr_sorted[1], thr_sorted[2])
            if lm in pair_maps and mh in pair_maps:
                plot_joint_hist(metric,
                                f"thr{lm[0]}-thr{lm[1]}",
                                f"thr{mh[0]}-thr{mh[1]}",
                                pair_maps[lm], pair_maps[mh])
