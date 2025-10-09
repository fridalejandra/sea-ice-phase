# cdf_sectoral_window_sensitivity_MS_FS.py
import os, re, glob
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil

# ----------------- USER CONFIG -----------------
SENSOR       = "SMMR"      # "SMMR" or "AMSRE"
THRESH_PCT   = 15          # e.g., 10/15/20
INPUT_ROOT   = f"/user/geog/falejandraperez/sea-ice-phase/results/{SENSOR}_phase"
RCLONE_DEST  = f"figures/cdf_ms_fs/{SENSOR}_thr{THRESH_PCT}"

PERIOD       = 366         # DOY wrap
MAX_X        = 30          # x-limit in days for |Δ|

# --- NEW: canonical sector file ---
CANONICAL = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"

# rclone destination (must include remote alias, not just a folder)
RCLONE_DEST = "gdrive:sea-ice-phase/results/cdf_ms_fs/SMMR_thr15"  # edit as needed

# Aesthetics
sns.set_context("talk")
sns.set_style("whitegrid")

# ----------------- HELPERS (from your original) -----------------
year_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    m = year_re.search(os.path.basename(path))
    if not m: return None
    return int(m.group(1))

def load_window_dict(metric, kdays):
    """Return {year: DataArray} for a given metric ('MS'/'FS') and window length (3/5/7)."""
    subdir = f"{metric}_thr{THRESH_PCT}_k{kdays}"
    folder = os.path.join(INPUT_ROOT, subdir)
    files = sorted(glob.glob(os.path.join(folder, f"{metric}_*.nc")))
    d = {}
    for f in files:
        yr = parse_year(f)
        if yr is None:
            print(f"Skip (no year): {f}")
            continue
        try:
            with xr.open_dataset(f) as ds:
                # variable must be exactly 'MS' or 'FS'. Change here if different.
                da = ds[metric].load()
            d[yr] = da
        except Exception as e:
            print(f"Skip {f}: {e}")
    if not d:
        raise FileNotFoundError(f"No {metric} files in {folder}")
    return d

def align_years(dicts):
    common = set.intersection(*[set(d.keys()) for d in dicts])
    years = sorted(common)
    if not years:
        raise ValueError("No overlapping years across windows.")
    return years

def stack_years(d, years, name):
    arrs = [d[y].expand_dims(year=[y]) for y in years]
    out = xr.concat(arrs, dim="year")
    out.name = name
    return out

def wrapped_diff_np(a_minus_b, period=PERIOD):
    return ((a_minus_b + period//2) % period) - (period//2)

def ecdf_data(abs_diff, max_x=MAX_X):
    abs_diff = abs_diff[~np.isnan(abs_diff)]
    return abs_diff[abs_diff <= max_x], abs_diff

# ----------------- SECTOR MASKS (from canonical file) -----------------
SECTOR_ID_TO_NAME = {
    1: "Amundsen–Bellingshausen",
    2: "Weddell",
    3: "King Haakon VII",
    4: "East Antarctic",
    5: "Ross–Amundsen",
}


def load_sector_masks(canonical_path=CANONICAL):
    ds = xr.open_dataset(canonical_path).load()
    sid = ds["sector_id"].astype(np.int16)
    valid_ocean = ds["valid_ocean"].astype(bool) if "valid_ocean" in ds else xr.ones_like(sid, dtype=bool)
    masks = {}
    for k, name in SECTOR_ID_TO_NAME.items():
        m = (sid == k) & valid_ocean
        masks[name] = m.values.astype(bool)
    ds.close()
    return masks

def rclone_copy(local_path, remote_dir=RCLONE_DEST):
    cmd = f"rclone copy '{local_path}' '{remote_dir}' --transfers=8 --checkers=8 --fast-list"
    rc = os.system(cmd)
    if rc != 0:
        print(f"!! rclone failed (code {rc}) for {local_path}")



# ----------------- DIFFS (whole domain + sector) -----------------
def diffs_for_metric(metric):
    """Whole Antarctica |Δ(3-5)| and |Δ(7-5)| flattened."""
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    years = align_years([d3, d5, d7])
    A3 = stack_years(d3, years, f"{metric}_k3")
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")
    valid = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7))
    d35 = wrapped_diff_np((A3 - A5).values)
    d75 = wrapped_diff_np((A7 - A5).values)
    v35 = np.abs(d35[valid.values]).ravel()
    v75 = np.abs(d75[valid.values]).ravel()
    return v35, v75

def diffs_for_metric_sector_mask(metric, sector_mask_2d):
    """Sectoral |Δ(3-5)| and |Δ(7-5)| flattened (mask broadcast over years)."""
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    years = align_years([d3, d5, d7])
    A3 = stack_years(d3, years, f"{metric}_k3")  # (year,y,x)
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")
    valid = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7))
    sm = xr.DataArray(np.asarray(sector_mask_2d, dtype=bool), dims=("y", "x"))
    if sm.shape != valid.isel(year=0).shape:
        raise ValueError(f"Sector mask shape {sm.shape} does not match grid {valid.isel(year=0).shape}")

    sector_mask = sm.broadcast_like(valid)
    valid = valid & sector_mask
    d35 = wrapped_diff_np((A3 - A5).values)
    d75 = wrapped_diff_np((A7 - A5).values)
    v35 = np.abs(d35[valid.values]).ravel()
    v75 = np.abs(d75[valid.values]).ravel()
    return v35, v75

# ----------------- DIAGNOSTIC -----------------
def debug_years(metric):
    """Prints year coverage and overlap for k3/k5/k7 for the metric."""
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)
    s3, s5, s7 = set(d3.keys()), set(d5.keys()), set(d7.keys())
    inter = s3 & s5 & s7
    def rng(s):
        return f"{min(s)}–{max(s)}" if s else "—"
    print(f"\n[{metric}] years per window:")
    print(f"  k3: {rng(s3)}  (n={len(s3)})")
    print(f"  k5: {rng(s5)}  (n={len(s5)})")
    print(f"  k7: {rng(s7)}  (n={len(s7)})")
    print(f"  overlap: n={len(inter)}  {sorted(list(inter))[:8]}{' …' if len(inter)>8 else ''}")
    if not inter:
        print("!! No overlapping years across k3/k5/k7 — check file availability / THRESH_PCT / paths.")


# ----------------- PLOTTING -----------------
def plot_cdf_sectors_with_masks(metric, masks_dict, ncols=3, dpi=300,
                                panel_size=(3.2, 2.6)):
    """Faceted plot per sector (no annotations). Shared axes; figure-level legend."""
    names = list(masks_dict.keys())
    n = len(names); nrows = ceil(n / ncols)

    fig_w = ncols * panel_size[0]
    fig_h = nrows * panel_size[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    print(f"\n[{metric}] sector pixel counts (valid & in mask):")
    print("sector".ljust(28), "n_pixels")

    legend_handles = None
    legend_labels  = ["3 vs 5-day window", "7 vs 5-day window"]

    for i, name in enumerate(names):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        v35, v75 = diffs_for_metric_sector_mask(metric, masks_dict[name])
        print(name.ljust(28), f"{(v35.size + v75.size)//2:,}")

        v35_clip, _ = ecdf_data(v35, MAX_X)
        v75_clip, _ = ecdf_data(v75, MAX_X)

        if v35_clip.size == 0 and v75_clip.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="0.4", fontsize=9)
        else:
            sns.ecdfplot(v35_clip, label=legend_labels[0], lw=2, ax=ax)
            sns.ecdfplot(v75_clip, label=legend_labels[1], lw=2, ax=ax)
            # capture line handles from FIRST axes only
            if legend_handles is None:
                legend_handles = [ax.lines[-2], ax.lines[-1]]

        ax.set_title(name, fontsize=10, pad=3, color="0.25")
        ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.82")

    # hide unused panels if any
    total = nrows * ncols
    for j in range(n, total):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    # identical axes + ticks across panels
    for ax in axes.ravel():
        if ax.get_visible():
            ax.set_xlim(0, MAX_X)
            ax.set_ylim(0, 1.0)
            ax.tick_params(labelsize=9)

    # figure-level x/y labels (no per-axes labels)
    fig.supxlabel("Absolute timing difference (days)", fontsize=11, fontweight="bold", color="0.3")
    fig.supylabel("Cumulative Fraction of Pixels",    fontsize=11, fontweight="bold", color="0.3")

    # legend outside the grid
    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, title="Window comparison",
                   loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=2, frameon=True)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    # save + upload
    fname = f"CDF_sectors_{metric}_{SENSOR}_thr{THRESH_PCT}.png"
    local_path = f"/tmp/{fname}"
    fig.savefig(local_path, dpi=dpi)
    plt.close(fig)
    os.system(f"rclone copy '{local_path}' '{RCLONE_DEST}'")
    print(f"✓ Uploaded: {fname}")

def plot_cdf_single_sector(metric, name, mask_2d,
                           figsize=(5.0, 3.5), dpi=300, save=True):
    """Bigger, single-sector panel (no annotations)."""
    v35, v75 = diffs_for_metric_sector_mask(metric, mask_2d)
    v35_clip, _ = ecdf_data(v35, MAX_X)
    v75_clip, _ = ecdf_data(v75, MAX_X)
    fig, ax = plt.subplots(figsize=figsize)
    if v35_clip.size == 0 and v75_clip.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.4", fontsize=10)
    else:
        # plot and create legend with real line handles
        sns.ecdfplot(v35_clip, label="3 vs 5-day window", lw=2, ax=ax)
        sns.ecdfplot(v75_clip, label="7 vs 5-day window", lw=2, ax=ax)
        handles = [ax.lines[-2], ax.lines[-1]]
        ax.legend(handles=handles, labels=["3 vs 5-day window", "7 vs 5-day window"],
                  title="Window comparison", frameon=True, loc="lower right")
    ax.set_xlim(0, MAX_X); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Absolute timing difference (days)", fontsize=11, fontweight="bold", color="0.3")
    ax.set_ylabel("Cumulative Fraction of Pixels",    fontsize=11, fontweight="bold", color="0.3")
    ax.set_title(f"{name} • {metric}", fontsize=11, color="0.25", pad=3)
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.82")
    fig.tight_layout()
    if save:
        fname = f"CDF_sector_{name.replace(' ','_')}_{metric}_{SENSOR}_thr{THRESH_PCT}.png"
        local_path = f"/tmp/{fname}"
        fig.savefig(local_path, dpi=dpi)
        plt.close(fig)
        os.system(f"rclone copy '{local_path}' '{RCLONE_DEST}'")
        print(f"✓ Uploaded: {fname}")
    else:
        plt.show()

# ----------------- RUN -----------------
if __name__ == "__main__":
    # Build sector masks from canonical file
    SECTOR_MASKS = load_sector_masks(CANONICAL)

    # Individual sector plots
    for metric in ["MS", "FS"]:
        print(f"\n=== Individual sector plots: {metric} ===")
        for name, mask in SECTOR_MASKS.items():
            plot_cdf_single_sector(metric, name, mask, figsize=(5.5, 4.0), dpi=300, save=True)

    # Diagnostics
    debug_years("MS")
    debug_years("FS")

    # Faceted figures
    plot_cdf_sectors_with_masks("MS", SECTOR_MASKS, ncols=3)
    plot_cdf_sectors_with_masks("FS", SECTOR_MASKS, ncols=3)
