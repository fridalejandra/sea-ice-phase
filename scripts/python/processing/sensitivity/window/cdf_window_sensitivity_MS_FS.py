# cdf_window_sensitivity_MS_FS.py
import os, re, glob
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------- USER CONFIG -----------------
SENSOR       = "SMMR"      # "SMMR" or "AMSRE"
THRESH_PCT   = 15          # e.g., 10/15/20
INPUT_ROOT   = f"/user/geog/falejandraperez/sea-ice-phase/results/{SENSOR}_phase"
RCLONE_DEST  = f"gdrive:sea-ice-phase/results/figures/cdf_ms_fs/{SENSOR}_thr{THRESH_PCT}"
PERIOD       = 366         # DOY wrap
MAX_X        = 30          # plot x-limit in days for |Δ|
MARKS        = [2, 5, 10]  # annotate CDF at these |Δ|
sns.set_context("talk"); sns.set_style("whitegrid")

# ----------------- HELPERS -----------------
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
                da = ds[metric].load()
            d[yr] = da
        except Exception as e:
            print(f"Skip {f}: {e}")
    if not d:
        raise FileNotFoundError(f"No {metric} files in {folder}")
    return d

def align_years(dicts):
    """Intersect years across window dicts and return sorted list."""
    common = set.intersection(*[set(d.keys()) for d in dicts])
    years = sorted(common)
    if not years:
        raise ValueError("No overlapping years across windows.")
    return years

def stack_years(d, years, name):
    """Stack {year: DataArray} into one DataArray with dim 'year' and coord years."""
    arrs = [d[y].expand_dims(year=[y]) for y in years]
    out = xr.concat(arrs, dim="year")
    out.name = name
    return out

def wrapped_diff_np(a_minus_b, period=PERIOD):
    return ((a_minus_b + period//2) % period) - (period//2)

def ecdf_data(abs_diff, max_x=MAX_X):
    """Return clipped array for plotting ECDF and full array for stats."""
    abs_diff = abs_diff[~np.isnan(abs_diff)]
    return abs_diff[abs_diff <= max_x], abs_diff

def frac_within(arr, m):
    if arr.size == 0: return np.nan
    return (arr <= m).mean() * 100.0

# ----------------- CORE: BUILD |Δ| FOR A METRIC -----------------
def diffs_for_metric(metric):
    """
    metric: 'MS' or 'FS'
    Returns: |Δ(3-5)| and |Δ(7-5)| as 1-D arrays over all common years/pixels.
    """
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)

    years = align_years([d3, d5, d7])
    A3 = stack_years(d3, years, f"{metric}_k3")  # (year,y,x)
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")

    # common valid mask
    mask = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7))

    # wrapped diffs then absolute
    d35 = wrapped_diff_np((A3 - A5).values)
    d75 = wrapped_diff_np((A7 - A5).values)

    # apply mask + flatten
    v35 = np.abs(d35[mask.values]).ravel()
    v75 = np.abs(d75[mask.values]).ravel()
    return v35, v75

# ----------------- PLOT CDF FOR A METRIC -----------------
def plot_cdf(metric):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- data ---
    v35, v75 = diffs_for_metric(metric)
    v35_clip, v35_full = ecdf_data(v35, MAX_X)
    v75_clip, v75_full = ecdf_data(v75, MAX_X)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    sns.ecdfplot(v35_clip, label="3 vs 5-day window", lw=2, ax=ax)
    sns.ecdfplot(v75_clip, label="7 vs 5-day window", lw=2, ax=ax)

    # helper: empirical CDF value at x
    def ecdf_at(vals, x):
        arr = np.asarray(vals)
        return float(np.mean(arr <= x))

    # --- annotations: fixed staggered y-levels near the bottom ---
    xoff = 0.3  # horizontal text offset (days)
    base_y = 0.12  # first label y (in data coords; 0..1 since CDF)
    step_y = 0.12  # vertical spacing between labels (in data coords)

    for i, m in enumerate(MARKS):
        ax.axvline(m, ls="--", c="k", lw=1)

        p35 = frac_within(v35_full, m)
        p75 = frac_within(v75_full, m)

        # put each label on its own horizontal band
        y_text = base_y + i * step_y
        y_text = max(0.06, min(0.94, y_text))  # stay on-axes

        # flip to left side if near the right edge
        on_right = m > 0.80 * MAX_X
        xt = m - xoff if on_right else m + xoff
        ha = "right" if on_right else "left"

        txt = f"{m} d:\n{p35:.1f}% (3–5)\n{p75:.1f}% (7–5)"
        ax.annotate(
            txt,
            xy=(m, y_text), xytext=(xt, y_text),  # horizontal arrow at a fixed y
            ha=ha, va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.75", alpha=0.95),
            arrowprops=dict(arrowstyle="-", lw=0.8, color="0.3"),
        )

    # --- styling unchanged from your original ---
    ax.set_xlim(0, MAX_X); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Absolute timing difference (days)")
    ax.set_ylabel("Cumulative Fraction of Pixels")
    ax.legend(frameon=True, title="Window comparison")
    plt.tight_layout()

    # save + upload
    fname = f"CDF_{metric}_{SENSOR}_thr{THRESH_PCT}_k3k7_vs_k5.png"
    local_path = f"/tmp/{fname}"
    plt.savefig(local_path, dpi=300)
    plt.close()
    os.system(f"rclone copy '{local_path}' '{RCLONE_DEST}'")
    print(f"✓ Uploaded: {fname}")


# ----------------- RUN -----------------
if __name__ == "__main__":
    plot_cdf("MS")  # melt start
    plot_cdf("FS")  # freeze start
