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
def plot_cdf(metric,
             figsize=(3.5, 3.0),   # single-column default
             dpi=600,
             use_legend=True):     # set False if you prefer direct labels
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MultipleLocator
    import seaborn as sns

    # Color-blind safe (Okabe–Ito): blue / orange
    C35 = "#0072B2"   # k3–k5
    C75 = "#E69F00"   # k7–k5

    v35, v75 = diffs_for_metric(metric)
    v35_clip, v35_full = ecdf_data(v35, MAX_X)
    v75_clip, v75_full = ecdf_data(v75, MAX_X)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # HIGHER CONTRAST LINES + distinct styles (works in grayscale & for CVD)
    sns.ecdfplot(v35_clip, label="|Δ| (k3 − k5)", lw=2.5, ls="-",  color=C35, ax=ax)
    sns.ecdfplot(v75_clip, label="|Δ| (k7 − k5)", lw=2.5, ls="--", color=C75, ax=ax)

    # ---- Smart annotations (collision-aware) ----
    def ecdf_at(values, m):
        arr = np.asarray(values)
        return float(np.mean(arr <= m))

    dx = 0.02 * MAX_X
    min_gap_y = 0.08
    label_positions = {"left": [], "right": []}

    for m in MARKS:
        ax.axvline(m, ls="--", c="0.2", lw=1)  # darker for print contrast
        p35 = frac_within(v35_full, m)
        p75 = frac_within(v75_full, m)

        f35 = ecdf_at(v35_full, m)
        f75 = ecdf_at(v75_full, m)
        y0 = max(f35, f75) + 0.05
        y0 = max(0.05, min(0.95, y0))

        on_right = m > 0.85 * MAX_X
        side = "left" if on_right else "right"
        ha = "right" if on_right else "left"
        xt = m - dx if on_right else m + dx

        _, y0_axes = ax.transAxes.inverted().transform(ax.transData.transform([0, y0]))
        ys_axes = label_positions[side]
        while any(abs(y0_axes - yy) < min_gap_y for yy in ys_axes):
            y0_axes += min_gap_y
            if y0_axes > 0.95:
                y0_axes = max(0.05, max(f35, f75) - 0.05)
                break
        label_positions[side].append(y0_axes)
        _, y_text = ax.transData.inverted().transform(ax.transAxes.transform([0, y0_axes]))

        txt = f"{m} d:\n{p35:.1f}% (3–5)\n{p75:.1f}% (7–5)"
        ax.annotate(
            txt, xy=(m, max(f35, f75)), xytext=(xt, y_text),
            ha=ha, va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", lw=0.6),
            arrowprops=dict(arrowstyle="-", lw=0.8, color="0.3"),
        )

    # ---- AXES: readable, high-contrast, journal-friendly ----
    ax.set(xlim=(0, MAX_X), ylim=(0, 1.0))
    ax.set_xlabel("Absolute timing difference |Δ| (days)", fontsize=9.5)
    ax.set_ylabel("Fraction of pixels (CDF)", fontsize=9.5)

    # ticks: larger labels, consistent intervals; show Y as percentages
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))  # 0–1 → 0–100%
    ax.tick_params(axis="both", which="major", labelsize=9, length=4, width=1.1, color="0.2")
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.9, color="0.3")

    # grid: helpful but subdued; spines darker for contrast in print
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.75")
    for s in ax.spines.values():
        s.set_linewidth(1.2)
        s.set_color("0.2")

    # title slightly smaller than labels at final size (avoid crowding)
    ax.set_title(f"{SENSOR} • {metric} • thr={THRESH_PCT}%  (k3/k7 vs k5)", fontsize=9.5, pad=6)

    # legend: high-contrast frame; larger handlelength for visibility
    if use_legend:
        leg = ax.legend(title="Window comparison",
                        frameon=True, facecolor="white", edgecolor="0.2",
                        framealpha=1.0, fontsize=9, title_fontsize=9)
        for lh in leg.legend_handles:
            lh.set_linewidth(3.0)
    else:
        # Optional: direct line labels near right edge (good for accessibility)
        for line, txt, xpad in [
            (ax.lines[0], "|Δ| (k3 − k5)", -0.02*MAX_X),
            (ax.lines[1], "|Δ| (k7 − k5)", -0.02*MAX_X),
        ]:
            x_end = line.get_xdata()[-1]
            y_end = line.get_ydata()[-1]
            ax.text(x_end + xpad, y_end, txt, va="center", ha="right", fontsize=9)

    fig.tight_layout()

    # save + upload
    fname = f"CDF_{metric}_{SENSOR}_thr{THRESH_PCT}_k3k7_vs_k5.png"
    local_path = f"/tmp/{fname}"
    fig.savefig(local_path, dpi=dpi)
    plt.close(fig)
    os.system(f"rclone copy '{local_path}' '{RCLONE_DEST}'")
    print(f"✓ Uploaded: {fname}")

# ----------------- RUN -----------------
if __name__ == "__main__":
    plot_cdf("MS")  # melt start
    plot_cdf("FS")  # freeze start
