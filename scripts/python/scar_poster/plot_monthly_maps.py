"""
plot_monthly_maps.py

Monthly pre/post-2016 difference maps for divergence, convergence, and sea ice
concentration, on the native EASE-Grid 2.0 South grid -- BEFORE any
sector/season aggregation.

WHY
The coupling tests collapse everything to 5 sectors x 4 seasons before testing.
That assumes each sector is spatially coherent enough for one relationship to
be meaningful. These maps check that assumption, and -- more importantly for
the poster -- show WHERE the atmosphere's leverage sits and whether it moved.

WHAT IT MAKES
For each variable and each calendar month:
    pre-2016 mean | post-2016 mean | difference (post - pre)
with stippling where the difference passes a per-cell screen.

THE SCREEN IS A SCREEN, NOT A TEST
Per-cell Welch t-test on YEARLY monthly means (n = number of years), not on
daily values -- daily cells are heavily autocorrelated and a daily-sample test
would be wildly overconfident. This shows WHERE to look. The defensible
statistics are the sector-level block-bootstrap tests.

MEMORY
Processes one month at a time rather than loading the full record, so this
runs comfortably on the 17k-timestep divergence file.
"""

import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

# ---------------- CONFIG ----------------
VARIABLES = {
    # label: (path, varname, colormap, units, negate)
    # negate=True flips the sign (convergence = -divergence)
    "divergence":  ("ice_divergence_daily_sh.nc", "divergence", "RdBu_r", "s^-1", False),
    "convergence": ("ice_divergence_daily_sh.nc", "divergence", "RdBu_r", "s^-1", True),
    "sic":         ("sic_bootstrap_on_ease_sh.nc", "sic", "RdBu_r", "fraction", False),
    "wind_stress": ("wind_stress_on_ease_sh.nc", "tau_mag", "YlOrRd", "Pa", False),
    "wind_curl":   ("wind_stress_curl_on_ease_sh.nc", "tau_curl", "RdBu_r", "Pa/m", False),
}

# EASE-Grid 2.0 South (EPSG:6932) = Lambert Azimuthal Equal-Area, lat_0=-90
EASE_CRS = ccrs.LambertAzimuthalEqualArea(
    central_latitude=-90.0, central_longitude=0.0
)
PLATE = ccrs.PlateCarree()

SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
MIN_YEARS_PER_PERIOD = 3

OUT_DIR = "maps"
MONTHS = [9]                   # September for a quick pass; range(1, 13) for all
DPI = 130
# -----------------------------------------


def load(path, var, negate=False):
    ds = xr.open_dataset(path)
    if var not in ds:
        raise KeyError(f"{var!r} not in {path}. Available: {list(ds.data_vars)}")
    da = ds[var]
    if negate:
        da = -da
    tname = "time" if "time" in da.dims else "valid_time"
    yrs = da[tname].dt.year
    da = da.sel({tname: ~yrs.isin(EXCLUDE_YEARS)})
    return da


def yearly_means_for_month(da, month):
    """-> (n_years, y, x) array of that month's mean for each year."""
    sub = da.sel(time=da["time"].dt.month == month)
    if sub.sizes["time"] == 0:
        return None
    return sub.groupby(sub["time"].dt.year).mean(dim="time").load()


def panel(fig, ax, field, title, cmap, vmax=None, x=None, y=None):
    if vmax is None:
        vmax = np.nanpercentile(np.abs(field), 98)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1e-9
    im = ax.pcolormesh(x, y, field, transform=EASE_CRS,
                       cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    ax.add_feature(cfeature.LAND, zorder=3, facecolor="0.85", edgecolor="none")
    ax.coastlines(resolution="50m", linewidth=0.4, zorder=4)
    ax.set_extent([-180, 180, -90, -50], crs=PLATE)
    ax.set_title(title, fontsize=10)
    return im


def make_maps(label, path, var, cmap, units, negate=False):
    print(f"\n=== {label} ===")
    da = load(path, var, negate=negate)
    x = da["x"].values
    y = da["y"].values
    os.makedirs(OUT_DIR, exist_ok=True)

    for m in MONTHS:
        ym = yearly_means_for_month(da, m)
        if ym is None:
            print(f"  [skip] month {m:02d}: no data")
            continue

        years = ym["year"].values
        post_m = years >= SPLIT_YEAR
        if post_m.sum() < MIN_YEARS_PER_PERIOD or (~post_m).sum() < MIN_YEARS_PER_PERIOD:
            print(f"  [skip] month {m:02d}: too few years "
                  f"(pre={(~post_m).sum()}, post={post_m.sum()})")
            continue

        pre = ym.values[~post_m]
        post = ym.values[post_m]
        pre_mean = np.nanmean(pre, axis=0)
        post_mean = np.nanmean(post, axis=0)
        diff = post_mean - pre_mean

        # per-cell screen on yearly means (n = years), NOT daily values
        with np.errstate(invalid="ignore"):
            _, pval = stats.ttest_ind(post, pre, axis=0,
                                      equal_var=False, nan_policy="omit")
        pval = np.ma.filled(np.ma.masked_invalid(pval), 1.0)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5.2),
                                 subplot_kw={"projection": EASE_CRS})

        # shared scale for the two means; independent scale for the difference
        mvmax = np.nanpercentile(np.abs(np.concatenate([pre_mean, post_mean])), 98)
        im0 = panel(fig, axes[0], pre_mean, f"pre-{SPLIT_YEAR} mean",
                    cmap, vmax=mvmax, x=x, y=y)
        panel(fig, axes[1], post_mean, f"post-{SPLIT_YEAR} mean",
              cmap, vmax=mvmax, x=x, y=y)
        im2 = panel(fig, axes[2], diff, "difference (post - pre)",
                    cmap, x=x, y=y)

        # stipple where the screen flags a difference
        sig = pval < 0.05
        yy, xx = np.meshgrid(y, x, indexing="ij")
        axes[2].scatter(xx[sig], yy[sig], s=0.12, c="k", alpha=0.45,
                        transform=EASE_CRS, zorder=5, linewidths=0)

        fig.colorbar(im0, ax=axes[:2], orientation="horizontal",
                     fraction=0.05, pad=0.06, label=f"{label} ({units})")
        fig.colorbar(im2, ax=axes[2], orientation="horizontal",
                     fraction=0.05, pad=0.06, label=f"Δ {label}")

        fig.suptitle(f"{label} — month {m:02d}   "
                     f"(pre n={(~post_m).sum()} yr, post n={post_m.sum()} yr; "
                     f"stipple = per-cell screen p<0.05, not the formal test)",
                     fontsize=11)
        out = os.path.join(OUT_DIR, f"{label}_month{m:02d}.png")
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


def main():
    for label, (path, var, cmap, units, negate) in VARIABLES.items():
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found")
            continue
        make_maps(label, path, var, cmap, units, negate=negate)
    print(f"\nAll maps in ./{OUT_DIR}/")
    print("Look for: does the divergence difference respect your sector "
          "boundaries or cut across them? Is the null spatially flat, or "
          "offsetting patches that cancel in the sector mean?")


if __name__ == "__main__":
    main()