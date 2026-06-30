import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from scipy.ndimage import uniform_filter
from scipy.interpolate import NearestNDInterpolator
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

DATA_ROOT   = Path("/user/geog/falejandraperez/sea-ice-phase/data")
ANOM_DIR    = DATA_ROOT / "anomalies" / "SMMR"
SECTOR_FILE = DATA_ROOT / "canonical_sectors.nc"
OUT         = Path("/tmp/Fig09_prepost_trends.png")
SMOOTH      = 9
PVAL_THRESH = 0.01
AUG15_DOY   = 227
PRE         = slice(1979, 2015)
POST        = slice(2016, 2024)

# ── Slope + pval ──────────────────────────────────────────────────────────────
def compute_slope_pval(da):
    years = da["year"].values.astype(float)
    def slope(y):
        m = np.isfinite(y)
        if m.sum() < 3: return np.nan
        return np.polyfit(years[m], y[m], 1)[0]
    def pval(y):
        m = np.isfinite(y)
        if m.sum() < 3: return np.nan
        return stats.linregress(years[m], y[m]).pvalue
    sl = xr.apply_ufunc(slope, da, input_core_dims=[["year"]],
                         vectorize=True, output_dtypes=[float])
    pv = xr.apply_ufunc(pval,  da, input_core_dims=[["year"]],
                         vectorize=True, output_dtypes=[float])
    return sl, pv

# ── Duration from raw files ───────────────────────────────────────────────────
def load_phase_year(base, phase, year):
    fpath = base / f"{phase}_{year}.nc"
    if not fpath.exists(): return None
    ds = xr.open_dataset(fpath)
    da = ds[phase].load(); ds.close()
    return da.transpose("y", "x")

def build_duration(fs_base, ms_base):
    arrays, years = [], []
    for y in range(1979, 2025):
        fs = load_phase_year(fs_base, "FS", y)
        ms = load_phase_year(ms_base, "MS", y)
        if fs is None or ms is None: continue
        ms_cont = xr.where(ms < AUG15_DOY, ms + 365, ms)
        dur = ms_cont - fs
        arrays.append(dur.expand_dims(year=[y]))
        years.append(y)
    return xr.concat(arrays, dim="year").assign_coords(year=("year", years))

# ── Load anomalies ────────────────────────────────────────────────────────────
def load_anom(fname):
    ds = xr.open_dataset(ANOM_DIR/fname, decode_times=False)
    da = ds[list(ds.data_vars)[0]].load(); ds.close()
    return da

print("Loading data...")
fs_anom = load_anom("FS_dynamic_k5_q70_anomalies.nc")
ms_anom = load_anom("MS_dynamic_k5_q70_anomalies.nc")
dur_anom = build_duration(
    DATA_ROOT/"SMMR_phase/dynamic/k5_q70/FS",
    DATA_ROOT/"SMMR_phase/dynamic/k5_q70/MS"
)

# ── Active mask ───────────────────────────────────────────────────────────────
ds_mask     = xr.open_dataset(SECTOR_FILE)
valid_ocean = ds_mask["valid_ocean"].astype(bool)
ds_mask.close()

n        = float(fs_anom.sizes["year"])
fs_active = (fs_anom.notnull().sum("year")/n >= 0.8) & valid_ocean
ms_active = (ms_anom.notnull().sum("year")/n >= 0.8) & valid_ocean
du_active = fs_active & ms_active

# ── Compute pre/post slopes ───────────────────────────────────────────────────
print("Computing slopes...")
results = {}
for phase, anom, active in [
    ("FS",  fs_anom,  fs_active),
    ("MS",  ms_anom,  ms_active),
    ("DUR", dur_anom, du_active),
]:
    for period, slc in [("pre", PRE), ("post", POST)]:
        sub = anom.sel(year=slc).where(active)
        sl, pv = compute_slope_pval(sub)
        results[f"{phase}_{period}_sl"] = sl.where(active)
        results[f"{phase}_{period}_pv"] = pv.where(active)
        med = float(np.nanmedian(sl.values))
        print(f"  {phase} {period}: median={med:.3f}")

# ── Smooth ────────────────────────────────────────────────────────────────────
def smooth(da, size=SMOOTH):
    vals = da.values.copy()
    mask = np.isfinite(vals)
    if mask.any() and (~mask).any():
        yy, xx = np.mgrid[0:vals.shape[0], 0:vals.shape[1]]
        interp = NearestNDInterpolator(
            list(zip(yy[mask], xx[mask])), vals[mask])
        vals[~mask] = interp(yy[~mask], xx[~mask])
    sm = uniform_filter(vals, size=size)
    sm[~mask] = np.nan
    return da.copy(data=sm)

for key in [k for k in results if k.endswith("_sl")]:
    results[key] = smooth(results[key])

# ── Discrete colormap ─────────────────────────────────────────────────────────
def make_cmap(vlim, step=0.25):
    bounds = np.arange(-vlim, vlim + step, step)
    n = len(bounds) - 1
    colors = [plt.cm.RdYlBu_r(i/(n-1)) for i in range(n)]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

# ── Lat/lon labels ────────────────────────────────────────────────────────────
def add_latlon_labels(ax):
    pc = ccrs.PlateCarree()
    for lat in [-60, -70, -80]:
        ax.text(-45, lat, f"{abs(lat)}°S", transform=pc,
                fontsize=6, ha="right", va="center", color="0.3", zorder=10)
    for lon, label, ha, va in [
        (0, "0°", "center", "bottom"),
        (90, "90°E", "left", "center"),
        (180, "180°", "center", "top"),
        (-90, "90°W", "right", "center"),
    ]:
        ax.text(lon, -52, label, transform=pc,
                fontsize=6, ha=ha, va=va, color="0.3", zorder=10)

# ── Figure ────────────────────────────────────────────────────────────────────
proj = ccrs.SouthPolarStereo()
pc   = ccrs.PlateCarree()

fig, axes = plt.subplots(3, 2, figsize=(10, 13),
                          subplot_kw={"projection": proj})
fig.suptitle("Linear trend in phase timing — Dynamic method (days/year)\n"
             "Left: 1979–2015 (pre-2016)    Right: 2016–2024 (post-2016)\n"
             "White contour = p < 0.01",
             fontsize=10, y=0.99)

row_specs = [
    ("FS",  4, "Freeze Start (days/year)", "Later",  "Earlier"),
    ("MS",  4, "Melt Start (days/year)",   "Later",  "Earlier"),
    ("DUR", 4, "Duration (days/year)",     "Longer", "Shorter"),
]

col_titles = ["(pre-2016) 1979–2015", "(post-2016) 2016–2024"]

panels = [
    (axes[0,0], "FS_pre_sl",  "FS_pre_pv",  "(a) FS pre-2016"),
    (axes[0,1], "FS_post_sl", "FS_post_pv", "(b) FS post-2016"),
    (axes[1,0], "MS_pre_sl",  "MS_pre_pv",  "(c) MS pre-2016"),
    (axes[1,1], "MS_post_sl", "MS_post_pv", "(d) MS post-2016"),
    (axes[2,0], "DUR_pre_sl", "DUR_pre_pv", "(e) Duration pre-2016"),
    (axes[2,1], "DUR_post_sl","DUR_post_pv","(f) Duration post-2016"),
]

ims = [None, None, None]

for idx, (ax, sl_key, pv_key, title) in enumerate(panels):
    row  = idx // 2
    phase, vlim, cb_label, top_label, bot_label = row_specs[row]
    sl = results[sl_key]
    pv = results[pv_key]

    cmap, norm, bounds = make_cmap(vlim)

    ax.set_extent([-180, 180, -90, -50], pc)
    ax.add_feature(cfeature.LAND, facecolor="0.75", edgecolor="none", zorder=3)
    ax.set_facecolor("white")

    gl = ax.gridlines(draw_labels=False, linewidth=0.4, color="0.6",
                      alpha=0.5, linestyle="--", zorder=4)
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 45))
    gl.ylocator = mticker.FixedLocator([-80, -70, -60])

    im = ax.contourf(sl["x"], sl["y"], sl,
                     levels=bounds, transform=proj,
                     cmap=cmap, norm=norm, extend="both", zorder=1)

    try:
        ax.contour(pv["x"], pv["y"], pv.values,
                   levels=[PVAL_THRESH], transform=proj,
                   colors="white", linewidths=0.8, zorder=5)
    except Exception:
        pass

    if ims[row] is None:
        ims[row] = im

    add_latlon_labels(ax)
    ax.set_title(title, fontsize=9, fontweight="bold")

# ── Colorbars ─────────────────────────────────────────────────────────────────
fig.subplots_adjust(left=0.02, right=0.85, top=0.94,
                    bottom=0.03, wspace=0.08, hspace=0.12)

for row, (phase, vlim, cb_label, top_label, bot_label) in enumerate(row_specs):
    ax_r = axes[row, 1]
    pos  = ax_r.get_position()
    cbar_ax = fig.add_axes([0.87, pos.y0, 0.018, pos.height])
    ticks = np.arange(-vlim, vlim + 1, 1).astype(int)
    cb = fig.colorbar(ims[row], cax=cbar_ax, orientation="vertical",
                      ticks=ticks)
    cb.ax.set_yticklabels([str(int(t)) for t in ticks], fontsize=7)
    cb.ax.set_ylabel(cb_label, fontsize=8, labelpad=8)
    # Later/Earlier labels as rotated text to right of colorbar
    x_right = 0.87 + 0.018 + 0.03
    fig.text(x_right, pos.y1, top_label, fontsize=7,
             ha="left", va="top", color="0.4", rotation=270)
    fig.text(x_right, pos.y0, bot_label, fontsize=7,
             ha="left", va="bottom", color="0.4", rotation=270)

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved → {OUT}")
