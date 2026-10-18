import numpy as np, xarray as xr, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, LinearSegmentedColormap, to_rgb
from matplotlib import font_manager

R = "/user/geog/falejandraperez/sea-ice-phase/results"     # edit if running on laptop
DRIFT_NC = f"{R}/NSIDC0116/NSIDC0116_uv_season_year.nc"
WIND_NC  = f"{R}/ERA5/ERA5_uv_season_year.nc"
SEC = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"
LAND = "/user/geog/falejandraperez/sea-ice-phase/data/landmask_sic.nc"
OUT = f"{R}/wind_drift_norm/drift_wind_vectors"
PRE, POST = (1992, 2015), (2016, 2024)
SEASONS = ["JJA", "SON"]
COLOR, ARROWS = "drift", "wind"
STEP = 8
MIN_PRE, MIN_YEARS = 0.03, 15
VMAX = 3.0                       # colour limit (cm/s for drift, m/s for wind)
LAND_COLOR = "#3b3b3b"
PANEL_TITLES = True
MARGIN = 0.04                    # fraction of data extent added around the pack

for f in ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"]:
    if any(f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
print("font:", plt.rcParams["font.family"])
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})

# ---- cblind diverging 'bird' as a continuous colormap, blue (negative) -> white -> red (positive)
try:
    import cblind as cb
    cols, _ = cb.Colorplots().bird(9)
    cols = [to_rgb(c) for c in cols]
    if cols[0][0] > cols[-1][0]:          # first colour is red -> reverse so blue is at the bottom
        cols = cols[::-1]
    cmap = LinearSegmentedColormap.from_list("bird", cols)
    print("cmap: cblind bird")
except Exception as e:
    cmap = "RdBu_r"; print(f"cblind unavailable ({e}); using RdBu_r")

def load(nc):
    d = xr.open_dataset(nc)
    pre, post = d.sel(year=slice(*PRE)), d.sel(year=slice(*POST))
    valid = ((pre["speed"].mean("year") > MIN_PRE) & (np.isfinite(pre["speed"]).sum("year") >= MIN_YEARS)
             & (np.isfinite(post["speed"]).sum("year") >= 5))
    ch = {k: (post[k].mean("year") - pre[k].mean("year")) for k in ("u", "v", "speed")}
    return d, ch, valid

dd, dch, dvalid = load(DRIFT_NC); wd, wch, wvalid = load(WIND_NC)
x, y = dd.x.values, dd.y.values
X, Y = np.meshgrid(x, y)
sector = xr.open_dataset(SEC)["sector_id"]
try:
    lm = xr.open_dataset(LAND); v = [k for k in lm.data_vars if lm[k].ndim == 2][0]
    land = (lm[v].values != 0) & np.isfinite(lm[v].values)
except Exception: land = None

col = dch if COLOR == "drift" else wch; arr = dch if ARROWS == "drift" else wch
col_scale, col_unit = (100.0, "cm s$^{-1}$") if COLOR == "drift" else (1.0, "m s$^{-1}$")
arr_scale, arr_unit = (100.0, "cm s$^{-1}$") if ARROWS == "drift" else (1.0, "m s$^{-1}$")
key_len = 1.0 if ARROWS == "drift" else 0.5

# ---- axis limits from the data, not a fixed crop
anyvalid = np.zeros_like(X, bool)
for s in SEASONS: anyvalid |= dvalid.sel(season=s).values
if land is not None: anyvalid |= land
xs, ys = X[anyvalid], Y[anyvalid]
dx, dy = (xs.max() - xs.min()) * MARGIN, (ys.max() - ys.min()) * MARGIN
xlim, ylim = (xs.min() - dx, xs.max() + dx), (ys.min() - dy, ys.max() + dy)

fig, axes = plt.subplots(1, len(SEASONS), figsize=(5.6 * len(SEASONS), 6.2), dpi=200)
fig.patch.set_alpha(0)
norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0, vmax=VMAX)
sl = (slice(None, None, STEP), slice(None, None, STEP))
for ax, s in zip(np.atleast_1d(axes), SEASONS):
    ax.patch.set_alpha(0)
    C = (col["speed"].sel(season=s) * col_scale).where(dvalid.sel(season=s)).values
    m = ax.pcolormesh(x, y, C, cmap=cmap, norm=norm, shading="auto", zorder=1, rasterized=True)
    if land is not None:
        ax.pcolormesh(x, y, np.where(land, 1.0, np.nan), cmap=ListedColormap([LAND_COLOR]), shading="auto", zorder=2)
    ax.contour(x, y, np.nan_to_num(sector.values, nan=0), levels=[1.5, 2.5, 3.5, 4.5], colors="#555", linewidths=0.5, zorder=3)
    U = (arr["u"].sel(season=s) * arr_scale).where(dvalid.sel(season=s)).values
    V = (arr["v"].sel(season=s) * arr_scale).where(dvalid.sel(season=s)).values
    q = ax.quiver(X[sl], Y[sl], U[sl], V[sl], color="#111", scale_units="xy", angles="xy",
                  scale=key_len / (STEP * 25000 * 0.9), width=0.0028, headwidth=3.5, headlength=4.5, zorder=4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    if PANEL_TITLES:
        ax.set_title({"JJA": "Winter (JJA)", "SON": "Spring (SON)"}.get(s, s))
ax.quiverkey(q, 0.88, 0.04, key_len, f"{key_len:g} {arr_unit}", labelpos="E", coordinates="axes", fontproperties={"size": 9})
cb_ = fig.colorbar(m, ax=axes, orientation="horizontal", fraction=0.045, pad=0.03, extend="both")
cb_.set_label(f"{'Ice-drift' if COLOR=='drift' else 'Wind'} speed change, {POST[0]}–{POST[1]} vs {PRE[0]}–{PRE[1]} ({col_unit})")
cb_.outline.set_visible(False)
plt.savefig(f"{OUT}_{COLOR}color_{ARROWS}arrows.png", bbox_inches="tight", transparent=True)
plt.savefig(f"{OUT}_{COLOR}color_{ARROWS}arrows.pdf", bbox_inches="tight", transparent=True)
print("saved", f"{OUT}_{COLOR}color_{ARROWS}arrows.png")
