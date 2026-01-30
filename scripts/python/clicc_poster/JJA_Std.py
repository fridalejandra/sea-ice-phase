import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ============================================================
# PATHS / SETTINGS
# ============================================================
sic_file = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
sic_var  = "N07_ICECON"

outdir = "/user/geog/falejandraperez/sea-ice-phase/results/figures/clic_poster/"

# ---- CHOOSE YEARS HERE ----
YEARS = [2001, 2014, 2022]   # example: neutral, high, low
SEASON_MONTHS = [6, 7, 8]    # JJA only

cmap_main = "magma_r"

# ============================================================
# HELPERS
# ============================================================
def drop_feb29(da):
    return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))

def ensure_01(sic):
    return sic / 100.0 if float(sic.max()) > 1.5 else sic

def poster_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor("0.85")
    ax.set_aspect("equal")

    circle = Circle((0.5, 0.5), 0.5, transform=ax.transAxes,
                    facecolor="none", edgecolor="none")
    ax.add_patch(circle)
    for artist in ax.get_children():
        artist.set_clip_path(circle)

# ============================================================
# LOAD DATA
# ============================================================
ds = xr.open_dataset(sic_file, chunks={"time": 365})
sic = ensure_01(drop_feb29(ds[sic_var]))

# JJA mask
jja = sic.time.dt.month.isin(SEASON_MONTHS)

# ============================================================
# COMPUTE STD(SIC) FOR EACH YEAR
# ============================================================
std_maps = {}

for yr in YEARS:
    sic_y = sic.sel(time=str(yr)).where(jja, drop=True)
    sic_y = sic_y.where(sic_y > 0)  # ocean-only mask

    std_maps[yr] = sic_y.std("time", skipna=True).compute()

# ============================================================
# COMMON COLOR SCALE
# ============================================================
vmax = max(float(m.max()) for m in std_maps.values())

# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(
    1, len(YEARS),
    figsize=(5 * len(YEARS), 5),
    constrained_layout=True
)

if len(YEARS) == 1:
    axes = [axes]

for ax, yr in zip(axes, YEARS):
    im = std_maps[yr].plot(
        ax=ax,
        cmap=cmap_main,
        vmin=0,
        vmax=vmax,
        add_colorbar=False
    )
    poster_ax(ax)
    ax.set_title(f"JJA {yr}", fontsize=14, weight="bold")

cbar = fig.colorbar(im, ax=axes, shrink=0.85)
cbar.set_label("Std of daily SIC", fontsize=12)

plt.savefig(
    f"{outdir}/JJA_std_SIC_selected_years.png",
    dpi=400,
    bbox_inches="tight"
)
plt.close()

print("✓ JJA Std(SIC) maps complete")
