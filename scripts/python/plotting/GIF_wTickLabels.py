import os
from glob import glob

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.path import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ==============================
# CONFIG
# ==============================
INPUT_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/"
OUT_DIR   = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/figures/yearly_phase_maps/"
YEAR_START, YEAR_END = 1979, 2023
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
# COLORBARS (ticks & labels)
# ==============================
ADV_NORM = Normalize(vmin=32,  vmax=274)   # ~Feb–Sep
RET_NORM = Normalize(vmin=274, vmax=424)   # ~Oct–Mar (wrapped)

ADV_TICKS   = [32, 60, 91, 121, 152, 182, 213, 244]
ADV_LABELS  = ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

RET_TICKS   = [274, 305, 335, 366, 395, 424]
RET_LABELS  = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']

# colormap with NaNs -> white
cmap = plt.cm.viridis.copy()
cmap.set_bad("white")

# ==============================
# CIRCULAR BOUNDARY
# ==============================
theta = np.linspace(0, 2 * np.pi, 360)
circle_verts = np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5
circle_path = Path(circle_verts)


def compute_advance_retreat(ds, year):
    """Return (advance, retreat_wrapped) or (None, None) if missing."""
    adv_var = f"advance_{year}"
    ret_var = f"retreat_{year}"
    if adv_var not in ds or ret_var not in ds:
        return None, None

    advance = ds[adv_var]
    retreat = ds[ret_var]

    # keep advance within ~Feb–Sep to avoid early/late noise
    advance = advance.where(advance >= 32)
    advance = advance.where(advance <= 274)

    # wrap retreat: if < 100 (Jan/Feb), add 365 so Oct–Mar is monotonic
    retreat_wrapped = retreat.where(retreat >= 100, retreat + 365)

    return advance, retreat_wrapped


def plot_single_field(year, data, norm, ticks, labels, title, cbar_title, save_path):
    """Plot one South Polar map with a labeled colorbar; no lat/lon ticks."""
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.set_boundary(circle_path, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND, zorder=100, facecolor="gray")
    ax.coastlines(linewidth=0.4)

    mesh = ax.pcolormesh(
        data.x, data.y, data,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap, norm=norm, shading="auto"
    )

    # keep map face clean
    ax.set_xticks([])
    ax.set_yticks([])

    # colorbar with custom month labels
    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    cbar.set_label(cbar_title, fontsize=10, labelpad=3)

    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    files = sorted(glob(os.path.join(INPUT_DIR, "seaice_phases_SMMR_*.nc")))
    if not files:
        print("No NetCDF files found. Check INPUT_DIR.")
        return

    for f in files:
        year_str = os.path.basename(f).split("_")[-1].split(".")[0]
        if not year_str.isdigit():
            continue
        year = int(year_str)
        if not (YEAR_START <= year <= YEAR_END):
            continue

        try:
            ds = xr.open_dataset(f)
        except Exception as e:
            print(f"⚠️ Failed to open {f}: {e}")
            continue

        advance, retreat_wrapped = compute_advance_retreat(ds, year)
        ds.close()

        if advance is None or retreat_wrapped is None:
            print(f"⚠️ Skipping {year}: required variables missing")
            continue

        # If everything is NaN, skip to avoid blank images
        if np.all(np.isnan(advance.values)) and np.all(np.isnan(retreat_wrapped.values)):
            print(f"⚠️ Skipping {year}: all-NaN fields")
            continue

        # Shared mask so the visual coverage matches
        shared_mask = advance.isnull() | retreat_wrapped.isnull()
        adv_plot = advance.where(~shared_mask)
        ret_plot = retreat_wrapped.where(~shared_mask)

        # --- ADVANCE FIG ---
        adv_path = os.path.join(OUT_DIR, f"advance_{year}.png")
        plot_single_field(
            year=year,
            data=adv_plot,
            norm=ADV_NORM,
            ticks=ADV_TICKS,
            labels=ADV_LABELS,
            title=f"Sea Ice Advance {year}",
            cbar_title="ADVANCE (DOY)",
            save_path=adv_path
        )

        # --- RETREAT FIG ---
        ret_path = os.path.join(OUT_DIR, f"retreat_{year}.png")
        plot_single_field(
            year=year,
            data=ret_plot,
            norm=RET_NORM,
            ticks=RET_TICKS,
            labels=RET_LABELS,
            title=f"Sea Ice Retreat {year}",
            cbar_title="RETREAT (DOY)",
            save_path=ret_path
        )

        print(f"✅ Saved: {adv_path}")
        print(f"✅ Saved: {ret_path}")


if __name__ == "__main__":
    main()
