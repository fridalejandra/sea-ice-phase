import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# SETTINGS
# -----------------------
sic_file = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
sic_var  = "N07_ICECON"

pre_period  = ("1979-01-01", "2016-12-31")
post_period = ("2017-01-01", "2024-12-31")

seasons = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

outdir = Path("/user/geog/falejandraperez/sea-ice-phase/results/figures/clic_poster")
outdir.mkdir(parents=True, exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
ds = xr.open_dataset(sic_file)
sic = ds[sic_var]

# ensure 0–1
if float(sic.max()) > 1.5:
    sic = sic / 100.0

# drop Feb 29
sic = sic.sel(time=~((sic.time.dt.month == 2) & (sic.time.dt.day == 29)))

sic_pre  = sic.sel(time=slice(*pre_period))
sic_post = sic.sel(time=slice(*post_period))

# -----------------------
# LOOP OVER SEASONS
# -----------------------
for season, months in seasons.items():

    pre_season  = sic_pre.sel(time=sic_pre.time.dt.month.isin(months))
    post_season = sic_post.sel(time=sic_post.time.dt.month.isin(months))

    std_pre  = pre_season.std("time", skipna=True)
    std_post = post_season.std("time", skipna=True)
    std_diff = std_post - std_pre

    vmax = np.nanpercentile(
        xr.concat([std_pre, std_post], dim="z"), 99
    )
    vmax_diff = np.nanpercentile(np.abs(std_diff), 99)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    std_pre.plot(ax=axes[0],
             cmap="magma_r",
             vmin=0,
             vmax=vmax,
             robust=False)
    axes[0].set_title(f"{season} Std(SIC) Pre")

    std_post.plot(ax=axes[1],
                  cmap="magma_r",
                  vmin=0,
                  vmax=vmax,
                  robust=False)
    axes[1].set_title(f"{season} Std(SIC) Post")

    std_diff.plot(ax=axes[2],
                  cmap="RdBu_r",
                  vmin=-vmax_diff,
                  vmax=vmax_diff,
                  robust=False)
    axes[2].set_title(f"{season} Post − Pre")

    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.savefig(outdir / f"SIC_std_{season}_pre_post_diff.png", dpi=300)
    plt.close()

    print(f"Saved {season}")
