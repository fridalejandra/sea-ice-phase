import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

REPO = "/user/geog/falejandraperez/sea-ice-phase"
BLOCK_SIZES = [75, 100, 150]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
OUT_DIR = f"{REPO}/results/ch4/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def load_all():
    ds = {}
    for km in BLOCK_SIZES:
        path = f"{REPO}/results/ch4/derived_nc/spatial_coupling_block{km}km.nc"
        ds[km] = xr.open_dataset(path)
        print(f"loaded {km} km: beta3 shape {ds[km]['beta3'].shape}, finite count {int(np.isfinite(ds[km]['beta3'].values).sum())}")
    return ds

def plot_season(ds, season):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    all_vals = np.concatenate([ds[km]["beta3"].sel(season=season).values.flatten() for km in BLOCK_SIZES])
    finite = all_vals[np.isfinite(all_vals)]
    vmax = np.nanpercentile(np.abs(finite), 95) if finite.size else 1.0
    for ax, km in zip(axes, BLOCK_SIZES):
        field = ds[km]["beta3"].sel(season=season)
        im = ax.pcolormesh(field.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        n_fin = int(np.isfinite(field.values).sum())
        ax.set_title(f"{km} km  (n={n_fin})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"beta3 (post-2016 change in wind-divergence sensitivity) - {season}")
    fig.colorbar(im, ax=axes, shrink=0.7, label="beta3")
    out = f"{OUT_DIR}/beta3_blocksize_compare_{season}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")

def summary_stats(ds):
    print("\n=== summary: beta3 by block size and season ===")
    for km in BLOCK_SIZES:
        for season in SEASONS:
            v = ds[km]["beta3"].sel(season=season).values
            fin = v[np.isfinite(v)]
            if fin.size == 0:
                print(f"  {km}km {season}: no finite values")
                continue
            print(f"  {km}km {season}: n={fin.size:5d}  mean={np.mean(fin):+.4e}  std={np.std(fin):.4e}  |mean|/std={abs(np.mean(fin))/np.std(fin):.3f}")

if __name__ == "__main__":
    ds = load_all()
    summary_stats(ds)
    for season in SEASONS:
        plot_season(ds, season)
    print("\nDone.")
