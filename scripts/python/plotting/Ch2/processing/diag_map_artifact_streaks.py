"""Map multi-day exactly-0 streaks inside the solid winter pack,
across several years. Fingerprint tells us where in the provenance
chain the artifacts originate. Also dumps (y,x,date) triplets for
manual ncview verification."""
from pathlib import Path
import numpy as np, xarray as xr, matplotlib.pyplot as plt

ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
YEARS = [1985, 1991, 1995, 2005, 2015, 2023]
MINRUN = 4

fig, axes = plt.subplots(2, 3, figsize=(16, 11), dpi=140)
for ax, year in zip(axes.flat, YEARS):
    f = ROOT / f"data/merged/smmr_yearly/SMMR_{year}.nc"
    if not f.exists():
        ax.set_title(f"{year}: missing"); ax.axis("off"); continue
    ds = xr.open_dataset(f)
    var = [v for v in ds.data_vars if "ICECON" in v.upper()][0]
    da = ds[var].sel(time=slice(f"{year}-07-01", f"{year}-08-31"))
    times = da["time"].values
    w = da.values.astype(float); ds.close()
    land = w[0] > 1.15
    w = np.where(w <= 1.0, w, np.nan)
    solid = (np.nanmedian(w, axis=0) > 0.8) & ~land
    z = (w == 0) & solid[None]
    acc = np.zeros(z.shape[1:], np.int16)
    longest = np.zeros(z.shape[1:], np.int16)
    for t in range(z.shape[0]):
        acc = np.where(z[t], acc + 1, 0)
        longest = np.maximum(longest, acc)
    m = np.where(solid, longest.astype(float), np.nan)
    im = ax.imshow(m, cmap="inferno", vmin=0, vmax=8)
    ax.set_title(f"{year}: longest 0-streak (Jul–Aug), solid pack")
    ax.axis("off"); plt.colorbar(im, ax=ax, shrink=0.7)
    # triplets for manual ncview check (first 3 pixels with runs>=MINRUN)
    ys, xs = np.where(longest >= MINRUN)
    for py, px in list(zip(ys, xs))[:3]:
        col = z[:, py, px]
        d0 = times[np.argmax(col)]
        print(f"{year}: pixel(y={py}, x={px}) streak>= {MINRUN}d starting ~{str(d0)[:10]}")
out = ROOT / "results/Ch2_Figures/diag_artifact_streak_maps.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved -> {out}")
