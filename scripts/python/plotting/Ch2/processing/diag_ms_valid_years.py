"""Diagnostic: per-pixel valid-year counts for MS, static vs dynamic,
pre/post-2016 split. Addresses: is Weddell-interior static MS shading
driven by post-2016 years?"""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
STATIC_MS = ROOT / "data/SMMR_phase/static/thr15_k5/MS"
DYN_MS = ROOT / "data/SMMR_phase/dynamic/k5_q70/MS"
SECT = ROOT / "data/canonical_sectors.nc"
YEARS = range(1979, 2025)
SPLIT = 2016  # post = >= 2016

ds = xr.open_dataset(SECT)
wed = (ds["sector_id"].values == 2) & ds["valid_ocean"].astype(bool).values
ds.close()

def counts(d, tag):
    pre = post = None
    n_pre = n_post = 0
    for y in YEARS:
        f = d / f"MS_{y}.nc"
        if not f.exists():
            continue
        da = xr.open_dataset(f)["MS"].values
        v = np.isfinite(da).astype(np.int16)
        if y < SPLIT:
            pre = v if pre is None else pre + v; n_pre += 1
        else:
            post = v if post is None else post + v; n_post += 1
    frac_pre, frac_post = pre / n_pre, post / n_post
    print(f"\n=== {tag} ===  (pre n={n_pre}, post n={n_post})")
    # Weddell pixels that are 'occasional': valid in >0 but <50% of all years
    tot = pre + post
    occ = wed & (tot > 0) & (tot < 0.5 * (n_pre + n_post))
    print(f"Weddell occasional-detection pixels: {occ.sum()}")
    if occ.sum() > 0:
        print(f"  mean valid-year frac PRE-2016 : {frac_pre[occ].mean():.3f}")
        print(f"  mean valid-year frac POST-2016: {frac_post[occ].mean():.3f}")
    return frac_pre, frac_post

fp_s, fq_s = counts(STATIC_MS, "MS static thr15_k5")
fp_d, fq_d = counts(DYN_MS, "MS dynamic k5_q70")

fig, ax = plt.subplots(2, 3, figsize=(15, 9), dpi=150)
for row, (fp, fq, t) in enumerate([(fp_s, fq_s, "static"), (fp_d, fq_d, "dynamic")]):
    for col, (arr, tt) in enumerate([(fp, "pre-2016 frac"), (fq, "post-2016 frac"),
                                     (fq - fp, "post minus pre")]):
        cmap, vlim = ("RdBu_r", (-1, 1)) if col == 2 else ("viridis", (0, 1))
        im = ax[row, col].imshow(arr, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax[row, col].set_title(f"MS {t}: {tt}"); ax[row, col].axis("off")
        plt.colorbar(im, ax=ax[row, col], shrink=0.7)
out = ROOT / "results/Ch2_Figures/diag_MS_valid_year_frac_prepost.png"
fig.savefig(out, bbox_inches="tight")
print(f"\nsaved -> {out}")
