"""
Stage-zero diagnostic: does the raw daily SIC over the Weddell INTERIOR
ever actually cross below 15%, and how does that square with the static
MS detections there?

Outputs:
  1) Per-year console table: interior min SIC, # interior pixels with
     any day < 15%, # with a k=5-day run < 15% (what static MS requires),
     plus data-health columns (zeros, NaNs, raw value range).
  2) Map PNG: static MS valid-year counts over Weddell vs. independently
     recomputed "has a 5-day sub-15% run" counts. If these two maps
     disagree, the detection path has a bug; if they agree, the SIC
     record genuinely dips and ncview needs a second look.
"""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
STATIC_MS = ROOT / "data/SMMR_phase/static/thr15_k5/MS"
YEARS = range(1979, 2025)
K = 5
THR = 0.15
# MS window per methods: Aug 15 (DOY 227) through Feb 28 of next calendar
# year — within a single yearly file we can only check DOY >= 227 plus
# Jan 1–Feb 28 (DOY <= 59) of the SAME file; note this in interpretation.
MS_DOYS = lambda n: np.r_[0:59, 226:n]

sect = xr.open_dataset(ROOT / "data/canonical_sectors.nc")
wed = (sect["sector_id"].values == 2) & sect["valid_ocean"].astype(bool).values
sect.close()
# INTERIOR = Weddell eroded 8 px inward (away from edge/coast effects)
interior = binary_erosion(wed, iterations=8)
print(f"Weddell sector px: {wed.sum()}, interior px: {interior.sum()}")

def sub_thr_run(below: np.ndarray, k: int) -> np.ndarray:
    """below: (t,y,x) bool. Returns (y,x) bool: any k consecutive True."""
    run = np.zeros(below.shape[1:], dtype=bool)
    acc = np.zeros(below.shape[1:], dtype=np.int16)
    for t in range(below.shape[0]):
        acc = np.where(below[t], acc + 1, 0)
        run |= acc >= k
    return run

runcount = np.zeros(wed.shape, dtype=np.int16)   # indep. recomputation
print(f"\n{'year':>4} {'rawmax':>7} {'minSIC':>7} {'anyday<15%':>10} "
      f"{'run5<15%':>9} {'zeros%':>7} {'NaNs%':>6}")
for y in YEARS:
    f = ROOT / f"data/merged/smmr_yearly/SMMR_{y}.nc"
    if not f.exists():
        print(f"{y:>4}  MISSING FILE"); continue
    ds = xr.open_dataset(f)
    var = [v for v in ds.data_vars if "ICECON" in v.upper()][0]
    sic = ds[var].values.astype(float)
    ds.close()
    rawmax = np.nanmax(sic)
    if rawmax > 1.5:            # percent -> fraction
        sic = sic / 100.0
    # flag guard: anything outside [0,1] after scaling -> NaN
    sic = np.where((sic < 0) | (sic > 1.0), np.nan, sic)
    sel = MS_DOYS(sic.shape[0])
    s = sic[sel][:, :, :]
    idx = interior
    smin = np.nanmin(np.where(idx, s, np.nan))
    below = np.where(idx, s < THR, False)
    anyday = below.any(axis=0)
    run5 = sub_thr_run(below, K)
    runcount += run5.astype(np.int16)
    ssel = np.where(idx, s, np.nan)
    zpct = 100 * np.nansum(ssel == 0) / max(np.isfinite(ssel).sum(), 1)
    npct = 100 * np.isnan(np.where(idx, s, 0)).sum() / (s.shape[0] * idx.sum())
    print(f"{y:>4} {rawmax:>7.1f} {smin:>7.3f} {int(anyday.sum()):>10} "
          f"{int(run5.sum()):>9} {zpct:>7.2f} {npct:>6.1f}")

# --- Map comparison: detector's valid years vs recomputed run counts ---
validcount = np.zeros(wed.shape, dtype=np.int16)
for y in YEARS:
    f = STATIC_MS / f"MS_{y}.nc"
    if not f.exists():
        continue
    da = xr.open_dataset(f)["MS"].values
    validcount += np.isfinite(da).astype(np.int16)

fig, ax = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
for a, (arr, t) in zip(ax, [
        (np.where(wed, validcount, np.nan), "static MS valid-year count (detector)"),
        (np.where(wed, runcount, np.nan), "recomputed 5-day sub-15% run count\n(interior only)"),
        (np.where(interior, validcount - runcount, np.nan), "detector minus recomputed"),
]):
    cmap = "RdBu_r" if "minus" in t else "magma"
    v = 10 if "minus" in t else 46
    im = a.imshow(arr, cmap=cmap, vmin=-v if "minus" in t else 0, vmax=v)
    a.set_title(t, fontsize=9); a.axis("off")
    plt.colorbar(im, ax=a, shrink=0.7)
out = ROOT / "results/Ch2_Figures/diag_weddell_MS_contradiction.png"
fig.savefig(out, bbox_inches="tight")
print(f"\nsaved -> {out}")
