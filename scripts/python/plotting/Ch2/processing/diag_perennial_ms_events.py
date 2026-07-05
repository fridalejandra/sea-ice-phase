"""
Final Weddell-interior diagnostic, properly specified.

1. Build a STABLE perennial mask: pixels with Jul-Aug median SIC > 80%
   in >= 90% of all years 1979-2024 (not one year's median).
2. Over that mask, count static MS valid years per pixel (map + histogram).
3. For the interior pixels WITH detections, pull their raw SIC series in
   their detection years and characterize the events (real dips vs. noise).
"""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
MERGED = ROOT / "data/merged/smmr_yearly"
STATIC_MS = ROOT / "data/SMMR_phase/static/thr15_k5/MS"
YEARS = list(range(1979, 2025))

# ---------- Pass 1: stable perennial mask ----------
print("Pass 1: building stable perennial mask...")
count_high = None
count_valid = None
for y in YEARS:
    f = MERGED / f"SMMR_{y}.nc"
    if not f.exists():
        continue
    ds = xr.open_dataset(f)
    var = [v for v in ds.data_vars if "ICECON" in v.upper()][0]
    da = ds[var].sel(time=slice(f"{y}-02-01", f"{y}-02-28"))
    w = da.values.astype(float)
    ds.close()
    w = np.where(w <= 1.0, w, np.nan)
    med = np.nanmedian(w, axis=0)
    high = (med > 0.15)   # survives summer minimum
    ok = np.isfinite(med)
    count_high = high.astype(np.int16) if count_high is None else count_high + high
    count_valid = ok.astype(np.int16) if count_valid is None else count_valid + ok

frac_high = count_high / np.maximum(count_valid, 1)
perennial = (frac_high >= 0.90) & (count_valid >= 40)
print(f"stable perennial pixels (Feb median>15% in >=90% of years): "
      f"{int(perennial.sum())}")

# ---------- Pass 2: MS valid-year counts over perennial mask ----------
print("\nPass 2: static MS detections over perennial pixels...")
validcount = np.zeros(perennial.shape, dtype=np.int16)
det_years = {}   # pixel -> list of years with a date
for y in YEARS:
    f = STATIC_MS / f"MS_{y}.nc"
    if not f.exists():
        continue
    ms = xr.open_dataset(f)["MS"].values
    hit = np.isfinite(ms) & perennial
    validcount += hit.astype(np.int16)
    for py, px in zip(*np.where(hit)):
        det_years.setdefault((py, px), []).append((y, float(ms[py, px])))

n_det = int((validcount > 0).sum())
print(f"perennial pixels with >=1 static MS date ever: {n_det} "
      f"of {int(perennial.sum())}")
if n_det:
    counts = validcount[validcount > 0]
    print(f"valid-year counts among them: median={np.median(counts):.0f}, "
          f"max={counts.max()}, "
          f"n with >=5 detections: {(counts>=5).sum()}")
    # year histogram of all detections
    allyears = [y for v in det_years.values() for (y, _) in v]
    hist = {y: allyears.count(y) for y in sorted(set(allyears))}
    print("detections per year:", hist)

# ---------- Pass 3: pull SIC series for 3 example detection events ----------
print("\nPass 3: example events (pixel, year, MS date, SIC around date)...")
examples = sorted(det_years.items(), key=lambda kv: -len(kv[1]))[:3]
fig, axes = plt.subplots(len(examples), 1, figsize=(12, 3*len(examples)),
                         dpi=140, squeeze=False)
for ax, ((py, px), events) in zip(axes[:, 0], examples):
    y, doy = events[0]
    # MS window wraps: load year y and y+1 if available
    series, tt = [], []
    for yy in (y, y + 1):
        f = MERGED / f"SMMR_{yy}.nc"
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        var = [v for v in ds.data_vars if "ICECON" in v.upper()][0]
        s = ds[var][:, py, px].values.astype(float)
        t = ds["time"].values
        ds.close()
        series.append(np.where(s <= 1.0, s, np.nan)); tt.append(t)
    s = np.concatenate(series); t = np.concatenate(tt)
    ax.plot(t, s, ".-", ms=3, lw=0.8)
    ax.axhline(0.15, color="red", lw=1, ls="--")
    ax.set_title(f"pixel(y={py},x={px}) detections={len(events)} "
                 f"first: year={y}, MS DOY={doy:.0f}")
    ax.set_ylim(-0.05, 1.05)
print("example pixels:", [(k, len(v), v[:3]) for k, v in examples])
out = ROOT / "results/Ch2_Figures/diag_perennial_ms_events.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved -> {out}")

# valid-count map
fig2, ax2 = plt.subplots(figsize=(7, 7), dpi=140)
m = np.where(perennial, validcount.astype(float), np.nan)
im = ax2.imshow(m, cmap="magma", vmin=0, vmax=15)
ax2.set_title("static MS valid-year count over TRUE perennial pixels (Feb>15%)")
plt.colorbar(im, ax=ax2, shrink=0.8)
out2 = ROOT / "results/Ch2_Figures/diag_perennial_ms_validcount_map.png"
fig2.savefig(out2, bbox_inches="tight")
print(f"saved -> {out2}")
