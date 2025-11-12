#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare dynamic-threshold FS/MS (k=5) against static 15% baseline.

Outputs in OUT_DIR:
  maps/:        mean/std/q95 of |Δ| for each scheme+tag (FS, MS)
  cdf/:         CDF(|Δ|) for each scheme+tag (FS, MS)
  box/:         Sectoral boxplots of |Δ| (FS, MS)
  summary.csv:  circumpolar + sector stats (median, p95, mean, N)

Revisions:
- Compact matplotlib rcParams; no oversized fonts
- bbox_inches='tight' and small padding to avoid cutoff
- Robust latitude orientation: uses 'lat' coord if available, else safe flip
- Boxplot x-tick labels rotated to prevent overlap
- Slightly smaller figures; consistent DPI
"""

import os, re, glob, json
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# ================= CONFIG (EDIT) =================
SENSOR = "SMMR"

# Static 15% (k=5) outputs from your slope+H run
STATIC_ROOT = f"/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase"

# Dynamic outputs written by run_dynamic_thresholds_staticSlope.py
DYN_ROOT = f"/user/geog/falejandraperez/sea-ice-phase/results/static_v2_slopeH/dynamic"

# Schemes you want to compare (will auto-discover param tags under each)
SCHEMES = ["mu_sigma_k5", "quantile_k5"]  # add others if present

# Canonical sectors file (for masks/labels)
CANONICAL = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"

# Output dir for figures/tables
OUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/results/Ch2_Figures/slope_dynamic_vs_static15"

# Limit to these years (optional); set to None for all common years
YEARS_LIMIT = None  # e.g., [1984,1992,2007,2016,2023]

# Plot bounds
PERIOD = 366
MAX_X  = 30     # cap for CDF/box x-axis in days

# rclone upload (optional)
RCLONE = dict(enabled=True, remote="gdrive",
              dst_dir="sea-ice-phase/results/Ch2_Figures/slope_dynamic_vs_static15",
              extra_flags=["--transfers=8","--checkers=8","--fast-list"],
              dry_run=False)

# =============== PLOTTING DEFAULTS ===============
plt.rcParams.update({
    "figure.constrained_layout.use": True,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.titlepad": 4,
})

# =============== HELPERS ===============
year_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    m = year_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def rclone_copy(local_path):
    if not RCLONE.get("enabled", False):
        return
    import subprocess
    dst = f"{RCLONE['remote']}:{RCLONE['dst_dir']}"
    cmd = ["rclone","copy",str(local_path),dst] + RCLONE.get("extra_flags",[])
    if RCLONE.get("dry_run"):
        cmd.insert(1,"--dry-run")
    print("[rclone]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!! rclone failed: {e}")

def load_static_dict(metric, k=5, thr_pct=15):
    subdir = f"{metric}_thr{thr_pct}_k{k}"
    folder = os.path.join(STATIC_ROOT, subdir)
    files  = sorted(glob.glob(os.path.join(folder, f"{metric}_*.nc")))
    d = {}
    for f in files:
        y = parse_year(f)
        if y is None:
            continue
        with xr.open_dataset(f) as ds:
            d[y] = ds[metric].load()
    if not d:
        raise FileNotFoundError(f"No static files in {folder}")
    return d

def list_dyn_paramtags(scheme_dir, metric):
    # e.g., .../mu_sigma_k5/FS/<tag>/FS_YYYY.nc
    base = os.path.join(scheme_dir, metric)
    tags = []
    for p in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(p) and glob.glob(os.path.join(p, f"{metric}_*.nc")):
            tags.append(os.path.basename(p))
    return tags

def load_dyn_dict(scheme_dir, metric, tag):
    folder = os.path.join(scheme_dir, metric, tag)
    files  = sorted(glob.glob(os.path.join(folder, f"{metric}_*.nc")))
    d = {}
    for f in files:
        y = parse_year(f)
        if y is None:
            continue
        with xr.open_dataset(f) as ds:
            d[y] = ds[metric].load()
    if not d:
        raise FileNotFoundError(f"No dynamic files in {folder}")
    return d

def align_years(dicts):
    common = set.intersection(*[set(d.keys()) for d in dicts])
    years = sorted(common)
    if YEARS_LIMIT:
        years = [y for y in years if y in YEARS_LIMIT]
    if not years:
        raise ValueError("No overlapping years after alignment/limit.")
    return years

def stack_years(d, years, name):
    arrs = [d[y].expand_dims(year=[y]) for y in years]
    return xr.concat(arrs, dim="year").rename(name)

def wrapped_abs(a, b, period=PERIOD):
    # circular absolute difference mapped to [0, period/2]
    return np.abs(((a - b + period//2) % period) - (period//2))

def ecdf(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0.0]), np.array([0.0])
    x.sort()
    y = np.linspace(0, 1, x.size, endpoint=False)
    return x, y

def to_plot_array(da, default_flip=True):
    """Return a 2D array oriented with *southern latitudes at the bottom*.
    Uses lat coordinate if present; otherwise flips by default."""
    arr = np.squeeze(da.values)
    # Coordinate-aware flip if possible
    try:
        # look for a latitude coord on either of the two spatial dims
        latname = None
        for cand in ["lat", "latitude", "y"]:
            if cand in da.coords and da[cand].ndim == 1 and da[cand].size == arr.shape[0]:
                latname = cand
                break
        if latname is not None:
            lats = da[latname].values
            # If lats decrease from south->north (i.e., first row is north), flip
            if lats[0] > lats[-1]:
                arr = np.flipud(arr)
            # else already south->north, leave as-is
        else:
            if default_flip:
                arr = np.flipud(arr)
    except Exception:
        if default_flip:
            arr = np.flipud(arr)
    return arr

# =============== CORE ===============
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    dirs = {k: Path(OUT_DIR)/k for k in ["maps","cdf","box"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    cano = xr.open_dataset(CANONICAL).load()
    VO   = cano["valid_ocean"].astype(bool).values
    SID  = cano["sector_id"].astype(np.int16).values
    SECT = {
        1:"Amundsen–Bellingshausen",
        2:"Weddell",
        3:"King Haakon VII",
        4:"East Antarctic",
        5:"Ross–Amundsen"
    }

    summary_rows = []

    for metric in ["FS","MS"]:
        static = load_static_dict(metric, k=5, thr_pct=15)

        for scheme in SCHEMES:
            scheme_dir = os.path.join(DYN_ROOT, scheme)
            tags = list_dyn_paramtags(scheme_dir, metric)
            if not tags:
                print(f"[warn] no tags found for {scheme} {metric}")
                continue

            for tag in tags:
                dyn = load_dyn_dict(scheme_dir, metric, tag)
                years = align_years([static, dyn])

                S = stack_years(static, years, f"{metric}_static")
                D = stack_years(dyn,    years, f"{metric}_dyn")

                valid = (~np.isnan(S)) & (~np.isnan(D))
                S = S.where(valid); D = D.where(valid)

                # absolute wrapped deltas
                A = xr.apply_ufunc(wrapped_abs, D, S, dask="allowed")
                if VO.shape == A.isel(year=0).shape:
                    A = A.where(VO)

                # ---- MAPS: mean/std/q95 across years ----
                meanA = A.mean("year", skipna=True)
                stdA  = A.std("year",  skipna=True)
                q95A  = A.quantile(0.95, dim="year", skipna=True)

                for da, statname, vmax, cmap in [
                    (meanA, "mean_absdiff", 10, "viridis"),
                    (stdA,  "std_absdiff",   8, "magma"),
                    (q95A,  "q95_absdiff",  10, "plasma")
                ]:
                    fig, ax = plt.subplots(figsize=(5.6, 4.4))
                    arr = to_plot_array(da, default_flip=True)
                    im = ax.imshow(arr, origin="lower", vmin=0, vmax=vmax, cmap=cmap)
                    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
                    cb.set_label(f"{statname} (days)")
                    ax.set_title(f"{metric} • {scheme} • {tag} • {statname}")
                    ax.axis("off")
                    fn = dirs["maps"]/f"{statname}_{metric}_{scheme}_{tag}.png"
                    fig.savefig(fn, dpi=300, bbox_inches="tight", pad_inches=0.03)
                    plt.close(fig)
                    rclone_copy(fn)

                # ---- 1D: CDF(|Δ|) over active ocean ----
                vals = A.values[np.isfinite(A.values)]
                vals_clip = vals[vals <= MAX_X]
                x,y = ecdf(vals_clip)
                fig, ax = plt.subplots(figsize=(5.4, 3.6))
                ax.plot(x,y,lw=1.6)
                ax.set_xlim(0,MAX_X); ax.set_ylim(0,1)
                ax.set_xlabel("|Δ| (days)"); ax.set_ylabel("Cumulative Fraction")
                ax.set_title(f"CDF |Δ| • {metric} • {scheme} • {tag}")
                ax.grid(True, ls=":", lw=0.6, alpha=0.6)
                fn = dirs["cdf"]/f"CDF_{metric}_{scheme}_{tag}.png"
                fig.savefig(fn, dpi=300, bbox_inches="tight", pad_inches=0.02)
                plt.close(fig); rclone_copy(fn)

                # ---- Box by sector + circumpolar ----
                box_data = []
                v_all = vals
                if v_all.size:
                    box_data.append(("Circumpolar", v_all))
                for k,name in SECT.items():
                    m = (SID==k)
                    vv = A.values[:, m]
                    vv = vv[np.isfinite(vv)]
                    if vv.size:
                        box_data.append((name, vv))
                labels = [d[0] for d in box_data]
                series = [d[1] for d in box_data]
                fig, ax = plt.subplots(figsize=(8.0, 3.6))
                ax.boxplot(series, showfliers=False)
                ax.set_xticklabels(labels, rotation=20, ha="right")
                ax.set_ylim(0, MAX_X); ax.set_ylabel("|Δ| (days)")
                ax.set_title(f"Sectoral |Δ| • {metric} • {scheme} • {tag}")
                ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
                fn = dirs["box"]/f"BOX_{metric}_{scheme}_{tag}.png"
                fig.savefig(fn, dpi=300, bbox_inches="tight", pad_inches=0.03)
                plt.close(fig); rclone_copy(fn)

                # ---- summary rows (circumpolar + sectors) ----
                def stats(v):
                    v = v[np.isfinite(v)]
                    if v.size==0:
                        return dict(median=np.nan, p95=np.nan, mean=np.nan, N=0)
                    return dict(median=float(np.nanmedian(v)),
                                p95=float(np.nanpercentile(v,95)),
                                mean=float(np.nanmean(v)), N=int(v.size))
                # circumpolar
                st = stats(v_all)
                summary_rows.append(dict(Metric=metric, Scheme=scheme, Tag=tag,
                                         Region="Circumpolar", **st))
                # sectors
                for k,name in SECT.items():
                    vv = A.values[:, (SID==k)]
                    vv = vv[np.isfinite(vv)]
                    st = stats(vv)
                    summary_rows.append(dict(Metric=metric, Scheme=scheme, Tag=tag,
                                             Region=name, **st))

    df = pd.DataFrame(summary_rows,
                      columns=["Metric","Scheme","Tag","Region","median","p95","mean","N"])
    csv_path = Path(OUT_DIR)/"summary.csv"
    df.to_csv(csv_path, index=False)
    print("[OK] wrote", csv_path)
    rclone_copy(csv_path)

if __name__ == "__main__":
    main()
