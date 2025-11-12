#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline (classic 15% no-slope) vs Dynamic Thresholds:
 - Core comparison: Classic vs Dynamic Percentile (p=0.7, k=5)
 - Optional: include Static+Slope (thr15, k=5) in skill plots
 - Volatility maps: |Δ| between Dynamic μ+σ (alpha=1.0, k=5) and Dynamic Percentile

Outputs in OUT_DIR:
  maps/:
      mean_DOY_{phase}_{method}.png
      trend_{phase}_{method}.png
      dmean_DOY_{phase}_dynP_minus_classic.png
      dtrend_{phase}_dynP_minus_classic.png
      volatility_{stat}_{phase}_muSigma_vs_percentile.png  (stat ∈ {mean,std,q95})
  cdf/:
      CDF_{phase}_vs_classic_{method}.png  (method ∈ {dynP, staticSlope?})
  box/:
      BOX_{phase}_vs_classic_{method}.png
  summary.csv:
      circumpolar + sector stats of |Δ| vs classic (median, p95, mean, N) for selected methods
"""

import os, re, glob
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import theilslopes

# ================== CONFIG ==================
PERIOD = 366
MAX_X  = 30

# --- Paths ---
CLASSIC_DIR   = "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase"  # holds seaice_phases_SMMR_YYYY.nc
STATIC_SLOPE = dict(  # optional comparator (thr15,k5)
    FS = "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase/FS_thr15_k5",
    MS = "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase/MS_thr15_k5",
    enabled = True
)
DYN_ROOT = "/user/geog/falejandraperez/sea-ice-phase/results/static_v2_slopeH/dynamic"  # has mu_sigma_k5/, quantile_k5/

# prefer these tags if they exist
PREFERRED_TAG = dict(mu_sigma_k5="alpha1.0", quantile_k5="p0.7")

CANONICAL = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"
OUT_DIR   = "/user/geog/falejandraperez/sea-ice-phase/results/Ch2_Figures/baseline_vs_dynamic"

# limit years (None = all)
YEARS_LIMIT = None

# plotting defaults
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

# rclone (optional)
RCLONE = dict(enabled=False, remote="gdrive",
              dst_dir="sea-ice-phase/results/Ch2_Figures/baseline_vs_dynamic",
              extra_flags=["--transfers=8","--checkers=8","--fast-list"],
              dry_run=False)

# =============== HELPERS ===============
yr_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    m = yr_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def rclone_copy(path):
    if not RCLONE.get("enabled", False): return
    import subprocess
    dst = f"{RCLONE['remote']}:{RCLONE['dst_dir']}"
    cmd = ["rclone", "copy", str(path), dst] + RCLONE.get("extra_flags", [])
    if RCLONE.get("dry_run"): cmd.insert(1, "--dry-run")
    try: subprocess.run(cmd, check=True)
    except Exception as e: print(f"[rclone] failed: {e}")

def wrapped_abs(a, b, period=PERIOD):
    return np.abs(((a - b + period//2) % period) - (period//2))

def ecdf(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    if x.size == 0: return np.array([0.0]), np.array([0.0])
    x.sort(); y = np.linspace(0, 1, x.size, endpoint=False)
    return x, y

def to_plot_array(da, default_flip=True):
    """Orient with southern lats at bottom."""
    arr = np.squeeze(da.values)
    try:
        for cand in ["lat", "latitude", "y"]:
            if cand in da.coords and da[cand].ndim == 1 and da[cand].size == arr.shape[0]:
                lats = da[cand].values
                if lats[0] > lats[-1]:  # north->south
                    arr = np.flipud(arr)
                return arr
    except Exception:
        pass
    return np.flipud(arr) if default_flip else arr

def list_dyn_tags(scheme_dir, phase):
    base = os.path.join(scheme_dir, phase)
    tags = []
    for p in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(p) and glob.glob(os.path.join(p, f"{phase}_*.nc")):
            tags.append(os.path.basename(p))
    return tags

def choose_tag(tags, preferred):
    if not tags: return None
    if preferred and preferred in tags: return preferred
    return sorted(tags)[0]

def load_stack_years(files_glob, varname):
    files = sorted(glob.glob(files_glob))
    d = {}
    for f in files:
        y = parse_year(f)
        if y is None: continue
        with xr.open_dataset(f) as ds:
            if varname not in ds:  # try upper/lower variants
                for alt in [varname.lower(), varname.upper()]:
                    if alt in ds: varname = alt; break
            d[y] = ds[varname].load()
    if not d: raise FileNotFoundError(f"No files for glob: {files_glob}")
    years = sorted(d.keys())
    if YEARS_LIMIT: years = [y for y in years if y in YEARS_LIMIT]
    arrs = [d[y].expand_dims(year=[y]) for y in years]
    return xr.concat(arrs, dim="year").rename(varname), years

def theilsen_trend(stack, years):
    years = np.asarray(years, dtype=float)
    trend = np.full(stack.shape[1:], np.nan, dtype=float)
    it = np.ndindex(stack.shape[1], stack.shape[2])
    for i,j in it:
        y = stack[:, i, j].values
        m = np.isfinite(y)
        if m.sum() > 10:
            slope, *_ = theilslopes(y[m], years[m])
            trend[i, j] = slope * 10.0  # days/decade
    da = xr.DataArray(trend, dims=stack.dims[1:], coords={stack.dims[1]: stack.coords[stack.dims[1]],
                                                          stack.dims[2]: stack.coords[stack.dims[2]]})
    return da

# =============== CORE ===============
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    dirs = {k: Path(OUT_DIR)/k for k in ["maps","cdf","box"]}
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

    # Canonical masks
    cano = xr.open_dataset(CANONICAL).load()
    VO   = cano["valid_ocean"].astype(bool).values
    SID  = cano["sector_id"].astype(np.int16).values
    SECT = {1:"Amundsen–Bellingshausen",2:"Weddell",3:"King Haakon VII",4:"East Antarctic",5:"Ross–Amundsen"}

    summary_rows = []

    # -------- Load CLASSIC baseline stacks (no slope) --------
    classic, years_c = load_stack_years(os.path.join(CLASSIC_DIR, "seaice_phases_SMMR_*.nc"), varname="FS")
    classic_MS, _    = load_stack_years(os.path.join(CLASSIC_DIR, "seaice_phases_SMMR_*.nc"), varname="MS")
    # align dims
    classic = classic.transpose("year", ...); classic_MS = classic_MS.transpose("year", ...)

    years_all = years_c

    # -------- Optional: Static+Slope stacks --------
    if STATIC_SLOPE.get("enabled", True):
        statFS, years_sfs = load_stack_years(os.path.join(STATIC_SLOPE["FS"], "FS_*.nc"), varname="FS")
        statMS, years_sms = load_stack_years(os.path.join(STATIC_SLOPE["MS"], "MS_*.nc"), varname="MS")
        years_all = sorted(set(years_all) & set(years_sfs) & set(years_sms))

    # -------- Dynamic stacks (percentile & mu+sigma) --------
    # Percentile
    tagsQ = list_dyn_tags(os.path.join(DYN_ROOT, "quantile_k5"), "FS")
    tagQ  = choose_tag(tagsQ, PREFERRED_TAG.get("quantile_k5"))
    FS_dynQ, years_qfs = load_stack_years(os.path.join(DYN_ROOT, "quantile_k5", "FS", tagQ, "FS_*.nc"), "FS")
    MS_dynQ, years_qms = load_stack_years(os.path.join(DYN_ROOT, "quantile_k5", "MS", tagQ, "MS_*.nc"), "MS")
    # mu+sigma
    tagsM = list_dyn_tags(os.path.join(DYN_ROOT, "mu_sigma_k5"), "FS")
    tagM  = choose_tag(tagsM, PREFERRED_TAG.get("mu_sigma_k5"))
    FS_dynM, years_mfs = load_stack_years(os.path.join(DYN_ROOT, "mu_sigma_k5", "FS", tagM, "FS_*.nc"), "FS")
    MS_dynM, years_mms = load_stack_years(os.path.join(DYN_ROOT, "mu_sigma_k5", "MS", tagM, "MS_*.nc"), "MS")

    # -------- Intersect years across everything we need --------
    sets = [set(years_all), set(years_qfs), set(years_qms), set(years_mfs), set(years_mms)]
    if STATIC_SLOPE.get("enabled", True):
        sets += [set(years_sfs), set(years_sms)]
    years = sorted(set.intersection(*sets))
    if YEARS_LIMIT: years = [y for y in years if y in YEARS_LIMIT]
    if not years: raise ValueError("No overlapping years across datasets.")

    # slice to common years
    def sel_years(stack): return stack.sel(year=years)
    FS_classic = classic.sel(year=years)
    MS_classic = classic_MS.sel(year=years)
    if STATIC_SLOPE.get("enabled", True):
        FS_statSlope = sel_years(statFS); MS_statSlope = sel_years(statMS)
    FS_dynP = sel_years(FS_dynQ); MS_dynP = sel_years(MS_dynQ)
    FS_dynM = sel_years(FS_dynM); MS_dynM = sel_years(MS_dynM)

    # mask to ocean & finite
    def mask_finite(A):
        m = np.isfinite(A.values)
        if VO.shape == A.isel(year=0).shape:
            m &= VO
        return xr.DataArray(np.where(m, A.values, np.nan), dims=A.dims, coords=A.coords)

    FS_classic = mask_finite(FS_classic); MS_classic = mask_finite(MS_classic)
    if STATIC_SLOPE.get("enabled", True):
        FS_statSlope = mask_finite(FS_statSlope); MS_statSlope = mask_finite(MS_statSlope)
    FS_dynP = mask_finite(FS_dynP); MS_dynP = mask_finite(MS_dynP)
    FS_dynM = mask_finite(FS_dynM); MS_dynM = mask_finite(MS_dynM)

    # ---------- FIGURES ----------
    def savefig(fig, path, dpi=300):
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig); rclone_copy(path)

    # A) CDFs and boxplots vs CLASSIC for selected methods
    def cdf_and_box_vs_classic(phase, baseline, method_stacks):
        # method_stacks: list of (label, stack)
        # CDFs
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for label, X in method_stacks:
            A = xr.apply_ufunc(wrapped_abs, X, baseline, dask="allowed").values
            v = A[np.isfinite(A) & (A <= MAX_X)]
            x, y = ecdf(v)
            ax.plot(x, y, lw=1.6, label=label)
        ax.set_xlim(0, MAX_X); ax.set_ylim(0, 1)
        ax.set_xlabel("|Δ| (days)"); ax.set_ylabel("Cumulative Fraction")
        ax.set_title(f"CDF |Δ| vs Classic • {phase}")
        ax.grid(True, ls=":", lw=0.6, alpha=0.6)
        ax.legend(loc="lower right", frameon=False)
        savefig(fig, dirs["cdf"]/f"CDF_{phase}_vs_classic.png")

        # Box by sector
        for label, X in method_stacks:
            A = xr.apply_ufunc(wrapped_abs, X, baseline, dask="allowed").values
            v_all = A[np.isfinite(A)]
            rows = [("Circumpolar", v_all)]
            for k,name in SECT.items():
                vv = A[:, (SID==k)]
                vv = vv[np.isfinite(vv)]
                if vv.size: rows.append((name, vv))
            labels = [r[0] for r in rows]; series = [r[1] for r in rows]
            fig, ax = plt.subplots(figsize=(8.2, 3.6))
            ax.boxplot(series, showfliers=False)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylim(0, MAX_X); ax.set_ylabel("|Δ| (days)")
            ax.set_title(f"Sectoral |Δ| vs Classic • {phase} • {label}")
            ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
            savefig(fig, dirs["box"]/f"BOX_{phase}_vs_classic_{label.replace(' ','_')}.png")

    # FS
    method_list_FS = [("Dynamic p=0.7", FS_dynP)]
    if STATIC_SLOPE.get("enabled", True):
        method_list_FS.insert(0, ("Static+Slope", FS_statSlope))
    cdf_and_box_vs_classic("FS", FS_classic, method_list_FS)
    # MS
    method_list_MS = [("Dynamic p=0.7", MS_dynP)]
    if STATIC_SLOPE.get("enabled", True):
        method_list_MS.insert(0, ("Static+Slope", MS_statSlope))
    cdf_and_box_vs_classic("MS", MS_classic, method_list_MS)

    # B) Mean DOY maps (Classic | DynP | Δ)
    def mean_map_panel(phase, baseline, dynP):
        meanB = baseline.mean("year", skipna=True)
        meanP = dynP.mean("year", skipna=True)
        dmean = (meanP - meanB)
        for da, name, vmin, vmax, cmap in [
            (meanB, f"mean_DOY_{phase}_classic",   0, 365, "cividis"),
            (meanP, f"mean_DOY_{phase}_dynP",      0, 365, "cividis"),
            (dmean, f"dmean_DOY_{phase}_dynP_minus_classic", -20, 20, "RdBu_r")
        ]:
            fig, ax = plt.subplots(figsize=(5.8, 4.5))
            arr = to_plot_array(da)
            vmin_, vmax_ = (vmin, vmax)
            im = ax.imshow(arr, origin="lower", vmin=vmin_, vmax=vmax_, cmap=cmap)
            cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label("DOY" if "mean_DOY" in name and "dmean" not in name else "ΔDOY (days)")
            ax.set_title(name.replace("_", " "))
            ax.axis("off")
            savefig(fig, dirs["maps"]/f"{name}.png")
    mean_map_panel("FS", FS_classic, FS_dynP)
    mean_map_panel("MS", MS_classic, MS_dynP)

    # C) Trend maps (Classic | DynP | Δ)
    def trend_map_panel(phase, baseline, dynP):
        trendB = theilsen_trend(baseline, years=years)
        trendP = theilsen_trend(dynP,   years=years)
        dtrend = trendP - trendB
        for da, name, vmin, vmax, cmap in [
            (trendB, f"trend_{phase}_classic",   -5, 5, "magma"),      # days/dec
            (trendP, f"trend_{phase}_dynP",      -5, 5, "magma"),
            (dtrend, f"dtrend_{phase}_dynP_minus_classic", -2, 2, "RdBu_r")
        ]:
            fig, ax = plt.subplots(figsize=(5.8, 4.5))
            arr = to_plot_array(da)
            im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
            cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label("days / decade" if "dtrend" not in name else "Δ (days / decade)")
            ax.set_title(name.replace("_", " "))
            ax.axis("off")
            savefig(fig, dirs["maps"]/f"{name}.png")
        return trendB, trendP, dtrend
    trend_map_panel("FS", FS_classic, FS_dynP)
    trend_map_panel("MS", MS_classic, MS_dynP)

    # D) Volatility / Sensitivity (μ+σ vs Percentile): mean/std/q95 of |Δ|
    def volatility_block(phase, dynM, dynP):
        A = xr.apply_ufunc(wrapped_abs, dynM, dynP, dask="allowed")  # per-year, per-pixel |Δ|
        meanA = A.mean("year", skipna=True)
        stdA  = A.std("year",  skipna=True)
        q95A  = A.quantile(0.95, dim="year", skipna=True)
        for da, name, vmax, cmap in [
            (meanA, f"volatility_mean_{phase}_muSigma_vs_percentile", 10, "viridis"),
            (stdA,  f"volatility_std_{phase}_muSigma_vs_percentile",   8, "magma"),
            (q95A,  f"volatility_q95_{phase}_muSigma_vs_percentile",  10, "plasma")
        ]:
            fig, ax = plt.subplots(figsize=(5.8, 4.5))
            arr = to_plot_array(da)
            im = ax.imshow(arr, origin="lower", vmin=0, vmax=vmax, cmap=cmap)
            cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label("|Δ| (days)")
            ax.set_title(name.replace("_", " "))
            ax.axis("off")
            savefig(fig, dirs["maps"]/f"{name}.png")
    volatility_block("FS", FS_dynM, FS_dynP)
    volatility_block("MS", MS_dynM, MS_dynP)

    # E) Summary CSV (|Δ| vs CLASSIC)
    def add_summary_rows(phase, baseline, label, stack):
        A = xr.apply_ufunc(wrapped_abs, stack, baseline, dask="allowed").values
        def stats(v):
            v = v[np.isfinite(v)]
            if v.size == 0:
                return dict(median=np.nan, p95=np.nan, mean=np.nan, N=0)
            return dict(median=float(np.nanmedian(v)),
                        p95=float(np.nanpercentile(v,95)),
                        mean=float(np.nanmean(v)), N=int(v.size))
        # circumpolar
        st = stats(A)
        summary_rows.append(dict(Phase=phase, Method=label, Region="Circumpolar", **st))
        # sectors
        for k,name in SECT.items():
            vv = A[:, (SID==k)]
            vv = vv[np.isfinite(vv)]
            st = stats(vv)
            summary_rows.append(dict(Phase=phase, Method=label, Region=name, **st))

    if STATIC_SLOPE.get("enabled", True):
        add_summary_rows("FS", FS_classic, "Static+Slope", FS_statSlope)
        add_summary_rows("MS", MS_classic, "Static+Slope", MS_statSlope)
    add_summary_rows("FS", FS_classic, "Dynamic p=0.7", FS_dynP)
    add_summary_rows("MS", MS_classic, "Dynamic p=0.7", MS_dynP)

    df = pd.DataFrame(summary_rows,
                      columns=["Phase","Method","Region","median","p95","mean","N"])
    csv_path = Path(OUT_DIR)/"summary.csv"
    df.to_csv(csv_path, index=False)
    print("[OK] wrote", csv_path)

if __name__ == "__main__":
    main()
