#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-ready figures for dynamic vs static threshold comparison
with rclone uploads (per-file + final recursive copy).

Outputs are written under OUT_DIR and uploaded to RCLONE['remote']:RCLONE['dst_dir'].
"""

import os, re, glob, shutil, subprocess
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import theilslopes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ------------------------- MAIN CONFIG -------------------------
OUT_DIR   = "/user/geog/falejandraperez/sea-ice-phase/results/Ch2_Figures/fig_set_v2"
CLASSIC_DIR = "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase"
DYN_ROOT    = "/user/geog/falejandraperez/sea-ice-phase/results/static_v2_slopeH/dynamic"
CANONICAL   = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"

# Preferred dynamic tags
PREFERRED_TAG = dict(mu_sigma_k5="alpha1.0", quantile_k5="p0.7")

# Blocks
MAKE_B_DYNvDYN      = True
MAKE_C_DYNvCLASSIC  = True
MAKE_D_CLIM_TRENDS  = True
MAKE_OPTIONAL_JOINT_CLASSIC = False

PHASES = ["FS", "MS"]
YEARS_LIMIT = None

# Plot limits
MAX_X = 30
DOY_VMIN, DOY_VMAX = 0, 365
DMAP_LIM = 10
SMAP_LIM = 8
TREND_LIM = 5

# Seaborn style
sns.set_theme(context="talk", style="whitegrid")
sns.set_palette("colorblind")
plt.rcParams.update({
    "figure.constrained_layout.use": True,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# ---- RCLONE CONFIG ----
RCLONE = dict(
    enabled=True,
    remote="gdrive",
    dst_dir="sea-ice-phase/Results/Ch2_Figures/fig_set_v2",
    extra_flags=["--transfers=8","--checkers=8","--fast-list","--copy-links"]
)

# ------------------------- UTILS -------------------------
PERIOD = 366
yr_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    m = yr_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def wrapped_abs(a, b, period=PERIOD):
    return np.abs(((a - b + period//2) % period) - (period//2))

def ecdf_vals(v, xmax=MAX_X):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    v = v[v <= xmax]
    if v.size == 0:
        return np.array([0.0]), np.array([0.0])
    v.sort()
    y = np.linspace(0, 1, v.size, endpoint=False)
    return v, y

def to_plot_array(da, default_flip=True):
    arr = np.squeeze(da.values)
    try:
        for cand in ["lat", "latitude", "y"]:
            if cand in da.coords and da[cand].ndim == 1 and da[cand].size == arr.shape[0]:
                lats = da[cand].values
                if lats[0] > lats[-1]:
                    arr = np.flipud(arr)
                return arr
    except Exception:
        pass
    return np.flipud(arr) if default_flip else arr

def theilsen_trend(stack, years):
    yrs = np.asarray(years, dtype=float)
    ny, nx = stack.shape[1], stack.shape[2]
    out = np.full((ny, nx), np.nan, dtype=float)
    for i in range(ny):
        Y = stack[:, i, :].values
        m = np.isfinite(Y)
        for j in range(nx):
            mask = m[:, j]
            if mask.sum() > 10:
                slope, *_ = theilslopes(Y[mask, j], yrs[mask])
                out[i, j] = slope * 10.0
    return xr.DataArray(out, dims=stack.dims[1:], coords={stack.dims[1]: stack.coords[stack.dims[1]],
                                                          stack.dims[2]: stack.coords[stack.dims[2]]})

def list_dyn_tag(root, scheme, phase):
    base = os.path.join(root, scheme, phase)
    tags = []
    for p in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(p) and glob.glob(os.path.join(p, f"{phase}_*.nc")):
            tags.append(os.path.basename(p))
    pref = PREFERRED_TAG.get(scheme)
    return pref if (pref in tags) else (tags[0] if tags else None)

def load_stack_years(files_glob, varname):
    files = sorted(glob.glob(files_glob))
    d = {}
    for f in files:
        y = parse_year(f)
        if y is None:
            continue
        with xr.open_dataset(f) as ds:
            v = varname
            if v not in ds:
                for alt in [varname.upper(), varname.lower()]:
                    if alt in ds:
                        v = alt; break
            if v not in ds:
                raise KeyError(f"{f} lacks variable '{varname}'")
            d[y] = ds[v].load().rename(varname)
    if not d:
        raise FileNotFoundError(f"No files matched {files_glob}")
    years = sorted(d.keys())
    if YEARS_LIMIT:
        years = [yy for yy in years if yy in YEARS_LIMIT]
    arrs = [d[yy].expand_dims(year=[yy]) for yy in years]
    return xr.concat(arrs, dim="year"), years

def load_classic_stack(classic_dir, phase):
    prefix = {"FS": "advance", "MS": "retreat"}[phase]
    files = sorted(glob.glob(os.path.join(classic_dir, "seaice_phases_SMMR_*.nc")))
    d = {}
    for f in files:
        y = parse_year(f)
        if y is None:
            continue
        with xr.open_dataset(f) as ds:
            vname = f"{prefix}_{y}"
            if vname not in ds:
                alts = [vname, vname.upper(), vname.lower()]
                hit = [a for a in alts if a in ds]
                if not hit:
                    raise KeyError(f"{f} missing '{vname}'")
                vname = hit[0]
            d[y] = ds[vname].load().rename(phase)
    if not d:
        raise FileNotFoundError("No classic files found.")
    years = sorted(d.keys())
    if YEARS_LIMIT:
        years = [yy for yy in years if yy in YEARS_LIMIT]
    arrs = [d[yy].expand_dims(year=[yy]) for yy in years]
    return xr.concat(arrs, dim="year"), years

def mask_valid(stack, VO):
    m = np.isfinite(stack.values)
    if VO.shape == stack.isel(year=0).shape:
        m &= VO
    return xr.DataArray(np.where(m, stack.values, np.nan), dims=stack.dims, coords=stack.coords)

# ----- rclone helpers -----
def rclone_available():
    try:
        subprocess.run(["rclone","version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def rclone_copy_file(local_path: Path):
    if not RCLONE.get("enabled", False): return
    if not rclone_available():
        print("[rclone] not found on PATH; skipping upload for", local_path)
        return
    dst = f"{RCLONE['remote']}:{RCLONE['dst_dir']}"
    cmd = ["rclone","copy",str(local_path),dst] + RCLONE.get("extra_flags",[])
    try:
        subprocess.run(cmd, check=True)
        print("[rclone] copied", local_path, "->", dst)
    except subprocess.CalledProcessError as e:
        print("[rclone] copy failed:", e)

def rclone_copy_tree(local_root: Path):
    if not RCLONE.get("enabled", False): return
    if not rclone_available():
        print("[rclone] not found on PATH; final sync skipped")
        return
    dst = f"{RCLONE['remote']}:{RCLONE['dst_dir']}"
    cmd = ["rclone","copy",str(local_root),dst,"--create-empty-src-dirs"] + RCLONE.get("extra_flags",[])
    try:
        subprocess.run(cmd, check=True)
        print("[rclone] synced folder", local_root, "->", dst)
    except subprocess.CalledProcessError as e:
        print("[rclone] folder sync failed:", e)

def save(fig, path, dpi=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    rclone_copy_file(Path(path))

# ------------------------- PLOTTING HELPERS -------------------------
def plot_ecdf_overlay(vals_dict, title, outpath, xmax=MAX_X):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for lab, v in vals_dict.items():
        x, y = ecdf_vals(v, xmax)
        ax.plot(x, y, lw=2, label=lab)
    for vline in (2, 5, 10):
        ax.axvline(vline, ls="--", lw=0.8, color="0.6")
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1)
    ax.set_xlabel("Absolute timing difference |Δ| (days)")
    ax.set_ylabel("Cumulative Fraction of Pixels")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    save(fig, outpath)

def plot_violin_by_sector(df, title, outpath, ymax=MAX_X):
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    sns.violinplot(data=df, x="sector", y="absdiff", hue="method",
                   inner="quartile", cut=0, bw=0.25, density_norm="width",
                   dodge=True, ax=ax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(""); ax.set_ylabel("|Δ| (days)")
    ax.set_title(title)
    ax.legend(title="", loc="upper right", frameon=True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(fig, outpath)

def plot_joint_hist(xvals, yvals, title, outpath):
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    sns.histplot(x=xvals, y=yvals, bins=200, pmax=0.999,
                 cbar=True, cbar_kws={"label":"pixel count (log)"},
                 norm=LogNorm(), cmap="mako", ax=ax)
    ax.plot([DOY_VMIN, DOY_VMAX], [DOY_VMIN, DOY_VMAX], ls="--", lw=1, color="0.5", label="1:1")
    try:
        m, b, *_ = theilslopes(yvals, xvals)
    except Exception:
        m, b = np.polyfit(xvals, yvals, 1)
    xx = np.array([DOY_VMIN, DOY_VMAX])
    ax.plot(xx, m*xx + b, lw=1.8, label=f"fit: y={m:.2f}x+{b:.2f}")
    mask = np.isfinite(xvals) & np.isfinite(yvals)
    if mask.sum() > 10:
        r = np.corrcoef(xvals[mask], yvals[mask])[0,1]
        ax.set_title(f"{title}\n$r = {r:.2f}$")
    else:
        ax.set_title(title)
    ax.set_xlim(DOY_VMIN, DOY_VMAX); ax.set_ylim(DOY_VMIN, DOY_VMAX)
    ax.set_xlabel("Mean DOY • x-axis method")
    ax.set_ylabel("Mean DOY • y-axis method")
    ax.legend(frameon=True)
    save(fig, outpath)

def plot_map_single(da, vmin, vmax, cmap, label, title, outpath):
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    arr = to_plot_array(da)
    im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axis("off"); ax.set_title(title, pad=6)
    cax = inset_axes(ax, width="60%", height="4%", loc="lower center", borderpad=1.8)
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(label)
    save(fig, outpath)

# ------------------------- DRIVER -------------------------
def run_for_phase(phase: str):
    base = Path(OUT_DIR)/phase
    paths = {
        "maps": base/"maps",
        "ecdf": base/"ecdf",
        "violin": base/"violin",
        "joint": base/"joint",
        "tables": base/"tables",
    }
    for p in paths.values(): p.mkdir(parents=True, exist_ok=True)

    cano = xr.open_dataset(CANONICAL).load()
    VO   = cano["valid_ocean"].astype(bool).values
    SID  = cano["sector_id"].astype(np.int16).values
    SECT = {0:"Circumpolar", 1:"Amundsen–Bellingshausen", 2:"Weddell",
            3:"King Haakon VII", 4:"East Antarctic", 5:"Ross–Amundsen"}

    # Classic baseline
    classic, years_c = load_classic_stack(CLASSIC_DIR, phase=phase)
    # Percentile dynamic
    tagP = list_dyn_tag(DYN_ROOT, "quantile_k5", phase)
    if tagP is None:
        raise RuntimeError(f"No percentile tag found for {phase}")
    dynP, years_p = load_stack_years(os.path.join(DYN_ROOT, "quantile_k5", phase, tagP, f"{phase}_*.nc"), phase)
    # mu+sigma dynamic
    tagM = list_dyn_tag(DYN_ROOT, "mu_sigma_k5", phase)
    if tagM is None:
        raise RuntimeError(f"No mu_sigma tag found for {phase}")
    dynM, years_m = load_stack_years(os.path.join(DYN_ROOT, "mu_sigma_k5", phase, tagM, f"{phase}_*.nc"), phase)

    years = sorted(set(years_c) & set(years_p) & set(years_m))
    if YEARS_LIMIT: years = [y for y in years if y in YEARS_LIMIT]
    if not years: raise ValueError(f"No overlapping years for {phase}.")
    classic = classic.sel(year=years)
    dynP    = dynP.sel(year=years)
    dynM    = dynM.sel(year=years)

    classic = mask_valid(classic, VO)
    dynP    = mask_valid(dynP, VO)
    dynM    = mask_valid(dynM, VO)

    # ------- B. Dynamic vs Dynamic -------
    if MAKE_B_DYNvDYN:
        A = xr.apply_ufunc(wrapped_abs, dynM, dynP, dask="allowed").values
        vals_dyn = {"μ+σ vs Percentile": A[np.isfinite(A)]}
        plot_ecdf_overlay(vals_dyn, title=f"ECDF of |Δ| • μ+σ vs Percentile • {phase}",
                          outpath=paths["ecdf"]/f"ecdf_muSigma_vs_percentile_{phase}.png")

        A_da = xr.apply_ufunc(wrapped_abs, dynM, dynP, dask="allowed")
        meanA = A_da.mean("year", skipna=True)
        stdA  = A_da.std("year",  skipna=True)
        q95A  = A_da.quantile(0.95, dim="year", skipna=True)
        plot_map_single(meanA, 0, DMAP_LIM, "viridis",
                        "|Δ| (days)", f"Volatility • mean(|Δ|) μ+σ vs Percentile • {phase}",
                        paths["maps"]/f"volatility_mean_muSigma_vs_percentile_{phase}.png")
        plot_map_single(stdA,  0, SMAP_LIM, "magma",
                        "|Δ| (days)", f"Volatility • std(|Δ|) μ+σ vs Percentile • {phase}",
                        paths["maps"]/f"volatility_std_muSigma_vs_percentile_{phase}.png")
        plot_map_single(q95A, 0, DMAP_LIM, "plasma",
                        "|Δ| (days)", f"Volatility • q95(|Δ|) μ+σ vs Percentile • {phase}",
                        paths["maps"]/f"volatility_q95_muSigma_vs_percentile_{phase}.png")

        meanP = dynP.mean("year", skipna=True).values.flatten()
        meanM = dynM.mean("year", skipna=True).values.flatten()
        msk = np.isfinite(meanP) & np.isfinite(meanM)
        plot_joint_hist(meanP[msk], meanM[msk],
                        title=f"Joint histogram • Mean DOY • Percentile (x) vs μ+σ (y) • {phase}",
                        outpath=paths["joint"]/f"joint_mean_doy_percentile_vs_muSigma_{phase}.png")

    # ------- C. Dynamic vs Classic -------
    if MAKE_C_DYNvCLASSIC:
        Aperc = xr.apply_ufunc(wrapped_abs, dynP, classic, dask="allowed").values
        vals = {"Percentile vs Classic": Aperc[np.isfinite(Aperc)]}
        plot_ecdf_overlay(vals, title=f"ECDF of |Δ| • Percentile vs Classic • {phase}",
                          outpath=paths["ecdf"]/f"ecdf_percentile_vs_classic_{phase}.png")

        rows = []
        v = Aperc
        if np.isfinite(v).any():
            rows.append(dict(sector="Circumpolar", method="Percentile vs Classic",
                             absdiff=v[np.isfinite(v)].ravel()))
            for sid, name in [(1,"Amundsen–Bellingshausen"), (2,"Weddell"),
                              (3,"King Haakon VII"), (4,"East Antarctic"), (5,"Ross–Amundsen")]:
                mask = (SID == sid)
                vv = v[:, mask]
                vv = vv[np.isfinite(vv)]
                if vv.size:
                    rows.append(dict(sector=name, method="Percentile vs Classic", absdiff=vv))
        if rows:
            recs = []
            for r in rows:
                vals1 = np.asarray(r["absdiff"]).ravel()
                for val in vals1:
                    if np.isfinite(val):
                        recs.append({"sector": r["sector"], "method": r["method"], "absdiff": float(val)})
            df = pd.DataFrame.from_records(recs)
            order = ["Circumpolar","Amundsen–Bellingshausen","Weddell","King Haakon VII",
                     "East Antarctic","Ross–Amundsen"]
            df["sector"] = pd.Categorical(df["sector"], categories=order, ordered=True)
            plot_violin_by_sector(df, title=f"Sectoral |Δ| • Percentile vs Classic • {phase}",
                                  outpath=paths["violin"]/f"violin_percentile_vs_classic_{phase}.png")

        if MAKE_OPTIONAL_JOINT_CLASSIC:
            meanP = dynP.mean("year", skipna=True).values.flatten()
            meanB = classic.mean("year", skipna=True).values.flatten()
            msk = np.isfinite(meanP) & np.isfinite(meanB)
            plot_joint_hist(meanB[msk], meanP[msk],
                            title=f"Joint histogram • Mean DOY • Classic (x) vs Percentile (y) • {phase}",
                            outpath=paths["joint"]/f"joint_mean_doy_classic_vs_percentile_{phase}.png")

    # ------- D. Climatology & Trends -------
    if MAKE_D_CLIM_TRENDS:
        meanClassic = classic.mean("year", skipna=True)
        plot_map_single(meanClassic, DOY_VMIN, DOY_VMAX, "cividis",
                        "DOY", f"Mean DOY • Classic • {phase}",
                        paths["maps"]/f"mean_doy_classic_{phase}.png")
        meanPerc = dynP.mean("year", skipna=True)
        plot_map_single(meanPerc, DOY_VMIN, DOY_VMAX, "cividis",
                        "DOY", f"Mean DOY • Percentile • {phase}",
                        paths["maps"]/f"mean_doy_percentile_{phase}.png")
        trendClassic = theilsen_trend(classic, years)
        plot_map_single(trendClassic, -TREND_LIM, TREND_LIM, "magma",
                        "days / decade", f"Trend • Classic • {phase}",
                        paths["maps"]/f"trend_classic_{phase}.png")
        trendPerc = theilsen_trend(dynP, years)
        plot_map_single(trendPerc, -TREND_LIM, TREND_LIM, "magma",
                        "days / decade", f"Trend • Percentile • {phase}",
                        paths["maps"]/f"trend_percentile_{phase}.png")

def main():
    for ph in PHASES:
        print(f"[INFO] Building figures for phase: {ph}")
        run_for_phase(ph)
    print(f"[OK] Finished. Outputs under: {OUT_DIR}")
    # final recursive copy
    rclone_copy_tree(Path(OUT_DIR))

if __name__ == "__main__":
    main()
