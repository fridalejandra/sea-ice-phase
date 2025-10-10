#!/usr/bin/env python3
"""
window_sensitivity_maps.py

Outputs:
  1) Per-pixel PNG maps (Cartopy) for mean(|Δ|), std(|Δ|), frac(|Δ|>5) for {MS,FS} × {3v5,7v5}
  2) CSV with sectoral + circumpolar summaries (median, p95, mean, N)
  3) (Optional) rclone upload of PNGs to a remote folder

Assumptions:
  - Phase date NetCDFs are in INPUT_ROOT with subdirs like "MS_thr15_k3", files like "MS_1980.nc" (var name == "MS" or "FS")
  - Canonical sector file provides: sector_id(0..5), valid_ocean(bool), lon, lat, lonE, area_m2
"""

from pathlib import Path
import os, glob, re, json
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# Cartopy (publication-quality maps)
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =========================
# CONFIG — EDIT THIS BLOCK
# =========================
CFG = {
    "SENSOR": "SMMR",                      # "SMMR", "AMSRE", etc
    "THRESH_PCT": 15,                      # e.g., 10/15/20
    "INPUT_ROOT": "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase",
    "CANONICAL":  "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc",

    "OUT_DIR": "/user/geog/falejandraperez/sea-ice-phase/results/window_sensitivity",
    "PERIOD": 366,                         # day-of-year wrap
    "GT_THRESH_DAYS": 5,                   # threshold for 'unstable' fraction
    "MEAN_VMAX": 10,                       # colorbar max for mean(|Δ|)
    "STD_VMAX": 8,                         # colorbar max for std(|Δ|)
    "DPI": 180,

    # rclone (PNG upload)
    "RCLONE": {
        "enabled": True,
        "remote": "gdrive",
        "dst_dir": "sea-ice-phase/results/window_sensitivity/",
        "extra_flags": ["--transfers=8","--checkers=8","--fast-list"],
        "dry_run": False
    }
}

# Sector ID → name (must match your canonical file construction)
SECTOR_ID_TO_NAME = {
    1: "Amundsen–Bellingshausen",
    2: "Weddell",
    3: "King Haakon VII",
    4: "East Antarctic",
    5: "Ross–Amundsen",
}

# Meridians (deg East) used to draw sector lines
SECTOR_BOUNDARIES_E = [71, 162, 250, 290, 346]

# ======================
# UTILITIES
# ======================
year_re = re.compile(r"_(\d{4})\.nc$")

def parse_year(path):
    m = year_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def rclone_copy(local_path, cfg=CFG):
    rc = cfg["RCLONE"]
    if not rc.get("enabled", False):
        return
    dst = f"{rc['remote']}:{rc['dst_dir']}"
    cmd = ["rclone", "copy", str(local_path), dst] + rc.get("extra_flags", [])
    if rc.get("dry_run"): cmd.insert(1, "--dry-run")
    print("[rclone]", " ".join(cmd))
    code = os.system(" ".join(cmd))
    if code != 0:
        print(f"!! rclone failed (code {code}) for {local_path}")

def open_phase_dict(metric, kdays, cfg=CFG):
    """Return {year: DataArray} for metric ('MS'/'FS') and window length (3/5/7)."""
    subdir = f"{metric}_thr{cfg['THRESH_PCT']}_k{kdays}"
    folder = os.path.join(cfg["INPUT_ROOT"], subdir)
    files = sorted(glob.glob(os.path.join(folder, f"{metric}_*.nc")))
    out = {}
    for f in files:
        yr = parse_year(f)
        if yr is None:
            continue
        try:
            with xr.open_dataset(f) as ds:
                da = ds[metric].load()     # var must be 'MS' or 'FS'
            out[yr] = da
        except Exception as e:
            print(f"Skip {f}: {e}")
    if not out:
        raise FileNotFoundError(f"No phase files in {folder}")
    return out

def align_years(dicts):
    common = set.intersection(*[set(d.keys()) for d in dicts])
    years = sorted(common)
    if not years:
        raise ValueError("No overlapping years across windows.")
    return years

def stack_years(d, years, name):
    arrs = [d[y].expand_dims(year=[y]) for y in years]
    out = xr.concat(arrs, dim="year").rename(name)
    return out

def wrapped_abs_diff(a, b, period):
    """Absolute wrapped difference |Δ| between two day-of-year arrays."""
    return np.abs(((a - b + period//2) % period) - (period//2))

def to_lon_deg(lonE):
    """Convert degrees East (0..360) to [-180,180] for PlateCarree plotting."""
    return xr.where(lonE <= 180, lonE, lonE - 360.0)

# ======================
# CORE COMPUTE
# ======================
def compute_maps_for_metric(metric, cano, cfg=CFG):
    """
    Returns a dict of per-pixel maps for the metric (MS/FS):
      { "3v5": {"mean": DA, "std": DA, "frac_gt": DA},
        "7v5": {"mean": DA, "std": DA, "frac_gt": DA} }
    All maps share dims (y,x) and the canonical grid/coords.
    """
    d3 = open_phase_dict(metric, 3, cfg)
    d5 = open_phase_dict(metric, 5, cfg)
    d7 = open_phase_dict(metric, 7, cfg)
    years = align_years([d3, d5, d7])

    A3 = stack_years(d3, years, f"{metric}_k3")  # (year,y,x)
    A5 = stack_years(d5, years, f"{metric}_k5")
    A7 = stack_years(d7, years, f"{metric}_k7")

    valid = (~np.isnan(A3)) & (~np.isnan(A5)) & (~np.isnan(A7))
    A3 = A3.where(valid); A5 = A5.where(valid); A7 = A7.where(valid)

    period = cfg["PERIOD"]
    D35 = xr.apply_ufunc(wrapped_abs_diff, A3, A5, period, dask="allowed")
    D75 = xr.apply_ufunc(wrapped_abs_diff, A7, A5, period, dask="allowed")

    # reduce across years
    mean35 = D35.mean(dim="year", skipna=True)
    std35  = D35.std(dim="year",  skipna=True)
    frac35 = (D35 > cfg["GT_THRESH_DAYS"]).sum(dim="year") / D35.count(dim="year")

    mean75 = D75.mean(dim="year", skipna=True)
    std75  = D75.std(dim="year",  skipna=True)
    frac75 = (D75 > cfg["GT_THRESH_DAYS"]).sum(dim="year") / D75.count(dim="year")

    # mask to valid ocean
    vo = cano["valid_ocean"].astype(bool)
    for da in [mean35, std35, frac35, mean75, std75, frac75]:
        da.values[~vo.values] = np.nan

    return {
        "3v5": {"mean": mean35, "std": std35, "frac_gt": frac35},
        "7v5": {"mean": mean75, "std": std75, "frac_gt": frac75},
        "years": years
    }

# ======================
# MAPPING (Cartopy)
# ======================
def draw_sector_meridians(ax):
    for bE in SECTOR_BOUNDARIES_E:
        lon = bE if bE <= 180 else bE - 360
        ax.plot([lon, lon], [-90, -45], transform=ccrs.PlateCarree(),
                color="k", linewidth=0.6, alpha=0.8)

def plot_map_cartopy(da, cano, title, out_png, vmin, vmax, cmap="viridis", cfg=CFG):
    proj = ccrs.SouthPolarStereo()
    fig, ax = plt.subplots(figsize=(6.6, 6.6), subplot_kw={"projection": proj})

    # base: land + coastlines
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="k", linewidth=0.3)
    ax.coastlines(resolution="110m", color="k", linewidth=0.4)

    # data (lat/lon centers)
    lon = cano["lon"]; lat = cano["lat"]
    pc = ax.pcolormesh(lon, lat, da, transform=ccrs.PlateCarree(),
                       vmin=vmin, vmax=vmax, cmap=cmap)
    cb = plt.colorbar(pc, ax=ax, shrink=0.6, pad=0.05)
    cb.ax.set_ylabel(title.split("•")[0].split("(")[0].strip())

    # sector meridians
    draw_sector_meridians(ax)

    # extent and cosmetics
    ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=cfg["DPI"])
    plt.close()
    print(f"[OK] wrote {out_png}")
    rclone_copy(out_png, cfg)

# ======================
# STATS → CSV
# ======================
def summarize_to_rows(da, cano, metric, difftype, statname):
    rows = []
    sid = cano["sector_id"].astype(np.int16)
    vo  = cano["valid_ocean"].astype(bool)

    # circumpolar
    vals = da.where(vo).values
    vals = vals[np.isfinite(vals)]
    if vals.size:
        rows.append({"Metric": metric, "DiffType": difftype, "Sector": "Circumpolar",
                     "Stat": statname,
                     "MedianDays": float(np.nanmedian(vals)),
                     "P95Days": float(np.nanpercentile(vals, 95)),
                     "MeanDays": float(np.nanmean(vals)),
                     "Npixels": int(vals.size)})

    # per sector
    for k, name in SECTOR_ID_TO_NAME.items():
        v = da.where((sid == k) & vo).values
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        rows.append({"Metric": metric, "DiffType": difftype, "Sector": name,
                     "Stat": statname,
                     "MedianDays": float(np.nanmedian(v)),
                     "P95Days": float(np.nanpercentile(v, 95)),
                     "MeanDays": float(np.nanmean(v)),
                     "Npixels": int(v.size)})
    return rows

# ======================
# MAIN
# ======================
def main(cfg=CFG):
    out_dir = Path(cfg["OUT_DIR"])
    maps_dir = out_dir / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    cano = xr.open_dataset(cfg["CANONICAL"]).load()  # lat, lon, lonE, valid_ocean, sector_id, area_m2

    all_rows = []

    for metric in ["MS", "FS"]:
        maps = compute_maps_for_metric(metric, cano, cfg)

        for difftype, group in [("3v5", maps["3v5"]), ("7v5", maps["7v5"])]:
            # 1) maps → PNG
            plot_map_cartopy(group["mean"], cano,
                             title=f"mean(|Δ|) days • {metric} • {difftype}",
                             out_png=maps_dir / f"mean_absdiff_{metric}_{difftype}.png",
                             vmin=0, vmax=cfg["MEAN_VMAX"], cmap="viridis", cfg=cfg)

            plot_map_cartopy(group["std"], cano,
                             title=f"std(|Δ|) days • {metric} • {difftype}",
                             out_png=maps_dir / f"std_absdiff_{metric}_{difftype}.png",
                             vmin=0, vmax=cfg["STD_VMAX"], cmap="magma", cfg=cfg)

            plot_map_cartopy(group["frac_gt"], cano,
                             title=f"fraction(|Δ|>{cfg['GT_THRESH_DAYS']}d) • {metric} • {difftype}",
                             out_png=maps_dir / f"frac_gt{cfg['GT_THRESH_DAYS']}_absdiff_{metric}_{difftype}.png",
                             vmin=0, vmax=1, cmap="plasma", cfg=cfg)

            # 2) stats → rows
            all_rows += summarize_to_rows(group["mean"], cano, metric, difftype, statname="mean_absdiff")
            all_rows += summarize_to_rows(group["std"],  cano, metric, difftype, statname="std_absdiff")
            # For fraction, report as “days” fields but they are fractions; keep columns consistent
            rows_frac = summarize_to_rows(group["frac_gt"], cano, metric, difftype, statname=f"frac_gt{cfg['GT_THRESH_DAYS']}")
            all_rows += rows_frac

    # Write CSV
    df = pd.DataFrame(all_rows,
        columns=["Metric","DiffType","Sector","Stat","MedianDays","P95Days","MeanDays","Npixels"])
    csv_path = out_dir / "summary_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] wrote {csv_path}")

    # Also dump a small JSON with year coverage for provenance
    meta = {"years_MS": None, "years_FS": None}
    try:
        meta["years_MS"] = compute_maps_for_metric("MS", cano, cfg)["years"]
        meta["years_FS"] = compute_maps_for_metric("FS", cano, cfg)["years"]
    except Exception:
        pass
    with open(out_dir / "summary_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
