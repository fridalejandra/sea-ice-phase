#!/usr/bin/env python3
"""
window_sensitivity_maps.py

Outputs:
  1) Per-pixel PNG maps (Cartopy) for mean(|Δ|), std(|Δ|), q95(|Δ|) for {MS,FS} × {3v5,7v5}
  2) CSV with sectoral + circumpolar summaries (median, p95, mean, N)
  3) (Optional) rclone upload of PNGs to a remote folder
"""

from pathlib import Path
import os, glob, re, json
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
from matplotlib.colors import LogNorm


# =========================
# CONFIG — EDIT THIS BLOCK
# =========================
CFG = {
    "SENSOR": "SMMR",
    "THRESH_PCT": 15,
    "INPUT_ROOT": "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase",
    "CANONICAL":  "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc",

    "OUT_DIR": "/user/geog/falejandraperez/sea-ice-phase/results/window_sensitivity",
    "PERIOD": 366,

    # styling
    "MEAN_VMAX": 10,   # colorbar max for mean(|Δ|)
    "STD_VMAX": 8,     # colorbar max for std(|Δ|)
    "Q95_VMAX": 10,    # colorbar max for q95(|Δ|)
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
    Returns per-pixel maps for the metric (MS/FS):
      { "3v5": {"mean": DA, "std": DA, "q95": DA},
        "7v5": {"mean": DA, "std": DA, "q95": DA},
        "years": [int,...] }
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

    # Reduce across years
    mean35 = D35.mean(dim="year", skipna=True)
    std35  = D35.std(dim="year",  skipna=True)
    q95_35 = D35.quantile(0.95, dim="year", skipna=True)

    mean75 = D75.mean(dim="year", skipna=True)
    std75  = D75.std(dim="year",  skipna=True)
    q95_75 = D75.quantile(0.95, dim="year", skipna=True)

    # mask to valid ocean
    vo = cano["valid_ocean"].astype(bool)
    for da in [mean35, std35, q95_35, mean75, std75, q95_75]:
        da.values[~vo.values] = np.nan

    return {
        "3v5": {"mean": mean35, "std": std35, "q95": q95_35},
        "7v5": {"mean": mean75, "std": std75, "q95": q95_75},
        "years": years
    }

def compute_year_series_for_metric(metric, cano, cfg=CFG):
    """
    Returns a DataFrame with per-year mean(|Δ|) for circumpolar and per-sector,
    for both 3v5 and 7v5. Columns: year, Metric, DiffType, Region, MeanAbsDiff
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
    D35 = xr.apply_ufunc(wrapped_abs_diff, A3, A5, period, dask="allowed")  # (year,y,x)
    D75 = xr.apply_ufunc(wrapped_abs_diff, A7, A5, period, dask="allowed")

    vo  = cano["valid_ocean"].astype(bool)
    sid = cano["sector_id"].astype(np.int16)

    rows = []
    for yi, yr in enumerate(D35["year"].values):
        # circumpolar
        v35 = D35.isel(year=yi).where(vo).values
        v75 = D75.isel(year=yi).where(vo).values
        v35 = v35[np.isfinite(v35)]
        v75 = v75[np.isfinite(v75)]
        if v35.size:
            rows.append({"year": int(yr), "Metric": metric, "DiffType": "3v5",
                         "Region": "Circumpolar", "MeanAbsDiff": float(np.nanmean(v35))})
        if v75.size:
            rows.append({"year": int(yr), "Metric": metric, "DiffType": "7v5",
                         "Region": "Circumpolar", "MeanAbsDiff": float(np.nanmean(v75))})

        # per-sector
        for k, name in SECTOR_ID_TO_NAME.items():
            mask_k = ((sid == k) & vo).values
            sv35 = D35.isel(year=yi).values
            sv75 = D75.isel(year=yi).values
            a35 = sv35[mask_k]; a75 = sv75[mask_k]
            a35 = a35[np.isfinite(a35)]; a75 = a75[np.isfinite(a75)]
            if a35.size:
                rows.append({"year": int(yr), "Metric": metric, "DiffType": "3v5",
                             "Region": name, "MeanAbsDiff": float(np.nanmean(a35))})
            if a75.size:
                rows.append({"year": int(yr), "Metric": metric, "DiffType": "7v5",
                             "Region": name, "MeanAbsDiff": float(np.nanmean(a75))})
    return pd.DataFrame(rows)


def plot_trend_lines(df, out_png, title, cfg=CFG, per_sector=False):
    """
    If per_sector=False: circumpolar only (two lines: 3v5, 7v5) per metric.
    If per_sector=True : small multiples, one panel per sector.
    """
    import matplotlib.pyplot as plt
    years = sorted(df["year"].unique())

    if not per_sector:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for difftype, style in [("3v5", "-"), ("7v5", "--")]:
            sub = df[(df["Region"]=="Circumpolar") & (df["DiffType"]==difftype)]
            # average both metrics if both present? — here caller should pass filtered df per metric.
            sub = sub.groupby("year")["MeanAbsDiff"].mean().reindex(years)
            ax.plot(years, sub.values, style, lw=2, label=difftype)
        ax.set_xlabel("Year"); ax.set_ylabel("mean(|Δ|) (days)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Diff")
        plt.tight_layout()
        plt.savefig(out_png, dpi=cfg["DPI"])
        plt.close()
        rclone_copy(out_png, cfg)
        print(f"[OK] wrote {out_png}")
        return

    # per-sector small multiples
    sectors = list(SECTOR_ID_TO_NAME.values())
    n = len(sectors)
    ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*2.6), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, name in enumerate(sectors):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        for difftype, style in [("3v5", "-"), ("7v5", "--")]:
            sub = df[(df["Region"]==name) & (df["DiffType"]==difftype)]
            sub = sub.groupby("year")["MeanAbsDiff"].mean().reindex(years)
            ax.plot(years, sub.values, style, lw=1.8, label=difftype)
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3)
    # hide extras
    total = nrows*ncols
    for j in range(n, total):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)
    fig.supxlabel("Year"); fig.supylabel("mean(|Δ|) (days)")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=True)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(out_png, dpi=cfg["DPI"])
    plt.close()
    rclone_copy(out_png, cfg)
    print(f"[OK] wrote {out_png}")


def plot_joint_hist(da_mean35, da_mean75, cano, metric, out_png, cfg=CFG):
    """
    Hexbin joint histogram of per-pixel mean |Δ₃₋₅| (x) vs mean |Δ₇₋₅| (y).
    Annotates slope, intercept, and Pearson r.
    """
    vo = cano["valid_ocean"].astype(bool).values
    x = da_mean35.where(vo).values.ravel()
    y = da_mean75.where(vo).values.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    # stats
    if x.size > 0:
        slope, intercept = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0,1]
    else:
        slope, intercept, r = np.nan, np.nan, np.nan

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    hb = ax.hexbin(x, y, gridsize=60, norm=LogNorm(), mincnt=1)
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("pixel count (log)")

    lim = max(np.nanmax(x), np.nanmax(y))
    lim = 0 if not np.isfinite(lim) else float(lim)
    lim = max(lim, 6)  # give some headroom
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="1:1")
    xx = np.linspace(0, lim, 100)
    ax.plot(xx, slope*xx + intercept, color="C1", lw=1.6, label=f"fit: y={slope:.2f}x+{intercept:.2f}")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("mean(|Δ₃₋₅|) (days)")
    ax.set_ylabel("mean(|Δ₇₋₅|) (days)")
    ax.set_title(f"Joint histogram • {metric}\n$r={r:.2f}$")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=cfg["DPI"])
    plt.close()
    rclone_copy(out_png, cfg)
    print(f"[OK] wrote {out_png}")

# ======================
# MAPPING (Cartopy)
# ======================
def draw_sector_meridians(ax):
    import numpy as np
    lats = np.linspace(-90, -45, 256)               # only the visible dome
    for bE in SECTOR_BOUNDARIES_E:
        # convert degE to [-180, 180] for plotting
        lon = bE if bE <= 180 else bE - 360
        lons = np.full_like(lats, lon, dtype=float)
        # plot as geodesic so Cartopy handles wrap/segmenting correctly
        ax.plot(lons, lats, transform=ccrs.Geodetic(),
                color="grey", linewidth=0.6, alpha=0.5, zorder=6,linestyle="--")


def _round_polar_axes(ax):
    # Circular boundary (like your screenshot)
    theta = np.linspace(0, 2*np.pi, 200)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

def plot_map_cartopy(da, cano, title, out_png, vmin, vmax, cmap="viridis", white_under=True, cfg=CFG):
    proj = ccrs.SouthPolarStereo()
    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw={"projection": proj})

    # base: land + coastlines
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="0.2", linewidth=0.4, zorder=2)
    ax.coastlines(resolution="110m", color="0.2", linewidth=0.5, zorder=3)

    # data
    lon = cano["lon"]; lat = cano["lat"]
    cmap_obj = plt.get_cmap(cmap).copy()
    if white_under:
        cmap_obj.set_under("white")
    # tiny epsilon so zeros render as 'under' (white)
    eps = 1e-6 if vmin == 0 else 0.0
    pc = ax.pcolormesh(lon, lat, da,
                       transform=ccrs.PlateCarree(),
                       vmin=vmin + eps, vmax=vmax,
                       cmap=cmap_obj, zorder=4)

    # colorbar (horizontal for cleaner layout)
    cb = plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.02, shrink=0.85)
    cb.ax.set_xlabel(title.split("•")[0].strip(), fontsize=9)

    # sector meridians
    draw_sector_meridians(ax)

    # extent + round frame
    ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
    _round_polar_axes(ax)

    # title
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
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

    cano = xr.open_dataset(cfg["CANONICAL"]).load()

    all_rows = []

    for metric in ["MS", "FS"]:
        maps = compute_maps_for_metric(metric, cano, cfg)

        for difftype, group in [("3v5", maps["3v5"]), ("7v5", maps["7v5"])]:
            # === maps → PNG ===
            plot_map_cartopy(group["mean"], cano,
                             title=f"mean(|Δ|) days • {metric} • {difftype}",
                             out_png=maps_dir / f"mean_absdiff_{metric}_{difftype}.png",
                             vmin=0, vmax=cfg["MEAN_VMAX"], cmap="viridis", white_under=True, cfg=cfg)

            plot_map_cartopy(group["std"], cano,
                             title=f"std(|Δ|) days • {metric} • {difftype}",
                             out_png=maps_dir / f"std_absdiff_{metric}_{difftype}.png",
                             vmin=0, vmax=cfg["STD_VMAX"], cmap="magma", white_under=True, cfg=cfg)

            plot_map_cartopy(group["q95"], cano,
                             title=f"q95(|Δ|) days • {metric} • {difftype}",
                             out_png=maps_dir / f"q95_absdiff_{metric}_{difftype}.png",
                             vmin=0, vmax=cfg["Q95_VMAX"], cmap="plasma", white_under=True, cfg=cfg)

            # === stats → rows ===
            all_rows += summarize_to_rows(group["mean"], cano, metric, difftype, statname="mean_absdiff")
            all_rows += summarize_to_rows(group["std"],  cano, metric, difftype, statname="std_absdiff")
            all_rows += summarize_to_rows(group["q95"],  cano, metric, difftype, statname="q95_absdiff")

    # CSV
    df = pd.DataFrame(all_rows,
        columns=["Metric","DiffType","Sector","Stat","MedianDays","P95Days","MeanDays","Npixels"])
    csv_path = out_dir / "summary_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] wrote {csv_path}")

    # === TEMPORAL TRENDS & JOINT HISTOGRAMS ===
    trend_rows = []
    trends_dir = out_dir / "trends"
    trends_dir.mkdir(exist_ok=True, parents=True)
    joint_dir = out_dir / "joint"
    joint_dir.mkdir(exist_ok=True, parents=True)

    for metric in ["MS", "FS"]:
        # Yearly circumpolar + sector series
        df_year = compute_year_series_for_metric(metric, cano, cfg)
        trend_rows.append(df_year)

        # Plot circumpolar trend (two lines: 3v5, 7v5)
        df_circ = df_year[df_year["Region"]=="Circumpolar"]
        plot_trend_lines(df_circ,
                         out_png=trends_dir / f"trend_circ_{metric}.png",
                         title=f"Circumpolar mean(|Δ|) per year • {metric}",
                         cfg=cfg, per_sector=False)

        # Optional: sector small multiples
        plot_trend_lines(df_year,
                         out_png=trends_dir / f"trend_sectors_{metric}.png",
                         title=f"Sectoral mean(|Δ|) per year • {metric}",
                         cfg=cfg, per_sector=True)

        # Joint histogram inputs: per-pixel mean across years
        maps = compute_maps_for_metric(metric, cano, cfg)
        mean35 = maps["3v5"]["mean"]  # already mean over years
        mean75 = maps["7v5"]["mean"]
        plot_joint_hist(mean35, mean75, cano,
                        metric=metric,
                        out_png=joint_dir / f"joint_hist_mean_absdiff_{metric}.png",
                        cfg=cfg)

    # Write trends CSV
    df_trends = pd.concat(trend_rows, ignore_index=True)
    df_trends.to_csv(out_dir / "trend_timeseries.csv", index=False)
    print(f"[OK] wrote {out_dir / 'trend_timeseries.csv'}")

    # Minimal provenance: year coverage
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
