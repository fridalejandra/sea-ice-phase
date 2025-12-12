#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostics for a pixel transect:
- SIC(t) and ΔSIC(t) for YEAR
- SIC and ΔSIC distributions (histograms)
- Vichi-style variability metric (sigma of daily anomalies vs monthly climatology)
- Simple classification: variable_MIZ_like if sigma >= SIGMA_THRESH
- Writes per-pixel PNGs + a summary CSV
- Pushes outputs to Google Drive via rclone (optional)
"""

import os
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ---------------- CONFIG ---------------- #
INPUT_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
CONC_VAR   = "N07_ICECON"

PIXEL_JSON = "/user/geog/falejandraperez/sea-ice-phase/results/diagnostics/ABS_transects/abs_transect_pixels.json"

OUT_DIR    = "/user/geog/falejandraperez/sea-ice-phase/results/diagnostics/ABS_transects/under_the_hood"

YEAR = 2005

# Vichi-style threshold (treat as tunable; this is just a starting point)
SIGMA_THRESH = 0.10

# Optional smoothing for readability only (raw is still plotted too)
SMOOTH_DAYS = 5  # set to None to disable

# ---- rclone config (override via env vars) ----
RCLONE_PUSH   = True
RCLONE_REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive")
RCLONE_DEST   = os.environ.get("RCLONE_DEST", "sea-ice-phase/results/diagnostics/ABS_transects")


# ---------------- HELPERS ---------------- #
def standardize_calendar(da: xr.DataArray):
    return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))

def prep_ice(ds: xr.Dataset) -> xr.DataArray:
    ice = ds[CONC_VAR].astype("float32")
    if float(ice.max()) > 1.5:
        ice = ice / 100.0
    ice = ice.where(ice < 1.1)
    return ice

def select_year(da: xr.DataArray, year: int):
    return da.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))

def vichi_sigma_monthly(ts: xr.DataArray) -> float:
    """
    Vichi-style: monthly climatology, daily anomalies, then SD of anomalies.
    Returns one pooled sigma across the whole record for that pixel.
    """
    ts = ts.where(np.isfinite(ts), drop=True)
    if ts.time.size < 365:  # be strict: need enough samples
        return np.nan
    clim = ts.groupby("time.month").mean("time", skipna=True)
    anom = ts.groupby("time.month") - clim
    return float(anom.std("time", skipna=True).values)

def flicker_count(ts: np.ndarray, thr: float) -> float:
    """Number of sign changes of (ts - thr), ignoring NaNs."""
    m = np.isfinite(ts)
    x = ts[m] - thr
    if x.size < 2:
        return np.nan
    s = np.sign(x)
    for k in range(1, s.size):
        if s[k] == 0:
            s[k] = s[k-1]
    return int(np.sum(s[1:] * s[:-1] < 0))

def rclone_push(local_dir: str, remote: str, remote_dir: str, include_ext=(".png", ".csv")):
    local_dir = str(local_dir)
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    filters = []
    for ext in include_ext:
        filters += ["--include", f"*{ext}"]
    filters += ["--exclude", "*"]

    cmd = [
        "rclone", "copy",
        local_dir,
        f"{remote}:{remote_dir}",
        "--update",
        "--create-empty-src-dirs",
        "--transfers", "8",
        "--checkers", "16",
        "--stats", "15s",
    ] + filters

    print("\n[RCLOUD] Pushing outputs to Google Drive via rclone:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    with open(PIXEL_JSON, "r") as f:
        meta = json.load(f)
    pixels = meta["pixels"]

    ds = xr.open_dataset(INPUT_FILE)[[CONC_VAR, "x", "y", "time"]]
    ice = prep_ice(ds)
    ice365 = standardize_calendar(ice)

    ice_year = select_year(ice365, YEAR)
    if ice_year.time.size == 0:
        raise RuntimeError(f"No data for YEAR={YEAR} after calendar standardization.")

    rows = []

    for p in pixels:
        name = p["name"]
        j, i = int(p["y"]), int(p["x"])

        ts_full = ice365.isel(y=j, x=i)
        ts_year = ice_year.isel(y=j, x=i)

        if np.all(np.isnan(ts_year.values)):
            print(f"{name}: all NaN in YEAR={YEAR}; skipping.")
            continue

        # optional smoothing for readability
        ts_smooth = None
        if SMOOTH_DAYS is not None:
            ts_smooth = ts_year.rolling(time=SMOOTH_DAYS, center=True).mean()

        vals = ts_year.values.astype(float)
        t    = ts_year.time.values
        m = np.isfinite(vals)
        vals = vals[m]
        t    = t[m]
        if vals.size < 20:
            print(f"{name}: too few valid points; skipping.")
            continue

        dC = np.diff(vals)
        t_mid = t[1:]

        sigma = vichi_sigma_monthly(ts_full)
        var_class = "variable_MIZ_like" if (np.isfinite(sigma) and sigma >= SIGMA_THRESH) else "stable_pack_like"

        rows.append({
            "name": name, "y": j, "x": i, "year": YEAR,
            "sigma_vichi": sigma, "class": var_class,
            "frac_SIC_lt15": float(np.mean(vals < 0.15)),
            "frac_SIC_15_80": float(np.mean((vals >= 0.15) & (vals <= 0.80))),
            "frac_SIC_gt80": float(np.mean(vals > 0.80)),
            "max_pos_dC": float(np.nanmax(dC)) if dC.size else np.nan,
            "max_neg_dC": float(np.nanmin(dC)) if dC.size else np.nan,
            "flicker_thr15": flicker_count(vals, 0.15),
            "flicker_thr60": flicker_count(vals, 0.60),
        })

        # ---- Plot: timeseries + dC ----
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        ax1.plot(ts_year.time.values, ts_year.values, lw=1, label="SIC (raw)")
        if ts_smooth is not None:
            ax1.plot(ts_smooth.time.values, ts_smooth.values, lw=1, label=f"SIC ({SMOOTH_DAYS}-day mean)")
        ax1.set_ylabel("SIC (0–1)")
        ax1.set_title(f"{name} (y={j}, x={i}) | YEAR={YEAR} | sigma={sigma:.3f} | {var_class}")
        ax1.legend(loc="best", fontsize=8)

        ax2.plot(t_mid, dC, lw=1)
        ax2.axhline(0.0, lw=0.8, ls="--")
        ax2.set_ylabel("ΔSIC/day")
        ax2.set_xlabel("Date")

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/ts_dC_{name}_Y{YEAR}.png", dpi=150)
        plt.close(fig)

        # ---- Plot: distributions ----
        fig, (axh1, axh2) = plt.subplots(1, 2, figsize=(11, 4))
        axh1.hist(vals, bins=30)
        axh1.set_title("SIC distribution")
        axh1.set_xlabel("SIC"); axh1.set_ylabel("count")

        axh2.hist(dC, bins=30)
        axh2.set_title("ΔSIC distribution")
        axh2.set_xlabel("ΔSIC/day"); axh2.set_ylabel("count")

        fig.suptitle(f"{name} | sigma={sigma:.3f} | {var_class}")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/hists_{name}_Y{YEAR}.png", dpi=150)
        plt.close(fig)

        print(f"Wrote plots for {name}")

    df = pd.DataFrame(rows).sort_values(["class", "sigma_vichi"], ascending=[True, False])
    out_csv = f"{OUT_DIR}/transect_summary_Y{YEAR}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    if RCLONE_PUSH:
        try:
            rclone_push(OUT_DIR, RCLONE_REMOTE, f"{RCLONE_DEST}/under_the_hood", include_ext=(".png", ".csv"))
        except Exception as e:
            print(f"[RCLOUD] rclone push failed (continuing): {e}")


if __name__ == "__main__":
    main()
