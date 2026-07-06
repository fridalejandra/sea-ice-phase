#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_sector_mean_trends.py

Sector-mean linear trend slopes (days/year) for FS and MS, static vs dynamic,
restricted to active pixels (>=80% valid detection in BOTH methods + valid ocean).

Self-contained version: the data-loading, activity-mask, and trend-slope
functions below are copied verbatim from fig07_trends_static_dynamic.py so
these numbers are guaranteed consistent with the Fig. 7 run, without relying
on this script's location relative to that file (fixes the prior
ModuleNotFoundError when run from a different directory, e.g. Ch2/processing/).

If PROJECT_ROOT below doesn't match your actual path, fix that one line —
everything else is unchanged from fig07.

Usage:
  python compute_sector_mean_trends.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------
# CONFIG — matches fig07_trends_static_dynamic.py; adjust if this doesn't
# match your actual cluster path (PROJECT_ROOT_CLUSTER in utils/plot_utils.py)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
ANOM_DIR = PROJECT_ROOT / "data" / "anomalies" / "SMMR"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

PRE_START, PRE_END = 1979, 2015
POST_START, POST_END = 2016, 2024
MIN_FRAC_ACTIVE = 0.80

sector_ids = [1, 2, 3, 4, 5]
sector_labels = {
    1: "A–B",
    2: "WED",
    3: "KHV",
    4: "EA",
    5: "RA",
}


# ---------------------------------------------------------------------
# Copied verbatim from fig07_trends_static_dynamic.py
# ---------------------------------------------------------------------
def _open_da(path: Path, candidates: list[str], decode_times: bool = True) -> xr.DataArray:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    ds = xr.open_dataset(path, decode_times=decode_times)
    for name in candidates:
        if name in ds:
            da = ds[name].load()
            ds.close()
            return da

    vars_ = list(ds.data_vars)
    ds.close()
    raise KeyError(f"None of {candidates} found in {path}. Vars={vars_}")


def load_fs_ms_clim_anom() -> dict[str, xr.DataArray]:
    fs_dyn_clim = _open_da(ANOM_DIR / "FS_dynamic_k5_q70_climatology.nc", ["FS_dynamic_k5_q70_clim"], decode_times=False)
    fs_dyn_anom = _open_da(ANOM_DIR / "FS_dynamic_k5_q70_anomalies.nc", ["FS_dynamic_k5_q70_anom"], decode_times=False)

    ms_dyn_clim = _open_da(ANOM_DIR / "MS_dynamic_k5_q70_climatology.nc", ["MS_dynamic_k5_q70_clim_dsa", "MS_dynamic_k5_q70_clim"], decode_times=False)
    ms_dyn_anom = _open_da(ANOM_DIR / "MS_dynamic_k5_q70_anomalies.nc", ["MS_dynamic_k5_q70_anom_dsa", "MS_dynamic_k5_q70_anom"], decode_times=False)

    fs_sta_clim = _open_da(ANOM_DIR / "FS_static_thr15_k5_climatology.nc", ["FS_static_thr15_k5_clim"], decode_times=False)
    fs_sta_anom = _open_da(ANOM_DIR / "FS_static_thr15_k5_anomalies.nc", ["FS_static_thr15_k5_anom"], decode_times=False)

    ms_sta_clim = _open_da(ANOM_DIR / "MS_static_thr15_k5_climatology.nc", ["MS_static_thr15_k5_clim_dsa", "MS_static_thr15_k5_clim"], decode_times=False)
    ms_sta_anom = _open_da(ANOM_DIR / "MS_static_thr15_k5_anomalies.nc", ["MS_static_thr15_k5_anom_dsa", "MS_static_thr15_k5_anom"], decode_times=False)

    ds_mask = xr.open_dataset(SECTOR_FILE)
    valid_ocean = ds_mask["valid_ocean"].astype(bool)
    sector_mask = ds_mask["sector_id"]
    ds_mask.close()

    return {
        "FS_dynamic_clim": fs_dyn_clim, "FS_dynamic_anom": fs_dyn_anom,
        "MS_dynamic_clim": ms_dyn_clim, "MS_dynamic_anom": ms_dyn_anom,
        "FS_static_clim": fs_sta_clim, "FS_static_anom": fs_sta_anom,
        "MS_static_clim": ms_sta_clim, "MS_static_anom": ms_sta_anom,
        "valid_ocean": valid_ocean, "sector_mask": sector_mask,
    }


def make_activity_mask(
    anom_dyn: xr.DataArray,
    anom_sta: xr.DataArray,
    valid_ocean: xr.DataArray,
    frac_required: float = 0.80,
) -> xr.DataArray:
    n_years = float(anom_dyn.sizes["year"])
    dyn_frac = anom_dyn.notnull().sum("year") / n_years
    sta_frac = anom_sta.notnull().sum("year") / n_years
    active = (dyn_frac >= frac_required) & (sta_frac >= frac_required) & valid_ocean
    active.name = "active_mask"
    return active


def _slope_from_series(y: np.ndarray, years: np.ndarray) -> float:
    m = np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(years[m], y[m], 1)[0])


def compute_trend_slopes(anom_da: xr.DataArray) -> xr.DataArray:
    years = anom_da["year"].values.astype(float)
    slopes = xr.apply_ufunc(
        _slope_from_series,
        anom_da,
        kwargs={"years": years},
        input_core_dims=[["year"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    slopes.name = "slope_days_per_year"
    return slopes


# ---------------------------------------------------------------------
# New: sector-mean aggregation of trend slopes (the missing piece)
# ---------------------------------------------------------------------
def sector_mean_trends(
    slope_dyn: xr.DataArray,
    slope_sta: xr.DataArray,
    sector_mask: xr.DataArray,
    valid_ocean: xr.DataArray,
    active_mask: xr.DataArray,
    phase: str,
) -> pd.DataFrame:
    records = []
    for sec in sector_ids:
        mask = (sector_mask == sec) & valid_ocean & active_mask

        dyn_vals = slope_dyn.where(mask).values
        sta_vals = slope_sta.where(mask).values
        n_pix = int(mask.values.sum())

        for method, vals in [("Dynamic", dyn_vals), ("Static", sta_vals)]:
            records.append({
                "phase": phase,
                "sector_id": sec,
                "sector_label": sector_labels[sec],
                "method": method,
                "mean_slope_days_per_yr": float(np.nanmean(vals)),
                "median_slope_days_per_yr": float(np.nanmedian(vals)),
                "std_slope": float(np.nanstd(vals)),
                "n_active_pixels": n_pix,
            })
    return pd.DataFrame.from_records(records)


def main():
    print(f"Loading fields from {ANOM_DIR} (active criterion = {MIN_FRAC_ACTIVE:.2f})...")
    fields = load_fs_ms_clim_anom()
    valid_ocean = fields["valid_ocean"]
    sector_mask = fields["sector_mask"]

    fs_active = make_activity_mask(
        fields["FS_dynamic_anom"], fields["FS_static_anom"], valid_ocean,
        frac_required=MIN_FRAC_ACTIVE,
    )
    ms_active = make_activity_mask(
        fields["MS_dynamic_anom"], fields["MS_static_anom"], valid_ocean,
        frac_required=MIN_FRAC_ACTIVE,
    )

    print("Computing per-pixel trend slopes...")
    fs_slope_dyn = compute_trend_slopes(fields["FS_dynamic_anom"])
    fs_slope_sta = compute_trend_slopes(fields["FS_static_anom"])
    ms_slope_dyn = compute_trend_slopes(fields["MS_dynamic_anom"])
    ms_slope_sta = compute_trend_slopes(fields["MS_static_anom"])

    df_fs = sector_mean_trends(fs_slope_dyn, fs_slope_sta, sector_mask, valid_ocean, fs_active, "FS")
    df_ms = sector_mean_trends(ms_slope_dyn, ms_slope_sta, sector_mask, valid_ocean, ms_active, "MS")
    df = pd.concat([df_fs, df_ms], ignore_index=True)

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n=== Sector-mean linear trend slopes (days/year), active80 pixels ===\n")
    print(df.to_string(index=False))

    out_csv = PROJECT_ROOT / "results" / "sector_mean_trends_static_vs_dynamic.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    print("\n=== Range summary (for manuscript text) ===")
    for phase in ["FS", "MS"]:
        for method in ["Dynamic", "Static"]:
            sub = df[(df["phase"] == phase) & (df["method"] == method)]
            lo = sub["mean_slope_days_per_yr"].min()
            hi = sub["mean_slope_days_per_yr"].max()
            print(f"{phase} {method}: {lo:+.3f} to {hi:+.3f} days/yr across sectors")

    print("\n=== Sign agreement check, MS static vs dynamic, per sector ===")
    ms_pivot = df_ms.pivot(index="sector_label", columns="method", values="mean_slope_days_per_yr")
    ms_pivot["same_sign"] = np.sign(ms_pivot["Static"]) == np.sign(ms_pivot["Dynamic"])
    print(ms_pivot.to_string())


if __name__ == "__main__":
    main()