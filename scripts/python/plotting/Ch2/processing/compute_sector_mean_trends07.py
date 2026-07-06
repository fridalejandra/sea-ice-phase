#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_sector_mean_trends.py

Sector-mean linear trend slopes (days/year) for FS and MS, static vs dynamic,
restricted to active pixels (>=80% valid detection in BOTH methods + valid ocean).

Drop this in the same directory as fig07_trends_static_dynamic.py. It imports
that script's own load_fs_ms_clim_anom(), make_activity_mask(), and
compute_trend_slopes() functions directly, so these sector-mean numbers are
guaranteed consistent with whatever Fig. 7 run you're citing — no risk of a
second, drifted implementation producing different values.

Usage:
  python compute_sector_mean_trends.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
PLOTTING_ROOT = HERE.parent
if str(PLOTTING_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTTING_ROOT))

from utils.plot_utils import PROJECT_ROOT_CLUSTER  # noqa: E402

# Reuse the exact same loading / masking / trend functions as Fig. 7
from fig07_trends_static_dynamic import (  # noqa: E402
    load_fs_ms_clim_anom,
    make_activity_mask,
    compute_trend_slopes,
    sector_ids,
    sector_labels,
    MIN_FRAC_ACTIVE,
)


def sector_mean_trends(
    slope_dyn: xr.DataArray,
    slope_sta: xr.DataArray,
    sector_mask: xr.DataArray,
    valid_ocean: xr.DataArray,
    active_mask: xr.DataArray,
    phase: str,
) -> pd.DataFrame:
    """Sector-mean trend slope (days/year), static vs dynamic, active pixels only."""
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
    print(f"Loading fields (active criterion = {MIN_FRAC_ACTIVE:.2f})...")
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

    out_csv = PROJECT_ROOT_CLUSTER / "results" / "sector_mean_trends_static_vs_dynamic.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Range summary in the exact prose style used in the current 4.1.6 draft
    # ("-0.05 to -0.23 days/year", "+0.05 to +0.35 days/year")
    print("\n=== Range summary (for manuscript text) ===")
    for phase in ["FS", "MS"]:
        for method in ["Dynamic", "Static"]:
            sub = df[(df["phase"] == phase) & (df["method"] == method)]
            lo = sub["mean_slope_days_per_yr"].min()
            hi = sub["mean_slope_days_per_yr"].max()
            print(f"{phase} {method}: {lo:+.3f} to {hi:+.3f} days/yr across sectors")

    # Explicit sign check per sector — this is the number that either
    # supports or kills the "opposite-sign trend" argument in the draft.
    print("\n=== Sign agreement check, MS static vs dynamic, per sector ===")
    ms_pivot = df_ms.pivot(index="sector_label", columns="method", values="mean_slope_days_per_yr")
    ms_pivot["same_sign"] = np.sign(ms_pivot["Static"]) == np.sign(ms_pivot["Dynamic"])
    print(ms_pivot.to_string())


if __name__ == "__main__":
    main()