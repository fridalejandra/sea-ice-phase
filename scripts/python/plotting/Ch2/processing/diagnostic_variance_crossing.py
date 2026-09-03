#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnostics_variance_crossing.py

Three diagnostics for the "contraction vs. increased volatility" question,
run together in one pass:

  PART 1 — Pre/post-2016 variance of phase-date anomalies (FS/MS, static &
           dynamic, by sector), tested with Levene's test. Cheap: uses the
           anomaly files you already have, no raw SIC access.

  PART 2 — Static-vs-dynamic disagreement rate per year (fraction of active
           pixels where |static - dynamic| > 7 days, matching the Fig. 7
           step-change sign-class threshold). Cheap: same anomaly files.

  PART 3 — Multi-crossing frequency within the FS/MS search windows,
           sector-mean, pre vs. post 2016. This is the only expensive part —
           it reads the raw daily SIC record — but is restricted to already-
           active pixels and processed one year at a time to stay light.

All three use the SAME active80 mask and the SAME 2016 breakpoint as
Fig. 7/8, so results are directly comparable to what's already in the draft.

If PROJECT_ROOT / RAW_SIC_FILE below don't match your actual paths, fix
those two lines first — everything else follows the same conventions as
fig07_trends_static_dynamic.py and compute_phase_dates_v2.py.

Usage:
  python diagnostics_variance_crossing.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# ---------------------------------------------------------------------
# CONFIG — matches fig07_trends_static_dynamic.py / compute_phase_dates_v2.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
ANOM_DIR = PROJECT_ROOT / "data" / "anomalies" / "SMMR"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"
RAW_SIC_FILE = PROJECT_ROOT / "data" / "merged" / "SMMR_merged_19781101_20251231_complete.nc"
CONC_VAR = "N07_ICECON"
MASK_ABOVE = 1.1
BAD_YEARS = [1987, 1991, 1995]

PRE_START, PRE_END = 1979, 2015
POST_START, POST_END = 2016, 2024
MIN_FRAC_ACTIVE = 0.80
DISAGREE_THRESH_DAYS = 7.0   # matches Fig. 7 step-change sign-class threshold
BASELINE_THR = 0.15

FS_START_MMDD, FS_END_MMDD = "-02-15", "-09-30"
MS_START_MMDD, MS_END_MMDD = "-08-15", "-02-28"

sector_ids = [1, 2, 3, 4, 5]
sector_labels = {1: "A–B", 2: "WED", 3: "KHV", 4: "EA", 5: "RA"}


# ---------------------------------------------------------------------
# Shared loaders (copied/adapted from fig07_trends_static_dynamic.py)
# ---------------------------------------------------------------------
def _open_da(path: Path, candidates: list[str]) -> xr.DataArray:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    ds = xr.open_dataset(path, decode_times=False)
    for name in candidates:
        if name in ds:
            da = ds[name].load()
            ds.close()
            return da
    vars_ = list(ds.data_vars)
    ds.close()
    raise KeyError(f"None of {candidates} found in {path}. Vars={vars_}")


def load_fs_ms_clim_anom() -> dict:
    fs_dyn_clim = _open_da(ANOM_DIR / "FS_dynamic_k5_q70_climatology.nc", ["FS_dynamic_k5_q70_clim"])
    fs_dyn_anom = _open_da(ANOM_DIR / "FS_dynamic_k5_q70_anomalies.nc", ["FS_dynamic_k5_q70_anom"])
    ms_dyn_clim = _open_da(ANOM_DIR / "MS_dynamic_k5_q70_climatology.nc", ["MS_dynamic_k5_q70_clim_dsa", "MS_dynamic_k5_q70_clim"])
    ms_dyn_anom = _open_da(ANOM_DIR / "MS_dynamic_k5_q70_anomalies.nc", ["MS_dynamic_k5_q70_anom_dsa", "MS_dynamic_k5_q70_anom"])
    fs_sta_clim = _open_da(ANOM_DIR / "FS_static_thr15_k5_climatology.nc", ["FS_static_thr15_k5_clim"])
    fs_sta_anom = _open_da(ANOM_DIR / "FS_static_thr15_k5_anomalies.nc", ["FS_static_thr15_k5_anom"])
    ms_sta_clim = _open_da(ANOM_DIR / "MS_static_thr15_k5_climatology.nc", ["MS_static_thr15_k5_clim_dsa", "MS_static_thr15_k5_clim"])
    ms_sta_anom = _open_da(ANOM_DIR / "MS_static_thr15_k5_anomalies.nc", ["MS_static_thr15_k5_anom_dsa", "MS_static_thr15_k5_anom"])

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


def make_activity_mask(anom_dyn, anom_sta, valid_ocean, frac_required=0.80):
    n_years = float(anom_dyn.sizes["year"])
    dyn_frac = anom_dyn.notnull().sum("year") / n_years
    sta_frac = anom_sta.notnull().sum("year") / n_years
    return (dyn_frac >= frac_required) & (sta_frac >= frac_required) & valid_ocean


# ---------------------------------------------------------------------
# PART 1 — pre/post-2016 variance of phase anomalies (Levene's test)
# ---------------------------------------------------------------------
def part1_variance_test(fields, fs_active, ms_active) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("PART 1: Pre/post-2016 variance of phase-date anomalies (Levene's test)")
    print("=" * 72)

    combos = [
        ("FS", "Dynamic", fields["FS_dynamic_anom"], fs_active),
        ("FS", "Static", fields["FS_static_anom"], fs_active),
        ("MS", "Dynamic", fields["MS_dynamic_anom"], ms_active),
        ("MS", "Static", fields["MS_static_anom"], ms_active),
    ]

    records = []
    for phase, method, anom, active in combos:
        years = anom["year"].values
        pre = anom.sel(year=years[(years >= PRE_START) & (years <= PRE_END)])
        post = anom.sel(year=years[(years >= POST_START) & (years <= POST_END)])

        for sec in sector_ids:
            mask = (fields["sector_mask"] == sec) & active
            pre_vals = pre.where(mask).values.ravel()
            post_vals = post.where(mask).values.ravel()
            pre_vals = pre_vals[np.isfinite(pre_vals)]
            post_vals = post_vals[np.isfinite(post_vals)]

            if len(pre_vals) < 10 or len(post_vals) < 10:
                continue

            stat, p = stats.levene(pre_vals, post_vals)
            records.append({
                "phase": phase, "method": method, "sector": sector_labels[sec],
                "pre_std": np.std(pre_vals), "post_std": np.std(post_vals),
                "std_ratio_post_over_pre": np.std(post_vals) / np.std(pre_vals),
                "levene_p": p, "n_pre": len(pre_vals), "n_post": len(post_vals),
            })

    df = pd.DataFrame.from_records(records)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df.to_string(index=False))
    print("\nNote: only 9 post-2016 years feed this test — same thin-sample")
    print("caveat that already applies to the Fig. 8 post-2016 trend slopes.")
    return df


# ---------------------------------------------------------------------
# PART 2 — static/dynamic disagreement rate per year
# ---------------------------------------------------------------------
def part2_disagreement_trend(fields, fs_active, ms_active) -> None:
    print("\n" + "=" * 72)
    print(f"PART 2: Static-vs-dynamic disagreement rate per year (|diff| > {DISAGREE_THRESH_DAYS:.0f} days)")
    print("=" * 72)

    for phase, clim_dyn, anom_dyn, clim_sta, anom_sta, active in [
        ("FS", fields["FS_dynamic_clim"], fields["FS_dynamic_anom"],
         fields["FS_static_clim"], fields["FS_static_anom"], fs_active),
        ("MS", fields["MS_dynamic_clim"], fields["MS_dynamic_anom"],
         fields["MS_static_clim"], fields["MS_static_anom"], ms_active),
    ]:
        raw_dyn = anom_dyn + clim_dyn
        raw_sta = anom_sta + clim_sta
        diff = np.abs(raw_dyn - raw_sta)

        years = diff["year"].values
        frac_disagree = []
        for y in years:
            d = diff.sel(year=y).where(active).values
            d = d[np.isfinite(d)]
            frac_disagree.append(float(np.mean(d > DISAGREE_THRESH_DAYS)) if len(d) else np.nan)

        s = pd.Series(frac_disagree, index=years, name=f"{phase}_disagree_frac")
        pre_mean = s[(s.index >= PRE_START) & (s.index <= PRE_END)].mean()
        post_mean = s[(s.index >= POST_START) & (s.index <= POST_END)].mean()

        print(f"\n{phase}: pre-2016 mean disagreement rate = {pre_mean:.3f}, "
              f"post-2016 = {post_mean:.3f}, delta = {post_mean - pre_mean:+.3f}")
        print(s.to_string())


# ---------------------------------------------------------------------
# PART 3 — multi-crossing frequency in the search window (needs raw SIC)
# ---------------------------------------------------------------------
def load_sic_lazy() -> xr.DataArray:
    ds = xr.open_dataset(RAW_SIC_FILE, chunks={"time": 200})
    ice = ds[CONC_VAR].astype("float32")
    ice = ice.where(ice <= MASK_ABOVE)
    ice = ice.sel(time=~((ice.time.dt.month == 2) & (ice.time.dt.day == 29)))
    ice = ice.sortby("time")
    ice = ice.sel(time=~ice.time.dt.year.isin(BAD_YEARS))
    return ice


def count_crossings_for_year(ice: xr.DataArray, year: int, phase: str,
                              threshold: float = BASELINE_THR) -> np.ndarray:
    """Count sign changes across `threshold` within one year's search window."""
    if phase == "FS":
        start, end = f"{year}{FS_START_MMDD}", f"{year}{FS_END_MMDD}"
    else:  # MS wraps the year boundary
        start, end = f"{year}{MS_START_MMDD}", f"{year + 1}{MS_END_MMDD}"

    sic = ice.sel(time=slice(start, end)).values  # (time, y, x) — this window only
    above = sic > threshold
    changes = np.abs(np.diff(above.astype(np.int8), axis=0))
    return changes.sum(axis=0)  # (y, x) crossing count for this pixel-year


def part3_crossing_frequency(fields, fs_active, ms_active) -> None:
    print("\n" + "=" * 72)
    print("PART 3: Multi-crossing frequency within search window (raw daily SIC)")
    print("This part reads the raw daily record and will take longer than Parts 1-2.")
    print("=" * 72)

    ice = load_sic_lazy()
    all_years = sorted(set(int(y) for y in ice.time.dt.year.values))

    for phase, active in [("FS", fs_active), ("MS", ms_active)]:
        active_np = active.values
        records = []
        years_to_run = [y for y in all_years if y not in BAD_YEARS and PRE_START <= y <= POST_END]

        for y in years_to_run:
            try:
                counts = count_crossings_for_year(ice, y, phase)
            except Exception as e:
                print(f"  skipping {phase} {y}: {e}")
                continue

            if counts.shape != fields["sector_mask"].shape:
                raise ValueError(
                    f"Grid mismatch: crossing counts {counts.shape} vs "
                    f"sector_mask {fields['sector_mask'].shape}. Check that the raw "
                    f"SIC grid matches canonical_sectors.nc."
                )

            for sec in sector_ids:
                mask = (fields["sector_mask"].values == sec) & active_np
                vals = counts[mask]
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                records.append({
                    "phase": phase, "year": y, "sector": sector_labels[sec],
                    "mean_crossings": float(np.mean(vals)),
                })

        df = pd.DataFrame.from_records(records)
        if df.empty:
            print(f"  No data produced for {phase}.")
            continue

        print(f"\n{phase} — mean crossing count per active pixel, pre vs. post 2016:")
        for sec in sector_labels.values():
            sub = df[df["sector"] == sec]
            pre = sub[(sub["year"] >= PRE_START) & (sub["year"] <= PRE_END)]["mean_crossings"].mean()
            post = sub[(sub["year"] >= POST_START) & (sub["year"] <= POST_END)]["mean_crossings"].mean()
            print(f"  {sec}: pre-2016 = {pre:.2f}, post-2016 = {post:.2f}, delta = {post - pre:+.2f}")

        out_csv = PROJECT_ROOT / "results" / f"crossing_frequency_{phase}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"  Saved per-year detail: {out_csv}")


def main():
    print(f"Loading phase-date fields (active criterion = {MIN_FRAC_ACTIVE:.2f})...")
    fields = load_fs_ms_clim_anom()
    fs_active = make_activity_mask(fields["FS_dynamic_anom"], fields["FS_static_anom"], fields["valid_ocean"], MIN_FRAC_ACTIVE)
    ms_active = make_activity_mask(fields["MS_dynamic_anom"], fields["MS_static_anom"], fields["valid_ocean"], MIN_FRAC_ACTIVE)

    part1_variance_test(fields, fs_active, ms_active)
    part2_disagreement_trend(fields, fs_active, ms_active)
    part3_crossing_frequency(fields, fs_active, ms_active)

    print("\nDone. All three diagnostics use the same active80 mask and 2016 breakpoint as Fig. 7/8.")


if __name__ == "__main__":
    main()