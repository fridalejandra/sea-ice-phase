#!/usr/bin/env python
"""
ch2_results_numbers.py

Numbers for Ch2 Results:
  4.1  climatological FS / MS timing, per method and sector
  4.2  dynamic-minus-static differences: climatological, interannual
       (spread + sign consistency), and inner-pack vs outer-edge split

Inputs: v3 climatology + anomaly files (Aug 27 run), 1979-2024 with 1986-87 absent.
Masking: identical to compute_sector_mean_trends07.py
  (canonical_sectors.nc, valid_ocean, active in >=80% of years under BOTH methods).

Per-year date = climatology + anomaly.
FS in day of year; MS in days since Aug 15 (_dsa variables), matching Fig. 2.
"""
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
ANOM_DIR = PROJECT_ROOT / "data" / "anomalies" / "SMMR"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"
OUT_DIR = PROJECT_ROOT / "results"
MIN_FRAC_ACTIVE = 0.80

# Inner/outer split: terciles of the STATIC climatology within each sector.
# FS: inner = earliest-freezing third, outer = latest-freezing third.
# MS: inner = latest-melting third,    outer = earliest-melting third.
# Provisional, data-driven definition -- to be agreed before use in text.
TERCILES = (1 / 3, 2 / 3)

SECTORS = {1: "A–B", 2: "WED", 3: "KHV", 4: "EA", 5: "RA"}
TAGS = {
    ("FS", "dynamic"): "FS_dynamic_k5_q70",
    ("FS", "static"): "FS_static_thr15_k5",
    ("MS", "dynamic"): "MS_dynamic_k5_q70",
    ("MS", "static"): "MS_static_thr15_k5",
}


def _open_da(path, candidates):
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


def load(phase, method):
    tag = TAGS[(phase, method)]
    if phase == "MS":
        clim_c, anom_c = [f"{tag}_clim_dsa", f"{tag}_clim"], [f"{tag}_anom_dsa", f"{tag}_anom"]
    else:
        clim_c, anom_c = [f"{tag}_clim"], [f"{tag}_anom"]
    clim = _open_da(ANOM_DIR / f"{tag}_climatology.nc", clim_c)
    anom = _open_da(ANOM_DIR / f"{tag}_anomalies.nc", anom_c)
    return clim, anom


def to_date(value, phase):
    if not np.isfinite(value):
        return "NaN"
    if phase == "FS":  # day of year, 1-based, non-leap reference
        d = dt.date(2001, 1, 1) + dt.timedelta(days=int(round(value)) - 1)
    else:  # days since Aug 15
        d = dt.date(2001, 8, 15) + dt.timedelta(days=int(round(value)))
    return d.strftime("%d %b")


def main():
    ds_mask = xr.open_dataset(SECTOR_FILE)
    valid_ocean = ds_mask["valid_ocean"].astype(bool).values
    sector_id = ds_mask["sector_id"].values
    ds_mask.close()

    timing_rows, diff_rows, iav_rows, split_rows = [], [], [], []

    for phase in ["FS", "MS"]:
        clim, anom = {}, {}
        for method in ["static", "dynamic"]:
            c, a = load(phase, method)
            clim[method], anom[method] = c.values, a.values
            years = a["year"].values
        assert clim["static"].shape == sector_id.shape, "grid mismatch with sector file"

        n_years = anom["static"].shape[0]
        frac = {m: np.isfinite(anom[m]).sum(0) / n_years for m in anom}
        active = (frac["dynamic"] >= MIN_FRAC_ACTIVE) & (frac["static"] >= MIN_FRAC_ACTIVE) & valid_ocean

        print(f"\n######## {phase} ########")
        print(f"Years: {years.min()}-{years.max()} (n={n_years}); missing: "
              f"{sorted(set(range(years.min(), years.max() + 1)) - set(years.tolist()))}")
        print(f"Active80 pixels: {int(active.sum())}   (text says FS 23,877 / MS 23,503)")
        for m in anom:
            mean_anom = np.nanmean(anom[m], axis=0)[active]
            print(f"Baseline check {m}: max |mean anomaly| over active pixels = "
                  f"{np.nanmax(np.abs(mean_anom)):.3f} days (should be ~0)")

        date = {m: clim[m][None, :, :] + anom[m] for m in anom}
        dclim = clim["dynamic"] - clim["static"]

        regions = [(s, SECTORS[s], active & (sector_id == s)) for s in SECTORS]
        regions.append((0, "ALL", active))

        for sid, label, sel in regions:
            for m in ["static", "dynamic"]:
                v = clim[m][sel]
                p10, p50, p90 = np.nanpercentile(v, [10, 50, 90])
                timing_rows.append(dict(
                    phase=phase, sector=label, method=m, n_pix=int(sel.sum()),
                    p10=round(p10, 1), p50=round(p50, 1), p90=round(p90, 1),
                    p10_date=to_date(p10, phase), p50_date=to_date(p50, phase),
                    p90_date=to_date(p90, phase)))

            d = dclim[sel]
            diff_rows.append(dict(
                phase=phase, sector=label,
                mean=round(np.nanmean(d), 1), median=round(np.nanmedian(d), 1),
                p10=round(np.nanpercentile(d, 10), 1), p90=round(np.nanpercentile(d, 90), 1),
                frac_pix_positive=round(np.nanmean(d > 0), 2)))

            series = []
            for k in range(n_years):
                both = sel & np.isfinite(date["dynamic"][k]) & np.isfinite(date["static"][k])
                series.append(np.mean(date["dynamic"][k][both] - date["static"][k][both])
                              if both.any() else np.nan)
            series = np.array(series)
            mu = np.nanmean(series)
            iav_rows.append(dict(
                phase=phase, sector=label, mean=round(mu, 1),
                interannual_sd=round(np.nanstd(series, ddof=1), 1),
                min=round(np.nanmin(series), 1), max=round(np.nanmax(series), 1),
                frac_years_same_sign=round(np.nanmean(np.sign(series) == np.sign(mu)), 2)))

            s = clim["static"][sel]
            q_lo, q_hi = np.nanpercentile(s, [100 * TERCILES[0], 100 * TERCILES[1]])
            early, late = s <= q_lo, s >= q_hi
            inner, outer = (early, late) if phase == "FS" else (late, early)
            split_rows.append(dict(
                phase=phase, sector=label,
                inner_mean_diff=round(np.nanmean(d[inner]), 1),
                outer_mean_diff=round(np.nanmean(d[outer]), 1),
                middle_mean_diff=round(np.nanmean(d[~inner & ~outer]), 1)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "4.1 climatological timing (FS: DOY; MS: days since Aug 15)": ("ch2_timing_by_sector", timing_rows),
        "4.2 climatological difference, dynamic - static (days)": ("ch2_diff_clim_by_sector", diff_rows),
        "4.2 interannual sector-mean difference, dynamic - static (days)": ("ch2_diff_interannual_by_sector", iav_rows),
        "4.2 inner vs outer difference, dynamic - static (days) [PROVISIONAL split]": ("ch2_diff_inner_outer", split_rows),
    }
    for title, (fname, rows) in tables.items():
        df = pd.DataFrame(rows)
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))
        df.to_csv(OUT_DIR / f"{fname}.csv", index=False)
    print(f"\nCSVs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
