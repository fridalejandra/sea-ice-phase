#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute wrapped Δ timing (days) for FS/MS sensitivity experiments and write:
- pooled distributions (NumPy .npz)
- per-year summary tables (CSV)

Assumes FS/MS annual fields exist under:
  $OUTPUT_ROOT/{FS,MS}_thrXX_kK/{FS,MS}_YYYY.nc
"""

import os, glob, csv
import numpy as np
import xarray as xr
from pathlib import Path

# --------- PATHS (env-driven) ---------
USER = os.environ.get("USER", "user")
ROOT     = os.environ.get("OUTPUT_ROOT", f"/scratch/{USER}/sea-ice-phase/results/phase/SMMR")
SENS_DIR = os.environ.get("SENS_DIR",  f"/scratch/{USER}/sea-ice-phase/results/sensitivity/SMMR")
Path(SENS_DIR).mkdir(parents=True, exist_ok=True)

# --------- CONFIG ---------
EVENTS = ["FS","MS"]       # Freeze Start, Melt Start
DAY = 365
HARD_CLIP = 60             # Δ beyond this (abs) masked as artifacts

# Comparisons
WINDOW_COMPS  = [("thr15_k3", "thr15_k5"), ("thr15_k7", "thr15_k5")]  # Δ(3–5), Δ(7–5)
THRESH_COMPS  = [("thr10_k5", "thr15_k5"), ("thr30_k5", "thr15_k5")]  # Δ(10–15), Δ(30–15)

# --------- HELPERS ---------
def wrapdiff(a, b):
    """Shortest signed DOY difference in [-182, +182]."""
    return ((a - b + DAY//2) % DAY) - DAY//2

def load_year(event, tag, year):
    """Load one (y,x) field for event+tag+year; return np.ndarray or None."""
    path = os.path.join(ROOT, f"{event}_{tag}", f"{event}_{year}.nc")
    if not os.path.exists(path):
        return None
    try:
        return xr.open_dataset(path)[event].values
    except Exception:
        return None

def years_available(event, tag):
    pat = os.path.join(ROOT, f"{event}_{tag}", f"{event}_*.nc")
    years = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in glob.glob(pat)]
    return sorted(set(years))

def summarize_delta(delta2d):
    v = delta2d[np.isfinite(delta2d)]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(np.nanmean(v))
    std  = float(np.nanstd(v))
    pct5 = 100.0 * float(np.mean(np.abs(v) > 5.0))
    return mean, std, pct5, int(v.size)

def collect(event, comp):
    """Return (years, list_of_Δ2d, per_year_rows)."""
    alt, base = comp
    years = sorted(set(years_available(event, alt)).intersection(years_available(event, base)))
    deltas, rows = [], []
    for y in years:
        A = load_year(event, alt, y)
        B = load_year(event, base, y)
        if A is None or B is None:
            continue
        m = np.isfinite(A) & np.isfinite(B)
        D = np.full_like(A, np.nan, float)
        D[m] = wrapdiff(A[m], B[m])
        D[np.abs(D) > HARD_CLIP] = np.nan
        mean, std, pct5, n = summarize_delta(D)
        rows.append([event, f"{alt}–{base}", y, mean, std, pct5, n])
        deltas.append(D)
    return years, deltas, rows

def save_npz(name, years, deltas):
    pool = np.concatenate([d[np.isfinite(d)].ravel() for d in deltas]) if deltas else np.array([])
    np.savez(os.path.join(SENS_DIR, f"{name}.npz"), pooled=pool, years=np.array(years, dtype=int))

def save_csv(name, rows):
    with open(os.path.join(SENS_DIR, f"{name}_yearly.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event","comparison","year","mean","std","pct_abs_gt5","n_valid"])
        w.writerows(rows)

# --------- MAIN ---------
def main():
    # Window comparisons
    for event in EVENTS:
        for comp in WINDOW_COMPS:
            years, deltas, rows = collect(event, comp)
            name = f"{event}_window_{comp[0]}__{comp[1]}"
            save_npz(name, years, deltas)
            save_csv(name, rows)

    # Threshold comparisons
    for event in EVENTS:
        for comp in THRESH_COMPS:
            years, deltas, rows = collect(event, comp)
            name = f"{event}_threshold_{comp[0]}__{comp[1]}"
            save_npz(name, years, deltas)
            save_csv(name, rows)

    print(f"[OK] Wrote npz/csv to {SENS_DIR}")

if __name__ == "__main__":
    main()
