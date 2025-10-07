#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute wrapped Δ timing (days) and write pooled NPZ + yearly CSV.
"""

import os, glob, csv, sys, datetime
import numpy as np
import xarray as xr
from pathlib import Path

# ==== ABSOLUTE PATHS  ====
ROOT = "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase"
SENS_DIR = "/user/geog/falejandraperez/sea-ice-phase/results/sensitivity/SMMR"
Path(SENS_DIR).mkdir(parents=True, exist_ok=True)

EVENTS = ["FS","MS"]
DAY = 365
HARD_CLIP = 60
WINDOW_COMPS  = [("thr15_k3", "thr15_k5"), ("thr15_k7", "thr15_k5")]
THRESH_COMPS  = [("thr10_k5", "thr15_k5"), ("thr30_k5", "thr15_k5")]

def log(m): print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)

def wrapdiff(a, b):  # shortest signed
    return ((a - b + DAY//2) % DAY) - DAY//2

def load_year(event, tag, year):
    """Return 2D array for given event/tag/year; robust to var naming."""
    path = os.path.join(ROOT, f"{event}_{tag}", f"{event}_{year}.nc")
    if not os.path.exists(path):
        log(f"[MISS] {path}")
        return None
    try:
        ds = xr.open_dataset(path)
        if event in ds:
            arr = ds[event].values
        else:
            # fallback: first 2D var
            cand = next((v for v in ds.data_vars if ds[v].ndim == 2), None)
            if cand is None:
                log(f"[ERR] No 2D vars in {os.path.basename(path)} -> {list(ds.data_vars)}")
                return None
            log(f"[WARN] Using '{cand}' (not '{event}') in {os.path.basename(path)}")
            arr = ds[cand].values
        return arr
    except Exception as e:
        log(f"[ERR] open {path}: {e}")
        return None

def years_available(event, tag):
    pat = os.path.join(ROOT, f"{event}_{tag}", f"{event}_*.nc")
    years = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in glob.glob(pat)]
    return sorted(set(years))

def summarize_delta(delta2d):
    v = delta2d[np.isfinite(delta2d)]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    return float(np.nanmean(v)), float(np.nanstd(v)), 100.0*float(np.mean(np.abs(v) > 5.0)), int(v.size)

def collect(event, comp):
    alt, base = comp
    ya = set(years_available(event, alt))
    yb = set(years_available(event, base))
    years = sorted(ya.intersection(yb))
    log(f"{event} {alt} vs {base}: years={len(years)} (alt:{len(ya)} base:{len(yb)})")
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
    outp = os.path.join(SENS_DIR, f"{name}.npz")
    np.savez(outp, pooled=pool, years=np.array(years, dtype=int))
    log(f"NPZ -> {outp} (pooled n={pool.size}, years={len(years)})")

def save_csv(name, rows):
    outp = os.path.join(SENS_DIR, f"{name}_yearly.csv")
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event","comparison","year","mean","std","pct_abs_gt5","n_valid"])
        w.writerows(rows)
    log(f"CSV -> {outp} (rows={len(rows)})")

def main():
    # sanity: base files exist?
    for ev in EVENTS:
        base_dir = os.path.join(ROOT, f"{ev}_thr15_k5")
        if not os.path.isdir(base_dir):
            log(f"[FATAL] Missing base dir: {base_dir}")
            sys.exit(2)

    for ev in EVENTS:
        for comp in WINDOW_COMPS:
            y, d, r = collect(ev, comp)
            save_npz(f"{ev}_window_{comp[0]}__{comp[1]}", y, d)
            save_csv(f"{ev}_window_{comp[0]}__{comp[1]}", r)

    for ev in EVENTS:
        for comp in THRESH_COMPS:
            y, d, r = collect(ev, comp)
            save_npz(f"{ev}_threshold_{comp[0]}__{comp[1]}", y, d)
            save_csv(f"{ev}_threshold_{comp[0]}__{comp[1]}", r)

    log(f"[OK] Wrote outputs to {SENS_DIR}")

if __name__ == "__main__":
    main()
