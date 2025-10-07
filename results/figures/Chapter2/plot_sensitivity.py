#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot sensitivity distributions (histograms) and spatial maps (median Δ, %|Δ|>5)
for SMMR FS/MS, for window (3–5, 7–5) and threshold (10–15, 30–15) comparisons.
Figures are saved locally and optionally copied to Google Drive via rclone.
"""

import os, glob, subprocess
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# --------- PATHS (env-driven) ---------
USER        = os.environ.get("USER", "user")
ROOT        = os.environ.get("OUTPUT_ROOT",   f"/scratch/{USER}/sea-ice-phase/results/phase/SMMR")
SENS        = os.environ.get("SENS_DIR",      f"/scratch/{USER}/sea-ice-phase/results/sensitivity/SMMR")
FIGS_LOCAL  = os.environ.get("FIGS_DIR_LOCAL",f"/scratch/{USER}/sea-ice-phase/figs/sensitivity/SMMR")
FIGS_GDRIVE = os.environ.get("FIGS_DIR_GDRIVE")  # e.g., 'gdrive:SeaIce/Chapter2/figs/sensitivity/SMMR'
Path(FIGS_LOCAL).mkdir(parents=True, exist_ok=True)

# --------- CONFIG ---------
EVENTS = ["FS","MS"]
WINDOW_COMPS = [("thr15_k3", "thr15_k5"), ("thr15_k7", "thr15_k5")]
THRESH_COMPS = [("thr10_k5", "thr15_k5"), ("thr30_k5", "thr15_k5")]
DAY = 365
RANGE = {"FS": 10, "MS": 20}  # colorbar range for median Δ map (days)
PCT_MAX = 40                  # % years |Δ|>5

# --------- HELPERS ---------
def wrapdiff(a, b):  # shortest signed
    return ((a - b + DAY//2) % DAY) - DAY//2

def years_available(event, tag):
    pat = os.path.join(ROOT, f"{event}_{tag}", f"{event}_*.nc")
    years = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in glob.glob(pat)]
    return sorted(set(years))

def stack_delta(event, alt, base):
    years = sorted(set(years_available(event, alt)).intersection(years_available(event, base)))
    stack = []
    for y in years:
        A = xr.open_dataset(os.path.join(ROOT, f"{event}_{alt}",  f"{event}_{y}.nc"))[event].values
        B = xr.open_dataset(os.path.join(ROOT, f"{event}_{base}", f"{event}_{y}.nc"))[event].values
        m = np.isfinite(A) & np.isfinite(B)
        D = np.full_like(A, np.nan, float)
        D[m] = wrapdiff(A[m], B[m])
        D[np.abs(D) > 60] = np.nan
        stack.append(D)
    return years, (np.stack(stack) if stack else None)  # (t,y,x)

def distribution_panel(ax, data, title):
    v = data[np.isfinite(data)]
    if v.size == 0:
        ax.text(0.5,0.5,"No data", ha="center", va="center", transform=ax.transAxes); return
    bins = np.arange(-30, 31, 1)
    ax.hist(v, bins=bins)
    med = float(np.nanmedian(v)) if v.size else np.nan
    q25, q75 = (np.nanpercentile(v, [25, 75]) if v.size else (np.nan, np.nan))
    pct = 100.0*np.mean(np.abs(v) > 5) if v.size else np.nan
    ax.axvline(0, ls="--", lw=1); ax.axvline(-5, ls=":", lw=1); ax.axvline(5, ls=":", lw=1)
    ax.set_xlim(-30, 30)
    ax.set_title(f"{title}\nmedian {med:+.1f} d | IQR {q25:+.1f}–{q75:+.1f} d | %|Δ|>5: {pct:.1f}%")
    ax.set_xlabel("Δ timing (days)")

def map_pair(fig, axs, stack, event, label):
    """Median Δ and % years |Δ|>5 from a (t,y,x) stack."""
    if stack is None:
        for a in axs: a.text(0.5,0.5,"No data", ha="center", va="center", transform=a.transAxes)
        return
    med = np.nanmedian(stack, axis=0)
    pct = 100.0*np.nanmean(np.abs(stack) > 5, axis=0)
    im0 = axs[0].imshow(med, vmin=-RANGE[event], vmax=RANGE[event])
    axs[0].set_title(f"{event} median Δ ({label})")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    im1 = axs[1].imshow(pct, vmin=0, vmax=PCT_MAX)
    axs[1].set_title(f"{event} % years |Δ|>5 ({label})")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    for a in axs: a.set_xticks([]); a.set_yticks([])

def save_and_sync(fig, filename):
    out_path = os.path.join(FIGS_LOCAL, filename)
    fig.savefig(out_path, dpi=180)
    if FIGS_GDRIVE:
        try:
            subprocess.run(["rclone", "copy", out_path, FIGS_GDRIVE], check=True)
        except Exception as e:
            print(f"[WARN] rclone copy failed for {filename}: {e}")

# --------- FIGURES ---------
def make_window_distributions():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    names = {
        ("FS","thr15_k3","thr15_k5"): "FS Δ(3–5)",
        ("FS","thr15_k7","thr15_k5"): "FS Δ(7–5)",
        ("MS","thr15_k3","thr15_k5"): "MS Δ(3–5)",
        ("MS","thr15_k7","thr15_k5"): "MS Δ(7–5)",
    }
    for (r,event) in enumerate(EVENTS):
        for (c,(alt,base)) in enumerate(WINDOW_COMPS):
            npz = os.path.join(SENS, f"{event}_window_{alt}__{base}.npz")
            data = np.load(npz)["pooled"] if os.path.exists(npz) else np.array([])
            distribution_panel(axs[r, c], data, names[(event,alt,base)])
    fig.suptitle("Window sensitivity distributions (SMMR, FS & MS)")
    save_and_sync(fig, "window_sensitivity_distributions.png")
    plt.close(fig)

def make_threshold_distributions():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    names = {
        ("FS","thr10_k5","thr15_k5"): "FS Δ(10–15)",
        ("FS","thr30_k5","thr15_k5"): "FS Δ(30–15)",
        ("MS","thr10_k5","thr15_k5"): "MS Δ(10–15)",
        ("MS","thr30_k5","thr15_k5"): "MS Δ(30–15)",
    }
    for (r,event) in enumerate(EVENTS):
        for (c,(alt,base)) in enumerate(THRESH_COMPS):
            npz = os.path.join(SENS, f"{event}_threshold_{alt}__{base}.npz")
            data = np.load(npz)["pooled"] if os.path.exists(npz) else np.array([])
            distribution_panel(axs[r, c], data, names[(event,alt,base)])
    fig.suptitle("Threshold sensitivity distributions (SMMR, FS & MS)")
    save_and_sync(fig, "threshold_sensitivity_distributions.png")
    plt.close(fig)

def make_window_maps():
    for event in EVENTS:
        fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        # Δ(3–5)
        _, stack = stack_delta(event, "thr15_k3", "thr15_k5")
        map_pair(fig, axs[0, :], stack, event, "3–5")
        # Δ(7–5)
        _, stack = stack_delta(event, "thr15_k7", "thr15_k5")
        map_pair(fig, axs[1, :], stack, event, "7–5")
        fig.suptitle(f"Window sensitivity maps ({event})")
        save_and_sync(fig, f"{event}_window_sensitivity_maps.png")
        plt.close(fig)

def make_threshold_maps():
    for event in EVENTS:
        fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        # Δ(10–15)
        _, stack = stack_delta(event, "thr10_k5", "thr15_k5")
        map_pair(fig, axs[0, :], stack, event, "10–15")
        # Δ(30–15)
        _, stack = stack_delta(event, "thr30_k5", "thr15_k5")
        map_pair(fig, axs[1, :], stack, event, "30–15")
        fig.suptitle(f"Threshold sensitivity maps ({event})")
        save_and_sync(fig, f"{event}_threshold_sensitivity_maps.png")
        plt.close(fig)

# --------- MAIN ---------
if __name__ == "__main__":
    make_window_distributions()
    make_threshold_distributions()
    make_window_maps()
    make_threshold_maps()
    print(f"[OK] Figures saved to {FIGS_LOCAL}" + (f" and synced to {FIGS_GDRIVE}" if FIGS_GDRIVE else ""))
