#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot sensitivity distributions and maps, save PNGs locally, and copy to Drive.
"""

import os, glob, subprocess, sys, datetime
import numpy as np
import xarray as xr
import matplotlib
if os.environ.get("DISPLAY","") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ==== ABSOLUTE PATHS (no env) ====
ROOT = "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase"
SENS = "/user/geog/falejandraperez/sea-ice-phase/results/sensitivity/SMMR"
FIGS_LOCAL = "/user/geog/falejandraperez/sea-ice-phase/results/figures/sensitivity"
FIGS_GDRIVE = "gdrive:SeaIce/Chapter2/figs/sensitivity/SMMR"
Path(FIGS_LOCAL).mkdir(parents=True, exist_ok=True)

EVENTS = ["FS","MS"]
WINDOW_COMPS = [("thr15_k3", "thr15_k5"), ("thr15_k7", "thr15_k5")]
THRESH_COMPS = [("thr10_k5", "thr15_k5"), ("thr30_k5", "thr15_k5")]
DAY = 365
RANGE = {"FS": 10, "MS": 20}
PCT_MAX = 40

def log(m): print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)

def require_dir(d, label):
    if not Path(d).exists():
        raise FileNotFoundError(f"{label} missing: {d}")
    log(f"{label} OK: {d}")

require_dir(SENS, "Sensitivity dir")
require_dir(ROOT, "Phase fields ROOT")

def wrapdiff(a, b):  # shortest signed
    return ((a - b + DAY//2) % DAY) - DAY//2

def stack_delta(event, alt, base):
    yearsA = sorted(int(os.path.basename(p).split("_")[-1].split(".")[0])
                    for p in glob.glob(os.path.join(ROOT, f"{event}_{alt}", f"{event}_*.nc")))
    yearsB = sorted(int(os.path.basename(p).split("_")[-1].split(".")[0])
                    for p in glob.glob(os.path.join(ROOT, f"{event}_{base}", f"{event}_*.nc")))
    years = sorted(set(yearsA).intersection(yearsB))
    log(f"{event} {alt} vs {base}: intersect years = {len(years)}")
    stack = []
    for y in years:
        A = xr.open_dataset(os.path.join(ROOT, f"{event}_{alt}",  f"{event}_{y}.nc"))
        B = xr.open_dataset(os.path.join(ROOT, f"{event}_{base}", f"{event}_{y}.nc"))
        varA = event if event in A else next((v for v in A.data_vars if A[v].ndim==2), None)
        varB = event if event in B else next((v for v in B.data_vars if B[v].ndim==2), None)
        if varA is None or varB is None:
            log(f"[WARN] No 2D var for year {y}"); continue
        a = A[varA].values; b = B[varB].values
        m = np.isfinite(a) & np.isfinite(b)
        D = np.full_like(a, np.nan, float); D[m] = wrapdiff(a[m], b[m])
        D[np.abs(D) > 60] = np.nan
        stack.append(D)
    return years, (np.stack(stack) if stack else None)

def distribution_panel(ax, data, title):
    v = data[np.isfinite(data)]
    if v.size == 0:
        ax.text(0.5,0.5,"No data", ha="center", va="center", transform=ax.transAxes); return
    bins = np.arange(-30, 31, 1)
    ax.hist(v, bins=bins)
    med = float(np.nanmedian(v)); q25, q75 = np.nanpercentile(v, [25, 75]); pct = 100.0*np.mean(np.abs(v) > 5)
    ax.axvline(0, ls="--", lw=1); ax.axvline(-5, ls=":", lw=1); ax.axvline(5, ls=":", lw=1)
    ax.set_xlim(-30, 30); ax.set_xlabel("Δ timing (days)")
    ax.set_title(f"{title}\nmedian {med:+.1f} d | IQR {q25:+.1f}–{q75:+.1f} d | %|Δ|>5: {pct:.1f}%")

def map_pair(fig, axs, stack, event, label):
    if stack is None:
        for a in axs: a.text(0.5,0.5,"No data", ha="center", va="center", transform=a.transAxes)
        return
    med = np.nanmedian(stack, axis=0)
    pct = 100.0*np.nanmean(np.abs(stack) > 5, axis=0)
    im0 = axs[0].imshow(med, vmin=-RANGE[event], vmax=RANGE[event]); axs[0].set_title(f"{event} median Δ ({label})")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    im1 = axs[1].imshow(pct, vmin=0, vmax=PCT_MAX); axs[1].set_title(f"{event} % years |Δ|>5 ({label})")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    for a in axs: a.set_xticks([]); a.set_yticks([])

def save_and_sync(fig, filename):
    out_path = os.path.join(FIGS_LOCAL, filename)
    Path(FIGS_LOCAL).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    sz = os.path.getsize(out_path); log(f"Saved figure: {out_path} ({sz} bytes)")
    if FIGS_GDRIVE:
        # Ensure remote path exists (mkdir is idempotent)
        try:
            subprocess.run(["rclone", "mkdir", FIGS_GDRIVE], check=True,
                           capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            log(f"[WARN] rclone mkdir: {e.stderr.strip() or e}")
        try:
            res = subprocess.run(["rclone", "copyto", out_path, f"{FIGS_GDRIVE}/{filename}", "-v"],
                                 check=True, capture_output=True, text=True)
            log(f"Synced -> {FIGS_GDRIVE}/{filename}")
            if res.stdout.strip(): log("[RCLONE]\n"+res.stdout.strip())
            if res.stderr.strip(): log("[RCLONE-ERR]\n"+res.stderr.strip())
        except subprocess.CalledProcessError as e:
            log("[ERROR] rclone copy failed:\n" + (e.stdout or "") + (e.stderr or ""))

def make_window_distributions():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    names = {("FS","thr15_k3","thr15_k5"): "FS Δ(3–5)",
             ("FS","thr15_k7","thr15_k5"): "FS Δ(7–5)",
             ("MS","thr15_k3","thr15_k5"): "MS Δ(3–5)",
             ("MS","thr15_k7","thr15_k5"): "MS Δ(7–5)"}
    for r,ev in enumerate(EVENTS):
        for c,(alt,base) in enumerate(WINDOW_COMPS):
            npz = os.path.join(SENS, f"{ev}_window_{alt}__{base}.npz")
            if not os.path.exists(npz): log(f"[WARN] Missing NPZ {npz}"); data = np.array([])
            else: data = np.load(npz, allow_pickle=False)["pooled"]
            distribution_panel(axs[r,c], data, names[(ev,alt,base)])
    fig.suptitle("Window sensitivity distributions (SMMR, FS & MS)")
    save_and_sync(fig, "window_sensitivity_distributions.png")

def make_threshold_distributions():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    names = {("FS","thr10_k5","thr15_k5"): "FS Δ(10–15)",
             ("FS","thr30_k5","thr15_k5"): "FS Δ(30–15)",
             ("MS","thr10_k5","thr15_k5"): "MS Δ(10–15)",
             ("MS","thr30_k5","thr15_k5"): "MS Δ(30–15)"}
    for r,ev in enumerate(EVENTS):
        for c,(alt,base) in enumerate(THRESH_COMPS):
            npz = os.path.join(SENS, f"{ev}_threshold_{alt}__{base}.npz")
            if not os.path.exists(npz): log(f"[WARN] Missing NPZ {npz}"); data = np.array([])
            else: data = np.load(npz, allow_pickle=False)["pooled"]
            distribution_panel(axs[r,c], data, names[(ev,alt,base)])
    fig.suptitle("Threshold sensitivity distributions (SMMR, FS & MS)")
    save_and_sync(fig, "threshold_sensitivity_distributions.png")

def make_window_maps():
    for ev in EVENTS:
        fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        _, stack = stack_delta(ev, "thr15_k3", "thr15_k5"); map_pair(fig, axs[0,:], stack, ev, "3–5")
        _, stack = stack_delta(ev, "thr15_k7", "thr15_k5"); map_pair(fig, axs[1,:], stack, ev, "7–5")
        fig.suptitle(f"Window sensitivity maps ({ev})")
        save_and_sync(fig, f"{ev}_window_sensitivity_maps.png")

def make_threshold_maps():
    for ev in EVENTS:
        fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        _, stack = stack_delta(ev, "thr10_k5", "thr15_k5"); map_pair(fig, axs[0,:], stack, ev, "10–15")
        _, stack = stack_delta(ev, "thr30_k5", "thr15_k5"); map_pair(fig, axs[1,:], stack, ev, "30–15")
        fig.suptitle(f"Threshold sensitivity maps ({ev})")
        save_and_sync(fig, f"{ev}_threshold_sensitivity_maps.png")

if __name__ == "__main__":
    # quick visibility on input availability
    log(f"NPZ window: {len(glob.glob(os.path.join(SENS,'*window*__.npz')))} "
        f"threshold: {len(glob.glob(os.path.join(SENS,'*threshold*__.npz')))}")
    log(f"FS base files: {len(glob.glob(os.path.join(ROOT,'FS_thr15_k5','FS_*.nc')))}")
    log(f"MS base files: {len(glob.glob(os.path.join(ROOT,'MS_thr15_k5','MS_*.nc')))}")

    make_window_distributions()
    make_threshold_distributions()
    make_window_maps()
    make_threshold_maps()
    log(f"[OK] Figures saved to {FIGS_LOCAL}")
