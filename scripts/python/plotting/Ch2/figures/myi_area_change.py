#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
myi_area_change.py

Answers two questions:

(1) How much ICE AREA never yields a phase date?
    Restricts to pixels that actually carry ice (SIC >= 15% on at least one
    day in the advance window in at least one year) but never produce a
    Freeze Start under either method. This is the quantity behind Marilyn's
    comment: how much ice is affected by undetectable freeze onset.

(2) Has that area CHANGED over time?
    For each year, counts pixels that carry ice in the advance window but
    contain no persistent (k-day) open interval, i.e. multi-year-like ice
    that does not seasonally open. Reports area per year by sector and a
    linear trend, for Weddell and Ross-Amundsen in particular.
"""

import os
import numpy as np
import xarray as xr

# ============================== EDIT-ME ==============================
ROOT     = "/user/geog/falejandraperez/sea-ice-phase"
MERGED   = f"{ROOT}/data/merged/SMMR_merged_19781101_20251231_complete.nc"
CONC_VAR = "N07_ICECON"
MASK_ABOVE = 1.1
SECTORS  = f"{ROOT}/data/canonical_sectors.nc"
PHASE    = f"{ROOT}/data/SMMR_phase"
OUT_DIR  = f"{ROOT}/results/myi_area"

THR = 0.15
K   = 5
FS_LO, FS_HI = 46, 273
BAD_YEARS = [1987]
PIXEL_KM2 = 625.0          # 25 km grid
LABELS = {1: "A-B", 2: "WED", 3: "KHV", 4: "EA", 5: "RA"}
# =====================================================================


def longest_run_map(below):
    run = np.zeros(below.shape[1:], dtype=np.int16)
    best = np.zeros(below.shape[1:], dtype=np.int16)
    for t in range(below.shape[0]):
        run = np.where(below[t], run + 1, 0)
        best = np.maximum(best, run)
    return best


def load_dates(method, sub, phase):
    import re
    d = f"{PHASE}/{method}/{sub}/{phase}"
    pat = re.compile(rf"^{phase}_(\d{{4}})\.nc$")
    files = sorted(f for f in os.listdir(d) if pat.match(f))
    yrs = [int(pat.match(f).group(1)) for f in files]
    arr = np.stack([xr.open_dataset(os.path.join(d, f))[phase].values for f in files])
    return np.array(yrs), arr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log = []
    def say(s=""):
        print(s); log.append(s)

    ds  = xr.open_dataset(MERGED)
    ice = ds[CONC_VAR].astype("float32")
    ice = ice.where(ice <= MASK_ABOVE)
    doys = ice.time.dt.dayofyear.values
    yrs  = ice.time.dt.year.values
    ny, nx = ice.y.size, ice.x.size
    sec = xr.open_dataset(SECTORS)["sector_id"].values

    years = [int(y) for y in np.unique(yrs) if 1979 <= y <= 2024 and y not in BAD_YEARS]

    say("=" * 72)
    say("  MULTI-YEAR ICE AREA AND ITS CHANGE")
    say("=" * 72)
    say(f"  pixel area = {PIXEL_KM2:.0f} km^2, {len(years)} years")

    # per-year: does the pixel carry ice, and does it open persistently?
    carries = np.zeros((len(years), ny, nx), dtype=bool)
    opens   = np.zeros((len(years), ny, nx), dtype=bool)

    say("\n  scanning advance windows ...")
    for yi, y in enumerate(years):
        m = (yrs == y) & (doys >= FS_LO) & (doys <= FS_HI)
        if not m.any():
            continue
        blk = ice.values[m]
        with np.errstate(invalid="ignore"):
            carries[yi] = np.nanmax(np.where(np.isfinite(blk), blk, -1), axis=0) >= THR
        below = np.where(np.isfinite(blk), blk < THR, False)
        opens[yi] = longest_run_map(below) >= K
        del blk, below

    # ---------------- Q1: ice area with no detectable freeze onset ---------
    say("\n" + "-" * 72)
    say("  Q1  ICE AREA THAT NEVER YIELDS A FREEZE START")
    say("-" * 72)

    ever_ice = carries.any(axis=0)
    say(f"  pixels carrying ice at some point: {int(ever_ice.sum()):,}"
        f"  ({ever_ice.sum()*PIXEL_KM2/1e6:.2f} million km^2)")

    for method, sub in [("static", "thr15_k5"), ("dynamic", "k5_q70")]:
        try:
            _, arr = load_dates(method, sub, "FS")
        except Exception as e:
            say(f"  [skip] {method}: {e}"); continue
        never = ever_ice & (~np.isfinite(arr).any(axis=0))
        say(f"\n  {method}: ice pixels with NO Freeze Start in any year: "
            f"{int(never.sum()):,}  ({never.sum()*PIXEL_KM2/1e6:.3f} million km^2)")
        for sid, lab in LABELS.items():
            n = int((never & (sec == sid)).sum())
            tot = int((ever_ice & (sec == sid)).sum())
            if tot:
                say(f"      {lab:>4s}: {n:6,d} px  {n*PIXEL_KM2/1e3:8.1f} thousand km^2"
                    f"   ({100*n/tot:4.1f}% of that sector's ice pixels)")

    # ---------------- Q2: change in non-opening ice area -------------------
    say("\n" + "-" * 72)
    say("  Q2  AREA OF ICE THAT DOES NOT SEASONALLY OPEN, BY YEAR")
    say("-" * 72)
    say("  (pixels carrying ice in the advance window with no k-day open run)")

    closed = carries & (~opens)
    rows = []
    for yi, y in enumerate(years):
        for sid, lab in LABELS.items():
            n = int((closed[yi] & (sec == sid)).sum())
            rows.append((y, lab, n, n * PIXEL_KM2))

    with open(f"{OUT_DIR}/nonopening_area_by_year.csv", "w") as fh:
        fh.write("year,sector,n_pixels,area_km2\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.0f}\n")

    say(f"\n  {'sector':>6s} {'1979-1999':>12s} {'2000-2015':>12s} "
        f"{'2016-2024':>12s} {'trend':>16s}")
    say(f"  {'':>6s} {'(1000 km2)':>12s} {'(1000 km2)':>12s} "
        f"{'(1000 km2)':>12s} {'(1000 km2/decade)':>16s}")
    for sid, lab in LABELS.items():
        sub_rows = [r for r in rows if r[1] == lab]
        yy = np.array([r[0] for r in sub_rows], dtype=float)
        aa = np.array([r[3] for r in sub_rows]) / 1e3
        def seg(a, b):
            m = (yy >= a) & (yy <= b)
            return np.nanmean(aa[m]) if m.any() else np.nan
        tr = np.polyfit(yy, aa, 1)[0] * 10
        say(f"  {lab:>6s} {seg(1979,1999):12.1f} {seg(2000,2015):12.1f} "
            f"{seg(2016,2024):12.1f} {tr:16.1f}")

    say("\n  Weddell and Ross-Amundsen year by year (1000 km^2):")
    for lab in ["WED", "RA"]:
        sub_rows = [r for r in rows if r[1] == lab]
        say(f"    {lab}: " + " ".join(f"{r[0]}:{r[3]/1e3:.0f}" for r in sub_rows))

    with open(f"{OUT_DIR}/receipt.txt", "w") as fh:
        fh.write("\n".join(log) + "\n")
    say(f"\n  wrote {OUT_DIR}/")
    say("=" * 72)


if __name__ == "__main__":
    main()
