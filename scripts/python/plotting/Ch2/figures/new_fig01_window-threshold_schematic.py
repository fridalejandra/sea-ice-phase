"""
Two-panel detection schematic from real Bootstrap SIC data.

Panel (a): a clean interior seasonal pixel-year. Daily c(t), the 15% threshold,
the completed k-day run below theta (blue shading), the k-day run above theta
that begins at Freeze Start (orange shading), FS and MS marked.
Panel (b): a high-crossing-count pixel-year, showing repeated transient
crossings that the persistence criterion rejects.

Pixel-years are chosen programmatically so the caption can state the rule:
(a) among pixels whose series crosses 15% exactly twice in the chosen year,
    the one whose FS date is the median of that set;
(b) the pixel with the maximum number of 15% crossings in the same year.

Usage on the cluster (from sea-ice-phase/):
    python figures/fig_schematic_detection.py \
        --nc data/merged/merged_bootstrap_SH_latest.nc --year 2019
Optional: --pixel-a IY,IX --pixel-b IY,IX to override the automatic choice.

Read-only with respect to the detection pipeline; safe to run alongside
anything except another job holding the merged file open for writing.
"""

import argparse
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

THETA = 0.15
K = 5
GREY = "0.35"

def unpack(da):
    """Mask flags (1100 missing, 1200 land) and scale to fraction."""
    v = da.where(da < 1050)
    if float(v.max()) > 1.5:
        v = v / 1000.0
    return v

def runs(mask, k):
    """Start indices of runs of >= k consecutive True."""
    m = np.asarray(mask, dtype=bool)
    if m.size < k:
        return np.array([], dtype=int)
    starts = []
    count = 0
    for i, val in enumerate(m):
        count = count + 1 if val else 0
        if count == k:
            starts.append(i - k + 1)
    # keep only run beginnings (previous day not part of a qualifying run)
    out = []
    for s in starts:
        if not out or s > out[-1]:
            # collapse consecutive window starts within one long run
            if out and s == out[-1] + 1 and m[out[-1]:s + k].all():
                continue
            out.append(s)
    return np.array(out, dtype=int)

def detect_fs(c, k=K, theta=THETA):
    """First start of a k-run above theta preceded by a completed k-run below."""
    above = runs(c >= theta, k)
    below = runs(c <= theta, k)
    for t in above:
        if any(b + k <= t for b in below):
            first_below = below[below + k <= t]
            return t, first_below[0]
    return None, None


def detect_ms(c, k=K, theta=THETA):
    """First start of a k-run below theta preceded by a completed k-run above.
    Searched after the FS-side maximum so the same calendar-year series can
    illustrate both transitions in one panel (the paper's true MS search uses
    the Aug 15 to Feb 28 window; for a single-year schematic we search from
    day 200 onward)."""
    c2 = np.asarray(c)
    start = 200
    above = runs(c2[start:] >= theta, k) + start
    below = runs(c2[start:] <= theta, k) + start
    for t in below:
        if any(a + k <= t for a in above):
            first_above = above[above + k <= t]
            return t, first_above[0]
    return None, None

def crossings(c, theta=THETA):
    s = np.sign(np.asarray(c) - theta)
    s[s == 0] = 1
    return int(np.sum(s[1:] != s[:-1]))

def pick_pixels(ds, year, stride=4):
    """Programmatic pixel choice on a strided subsample for speed."""
    t = pd.to_datetime(ds.time.values)
    sel = (t >= f"{year}-02-15") & (t <= f"{year}-12-31")
    c = unpack(ds["N07_ICECON"].isel(time=np.where(sel)[0])).values
    ny, nx = c.shape[1], c.shape[2]
    clean, messy, max_cross = [], None, -1
    for iy in range(0, ny, stride):
        for ix in range(0, nx, stride):
            series = c[:, iy, ix]
            if np.isnan(series).mean() > 0.2:
                continue
            if np.nanmin(series) > 0.10 or np.nanmax(series) < 0.80:
                continue
            n = crossings(np.nan_to_num(series, nan=0.0))
            if n == 2:
                fs, _ = detect_fs(np.nan_to_num(series, nan=0.0))
                if fs is not None:
                    clean.append((fs, iy, ix))
            if n > max_cross:
                max_cross, messy = n, (iy, ix)
    clean.sort()
    a = clean[len(clean) // 2][1:] if clean else None
    return a, messy, max_cross

def shade_pair(ax, days, first, second, color_first, color_second,
               mark, label):
    if first is not None:
        ax.axvspan(days[first], days[first + K - 1],
                   color=color_first, alpha=0.18, lw=0)
    if second is not None:
        ax.axvspan(days[second], days[second + K - 1],
                   color=color_second, alpha=0.18, lw=0)
        ax.axvline(days[second], color=mark, lw=1.2)
        ax.annotate(label, xy=(days[second], 1.0),
                    xytext=(days[second], 1.06),
                    ha="center", fontsize=8, fontweight="bold",
                    color=mark, annotation_clip=False)

def style(ax, letter, title):
    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.text(0.02, 0.95, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=11, va="top")
    for lab in (ax.xaxis.label, ax.yaxis.label):
        lab.set_fontweight("bold")
        lab.set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GREY)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc", required=True)
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--pixel-a", default=None)
    p.add_argument("--pixel-b", default=None)
    p.add_argument("--out", default="figures/schematic_detection.png")
    args = p.parse_args()

    ds = xr.open_dataset(args.nc)
    t = pd.to_datetime(ds.time.values)
    sel = np.where((t >= f"{args.year}-01-01") & (t <= f"{args.year}-12-31"))[0]
    days = t[sel].dayofyear

    if args.pixel_a and args.pixel_b:
        pa = tuple(int(v) for v in args.pixel_a.split(","))
        pb = tuple(int(v) for v in args.pixel_b.split(","))
        ncross = None
    else:
        pa, pb, ncross = pick_pixels(ds, args.year)
        print(f"panel a pixel (iy,ix) = {pa}, panel b = {pb}, "
              f"max crossings = {ncross}")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1), sharey=True)
    for ax, px, letter, title in zip(
            axes, (pa, pb), ("(a)", "(b)"),
            ("Interior seasonal pixel", "Repeated-crossing pixel")):
        c = unpack(ds["N07_ICECON"].isel(
            time=sel, y=px[0], x=px[1])).values
        cz = np.nan_to_num(c, nan=0.0)
        ax.plot(days, c, lw=0.9, color="#54278f")
        ax.axhline(THETA, color=GREY, lw=0.8, ls="--")
        d = np.asarray(days)
        fs, fb = detect_fs(cz)
        shade_pair(ax, d, fb, fs, "#377eb8", "#e6550d", "#e6550d", "FS")
        ms, ma = detect_ms(cz)
        shade_pair(ax, d, ma, ms, "#e6550d", "#377eb8", "#377eb8", "MS")
        ax.set_xlabel("Day of year")
        ax.set_ylim(-0.02, 1.12)
        style(ax, letter, title)
        n = crossings(cz)
        ax.text(0.98, 0.05, f"{n} crossings of θ",
                transform=ax.transAxes, ha="right", fontsize=8, color=GREY)
    axes[0].set_ylabel("SIC (fraction)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()