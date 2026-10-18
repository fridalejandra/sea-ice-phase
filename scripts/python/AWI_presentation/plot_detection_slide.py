#!/usr/bin/env python3
"""
plot_detection_slide.py -- slide version of Ch. 2 Fig. 2.1: how the two methods date the same pixel.

Two panels from the real Bootstrap SIC record (same pixel-selection rule and static detection as
fig_schematic_detection.py):
  (a) a clean interior seasonal pixel (exactly two crossings of 15%, median Freeze Start date)
  (b) the repeated-crossing pixel (most crossings of 15% in the year)

Each panel shows the daily SIC, the fixed 15% threshold (static) with its Freeze Start / Melt Start
as dots, and the pixel's own reference levels (dynamic) with its Freeze Start / Melt Start as diamonds.

The dynamic method is implemented as in Sect. 2.2.3.2 of the chapter:
  - reference per pixel and transition: the 70th (FS) / 30th (MS) percentile of SIC within +-2 days of
    each day of year across every year, averaged over the search window; FS reference floored at 0.15
  - smoothed series = centred 5-day mean; rate = centred difference over +-3 days; the onset needs
    |rate| > 0.02 per day on the first day of the k-day run (positive for FS, negative for MS)
Check the dates it prints against what your pipeline gives for these two pixels before using it.

Slide changes: transparent background, no title, search-window bars, shaded runs and crossing dots
removed (SHOW_RUNS = True brings the shaded runs back), large text, direct labels, no legend.

Usage:
    python plot_detection_slide.py --nc /path/to/merged_bootstrap_SH_latest.nc --year 2019
    python plot_detection_slide.py --nc ... --pixel-a IY,IX --pixel-b IY,IX     # fix the pixels

Writes {out-dir}/detection_slide.png (300 dpi, transparent) and .svg.
"""
import argparse
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---- paths ------------------------------------------------------------------------------------
OUT_DIR = "/Users/fridaperez/Research/repos/sea-ice-phase/scripts/python/AWI_presentation"
NC_DEFAULT = None            # or set the path to merged_bootstrap_SH_latest.nc here
VAR = "N07_ICECON"

# ---- method settings (as in the chapter) ------------------------------------------------------
THETA, K, H = 0.15, 5, 3
PCT_FS, PCT_MS, RATE = 70, 30, 0.02
FS_WINDOW_DOY = (46, 273)    # Feb 15 - Sep 30
MS_WINDOW_DOY = (227, 59)    # Aug 15 - Feb 28 (wraps the new year)

# ---- look -------------------------------------------------------------------------------------
INK = "#222222"              # use "#EEEEEE" on a dark slide
MUTED = "#6B6B6B"
COL_SIC = "#444444"
COL_STATIC = "#2A9D8F"       # teal, as in Fig. 2.3
COL_DYN = "#8E44AD"          # purple, as in Fig. 2.3
FONT = None                  # e.g. "Arial"; match the other slide figures
SHOW_RUNS = False            # shade the k-day runs on either side of each static date
W, H_IN = 12.0, 4.8


def unpack(da):
    v = da.where(da < 1050)
    if float(v.max()) > 1.5:
        v = v / 1000.0
    return v


def run_segments(mask, k):
    m = np.asarray(mask, dtype=bool).astype(int)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return [(s, e) for s, e in zip(starts, ends) if e - s + 1 >= k]


def detect(c, order, ref, search_from=0, search_to=None, k=K, ok=None):
    """First k-day run beyond `ref` (above for 'fs', below for 'ms') that follows a completed k-day run
    on the other side. ok(i) is an extra condition on the first day (the rate condition)."""
    c = np.asarray(c)
    above, below = run_segments(c >= ref, k), run_segments(c <= ref, k)
    first, second = (below, above) if order == "fs" else (above, below)
    search_to = len(c) if search_to is None else search_to
    for s2, _ in second:
        if s2 < search_from or s2 > search_to:
            continue
        if ok is not None and not ok(s2):
            continue
        pre = [r for r in first if r[0] + k - 1 < s2]
        if pre:
            return s2, pre[-1]
    return None, None


def crossings(c, theta=THETA):
    s = np.sign(np.asarray(c) - theta)
    s[s == 0] = 1
    return int(np.sum(s[1:] != s[:-1]))


def pick_pixels(c_all, stride=4):
    ny, nx = c_all.shape[1], c_all.shape[2]
    clean, messy, max_cross = [], None, -1
    for iy in range(0, ny, stride):
        for ix in range(0, nx, stride):
            s = c_all[:, iy, ix]
            if np.isnan(s).mean() > 0.2:
                continue
            if np.nanmin(s) > 0.10 or np.nanmax(s) < 0.80:
                continue
            sz = np.nan_to_num(s, nan=0.0)
            n = crossings(sz)
            if n == 2:
                fs, _ = detect(sz, "fs", THETA)
                if fs is not None:
                    clean.append((fs, iy, ix))
            if n > max_cross:
                max_cross, messy = n, (iy, ix)
    clean.sort()
    return (clean[len(clean) // 2][1:] if clean else None), messy, max_cross


def references(series, times):
    """Dynamic FS / MS reference levels for one pixel from its whole record."""
    t = pd.DatetimeIndex(times)
    keep = ~((t.month == 2) & (t.day == 29))
    v, t = np.asarray(series)[keep], t[keep]
    doy = t.dayofyear.values - ((t.is_leap_year) & (t.month > 2)).astype(int)
    p_fs, p_ms = np.full(365, np.nan), np.full(365, np.nan)
    for d in range(1, 366):
        dd = np.abs(doy - d)
        near = v[np.minimum(dd, 365 - dd) <= 2]
        near = near[~np.isnan(near)]
        if near.size:
            p_fs[d - 1], p_ms[d - 1] = np.percentile(near, PCT_FS), np.percentile(near, PCT_MS)
    fs_days = np.arange(FS_WINDOW_DOY[0], FS_WINDOW_DOY[1] + 1) - 1
    ms_days = np.r_[np.arange(MS_WINDOW_DOY[0], 366), np.arange(1, MS_WINDOW_DOY[1] + 1)] - 1
    return max(np.nanmean(p_fs[fs_days]), THETA), np.nanmean(p_ms[ms_days])


def smooth_and_rate(c, k=K, h=H):
    cb = np.convolve(np.nan_to_num(c, nan=0.0), np.ones(k) / k, mode="same")
    r = np.full(len(cb), np.nan)
    r[h:-h] = (cb[2 * h:] - cb[:-2 * h]) / (2 * h)
    return cb, r


def shade(ax, dates, run, color):
    if run is not None:
        ax.axvspan(dates[run[0]], dates[min(run[1], len(dates) - 1)], color=color, alpha=0.18, lw=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc", default=NC_DEFAULT, required=NC_DEFAULT is None)
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--pixel-a", default=None)
    p.add_argument("--pixel-b", default=None)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()
    if FONT:
        matplotlib.rcParams["font.family"] = FONT
    matplotlib.rcParams.update({"text.color": INK, "axes.labelcolor": INK,
                                "xtick.color": INK, "ytick.color": INK})
    os.makedirs(args.out_dir, exist_ok=True)

    ds = xr.open_dataset(args.nc)
    t = pd.to_datetime(ds.time.values)
    y = args.year
    sel = np.where((t >= f"{y}-02-15") & (t <= f"{y + 1}-02-28"))[0]
    dates = t[sel]
    ms_from = int(np.searchsorted(dates, pd.Timestamp(f"{y}-08-15")))
    fs_to = int(np.searchsorted(dates, pd.Timestamp(f"{y}-09-30")))

    da = ds[VAR]
    c_all = unpack(da.isel(time=sel)).values
    if args.pixel_a and args.pixel_b:
        pa = tuple(int(v) for v in args.pixel_a.split(","))
        pb = tuple(int(v) for v in args.pixel_b.split(","))
    else:
        pa, pb, ncr = pick_pixels(c_all)
        print(f"panel a pixel (iy,ix) = {pa}, panel b = {pb}, max crossings = {ncr}")

    dy, dx = da.dims[1], da.dims[2]
    fig, axes = plt.subplots(1, 2, figsize=(W, H_IN), sharey=True)
    fig.patch.set_alpha(0)

    for ax, px, tag in zip(axes, (pa, pb), ("interior seasonal pixel", "repeated-crossing pixel")):
        full = unpack(da.isel({dy: px[0], dx: px[1]})).values
        ref_fs, ref_ms = references(full, t)
        c = c_all[:, px[0], px[1]]
        cz = np.nan_to_num(c, nan=0.0)
        cb, r = smooth_and_rate(c)

        fs_s, fs_pre = detect(cz, "fs", THETA, search_to=fs_to)
        ms_s, ms_pre = detect(cz, "ms", THETA, search_from=ms_from)
        fs_d, _ = detect(cb, "fs", ref_fs, search_to=fs_to, ok=lambda i: r[i] > RATE)
        ms_d, _ = detect(cb, "ms", ref_ms, search_from=ms_from, ok=lambda i: r[i] < -RATE)
        d = lambda i: dates[i].strftime("%d %b") if i is not None else "none"
        print(f"{tag} {px}: crossings {crossings(cz)} | static FS {d(fs_s)}, MS {d(ms_s)} | "
              f"dynamic FS {d(fs_d)}, MS {d(ms_d)} | refs {ref_fs:.2f}, {ref_ms:.2f}")

        ax.set_facecolor("none")
        if SHOW_RUNS:
            for run, col in ((fs_pre, "0.5"), ((fs_s, fs_s + K - 1) if fs_s is not None else None, COL_STATIC),
                             (ms_pre, "0.5"), ((ms_s, ms_s + K - 1) if ms_s is not None else None, COL_STATIC)):
                shade(ax, dates, run, col)
        ax.plot(dates, c, lw=1.6, color=COL_SIC, zorder=3)
        ax.axhline(THETA, color=COL_STATIC, lw=2.0, ls=(0, (5, 3)), zorder=2)
        ax.plot([dates[0], dates[fs_to]], [ref_fs, ref_fs], color=COL_DYN, lw=2.0, ls=(0, (1, 2)), zorder=2)
        ax.plot([dates[ms_from], dates[-1]], [ref_ms, ref_ms], color=COL_DYN, lw=2.0, ls=(0, (1, 2)), zorder=2)

        for idx, col, mk, ms, lab, dy_off in ((fs_s, COL_STATIC, "o", 11, "FS", -22), (ms_s, COL_STATIC, "o", 11, "MS", -22),
                                              (fs_d, COL_DYN, "D", 11, "FS", 16), (ms_d, COL_DYN, "D", 11, "MS", 16)):
            if idx is None:
                continue
            yv = THETA if col == COL_STATIC else float(cb[idx])
            ax.plot(dates[idx], yv, marker=mk, ms=ms, mfc=col, mec="white", mew=1.2, ls="none", zorder=6)
            ax.annotate(lab, xy=(dates[idx], yv), xytext=(0, dy_off), textcoords="offset points",
                        ha="center", va="center", fontsize=15, fontweight="bold", color=col, zorder=7)

        ax.text(0.02, 0.97, tag, transform=ax.transAxes, ha="left", va="top", fontsize=15, color=MUTED)
        ax.set_ylim(-0.12, 1.12)
        ax.set_xlim(dates[0], dates[-1])
        ax.set_xticks(pd.to_datetime([f"{y}-03-01", f"{y}-05-01", f"{y}-07-01", f"{y}-09-01",
                                      f"{y}-11-01", f"{y + 1}-01-01"]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("0.6")
        ax.tick_params(labelsize=15, length=4)

    axes[0].set_ylabel("Sea ice concentration (fraction)", fontsize=16, color=MUTED)
    # key at the top of the figure: which marker belongs to which method
    fig.text(0.075, 0.975, "\u25CF static: fixed 15%", color=COL_STATIC, ha="left", va="top",
             fontsize=16, fontweight="bold")
    fig.text(0.30, 0.975, "\u25C6 dynamic: the pixel's own reference", color=COL_DYN, ha="left", va="top",
             fontsize=16, fontweight="bold")
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.92))

    stem = os.path.join(args.out_dir, "detection_slide")
    fig.savefig(stem + ".svg", transparent=True)
    fig.savefig(stem + ".png", dpi=300, transparent=True)
    print("wrote", stem + ".png / .svg")


if __name__ == "__main__":
    main()
