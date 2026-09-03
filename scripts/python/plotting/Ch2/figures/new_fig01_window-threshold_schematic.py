"""
Figure 1: detection schematic from real Bootstrap SIC data. Three panels.

(a) Circumpolar SIE annual cycle for the chosen ice year (Feb 15 to Feb 14),
    the familiar smooth aggregate.
(b) A clean interior seasonal pixel: daily c(t), theta, the completed k-day
    run on the departing side (shaded, state color) and the qualifying k-day
    run on the arriving side (shaded), FS and MS marked at run starts.
(c) A repeated-crossing pixel: same construction, showing the transient
    crossings the persistence criterion rejects.

Pixel selection rule (stated in caption): among pixels crossing theta exactly
twice within the ice year, panel (b) is the pixel with the median FS date;
panel (c) is the pixel with the maximum crossing count.

Usage:
    python fig_schematic_detection.py \
        --nc /path/to/merged_bootstrap_SH_latest.nc --year 2019 \
        --out schematic_detection.png
    Optional: --pixel-b IY,IX --pixel-c IY,IX
"""

import argparse
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

THETA = 0.15
K = 5
GREY = "0.35"
CELL_KM2 = 25.0 * 25.0
COL_BELOW = "#377eb8"   # below-theta state
COL_ABOVE = "#e6550d"   # above-theta state
COL_LINE = "#54278f"

def unpack(da):
    v = da.where(da < 1050)
    if float(v.max()) > 1.5:
        v = v / 1000.0
    return v

def run_segments(mask, k):
    """(start, end) inclusive of maximal runs of True with length >= k."""
    m = np.asarray(mask, dtype=bool).astype(int)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return [(s, e) for s, e in zip(starts, ends) if e - s + 1 >= k]

def detect(c, order, search_from=0, search_to=None, k=K, theta=THETA):
    """order='fs': below-run completed, then above-run begins (FS at its start).
    order='ms': above-run completed, then below-run begins. Returns
    (date_idx, precondition_run) with runs as (start, end) tuples."""
    c = np.asarray(c)
    above = run_segments(c >= theta, k)
    below = run_segments(c <= theta, k)
    first, second = (below, above) if order == "fs" else (above, below)
    if search_to is None:
        search_to = len(c)
    for s2, e2 in second:
        if s2 < search_from or s2 > search_to:
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
                fs, _ = detect(sz, "fs")
                if fs is not None:
                    clean.append((fs, iy, ix))
            if n > max_cross:
                max_cross, messy = n, (iy, ix)
    clean.sort()
    b = clean[len(clean) // 2][1:] if clean else None
    return b, messy, max_cross

def shade(ax, dates, run, color):
    if run is not None:
        ax.axvspan(dates[run[0]], dates[min(run[1], len(dates) - 1)],
                   color=color, alpha=0.35, lw=0)

def mark(ax, dates, idx, c, color, label):
    if idx is not None:
        ytop = min(float(np.nan_to_num(c[idx], nan=0.0)) + 0.28, 0.95)
        ax.plot([dates[idx], dates[idx]], [0, ytop], color=color, lw=1.4)
        ax.annotate(label, xy=(dates[idx], ytop), xytext=(dates[idx], ytop + 0.04),
                    ha="center", fontsize=8, fontweight="bold",
                    color=color, annotation_clip=False)

def style(ax, letter, title, ylab=None):
    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.text(0.03, 0.96, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=11, va="top")
    if ylab:
        ax.set_ylabel(ylab, fontweight="bold", color=GREY)
    ax.xaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=7.5)
    for s in ax.spines.values():
        s.set_color(GREY)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc", required=True)
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--pixel-b", default=None)
    p.add_argument("--pixel-c", default=None)
    p.add_argument("--out", default="schematic_detection.png")
    args = p.parse_args()

    ds = xr.open_dataset(args.nc)
    t = pd.to_datetime(ds.time.values)
    y = args.year
    sel = np.where((t >= f"{y}-02-15") & (t <= f"{y + 1}-02-28"))[0]
    dates = t[sel]
    ms_from = int(np.searchsorted(dates, pd.Timestamp(f"{y}-08-15")))
    fs_to = int(np.searchsorted(dates, pd.Timestamp(f"{y}-09-30")))

    c_all = unpack(ds["N07_ICECON"].isel(time=sel)).values

    if args.pixel_b and args.pixel_c:
        pb = tuple(int(v) for v in args.pixel_b.split(","))
        pc = tuple(int(v) for v in args.pixel_c.split(","))
    else:
        pb, pc, ncr = pick_pixels(c_all)
        print(f"panel b pixel (iy,ix) = {pb}, panel c = {pc}, "
              f"max crossings = {ncr}")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), sharey=True)

    for ax, px, letter, title in zip(
            axes, (pb, pc), ("(a)", "(b)"),
            ("Interior seasonal pixel", "Repeated-crossing pixel")):
        c = c_all[:, px[0], px[1]]
        cz = np.nan_to_num(c, nan=0.0)
        ax.axvspan(dates[0], dates[fs_to], color=COL_ABOVE, alpha=0.06, lw=0)
        ax.axvspan(dates[ms_from], dates[-1], color=COL_BELOW, alpha=0.06, lw=0)
        ax.plot(dates, c, lw=0.8, color=COL_LINE)
        ax.axhline(THETA, color=GREY, lw=0.8, ls="--")
        s = np.sign(cz - THETA); s[s == 0] = 1
        xi = np.where(s[1:] != s[:-1])[0]
        ax.plot(dates[xi], np.full(xi.size, THETA), ls="none", marker="o",
                ms=2.2, mfc="0.2", mec="none", zorder=5)
        fs, fs_pre = detect(cz, "fs", search_to=fs_to)
        ms, ms_pre = detect(cz, "ms", search_from=ms_from)
        if fs_pre is not None:
            shade(ax, dates, (fs_pre[1] - K + 1, fs_pre[1]), COL_BELOW)
        if fs is not None:
            shade(ax, dates, (fs, fs + K - 1), COL_ABOVE)
        if ms_pre is not None:
            shade(ax, dates, (ms_pre[1] - K + 1, ms_pre[1]), COL_ABOVE)
        if ms is not None:
            shade(ax, dates, (ms, ms + K - 1), COL_BELOW)
        mark(ax, dates, fs, cz, COL_ABOVE, "FS")
        mark(ax, dates, ms, cz, COL_BELOW, "MS")
        ax.set_ylim(-0.02, 1.12)
        style(ax, letter, title)
        ax.text(0.03, 0.90, f"{crossings(cz)} crossings of " + r"$\theta$" + " (dots)",
                transform=ax.transAxes, fontsize=7.5, color=GREY,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))
    axes[0].set_ylabel("SIC (fraction)", fontweight="bold", color=GREY)

    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    handles = [
        mpatches.Patch(fc=COL_ABOVE, alpha=0.25, label="FS search window (Feb 15 to Sep 30)"),
        mpatches.Patch(fc=COL_BELOW, alpha=0.25, label="MS search window (Aug 15 to Feb 28)"),
        mpatches.Patch(fc=COL_BELOW, alpha=0.35, label=r"$k$-day run at or below $\theta$"),
        mpatches.Patch(fc=COL_ABOVE, alpha=0.35, label=r"$k$-day run at or above $\theta$"),
        mlines.Line2D([], [], color=GREY, ls="--", lw=0.8, label=r"$\theta$ = 0.15"),
        mlines.Line2D([], [], color="0.2", marker="o", ls="none", ms=3,
                      label=r"crossing of $\theta$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()