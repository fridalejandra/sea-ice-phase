"""
Figure 1: detection schematic from real Bootstrap SIC data. Two panels.

(a) A clean interior seasonal pixel and (b) a repeated-crossing pixel, each showing:
    daily c(t); the static threshold theta = 0.15 with its completed k-day run on the
    departing side and qualifying k-day run on the arriving side (shaded); static FS and
    MS (filled circles); the dynamic reference levels for FS and MS (dotted lines); and
    the dynamic FS and MS (open diamonds).

Dynamic method, reproduced locally for the two pixels (same recipe as
compute_phase_dates_v3.py): per-day-of-year percentile (q = 0.70 FS, 0.30 MS) of SIC
within +/-2 days across all years (1987 and Feb 29 excluded), averaged over the search
window; FS reference floored at 0.15. Detection on the 5-day centred mean of SIC within
each search window, with two-sided k-day persistence and a centred slope over +/-3 days
exceeding 0.02 per day (positive for FS, negative for MS).

Pixel selection rule (stated in caption): among pixels crossing theta exactly twice
within the ice year, panel (a) is the pixel with the median FS date; panel (b) is the
pixel with the maximum crossing count.

Usage:
    python fig01_schematic_detection.py \
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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

THETA = 0.15
K = 5
GREY = "0.35"
CELL_KM2 = 25.0 * 25.0
COL_BELOW = "#0077BB"   # below-theta state / MS
COL_ABOVE = "#EE7733"   # above-theta state / FS
COL_LINE = "#332288"

# dynamic method settings (as in compute_phase_dates_v3.py)
Q_FS, Q_MS = 0.70, 0.30
DOY_HW = 2
SLOPE_HW = 3
SLOPE_MIN = 0.02
ABS_MIN = 0.15
BAD_YEARS = (1987,)
FS_DOYS = np.arange(46, 274)                                   # Feb 15 - Sep 30
MS_DOYS = np.concatenate([np.arange(227, 366), np.arange(1, 60)])  # Aug 15 - Feb 28


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
    """Static method. order='fs': below-run completed, then above-run begins (FS at its
    start). order='ms': the reverse. Returns (date_idx, precondition_run)."""
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


# ---------------------------------------------------------------- dynamic method
def doy365(t):
    """Day of year on a 365-day calendar (Feb 29 removed)."""
    t = pd.DatetimeIndex(t)
    return (t.dayofyear - ((t.is_leap_year) & (t.month > 2)).astype(int)).values


def dynamic_references(ds, px):
    """FS and MS references for one pixel from its full record."""
    s = unpack(ds["N07_ICECON"][:, px[0], px[1]]).values.astype(float)
    t = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    keep = ~((t.month == 2) & (t.day == 29)) & ~np.isin(t.year, BAD_YEARS)
    s, doy = s[keep], doy365(t[keep])
    clim = {}
    for q in (Q_FS, Q_MS):
        c = np.full(366, np.nan)
        for d in range(1, 366):
            dist = np.abs(doy - d)
            dist = np.minimum(dist, 365 - dist)
            v = s[dist <= DOY_HW]
            if np.isfinite(v).any():
                c[d] = np.nanquantile(v, q)
        clim[q] = c
    ref_fs = max(np.nanmean(clim[Q_FS][FS_DOYS]), ABS_MIN)
    ref_ms = np.nanmean(clim[Q_MS][MS_DOYS])
    return ref_fs, ref_ms


def detect_dynamic(c_win, ref, order, k=K):
    """Dynamic method on one search-window slice. Returns index within the slice."""
    cs = pd.Series(c_win).rolling(k, center=True, min_periods=max(1, k // 2)).mean().values
    ahead = np.concatenate([cs[SLOPE_HW:], np.full(SLOPE_HW, np.nan)])
    behind = np.concatenate([np.full(SLOPE_HW, np.nan), cs[:-SLOPE_HW]])
    slope = (ahead - behind) / (2.0 * SLOPE_HW)
    if order == "fs":
        cond, opp, ok = cs >= ref, cs < ref, slope > SLOPE_MIN
    else:
        cond, opp, ok = cs <= ref, cs > ref, slope < -SLOPE_MIN
    n = len(cs)
    opp_done = np.zeros(n, dtype=bool)          # a complete opposite run ended before t
    seen = False
    for t in range(n):
        opp_done[t] = seen
        if t >= k - 1 and opp[t - k + 1:t + 1].all():
            seen = True
    for t in range(n - k + 1):
        if opp_done[t] and cond[t:t + k].all() and ok[t]:
            return t
    return None


# ---------------------------------------------------------------- plotting helpers
def shade(ax, dates, run, color):
    if run is not None:
        ax.axvspan(dates[run[0]], dates[min(run[1], len(dates) - 1)],
                   color=color, alpha=0.35, lw=0)


def mark(ax, dates, idx, c, color, label, dynamic=False):
    if idx is None:
        return
    yv = float(np.nan_to_num(c[idx], nan=0.0))
    if dynamic:
        ax.plot(dates[idx], yv, marker="D", ms=5.5, mfc="white", mec=color,
                mew=1.4, ls="none", zorder=7)
        ax.annotate(label, xy=(dates[idx], yv), xytext=(6, 5),
                    textcoords="offset points", ha="left", fontsize=7.5,
                    fontweight="bold", color=color, annotation_clip=False, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    else:
        ax.plot(dates[idx], yv, marker="o", ms=6, mfc=color, mec="white",
                mew=0.8, ls="none", zorder=6)
        ax.annotate(label, xy=(dates[idx], yv), xytext=(-2, 10),
                    textcoords="offset points", ha="right", fontsize=8,
                    fontweight="bold", color=color, annotation_clip=False, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))


def style(ax, letter, title, ylab=None):
    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.text(-0.04, 1.10, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=11, va="bottom", ha="left")
    if ylab:
        ax.set_ylabel(ylab, fontweight="bold", color=GREY)
    ax.tick_params(colors=GREY, labelsize=7.5)
    for s in ax.spines.values():
        s.set_color(GREY)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc", default="/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc")
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
        print(f"panel a pixel (iy,ix) = {pb}, panel b = {pc}, max crossings = {ncr}")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.6), sharey=True)

    for ax, px, letter, title in zip(
            axes, (pb, pc), ("(a)", "(b)"),
            ("Interior seasonal pixel", "Repeated-crossing pixel")):
        c = c_all[:, px[0], px[1]]
        cz = np.nan_to_num(c, nan=0.0)
        for (i0, i1), col, yy in (((0, fs_to), COL_ABOVE, 1.06),
                                  ((ms_from, len(dates) - 1), COL_BELOW, 1.11)):
            ax.plot([dates[i0], dates[i1]], [yy, yy], lw=1.2, color=col,
                    marker="|", ms=4, mew=1.2, solid_capstyle="butt", clip_on=False)
        ax.plot(dates, c, lw=0.8, color=COL_LINE)
        ax.axhline(THETA, color=GREY, lw=0.8, ls="--")
        s = np.sign(cz - THETA); s[s == 0] = 1
        xi = np.where(s[1:] != s[:-1])[0]
        ax.plot(dates[xi], np.full(xi.size, THETA), ls="none", marker="o",
                ms=2.2, mfc="0.2", mec="none", zorder=5)

        # static method
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

        # dynamic method
        ref_fs, ref_ms = dynamic_references(ds, px)
        ax.hlines(ref_fs, dates[0], dates[fs_to], colors=COL_ABOVE, linestyles=":", lw=1.2)
        ax.hlines(ref_ms, dates[ms_from], dates[-1], colors=COL_BELOW, linestyles=":", lw=1.2)
        d_fs = detect_dynamic(c[:fs_to + 1], ref_fs, "fs")
        d_ms = detect_dynamic(c[ms_from:], ref_ms, "ms")
        d_ms = None if d_ms is None else d_ms + ms_from
        mark(ax, dates, d_fs, cz, COL_ABOVE, "FS$_{dyn}$", dynamic=True)
        mark(ax, dates, d_ms, cz, COL_BELOW, "MS$_{dyn}$", dynamic=True)

        fmt = lambda i: dates[i].strftime("%d %b") if i is not None else "none"
        print(f"{title}: static FS {fmt(fs)}, MS {fmt(ms)} | dynamic FS {fmt(d_fs)}, "
              f"MS {fmt(d_ms)} | references FS {ref_fs:.2f}, MS {ref_ms:.2f}")

        ax.set_ylim(-0.02, 1.16)
        style(ax, letter, f"{title} ({crossings(cz)} crossings)")

    axes[0].set_ylabel("SIC (fraction)", fontweight="bold", color=GREY)

    handles = [
        mlines.Line2D([], [], color=COL_ABOVE, lw=1.2, marker="|", ms=4,
                      label="FS search window (Feb 15 to Sep 30)"),
        mlines.Line2D([], [], color=COL_BELOW, lw=1.2, marker="|", ms=4,
                      label="MS search window (Aug 15 to Feb 28)"),
        mpatches.Patch(fc=COL_BELOW, alpha=0.35, label=r"$k$-day run at or below $\theta$"),
        mpatches.Patch(fc=COL_ABOVE, alpha=0.35, label=r"$k$-day run at or above $\theta$"),
        mlines.Line2D([], [], color=GREY, ls="--", lw=0.8, label=r"static $\theta$ = 0.15"),
        mlines.Line2D([], [], color="0.2", marker="o", ls="none", ms=3,
                      label=r"crossing of $\theta$"),
        mlines.Line2D([], [], color=COL_ABOVE, marker="o", ls="none", ms=5,
                      mec="white", label="static FS"),
        mlines.Line2D([], [], color=COL_BELOW, marker="o", ls="none", ms=5,
                      mec="white", label="static MS"),
        mlines.Line2D([], [], color=COL_ABOVE, ls=":", lw=1.2, label="dynamic FS reference"),
        mlines.Line2D([], [], color=COL_BELOW, ls=":", lw=1.2, label="dynamic MS reference"),
        mlines.Line2D([], [], color=COL_ABOVE, marker="D", ls="none", ms=4.5,
                      mfc="white", mew=1.3, label="dynamic FS"),
        mlines.Line2D([], [], color=COL_BELOW, marker="D", ls="none", ms=4.5,
                      mfc="white", mew=1.3, label="dynamic MS"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.20, 1, 0.98))
    fig.savefig(args.out, dpi=300)
    from pathlib import Path as _P
    _pp = _P(args.out).with_name("ppt_" + _P(args.out).name)
    fig.savefig(_pp, dpi=300, transparent=True)
    print(f"wrote {_pp}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
