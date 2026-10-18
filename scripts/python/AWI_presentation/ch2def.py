#!/usr/bin/env python3
"""
fig_two_definitions_slide.py -- Ch2 slide figure from real Bootstrap SIC.

Layout (one figure, transparent background, slide colours):
  top row   (a) interior seasonal pixel   (b) repeated-crossing pixel
            daily c(t) for the chosen ice year, the 15% line, the pixel's own reference,
            static dates (circles, at 15%) and dynamic dates (diamonds, at the reference).
            No search windows, no k-day shading, no crossing dots.
  bottom    season-length bars, active-pixel means from Chapter 2 (ice present vs consolidated,
            1979-2015 vs 2016-2024), or computed at the two pixels with --bars pixel.

Data handling and the static detection are taken from fig_schematic_detection.py unchanged.
The dynamic detection here is a simplified stand-in (reference = pixel's median in-season SIC over the
baseline years, onset = first k-day run at or beyond it with a rise/fall faster than RATE per day).
Replace `dynamic_detect` with the chapter's routine, or pass the dates with --dates, for the final slide.

Usage:
    python fig_two_definitions_slide.py --nc merged_bootstrap_SH_latest.nc --year 2019
    Optional: --pixel-b IY,IX --pixel-c IY,IX   --bars chapter|pixel   --out two_definitions_slide.png
              --dates "b:FSs,MSs,FSd,MSd;c:FSs,MSs,FSd,MSd"  (YYYY-MM-DD, '-' for none) to override detection
"""
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

THETA, K, RATE = 0.15, 5, 0.02
INK, INK2, GRID = "#222222", "#555555", "#C8CBD0"
STATIC, DYNAMIC, TRACE = "#0072B2", "#CC79A7", "#3B2F6B"
CHAPTER_BARS = {  # active-pixel means, days (Ch2 Sect. 2.3.2-2.3.3): static FS/MS, dynamic FS/MS as day of year
    "1979–2015": {"static": (115, 335), "dynamic": (124.5, 327)},
    "2016–2024": {"static": (123, 328), "dynamic": (132.5, 320)},
}

# ---- from fig_schematic_detection.py -------------------------------------
def unpack(da):
    v = da.where(da < 1050)
    if float(v.max()) > 1.5:
        v = v / 1000.0
    return v

def run_segments(mask, k):
    m = np.asarray(mask, dtype=bool).astype(int)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0] - 1
    return [(s, e) for s, e in zip(starts, ends) if e - s + 1 >= k]

def detect(c, order, search_from=0, search_to=None, k=K, theta=THETA):
    c = np.asarray(c)
    above = run_segments(c >= theta, k); below = run_segments(c <= theta, k)
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
    s = np.sign(np.asarray(c) - theta); s[s == 0] = 1
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

# ---- dynamic detection: simplified stand-in --------------------------------
def smooth(x, n=5):
    xp = np.pad(x, n, mode="edge")
    return np.convolve(xp, np.ones(n) / n, mode="same")[n:-n]

def dynamic_detect(c, ref, fs_to, ms_from, k=K, rate=RATE):
    """FS: first day of a k-day run at/above ref, within the FS window, whose preceding k days rose faster
    than `rate` per day. MS: first day of a k-day run below ref, within the MS window, after FS, with a fall
    faster than `rate`. Stand-in for the chapter's routine."""
    cs = smooth(np.nan_to_num(c, nan=0.0), 5)
    fs = ms = None
    for s, e in run_segments(cs >= ref, k):
        if s <= fs_to and s >= k and (cs[s] - cs[s - k]) / k >= rate * 0.5:
            fs = s; break
    if fs is not None:
        for s, e in run_segments(cs < ref, k):
            if s >= max(ms_from, fs + k) and s >= k and (cs[s - k] - cs[s]) / k >= rate * 0.5:
                ms = s; break
    return fs, ms

def reference_level(ds, iy, ix, t, years=(1988, 2015)):
    """pixel's own reference, stand-in: 0.7 x the median in-season (Jun-Oct) SIC over the baseline years.
    The chapter derives FS and MS references from the pixel's own climatology; substitute those here."""
    sel = np.where((t.year >= years[0]) & (t.year <= years[1]) & (t.month >= 6) & (t.month <= 10))[0]
    v = unpack(ds["N07_ICECON"].isel(time=sel, y=iy, x=ix)).values
    return float(0.7 * np.nanmedian(v))

# ---- drawing ---------------------------------------------------------------
def panel(ax, dates, c, ref, marks, title):
    sF, sM, dF, dM = marks
    ax.plot(dates, c, color=TRACE, lw=1.3, zorder=3)
    ax.axhline(THETA, color=STATIC, lw=1, ls="--", zorder=1)
    ax.axhline(ref, color=DYNAMIC, lw=1, ls=":", zorder=1)
    for i, lab, col, m, lv in [(sF, "FS", STATIC, "o", THETA), (sM, "MS", STATIC, "o", THETA),
                               (dF, "FS", DYNAMIC, "D", ref), (dM, "MS", DYNAMIC, "D", ref)]:
        if i is None:
            continue
        ax.scatter([dates[i]], [lv], s=95, marker=m, color=col, edgecolor="white", lw=1.2, zorder=6)
        ax.annotate(lab, (dates[i], lv), xytext=(7 if lab == "FS" else -7, -12), textcoords="offset points",
                    ha="left" if lab == "FS" else "right", va="top", color=col, fontsize=11.5, fontweight="bold")
    ax.set_title(title, loc="left", fontsize=13.5, color=INK, fontweight="bold")
    ax.set_ylim(-0.03, 1.08); ax.set_yticks([0, 0.15, 0.5, 1.0]); ax.set_yticklabels(["0", "15", "50", "100"], color=INK2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)

def bars(ax, table):
    for i, (per, v) in enumerate(table.items()):
        a = 0.45 if i == 0 else 0.95
        ys, yd = 3.2 - i * 1.5, 2.65 - i * 1.5
        (sF, sM), (dF, dM) = v["static"], v["dynamic"]
        ax.barh(ys, sM - sF, left=sF, height=0.42, color=STATIC, alpha=a, zorder=3)
        ax.barh(yd, dM - dF, left=dF, height=0.42, color=DYNAMIC, alpha=a, zorder=3)
        ax.text(sF - 8, (ys + yd) / 2, per, ha="right", va="center", color=INK, fontsize=12.5, fontweight="bold")
        ax.text(sM + 8, ys, f"ice present  {sM - sF:.0f} d", va="center", color=INK2, fontsize=11.5)
        ax.text(dM + 8, yd, f"consolidated  {dM - dF:.0f} d", va="center", color=INK2, fontsize=11.5)
    (sF1, sM1), (dF1, dM1) = table[list(table)[-1]]["static"], table[list(table)[-1]]["dynamic"]
    for x0, x1, lab, ha in [(sF1, dF1, f"starts {dF1 - sF1:.0f} d later", "left"), (dM1, sM1, f"ends {sM1 - dM1:.0f} d earlier", "right")]:
        ax.plot([x0, x1], [0.5, 0.5], color=INK, lw=1.4)
        for x in (x0, x1): ax.plot([x, x], [0.4, 0.6], color=INK, lw=1.2)
        ax.text(x1 + 5 if ha == "left" else x0 - 5, 0.5, lab, ha=ha, va="center", color=INK, fontsize=11)
    ax.text(sF1 - 8, 0.5, "consolidated season vs ice present:", ha="right", va="center", color=INK2, fontsize=11)
    ax.set_xlim(30, 400); ax.set_ylim(0.2, 3.6); ax.set_yticks([])
    ax.set_xticks([46, 105, 166, 227, 288, 349]); ax.set_xticklabels(["Feb", "Apr", "Jun", "Aug", "Oct", "Dec"], color=INK2)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0)

def pixel_bars(ds, t, px, ref):
    """season bars computed at one pixel: mean dates over the two periods."""
    out = {}
    for per, (y0, y1) in {"1979–2015": (1979, 2015), "2016–2024": (2016, 2024)}.items():
        acc = {"static": [], "dynamic": []}
        for y in range(y0, y1 + 1):
            sel = np.where((t >= f"{y}-02-15") & (t <= f"{y + 1}-02-28"))[0]
            if sel.size < 300: continue
            dates = t[sel]; c = np.nan_to_num(unpack(ds["N07_ICECON"].isel(time=sel, y=px[0], x=px[1])).values, nan=0.0)
            ms_from = int(np.searchsorted(dates, pd.Timestamp(f"{y}-08-15"))); fs_to = int(np.searchsorted(dates, pd.Timestamp(f"{y}-09-30")))
            sF, _ = detect(c, "fs", search_to=fs_to); sM, _ = detect(c, "ms", search_from=ms_from)
            dF, dM = dynamic_detect(c, ref, fs_to, ms_from)
            if None not in (sF, sM): acc["static"].append((sF + 46, sM + 46))
            if None not in (dF, dM): acc["dynamic"].append((dF + 46, dM + 46))
        out[per] = {k: tuple(np.mean(v, axis=0)) for k, v in acc.items() if v}
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc", required=True); p.add_argument("--year", type=int, default=2019)
    p.add_argument("--pixel-b", default=None); p.add_argument("--pixel-c", default=None)
    p.add_argument("--bars", default="chapter", choices=["chapter", "pixel", "none"])
    p.add_argument("--dates", default=None); p.add_argument("--out", default="two_definitions_slide.png")
    args = p.parse_args()

    ds = xr.open_dataset(args.nc); t = pd.to_datetime(ds.time.values); y = args.year
    sel = np.where((t >= f"{y}-02-15") & (t <= f"{y + 1}-02-28"))[0]; dates = t[sel]
    ms_from = int(np.searchsorted(dates, pd.Timestamp(f"{y}-08-15"))); fs_to = int(np.searchsorted(dates, pd.Timestamp(f"{y}-09-30")))
    c_all = unpack(ds["N07_ICECON"].isel(time=sel)).values

    if args.pixel_b and args.pixel_c:
        pb = tuple(int(v) for v in args.pixel_b.split(",")); pc = tuple(int(v) for v in args.pixel_c.split(","))
    else:
        pb, pc, ncr = pick_pixels(c_all); print(f"panel a pixel (iy,ix) = {pb}, panel b = {pc}, max crossings = {ncr}")

    given = {}
    if args.dates:
        for blk in args.dates.split(";"):
            key, vals = blk.split(":")
            given[key] = [None if v.strip() == "-" else int(np.searchsorted(dates, pd.Timestamp(v))) for v in vals.split(",")]

    plt.rcParams.update({"font.family": "Arial", "font.size": 12.5})
    nrows = 2 if args.bars != "none" else 1
    fig = plt.figure(figsize=(12, 8.2 if nrows == 2 else 4.6), dpi=200); fig.patch.set_alpha(0)
    gs = fig.add_gridspec(nrows, 2, height_ratios=[1.0, 0.75] if nrows == 2 else [1], hspace=0.45, wspace=0.08)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes[1].sharey(axes[0])

    refs = {}
    for ax, px, key, title in zip(axes, (pb, pc), ("b", "c"), ("(a)  Interior seasonal pixel", "(b)  Repeated-crossing pixel")):
        c = c_all[:, px[0], px[1]]; cz = np.nan_to_num(c, nan=0.0)
        ref = reference_level(ds, px[0], px[1], t); refs[key] = ref
        if key in given:
            marks = tuple(given[key])
        else:
            sF, _ = detect(cz, "fs", search_to=fs_to); sM, _ = detect(cz, "ms", search_from=ms_from)
            dF, dM = dynamic_detect(cz, ref, fs_to, ms_from); marks = (sF, sM, dF, dM)
        ax.set_facecolor("none")
        panel(ax, dates, c, ref, marks, f"{title}  ({crossings(cz)} crossings)")
        print(key, "static", [None if i is None else str(dates[i].date()) for i in marks[:2]],
              "dynamic", [None if i is None else str(dates[i].date()) for i in marks[2:]], f"ref {ref:.2f}")
    axes[0].set_ylabel("Sea ice concentration (%)", color=INK2)
    axes[0].text(dates[2], refs["b"] + 0.05, "pixel's own reference", color=DYNAMIC, fontsize=11)
    axes[0].text(dates[2], THETA + 0.04, "fixed 15 %", color=STATIC, fontsize=11)
    plt.setp(axes[1].get_yticklabels(), visible=False)

    if nrows == 2:
        axb = fig.add_subplot(gs[1, :]); axb.set_facecolor("none")
        table = CHAPTER_BARS if args.bars == "chapter" else pixel_bars(ds, t, pb, refs["b"])
        bars(axb, table)
        axb.set_title("(c)  Season length, " + ("active-pixel mean" if args.bars == "chapter" else "at the interior pixel"),
                      loc="left", fontsize=13.5, color=INK, fontweight="bold")

    handles = [Line2D([], [], marker="o", ls="", color=STATIC, ms=9, label="fixed 15 %: first / last presence of ice"),
               Line2D([], [], marker="D", ls="", color=DYNAMIC, ms=8, label="locally referenced: consolidation / decay")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.995), columnspacing=2.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.07)
    fig.savefig(args.out, transparent=True); fig.savefig(args.out.replace(".png", ".svg"), transparent=True)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()