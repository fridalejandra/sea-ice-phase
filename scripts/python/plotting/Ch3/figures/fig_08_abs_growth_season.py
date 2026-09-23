#!/usr/bin/env python3
"""
fig_08_abs_growth_season.py -- Fig. 8: Amundsen-Bellingshausen amplitude against
growth-season length (day of maximum minus day of minimum), 1979-2015 and
2016-2025 (Sect. 3.4.2).

Both periods are drawn: the earlier years as small grey points (unrelated,
r = +0.09), the years since 2016 as labelled points coloured by year (viridis) with their fit line.
The correlation in each period (on linearly detrended series, the chapter's
convention) and the p of their difference (Fisher z) are printed in the panel
and to the console; they match ch3_stats.py's 3.3c block / the ledger.
The points and the two fit lines are the raw values, in physical units:
the grey line (1979-2015) is flat, the blue one (2016-2025) is not.

Reads   annual_params.csv  (period == FULL; raw amplitude and raw day-of-year
        columns, not the anomalies)
Writes  results/ch3/figures/fig08_abs_growth_season.png

Run from scripts/python/plotting/Ch3/figures/.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import ANNUAL_CSV, SECTORS, OUTPUT_DIR, BREAK_YEAR
import ch3_style  # font + spines; bold_font_properties() for real bold

INK = "0.35"
COLOR_PRE = "0.62"


def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"no column for '{label}'; tried {candidates}; have {list(df.columns)}")


def fisher_shift_p(r1, n1, r2, n2):
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(2 * stats.norm.sf(abs(z)))


def detrend(x):
    """Remove a linear trend in time -- the chapter's convention for every
    correlation (Sect. 2.3), and what ch3_stats.py's 3.3c block does."""
    x = np.asarray(x, float); t = np.arange(len(x), dtype=float)
    b, a = np.polyfit(t, x, 1)
    return x - (a + b * t)


def pear_dt(x, y):
    return stats.pearsonr(detrend(x), detrend(y))


def fmt_p(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"].astype(str) == "FULL"].copy()
dup = ann.duplicated(["sector", "Year"], keep=False)
if dup.any():
    sys.exit(f"annual_params has {int(dup.sum())} rows sharing a (sector, Year) after the FULL filter")

abs_sector = [s for s in SECTORS if "Amundsen" in s][0]
a = ann[ann["sector"] == abs_sector].sort_values("Year")
if a.empty:
    sys.exit(f"no rows for {abs_sector!r}; sectors present: {sorted(ann['sector'].unique())}")

amp_col = pick_col(a, ["amplitude_raw_yr", "amplitude_raw", "amplitude"], "amplitude (raw)")
max_col = pick_col(a, ["max_doy_raw", "max_doy"], "day of max (raw)")
min_col = pick_col(a, ["min_doy_raw", "min_doy"], "day of min (raw)")

a = a.assign(length=a[max_col] - a[min_col], amp=a[amp_col]).dropna(subset=["length", "amp"])
pre, post = a[a["Year"] < BREAK_YEAR], a[a["Year"] >= BREAK_YEAR]
YR_MAX = int(a["Year"].max())
PRE_LABEL, POST_LABEL = f"{int(a['Year'].min())}–{BREAK_YEAR - 1}", f"{BREAK_YEAR}–{YR_MAX}"

# correlations on detrended series within each period, exactly as the ledger
# (t33c_growth_length_vs_amplitude.csv) computes them, so figure and text agree
r_pre, p_pre = pear_dt(pre["length"], pre["amp"])
r_post, p_post = pear_dt(post["length"], post["amp"])
p_shift = fisher_shift_p(r_pre, len(pre), r_post, len(post))

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.scatter(pre["length"], pre["amp"], s=28, color=COLOR_PRE, edgecolor="none", zorder=2,
           label=f"{PRE_LABEL}  (r = {r_pre:+.2f})")
# recent years coloured by year (viridis), so the sequence 2016 -> 2025 can be read off
sc = ax.scatter(post["length"], post["amp"], c=post["Year"], cmap="viridis",
                vmin=BREAK_YEAR, vmax=YR_MAX, s=80, edgecolor="white", linewidth=0.8, zorder=4)
from matplotlib.lines import Line2D
post_handle = Line2D([], [], marker="o", ls="none", ms=8, markeredgecolor="white",
                     markerfacecolor=plt.get_cmap("viridis")(0.55),
                     label=f"{POST_LABEL}  (r = {r_post:+.2f}, {fmt_p(p_post)})")
# a fit line for each period, so the contrast is visible: flat before 2016, steep after
for grp, color, lw in ((pre, COLOR_PRE, 1.4), (post, "0.2", 1.8)):
    m, b = np.polyfit(grp["length"], grp["amp"], 1)
    xx = np.linspace(grp["length"].min() - 4, grp["length"].max() + 4, 50)
    ax.plot(xx, m * xx + b, color=color, lw=lw, ls="--", zorder=3)
cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, ticks=list(range(BREAK_YEAR, YR_MAX + 1)))
cbar.ax.tick_params(labelsize=8, colors=INK, length=0)
cbar.outline.set_visible(False)

# year labels on the recent points, nudged so neighbours do not collide
small_bold = ch3_style.bold_font_properties(size=8)
placed = []
for x, y, yr in zip(post["length"], post["amp"], post["Year"]):
    crowded = any(abs(x - px) < 10 and abs(y - py) < 0.07 for px, py in placed)
    xy, ha = ((-6, -11), "right") if crowded else ((6, 5), "left")   # second of a close pair goes below-left
    ax.annotate(str(int(yr)), (x, y), textcoords="offset points", xytext=xy, ha=ha,
                fontsize=8, color="0.15", fontproperties=small_bold)
    placed.append((x, y))

ax.set_xlabel("growth-season length, day of maximum − day of minimum (days)", color=INK)
ax.set_ylabel("amplitude (10⁶ km²)", color=INK)
ax.set_title("Amundsen-Bellingshausen", loc="left",
             fontproperties=ch3_style.bold_font_properties(size=11), pad=10)
pre_handle = ax.collections[0]
ax.legend(handles=[pre_handle, post_handle], frameon=False, fontsize=9, loc="upper left")
ax.text(0.98, 0.03, f"change between periods {fmt_p(p_shift)}\n"
        "correlations on linearly detrended series, as in the text",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK, linespacing=1.4)
ax.tick_params(colors=INK, labelsize=9, length=3)
for s in ("left", "bottom"):
    ax.spines[s].set_color(INK)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, "fig08_abs_growth_season.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
print(f"  {PRE_LABEL}: r = {r_pre:+.3f} (n = {len(pre)}, {fmt_p(p_pre)})")
print(f"  {POST_LABEL}: r = {r_post:+.3f} (n = {len(post)}, {fmt_p(p_post)})")
print(f"  change between periods: {fmt_p(p_shift)}")
print(post[["Year", min_col, max_col, "length", "amp"]].to_string(index=False))