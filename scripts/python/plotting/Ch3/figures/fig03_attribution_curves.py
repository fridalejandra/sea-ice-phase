"""
fig03_attribution_curves.py  —  §3.1 attribution as daily component curves.

For each sector, the mean departure of observed SIE from the invariant cycle
(anomaly_from_iac) is split, day of year by day of year, into its four additive
daily components (trend, amplitude, phase, residual), which sum to it. Curves,
not bars: the shape of each component shows WHEN in the year it acts and which
one dominates each sector.

Layout (Option 2): columns = sectors, rows = [2016–2023 mean, worst single year].
"Worst year" = the year of largest absolute annual-mean anomaly_from_iac in that
sector (a rule, not a choice; the year is labelled in the panel). Fills are drawn
for the two largest components in each panel so crossings stay legible.

Run from scripts/python/plotting/Ch3/figures/ :  python fig03_attribution_curves.py
Set SECTORS_MAIN below to the six-sector list for the supplement version.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
from ch3_config import OUTPUT_DIR, BREAK_YEAR, YEAR_MIN, YEAR_MAX  # paths/constants
from ch3_config import DAILY_CSV  # canonical daily_fitted.csv (old _E fallback removed 2026-10-04)

# sectors to show (main text = 3); swap to the six-name list for the supplement
SECTORS_MAIN = ["SIE_Amundsen_Bellingshausen", "SIE_East_Antarctica", "SIE_Weddell"]
LABELS = {"SIE_Amundsen_Bellingshausen": "Amundsen–Bellingshausen",
          "SIE_East_Antarctica": "East Antarctica", "SIE_Weddell": "Weddell",
          "SIE_Ross": "Ross", "SIE_King_Haakon": "King Haakon", "SIE_circumpolar": "Circumpolar"}

COMPS = [("trend_component", "trend", "#5b6b7a"),
         ("amplitude_component", "amplitude", "#2c7fb8"),
         ("phase_component", "phase", "#e8703a"),
         ("residual_apac", "residual", "#c4c4c4")]
BLACK = "anomaly_from_iac"

d = pd.read_csv(DAILY_CSV)
d = d[(d.Year >= YEAR_MIN) & (d.Year <= YEAR_MAX)].copy()
# guard: components must sum to anomaly_from_iac
chk = (d[[c for c, _, _ in COMPS]].sum(axis=1) - d[BLACK]).abs().max()
assert chk < 1e-6, f"components do not sum to {BLACK} (max err {chk:.2e})"

# MODE: "mean_worst" (two rows: 2016-2023 mean, and the extreme year), or
#       "years" (one row per named year in YEARS), or "mean_only" (single row).
MODE = "years"
YEARS = [2016, 2023]                 # used when MODE == "years"
WORST_WINDOW = (BREAK_YEAR, YEAR_MAX)  # worst-year search restricted to the recent, fully-sampled period

def extreme_year(sec, lo, hi):
    a = d[(d.sector == sec) & (d.Year >= lo) & (d.Year <= hi)].groupby("Year")[BLACK].mean()
    return int(a.abs().idxmax()), float(a.loc[a.abs().idxmax()])

def curves(sub):
    g = sub.groupby("DOY")
    return {c: g[c].mean().reindex(range(1, 366)).values for c, _, _ in COMPS} | \
           {BLACK: g[BLACK].mean().reindex(range(1, 366)).values}

t = np.arange(1, 366)
mon = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]; ml = list("JFMAMJJASOND")

# build the list of rows: each row is (row_label, function(sector)->(curves, panel_note))
if MODE == "mean_only":
    rows = [(f"{BREAK_YEAR}–{YEAR_MAX} mean",
             lambda s: (curves(d[(d.sector == s) & (d.Year >= BREAK_YEAR)]), None))]
elif MODE == "mean_worst":
    def _worst(s):
        wy, _ = extreme_year(s, *WORST_WINDOW); return curves(d[(d.sector == s) & (d.Year == wy)]), str(wy)
    rows = [(f"{BREAK_YEAR}–{YEAR_MAX} mean",
             lambda s: (curves(d[(d.sector == s) & (d.Year >= BREAK_YEAR)]), None)),
            ("extreme year", _worst)]
else:  # "years"
    rows = [(str(y), (lambda yy: lambda s: (curves(d[(d.sector == s) & (d.Year == yy)]), None))(y)) for y in YEARS]

nrow, ncol = len(rows), len(SECTORS_MAIN)
# share y within a row (each row auto-scales to itself), not across rows
fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.1 * nrow), sharex=True, sharey="row")
axes = np.atleast_2d(axes)

def draw(ax, cur, title=None, ylab=None, note=None):
    mag = {c: np.nanmean(np.abs(cur[c])) for c, _, _ in COMPS}
    top2 = sorted(mag, key=mag.get, reverse=True)[:2]
    for c, lab, col in COMPS:
        ax.plot(t, cur[c], color=col, lw=1.6, zorder=3)
        if c in top2:
            ax.fill_between(t, 0, cur[c], color=col, alpha=0.13, zorder=1)
    ax.plot(t, cur[BLACK], color="k", lw=2.1, zorder=4)
    ax.axhline(0, color="k", lw=0.6, ls=(0, (4, 3)))
    ax.set_xlim(1, 365); ax.set_xticks(mon); ax.set_xticklabels(ml, fontsize=7)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if title: ax.set_title(title, fontsize=10.5, pad=4)
    if ylab: ax.set_ylabel(ylab, fontsize=8)
    if note: ax.annotate(note, xy=(0.04, 0.06), xycoords="axes fraction", fontsize=10, fontweight="bold")

for i, (rlab, fn) in enumerate(rows):
    for j, sec in enumerate(SECTORS_MAIN):
        cur, note = fn(sec)
        draw(axes[i, j], cur,
             title=LABELS.get(sec, sec) if i == 0 else None,
             ylab=f"{rlab}\nSIE − invariant cycle (10$^6$ km$^2$)" if j == 0 else None,
             note=note)

leg = [Line2D([], [], color="k", lw=2.1, label="observed − invariant cycle")] + \
      [Line2D([], [], color=col, lw=1.6, label=lab) for _, lab, col in COMPS]
fig.legend(handles=leg, ncol=5, loc="lower center", frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.05, 1, 0.99])
out = os.path.join(OUTPUT_DIR, "fig03_attribution_curves.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out, "  MODE =", MODE)
for sec in SECTORS_MAIN:
    wy, wv = extreme_year(sec, *WORST_WINDOW)
    print(f"  extreme {BREAK_YEAR}-{YEAR_MAX}  {LABELS.get(sec, sec):24s} {wy}  (annual anomaly {wv:+.2f})")
