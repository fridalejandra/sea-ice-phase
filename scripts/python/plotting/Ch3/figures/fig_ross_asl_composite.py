#!/usr/bin/env python3
"""
fig_asl_ross_mechanism.py -- DRAFT, unnumbered. Candidate figure for Sect. 3.3.

Rebuilt from scratch (2026-10-04): the script behind the reference figure
("Weddell sea ice: same sector, two mechanisms" -- SAM (JJA) vs Nino3.4
(annual), dominant: amplitude vs dominant: level) could not be found
anywhere synced to this session, so this reconstructs the same design from
the image alone:

  - split years into low/high TERCILES of an index (bottom third / top
    third by value; middle third dropped -- matches the n=16 groups out of
    47 years in the reference image, 47 // 3 == 15, close enough that the
    reference was very likely built the same way)
  - build each group's composite seasonal cycle (mean daily extent, by day
    of year, across the years in that group)
  - compute, from each group's composite curve, the same two quantities
    check_level_shift.py already uses elsewhere in this repo: RANGE
    (Jun-Nov max minus Jan-Mar min -- the amplitude) and LEVEL (the annual
    mean) -- so these stay consistent with the rest of the chapter's
    "level vs. amplitude" vocabulary rather than inventing a third
    definition
  - whichever quantity changes by the larger percentage between low and
    high is the "dominant" one, and that choice also decides how the panel
    is DRAWN: if amplitude dominates, both curves are shifted down to their
    own minimum (baselines aligned to zero) so the range difference is
    what's visible; if level dominates, the curves are left at their raw
    extent so the vertical offset between them is what's visible. This
    matches the reference image's two panels exactly (SAM panel aligned to
    zero with "dominant: amplitude"; Nino3.4 panel left raw with
    "dominant: level").

Applied here to ASL (annual) vs. Ross Sea -- the relationship Table 3 /
Fig. 6 show reversing after 2001 -- split into the two eras rather than two
different indices, so the two panels show directly whether "dominant:
amplitude" is specific to the pre-2001 half of the record.

CAVEAT: tercile groups shrink once split by era (pre-2001 n~22, post
n~25, so each low/high group here is only ~7-8 years) -- this is a
descriptive/illustrative figure for Sect. 3.3's prose, not a hypothesis
test on its own; Fig. 6 and Table 3 carry the actual significance tests
this figure is illustrating.

Reads   DAILY_CSV, INDEX_CSV (via ch3_config)
Writes  results/ch3/figures/fig_asl_ross_mechanism.png              (white bg, DRAFT)
        results/ch3/figures/fig_asl_ross_mechanism_presentation.png (transparent bg, slides)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import DAILY_CSV, INDEX_CSV, SPLIT_YEAR, OUTPUT_DIR
import ch3_style

SECTOR = "SIE_Ross"
INDEX_COL = "ASL_annual"
ERAS = [(f"1979–{SPLIT_YEAR - 1}", 1979, SPLIT_YEAR - 1),
        (f"{SPLIT_YEAR}–2025", SPLIT_YEAR, 2025)]
INK = "0.35"


def low_high_terciles(idx_vals, years):
    """bottom/top tercile years by idx_vals -- same n/3 split as the reference figure"""
    order = np.argsort(idx_vals)
    n = len(years)
    k = max(2, n // 3)
    lo = np.asarray(years)[order[:k]]
    hi = np.asarray(years)[order[-k:]]
    return lo, hi


daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
if "period" in daily.columns:
    daily = daily[daily["period"] == "FULL"]
_dl = {str(x).lower(): x for x in daily["sector"].unique()}
_key = next((c for c in (SECTOR, "Ross", "SIE_Ross") if c.lower() in _dl), None)
if _key is None:
    sys.exit(f"Could not find Ross in DAILY_CSV's sector column (saw {sorted(daily['sector'].unique())})")
d = daily[daily["sector"] == _dl[_key.lower()]].copy()

# same extent-column auto-detect as check_level_shift.py / check_level_variance.py,
# so this doesn't need to guess a column name that might not match your file
EXTENT_COL = next((c for c in ("Extent", "extent", "SIE", "sie", "extent_obs", "obs") if c in d.columns), None)
if EXTENT_COL is None:
    sys.exit(f"No extent column found in DAILY_CSV (saw {list(d.columns)}) -- "
             f"edit the candidate list above to match your file.")

d["doy"] = d["Date"].dt.dayofyear
d["Y"], d["m"] = d["Date"].dt.year, d["Date"].dt.month

idx = pd.read_csv(INDEX_CSV)
if INDEX_COL not in idx.columns:
    sys.exit(f"{INDEX_COL} not in INDEX_CSV (saw {list(idx.columns)})")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
bold = ch3_style.bold_font_properties(size=11)
summary = []

for ax, (label, y0, y1) in zip(axes, ERAS):
    sub_idx = idx[idx.Year.between(y0, y1) & idx[INDEX_COL].notna()]
    years, vals = sub_idx["Year"].values, sub_idx[INDEX_COL].values
    lo_years, hi_years = low_high_terciles(vals, years)

    doy_full = np.arange(1, 367)
    curves, maxv, minv, meanv = {}, {}, {}, {}
    for grp_name, grp_years in (("low", lo_years), ("high", hi_years)):
        sub = d[d["Y"].isin(grp_years)]
        curves[grp_name] = sub.groupby("doy")[EXTENT_COL].mean().reindex(doy_full)
        # same windows as check_level_shift.py: max over Jun-Nov, min over Jan-Mar
        maxv[grp_name] = float(sub[sub.m.between(6, 11)].groupby("doy")[EXTENT_COL].mean().max())
        minv[grp_name] = float(sub[sub.m.between(1, 3)].groupby("doy")[EXTENT_COL].mean().min())
        meanv[grp_name] = float(curves[grp_name].mean())

    rng = {g: maxv[g] - minv[g] for g in ("low", "high")}
    rng_pct = (rng["high"] - rng["low"]) / rng["low"] * 100
    lvl_pct = (meanv["high"] - meanv["low"]) / meanv["low"] * 100
    dominant = "amplitude" if abs(rng_pct) > abs(lvl_pct) else "level"
    summary.append(dict(era=label, n_low=len(lo_years), n_high=len(hi_years),
                        range_low=rng["low"], range_high=rng["high"], range_pct=rng_pct,
                        level_low=meanv["low"], level_high=meanv["high"], level_pct=lvl_pct,
                        dominant=dominant))

    if dominant == "amplitude":
        plot_curves = {g: curves[g] - np.nanmin(curves[g]) for g in curves}
        ylabel = "Seasonal range, baselines aligned to zero"
    else:
        plot_curves = curves
        ylabel = "Sea ice extent"

    ax.fill_between(doy_full, plot_curves["low"], plot_curves["high"], color="0.88", zorder=0)
    ax.plot(doy_full, plot_curves["low"], color=ch3_style.BLUE, lw=1.8, label=f"low (n={len(lo_years)})")
    ax.plot(doy_full, plot_curves["high"], color=ch3_style.ORANGE, lw=1.8, label=f"high (n={len(hi_years)})")
    ax.set_title(label, fontproperties=bold, color="0.1", pad=10)
    ax.set_xlabel("Day of year", color=INK)
    ax.set_ylabel(ylabel, color=INK, fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.tick_params(colors=INK, labelsize=9)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.spines[["top", "right"]].set_visible(False)
    box = (f"range: {rng['low']:.2f} → {rng['high']:.2f} ({rng_pct:+.0f}%)\n"
           f"level: {meanv['low']:.2f} → {meanv['high']:.2f} ({lvl_pct:+.0f}%)\n"
           f"dominant: {dominant}")
    ax.text(0.97, 0.03, box, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color="0.15", bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))

fig.suptitle("Ross Sea ice, split by ASL (annual): before and after 2001", y=1.03, fontsize=12, color="0.1")
fig.tight_layout()
stem = os.path.join(OUTPUT_DIR, "fig_asl_ross_mechanism")
fig.savefig(f"{stem}_presentation.png", dpi=220, bbox_inches="tight", transparent=True)
fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white", transparent=False)
print(f"wrote {stem}.png (+ _presentation.png)\n")
for s in summary:
    print(f"{s['era']}  (low n={s['n_low']}, high n={s['n_high']})")
    print(f"   range: {s['range_low']:.2f} -> {s['range_high']:.2f}  ({s['range_pct']:+.0f}%)")
    print(f"   level: {s['level_low']:.2f} -> {s['level_high']:.2f}  ({s['level_pct']:+.0f}%)")
    print(f"   dominant: {s['dominant']}\n")
plt.close(fig)
