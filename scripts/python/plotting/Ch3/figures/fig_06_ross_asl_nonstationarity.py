#!/usr/bin/env python3
"""
fig_06_ross_asl_nonstationarity.py -- Fig. 6: the Amundsen Sea Low - Ross Sea
amplitude relationship, 1979-2000 vs 2001-2025 (Sect. 3.3).

Panel (a): the correlation between the ASL relative central pressure and the
Ross Sea amplitude, by season, in each period.
Panel (b): the same annual correlation before and after a candidate break
year, for five candidate years. If the break were an artefact of choosing
2001, moving the boundary would remove it; it does not for 2001-2011.

Reads   results/ch3/tables/t36_ross_asl_detail.csv   (from ch3_stats.py, Sect. 3.6 block)
Writes  results/ch3/figures/fig06_ross_asl_nonstationarity.png

Run from scripts/python/plotting/Ch3/figures/ after ch3_stats.py.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR, OUTPUT_DIR
import ch3_style  # font + spines; bold_font_properties() for real bold

PRE, POST = "1979–2000", "2001–2025"
COLOR_PRE, COLOR_POST = "#2a78d6", "#eb6834"     # same pair as Fig. S3
INK = "0.35"
SEASONS = ["annual", "DJF", "MAM", "JJA", "SON", "ADV", "RET"]
SEASON_LABEL = {"annual": "annual", "DJF": "DJF", "MAM": "MAM", "JJA": "JJA",
                "SON": "SON", "ADV": "Mar–Aug", "RET": "Oct–Jan"}


def fmt_p(p):
    """p as a reader expects it: p < 0.001, p = 0.002, p = 0.13."""
    if not np.isfinite(p):
        return "p n/a"
    if p < 0.001:
        return "p < 0.001"
    if p < 0.1:
        return f"p = {p:.3f}"          # 0.048 is not "0.05"
    return f"p = {p:.2f}"


path = os.path.join(TABLES_DIR, "t36_ross_asl_detail.csv")
if not os.path.exists(path):
    sys.exit(f"{path} not found -- run ch3_stats.py first")
detail = pd.read_csv(path)

season_rows = detail[detail["test"] == "season"].set_index("key").reindex(SEASONS)
if season_rows["r_pre"].isna().any():
    sys.exit(f"t36_ross_asl_detail.csv lacks a season row for {season_rows.index[season_rows['r_pre'].isna()].tolist()}")
sweep = detail[detail["test"] == "split_sweep"].copy()
sweep["key"] = sweep["key"].astype(float).astype(int)
sweep = sweep.sort_values("key")
loo = detail.loc[detail["test"] == "loo_shift", "p_shift"]
loo_p = float(loo.iloc[0]) if len(loo) else np.nan

title_font = ch3_style.bold_font_properties(size=11)
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.2))

# ── (a) by season ────────────────────────────────────────────────────────────
x = np.arange(len(SEASONS)); w = 0.38
ax_a.bar(x - w / 2, season_rows["r_pre"], w, color=COLOR_PRE, label=PRE)
ax_a.bar(x + w / 2, season_rows["r_post"], w, color=COLOR_POST, label=POST)
ax_a.axhline(0, color=INK, lw=0.8)
ax_a.set_xticks(x)
ax_a.set_xticklabels([SEASON_LABEL[s] for s in SEASONS])
ax_a.set_ylabel("correlation, ASL relative central pressure\nvs Ross Sea amplitude", color=INK)
ax_a.set_title("(a)  By season, in each period", loc="left", fontproperties=title_font, pad=10)
ax_a.legend(frameon=False, fontsize=9, loc="lower left")
# the headline number, in words, where the reader looks first
p_ann = float(season_rows.loc["annual", "p_shift"])
ax_a.text(0.98, 0.97,
          f"annual: {season_rows.loc['annual', 'r_pre']:+.2f} to {season_rows.loc['annual', 'r_post']:+.2f}\n"
          f"change {fmt_p(p_ann)}",
          transform=ax_a.transAxes, ha="right", va="top", fontsize=9, color=INK)

# ── (b) moving the break year ────────────────────────────────────────────────
yrs = sweep["key"].to_numpy()
ax_b.plot(yrs, sweep["r_pre"], "o-", color=COLOR_PRE, lw=2, ms=6, label="years before the break")
ax_b.plot(yrs, sweep["r_post"], "o-", color=COLOR_POST, lw=2, ms=6, label="years after the break")
# p of the before/after difference, in a row along the bottom so nothing sits on a line
for xv, pv in zip(yrs, sweep["p_shift"]):
    sig = np.isfinite(pv) and pv < 0.05
    ax_b.text(xv, 0.03, fmt_p(pv).replace("p = ", "").replace("p < ", "<"),
              transform=ax_b.get_xaxis_transform(), ha="center", va="bottom",
              fontsize=8.5, color="0.15" if sig else "0.55",
              fontproperties=ch3_style.bold_font_properties(size=8.5) if sig else None)
ax_b.text(yrs.min() - 5.7, 0.03, "p for the\nchange:", transform=ax_b.get_xaxis_transform(),
          ha="left", va="bottom", fontsize=8, color=INK)
ax_b.axhline(0, color=INK, lw=0.8)
ax_b.set_xticks(yrs)                       # only the five candidate years, no decimals
ax_b.set_xticklabels([str(int(v)) for v in yrs])
ax_b.set_xlim(yrs.min() - 6, yrs.max() + 2.5)
ax_b.set_xlabel("break year (first year of the later period)", color=INK)
ax_b.set_ylabel("correlation, annual", color=INK)
ax_b.set_title("(b)  Moving the break year", loc="left", fontproperties=title_font, pad=10)
ax_b.legend(frameon=False, fontsize=9, loc="upper right")

lo = min(season_rows[["r_pre", "r_post"]].min().min(), sweep[["r_pre", "r_post"]].min().min())
hi = max(season_rows[["r_pre", "r_post"]].max().max(), sweep[["r_pre", "r_post"]].max().max())
for ax in (ax_a, ax_b):
    ax.set_ylim(lo - 0.22, hi + 0.15)
    ax.tick_params(colors=INK, labelsize=9, length=3)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(w_pad=2.5)
out = os.path.join(OUTPUT_DIR, "fig06_ross_asl_nonstationarity.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
print(f"  annual: r_pre={season_rows.loc['annual', 'r_pre']:+.3f}  r_post={season_rows.loc['annual', 'r_post']:+.3f}  "
      f"shift {fmt_p(p_ann)}")
print("  break-year sweep: " + ", ".join(f"{k}: {fmt_p(p)}" for k, p in zip(yrs, sweep["p_shift"])))
print(f"  leave-one-year-out worst shift {fmt_p(loo_p)}")