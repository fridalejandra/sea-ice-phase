#!/usr/bin/env python3
"""
fig_09_ross_asl_nonstationarity.py — Figure 9: the Ross-ASL amplitude
relationship's non-stationarity (§3.3/§4.3). Panel (a): by season, pre- vs.
post-2001. Panel (b): the split-year sweep across five candidate
boundaries, showing the break isn't an artifact of picking 2001
specifically.

Split out of build_ch3_figures.py (2026-09-18) so each manuscript figure has
its own script, named to match: fig_##_name.py. Figure logic unchanged from
that file's Fig-9 block. (compute_asl_ross_sweep.py is a separate,
standalone reimplementation of the same underlying test built earlier for a
quick advisor-meeting figure — this is the canonical version, reading the
canonical table directly rather than recomputing.)

Run this from the same directory as ch3_config.py
(scripts/python/plotting/Ch3/figures/), after ch3_stats.py has produced
t36_ross_asl_detail.csv.

Reads:
    results/ch3/tables/t36_ross_asl_detail.csv
Writes:
    results/ch3/figures/fig9_ross_asl_nonstationarity.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR, OUTPUT_DIR
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure


def load_table(name):
    path = os.path.join(TABLES_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run the canonical pipeline first.")
    return pd.read_csv(path)


detail = load_table("t36_ross_asl_detail.csv")
seasons_order = ["annual", "DJF", "MAM", "JJA", "SON", "ADV", "RET"]
season_rows = detail[detail["test"] == "season"].set_index("key").reindex(seasons_order)

sweep = detail[detail["test"] == "split_sweep"].copy()
sweep["key"] = sweep["key"].astype(int)
sweep = sweep.sort_values("key")

loo_row = detail[detail["test"] == "loo_shift"].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

ax = axes[0]
x = np.arange(len(seasons_order)); w = 0.36
ax.bar(x - w / 2, season_rows["r_pre"], w, label="1979-2000", color="#2c7fb8")
ax.bar(x + w / 2, season_rows["r_post"], w, label="2001-2023", color="#e8703a")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(seasons_order)
ax.set_ylabel("r (ASL relative central pressure, Ross amplitude)")
ax.set_title(
    f"(a) By season (annual shift p={season_rows.loc['annual', 'p_shift']:.1g})", fontsize=9.5)
ax.legend(fontsize=8.5, frameon=False)

ax = axes[1]
ax.plot(sweep["key"], sweep["r_pre"], "o-", color="#2c7fb8", label="r before boundary", lw=2)
ax.plot(sweep["key"], sweep["r_post"], "o-", color="#e8703a", label="r after boundary", lw=2)
for xv, yv, pv in zip(sweep["key"], sweep["r_post"], sweep["p_shift"]):
    ax.annotate(f"p={pv:.2g}", (xv, yv), textcoords="offset points", xytext=(0, -14),
                fontsize=7.5, ha="center", color="#555")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Candidate breakpoint year")
ax.set_ylabel("r (ASL annual, Ross amplitude)")
sig_years = sweep.loc[sweep["p_shift"] < 0.05, "key"].tolist()
ax.set_title(f"(b) Split-year sweep—shift p<0.05 for boundaries {sig_years}", fontsize=9.5)
ax.legend(fontsize=8.5, frameon=False)

fig.suptitle("Amundsen Sea Low – Ross Sea amplitude correlation, by season and split-year boundary",
              fontweight="bold", y=1.03)
fig.tight_layout()
out9 = os.path.join(OUTPUT_DIR, "fig9_ross_asl_nonstationarity.png")
fig.savefig(out9, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out9}")
print(f"  annual: r_pre={season_rows.loc['annual','r_pre']:.3f}, "
      f"r_post={season_rows.loc['annual','r_post']:.3f}, "
      f"shift p={season_rows.loc['annual','p_shift']:.2g}")
print(f"  split-year sweep, shift p by boundary: "
      + ", ".join(f"{int(k)}: {p:.2g}" for k, p in zip(sweep['key'], sweep['p_shift'])))
print(f"  LOO worst shift p: {loo_row['p_shift']:.2g}")