"""
fig03_attribution_annual.py — §3.2 attribution, per sector, every cycle.

Two figures and two tables.

  fig03_attribution_by_cycle.png
      Per sector, one stacked bar per CYCLE (21 Feb to 20 Feb): the signed
      cycle-mean of each component, positives stacked above zero, negatives
      below, summing to the black line (observed − invariant cycle). Each
      panel has its own y-axis.

  fig03_attribution_era_shares.png
      Per sector, two 100%-stacked bars (pre-2016 / 2016+) of the component
      shares, drawn for two metrics:
        NET    mean over cycles of |cycle-mean contribution|
               — what the annual-mean extent anomaly sees. A timing shift
               moves extent from one part of the year to another and nearly
               cancels in the cycle mean, so phase is SMALL here BY
               CONSTRUCTION; trend (one-signed offset) and amplitude
               (one-signed change in the maximum) do not cancel.
        GROSS  mean over cycles of the within-cycle mean |daily contribution|
               — what the seasonal anatomy (Fig. 7) sees. No cancellation.
      The contrast between the two is the result: which component appears
      to dominate depends on the timescale you average over.

  t32_attribution_by_cycle.csv        signed cycle means, every sector-cycle
  t32_attribution_by_cycle_gross.csv  within-cycle mean |daily|, same
  t32_component_dominance_era.csv     era x sector x metric, magnitudes,
                                      shares, dominant component, n_cycles

Cycle completeness is judged by DATE SPAN (>= 360 days), not by row count:
1979 to mid-1987 are every-other-day data and a row-count test throws those
cycles away. The 1987 cycle has the Dec 1987–Jan 1988 gap; it is kept, and
its mean is over the days that exist.

Components sum to the anomaly by construction. What is not fixed by
construction is the distribution, but the shares are ORDER-DEPENDENT
(amplitude is extracted before phase). Say so in the text.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import ch3_data as D
from ch3_config import SECTORS, SECTOR_LABELS, TABLES_DIR
from ch3_plot import save
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure

COMPS = [("trend_component",     "trend",     "#5b6b7a"),
         ("amplitude_component", "amplitude", "#2c7fb8"),
         ("phase_component",     "phase",     "#e8703a"),
         ("residual_apac",       "residual",  "#c4c4c4")]
BLACK = "anomaly_from_iac"
BREAK = 2016
CYCLE_START_MONTH, CYCLE_START_DAY = 2, 21
ERAS = [f"pre-{BREAK}", f"{BREAK}+"]

print("fig03 — attribution by cycle")
d = D.load_daily(period="FULL")
D.summary(daily=d)

# ── assign each day to a cycle ──────────────────────────────────────────────
cut = pd.to_datetime(dict(year=d["Date"].dt.year,
                          month=CYCLE_START_MONTH, day=CYCLE_START_DAY))
d = d.assign(cycle=np.where(d["Date"] >= cut,
                            d["Date"].dt.year, d["Date"].dt.year - 1))

# ── keep complete cycles: by date span, not row count ───────────────────────
grp  = d.groupby(["sector", "cycle"])["Date"]
span = grp.transform(lambda s: (s.max() - s.min()).days)
nrow = grp.transform("size")
keep = (span >= 360) & (nrow >= 150)
dropped = sorted(d.loc[~keep, "cycle"].unique())
d = d[keep]
if dropped:
    print(f"  dropped incomplete cycles: {dropped}")
cyc = sorted(d.cycle.unique())
print(f"  cycles retained: {cyc[0]}–{cyc[-1]}  (n={len(cyc)})")
days_per = d.groupby("cycle")["Date"].size() / d.sector.nunique()
sparse = days_per[days_per < 300].index.tolist()
if sparse:
    print(f"  cycles with < 300 days of data (every-other-day era / gaps): "
          f"{sparse[0]}–{sparse[-1]}")

# ── cycle means: NET (signed) and GROSS (mean |daily|) ──────────────────────
comp_cols = [c for c, _, _ in COMPS]
net = (d.groupby(["sector", "cycle"])[comp_cols + [BLACK]].mean().reset_index())
gross = (d.assign(**{c: d[c].abs() for c in comp_cols})
           .groupby(["sector", "cycle"])[comp_cols].mean().reset_index())
err = (net[comp_cols].sum(axis=1) - net[BLACK]).abs().max()
print(f"  components sum to anomaly, max abs error {err:.2e}")

os.makedirs(TABLES_DIR, exist_ok=True)
net.rename(columns={"cycle": "Year"}).to_csv(
    os.path.join(TABLES_DIR, "t32_attribution_by_cycle.csv"), index=False)
gross.rename(columns={"cycle": "Year"}).to_csv(
    os.path.join(TABLES_DIR, "t32_attribution_by_cycle_gross.csv"), index=False)

# ── figure 1: stacked bars per cycle, per-panel y ───────────────────────────
ncol = 3
nrow_ = int(np.ceil(len(SECTORS) / ncol))
fig, axes = plt.subplots(nrow_, ncol, figsize=(4.8 * ncol, 3.0 * nrow_), sharex=True)
axes = np.atleast_1d(axes).ravel()
for k, sec in enumerate(SECTORS):
    ax = axes[k]
    a = net[net.sector == sec].sort_values("cycle")
    yr = a.cycle.values
    pos = np.zeros(len(yr)); neg = np.zeros(len(yr))
    for c, lab, col in COMPS:
        v  = a[c].values
        vp = np.where(v > 0, v, 0.0); vn = np.where(v < 0, v, 0.0)
        ax.bar(yr, vp, bottom=pos, color=col, width=0.85, lw=0)
        ax.bar(yr, vn, bottom=neg, color=col, width=0.85, lw=0)
        pos += vp; neg += vn
    ax.plot(yr, a[BLACK].values, color="k", lw=1.3, marker="o", ms=2.2, zorder=5)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(BREAK - 0.5, color="0.35", lw=0.9, ls=(0, (2, 2)), zorder=1)
    ax.set_title(SECTOR_LABELS[sec], pad=3,
                 fontproperties=ch3_style.bold_font_properties(size=10.5))
    ax.set_xlim(yr.min() - 0.7, yr.max() + 0.7)
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.01, 0.97, f"({chr(97 + k)})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")
    if k % ncol == 0:
        ax.set_ylabel("cycle-mean contribution\n(10$^6$ km$^2$)", fontsize=8)
for k in range(len(SECTORS), len(axes)):
    axes[k].set_visible(False)
leg = [Line2D([], [], color="k", lw=1.3, marker="o", ms=2.5,
              label="observed − invariant cycle")] + \
      [Patch(facecolor=col, label=lab) for _, lab, col in COMPS]
fig.legend(handles=leg, ncol=5, loc="lower center", frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.05, 1, 0.99])
save(fig, "fig03_attribution_by_cycle.png", sync=False)

# ── era table: NET and GROSS, magnitudes and shares ─────────────────────────
def era_of(y):
    return ERAS[0] if y < BREAK else ERAS[1]

rows = []
for metric, tab in [("net", net), ("gross", gross)]:
    t = tab.assign(era=tab.cycle.map(era_of))
    for era in ERAS:
        for sec in SECTORS:
            a = t[(t.sector == sec) & (t.era == era)]
            mags = {lab: float(a[c].abs().mean()) for c, lab, _ in COMPS}
            tot  = sum(mags.values())
            rows.append(dict(metric=metric, era=era, sector=SECTOR_LABELS[sec],
                             n_cycles=len(a), **mags,
                             **{f"share_{lab}": mags[lab] / tot for lab in mags},
                             dominant=max(mags, key=mags.get)))
dom = pd.DataFrame(rows)
dom.to_csv(os.path.join(TABLES_DIR, "t32_component_dominance_era.csv"), index=False)

labs_c = [lab for _, lab, _ in COMPS]
for metric, title in [
        ("net",   "NET   |cycle-mean contribution|  (10^6 km^2) — what the annual-mean anomaly sees"),
        ("gross", "GROSS mean |daily contribution| within the cycle (10^6 km^2) — what the seasonal anatomy sees")]:
    x = dom[dom.metric == metric]
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)
    print(x[["era", "sector"] + labs_c + ["dominant", "n_cycles"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("  shares:")
    print(x[["era", "sector"] + [f"share_{l}" for l in labs_c]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# ── figure 2: era shares, net vs gross ──────────────────────────────────────
labs_s = [SECTOR_LABELS[s] for s in SECTORS]
xpos = np.arange(len(SECTORS)); w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
for ax, (metric, title) in zip(axes, [
        ("net",   "Net: shares of the cycle-mean anomaly"),
        ("gross", "Gross: shares of within-cycle |contribution|")]):
    for j, era in enumerate(ERAS):
        t = dom[(dom.metric == metric) & (dom.era == era)].set_index("sector").loc[labs_s]
        bottom = np.zeros(len(labs_s))
        for _, lab, col in COMPS:
            v = t[f"share_{lab}"].values
            ax.bar(xpos + (j - 0.5) * w, v, w, bottom=bottom, color=col, lw=0,
                   hatch=None if j else "///", edgecolor="white")
            if lab == "phase":
                for i in range(len(labs_s)):
                    ax.text(xpos[i] + (j - 0.5) * w, bottom[i] + v[i] / 2,
                            f"{v[i] * 100:.0f}", ha="center", va="center",
                            fontsize=6.5, color="white", fontweight="bold")
            bottom += v
    ax.set_xticks(xpos)
    ax.set_xticklabels(labs_s, rotation=20, ha="right",
                        fontproperties=ch3_style.bold_font_properties(size=8.5))
    ax.set_ylim(0, 1); ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("share of total |contribution|", fontsize=9)
leg = [Patch(facecolor=col, label=lab) for _, lab, col in COMPS] + \
      [Patch(facecolor="0.75", hatch="///", edgecolor="white", label=ERAS[0]),
       Patch(facecolor="0.75", label=ERAS[1])]
fig.legend(handles=leg, ncol=6, loc="lower center", frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Numbers on bars: phase share (%)", fontsize=8, color="0.35", y=0.995)
fig.tight_layout(rect=[0, 0.07, 1, 0.97])
save(fig, "fig04_attribution_era_shares.png", sync=False)

print("\nwrote t32_attribution_by_cycle.csv, t32_attribution_by_cycle_gross.csv, "
      "t32_component_dominance_era.csv")