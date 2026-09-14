"""
fig04_dot_timeline.py  — REBUILT (original script lost)
========================================================
Per sector-year, which component of the decomposition dominates that year's
variability, and how dominant is it.

Reads daily_fitted_B.csv (Pipeline B). This matters: in Pipeline A the
components did NOT sum to the anomaly (trend double-counted, phase_component
built from the wrong curve), so dominance shares were not interpretable.
In Pipeline B:

    anomaly_from_iac = trend_component + amplitude_component
                     + phase_component + residual_apac      (exact, ~1e-19)

Dominance is the share of that year's total component variance held by the
largest component. Dot colour = dominant component; dot size = share.

NOTE on the trend category: trend and amplitude compete for the same variance
(a flexible trend absorbs interannual level). Pipeline B uses
s(tdate, bs="tp", k=8), edf ~6.8-7.0, with within-year SD now far below
between-year SD. Under Pipeline A's k=150 the trend was absorbing far more,
so orange dots here should be fewer than in the original version of this
figure. That is the fix working, not a discrepancy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ch3_config import DATA_DIR, OUTPUT_DIR, GDRIVE, DAILY_CSV

SECTOR_ORDER = [
    ("SIE_Weddell",                 "Weddell",         "#2196F3"),
    ("SIE_Amundsen_Bellingshausen", "ABS",             "#F44336"),
    ("SIE_Ross",                    "Ross",            "#4CAF50"),
    ("SIE_East_Antarctica",         "East Antarctica", "#FF9800"),
    ("SIE_King_Haakon",             "King Haakon",     "#9C27B0"),
    ("SIE_circumpolar",             "Circumpolar",     "#2C2C2A"),
]

COMPONENT_COLORS = {
    "Phase":     "#D4537E",
    "Amplitude": "#378ADD",
    "Trend":     "#C77B26",
    "Residual":  "#BDBDBD",
}
COMPONENT_ORDER = ["Phase", "Amplitude", "Trend", "Residual"]

BREAK_YEAR = 2016

# Dot size scaling: share 0.35 -> small, 0.50+ -> large
SIZE_MIN, SIZE_MAX = 90, 340
SHARE_LO, SHARE_HI = 0.30, 0.60


def share_to_size(share):
    x = np.clip((share - SHARE_LO) / (SHARE_HI - SHARE_LO), 0, 1)
    return SIZE_MIN + x * (SIZE_MAX - SIZE_MIN)


# ── Load and compute dominance ────────────────────────────────────────────────
print(f"Loading {DAILY_CSV} ...")
daily = pd.read_csv(DAILY_CSV)

required = ["trend_component", "amplitude_component", "phase_component",
            "residual_apac", "anomaly_from_iac", "sector", "Year"]
missing = [c for c in required if c not in daily.columns]
if missing:
    raise SystemExit(
        f"Missing columns: {missing}\n"
        "This script needs Pipeline B output (daily_fitted_B.csv)."
    )

# Verify the decomposition actually sums — refuse to plot if it doesn't
chk = (daily["anomaly_from_iac"]
       - (daily["trend_component"] + daily["amplitude_component"]
          + daily["phase_component"] + daily["residual_apac"]))
sum_err = np.nanmean(np.abs(chk))
print(f"Decomposition sum check: mean |error| = {sum_err:.3e}")
if sum_err > 1e-6:
    raise SystemExit(
        "Components do not sum to the anomaly. This is Pipeline A output "
        "or a partial fix — regenerate with APAC_Sector_Pipeline_B.R first."
    )

rows = []
for (sec, yr), g in daily.groupby(["sector", "Year"]):
    var = {
        "Trend":     np.nanvar(g["trend_component"]),
        "Amplitude": np.nanvar(g["amplitude_component"]),
        "Phase":     np.nanvar(g["phase_component"]),
        "Residual":  np.nanvar(g["residual_apac"]),
    }
    total = sum(var.values())
    if not np.isfinite(total) or total <= 0:
        continue
    frac = {k: v / total for k, v in var.items()}
    dom = max(frac, key=frac.get)
    rows.append(dict(sector=sec, Year=int(yr), dominant=dom, share=frac[dom],
                     **{f"f_{k}": frac[k] for k in COMPONENT_ORDER}))

dom_df = pd.DataFrame(rows)

out_csv = os.path.join(DATA_DIR, "component_dominance.csv")
dom_df.to_csv(out_csv, index=False)
print(f"Wrote {out_csv}  ({len(dom_df)} sector-years)")
print("\nDominant component counts:")
print(dom_df["dominant"].value_counts().to_string())

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 8))

y_positions = {}
for i, (sec_key, label, label_color) in enumerate(SECTOR_ORDER):
    y = len(SECTOR_ORDER) - i
    y_positions[sec_key] = y

    sub = dom_df[dom_df["sector"] == sec_key].sort_values("Year")
    if len(sub) == 0:
        print(f"  WARNING: no rows for {sec_key}")
        continue

    ax.scatter(
        sub["Year"], np.full(len(sub), y),
        c=[COMPONENT_COLORS[d] for d in sub["dominant"]],
        s=share_to_size(sub["share"].values),
        edgecolors="white", linewidth=0.6, zorder=3,
    )

    ax.text(sub["Year"].min() - 2.5, y, label,
            ha="right", va="center", fontsize=12, fontweight="bold",
            color=label_color)

# Decade gridlines + 2016 break
for decade in [1980, 1990, 2000, 2010, 2020]:
    ax.axvline(decade, color="#CCCCCC", lw=1.0, zorder=1)
ax.axvline(BREAK_YEAR, color="#D4537E", lw=1.6, ls="--", zorder=2, alpha=0.9)

ax.set_yticks([])
ax.set_ylim(0.3, len(SECTOR_ORDER) + 0.7)
ax.set_xlabel("Year", fontsize=12)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="x", labelsize=11)

# Legend: components + size key
comp_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=COMPONENT_COLORS[c],
           markersize=11, label=c)
    for c in COMPONENT_ORDER
]
size_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777",
           markersize=np.sqrt(share_to_size(0.35)) * 0.8, label="35% dominant"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777",
           markersize=np.sqrt(share_to_size(0.50)) * 0.8, label="50% dominant"),
]
ax.legend(handles=comp_handles + size_handles, loc="upper center",
          bbox_to_anchor=(0.5, -0.10), ncol=6, frameon=False, fontsize=11)

plt.tight_layout()

outfile = os.path.join(OUTPUT_DIR, "fig04_dot_timeline.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)
fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved → {outfile}")

if os.system(f'rclone copy "{outfile}" "{GDRIVE}"') == 0:
    print(f"Synced → {GDRIVE}")
