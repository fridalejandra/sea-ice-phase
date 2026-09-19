"""
fig03_anatomy_7a.py — H&R Fig 7a, reproduced with the VALIDATED recipe, then
extended to all sectors.

The component math is copied VERBATIM from fig7a_final_HR.py, the script that
reproduced H&R Fig 7a against the 400-DPI digitization (do not "simplify" it):

  * Cycle window: 21 Feb (year) to 20 Feb (year+1); x = day of the cycle.
  * SCALING FIX 1: fitted_amp / fitted_apac in the daily file are back-
    transformed with each ROW's calendar-year min/amplitude, which jumps at
    Jan 1 mid-cycle (cliff at cycle day ~315). Invert row-by-row to u-space,
    re-express the whole cycle with the CYCLE year's amplitude and minimum.
  * SCALING FIX 2 (H&R mixed axis): invariant cycle in raw Mkm^2, centred;
    trend/amplitude/phase components as PERCENT of the cycle-year amplitude;
    raw anomaly = (Extent - fapac_c)/amp_c*100; estimated anomaly = 11-day
    centred rolling mean of the raw anomaly.

Outputs
  fig03_check_circ2016.png       exact single-panel reproduction (compare with
                                 FIG7A_FINAL.png and the digitized ranges it prints)
  fig03_anatomy_<year>.png       six-sector grid, one per YEAR, same math per
                                 panel; the invariant is normalised (/amp*100)
                                 and drawn light+clipped so small sectors stay
                                 legible (H&R's raw-invariant convention only
                                 balances for the circumpolar series).

Run from scripts/python/plotting/Ch3/figures/ :  python fig03_anatomy_7a.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); import sys; sys.path.insert(0, HERE)
from ch3_config import OUTPUT_DIR
try:
    from ch3_config import DAILY_CSV
except ImportError:
    from ch3_config import DATA_DIR; DAILY_CSV = os.path.join(DATA_DIR, "daily_fitted_E.csv")

YEARS = [2016, 2023]
SECTORS = ["SIE_circumpolar", "SIE_Amundsen_Bellingshausen", "SIE_East_Antarctica",
           "SIE_Weddell", "SIE_Ross", "SIE_King_Haakon"]
LABELS = {"SIE_Amundsen_Bellingshausen": "Amundsen–Bellingshausen", "SIE_East_Antarctica": "East Antarctica",
          "SIE_Weddell": "Weddell", "SIE_Ross": "Ross", "SIE_King_Haakon": "King Haakon",
          "SIE_circumpolar": "Circumpolar"}
COLORS = {"invariant": "#A8D8EA", "trend": "#1B7A1B", "amplitude": "#14148C",
          "phase": "#E8140C", "raw_anom": "#000000", "est_anom": "#FFA500"}

df = pd.read_csv(DAILY_CSV)
df["Date"] = pd.to_datetime(df["Date"])
# daily_HR_v2 had `invariant_component`; the E file's equivalent is iac_notrend
# (only ever used centred, so any constant offset is irrelevant).
INV = "invariant_component" if "invariant_component" in df.columns else "iac_notrend"


def cycle_frame(sector, year):
    """VERBATIM math from fig7a_final_HR.py, on the E-file columns."""
    cycle_start = pd.Timestamp(f"{year}-02-21")
    cycle_end = pd.Timestamp(f"{year + 1}-02-20")
    m = (df["Date"] >= cycle_start) & (df["Date"] <= cycle_end) & (df["sector"] == sector)
    g = df[m].copy()
    if len(g) == 0:
        raise SystemExit(f"no rows for {sector} cycle {year}")
    g["cycle_day"] = (g["Date"] - cycle_start).dt.days
    g = g.sort_values("cycle_day").reset_index(drop=True)

    # -- SCALING FIX 1: cycle-year stats, undo the calendar-year jump ---------
    cyc = g[g["Year"] == year]
    amp_c = cyc["amplitude"].iloc[0]
    min_c = cyc["min_extent"].iloc[0]
    g["u_amp"] = (g["fitted_amp"] - g["min_extent"]) / g["amplitude"]     # exact row-wise inversion
    g["u_apac"] = (g["fitted_apac"] - g["min_extent"]) / g["amplitude"]
    g["famp_c"] = g["u_amp"] * amp_c + min_c                              # re-expressed, cycle-year stats
    g["fapac_c"] = g["u_apac"] * amp_c + min_c

    # -- SCALING FIX 2: H&R mixed axis ----------------------------------------
    g["inv_raw"] = g[INV] - g[INV].mean()                                  # raw Mkm^2, centred
    g["inv_norm"] = g["inv_raw"] / amp_c * 100.0                           # normalised variant (sector grid)
    g["trend_plot"] = g["trend_component"] / amp_c * 100.0
    g["amp_plot"] = (g["famp_c"] - g["iac_notrend"] - g["trend_component"]) / amp_c * 100.0
    g["phase_plot"] = (g["u_apac"] - g["u_amp"]) * 100.0
    g["raw_plot"] = (g["Extent"] - g["fapac_c"]) / amp_c * 100.0
    g["est_plot"] = g["raw_plot"].rolling(11, center=True, min_periods=1).mean()
    return g, amp_c, min_c


def draw(ax, g, invariant="raw"):
    ax.axhline(0, color="black", lw=0.8, linestyle="--", zorder=2)
    if invariant == "raw":
        ax.plot(g.cycle_day, g.inv_raw, color=COLORS["invariant"], lw=3.5, zorder=3, solid_capstyle="round")
    elif invariant == "norm":
        ax.plot(g.cycle_day, g.inv_norm, color=COLORS["invariant"], lw=2.4, alpha=0.6,
                zorder=3, solid_capstyle="round", clip_on=True)
    ax.plot(g.cycle_day, g.trend_plot, color=COLORS["trend"], lw=2.5, zorder=5)
    ax.plot(g.cycle_day, g.amp_plot, color=COLORS["amplitude"], lw=2.5, zorder=5)
    ax.plot(g.cycle_day, g.phase_plot, color=COLORS["phase"], lw=2.8, zorder=6)
    ax.plot(g.cycle_day, g.raw_plot, color=COLORS["raw_anom"], lw=1.0, zorder=4)
    ax.plot(g.cycle_day, g.est_plot, color=COLORS["est_anom"], lw=1.6, zorder=4)
    ax.set_xlim(0, 365); ax.set_xticks([0, 100, 200, 300])


# ---- 1) VALIDATION: Circumpolar 2016, exact H&R recipe ----------------------
g, amp_c, min_c = cycle_frame("SIE_circumpolar", 2016)
print(f"Circumpolar 2016: amplitude={amp_c:.3f}, min={min_c:.3f} Mkm^2, {len(g)} days")
print("\nPlotted ranges (components in % of amplitude):")
for col in ("inv_raw", "trend_plot", "amp_plot", "phase_plot", "raw_plot"):
    v = g[col].dropna(); print(f"  {col:12} {v.min():7.2f} to {v.max():7.2f}")
print("\nH&R digitized:  trend -1.8..+0.7 | amp -3.5..-1.0 | "
      "phase -7.8..+3.5 | raw ~-5..+2.5 | invariant -8.5..+7.0")

fig, ax = plt.subplots(figsize=(9.5, 7))
draw(ax, g, invariant="raw")
ax.set_ylim(-10.5, 7.5)
ax.set_xlabel("Day of the 2016 cycle", fontsize=13)
ax.set_ylabel("Anomaly for sea ice extent", fontsize=13)
ax.set_title("(a)  circumpolar — 2016", fontsize=13, fontweight="bold", loc="left")
ax.legend(handles=[Line2D([], [], color=COLORS["invariant"], lw=3.5, label="Invariant annual cycle"),
                   Line2D([], [], color=COLORS["trend"], lw=2.5, label="Trend component"),
                   Line2D([], [], color=COLORS["amplitude"], lw=2.5, label="Amplitude component"),
                   Line2D([], [], color=COLORS["phase"], lw=2.8, label="Phase component"),
                   Line2D([], [], color=COLORS["raw_anom"], lw=1.0, label="Raw anomaly"),
                   Line2D([], [], color=COLORS["est_anom"], lw=1.6, label="Estimated anomaly")],
          loc="lower center", fontsize=9.5, frameon=False)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "fig03_check_circ2016.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
print(f"\nSaved validation panel: {out}  (compare with FIG7A_FINAL.png)")

# ---- 2) EXTENSION: six sectors per year, same math per panel ----------------
leg = [Line2D([], [], color=COLORS["invariant"], lw=2.4, alpha=0.6, label="invariant cycle (/amp)"),
       Line2D([], [], color=COLORS["trend"], lw=2.5, label="trend"),
       Line2D([], [], color=COLORS["amplitude"], lw=2.5, label="amplitude"),
       Line2D([], [], color=COLORS["phase"], lw=2.8, label="phase"),
       Line2D([], [], color=COLORS["raw_anom"], lw=1.0, label="raw anomaly"),
       Line2D([], [], color=COLORS["est_anom"], lw=1.6, label="estimated anomaly")]
for yr in YEARS:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    for k, sec in enumerate(SECTORS):
        ax = axes.ravel()[k]
        gy, a_c, _ = cycle_frame(sec, yr)
        draw(ax, gy, invariant="norm")
        # panel scaled to the COMPONENTS; the (normalised) invariant clips
        comp = np.concatenate([gy.trend_plot, gy.amp_plot, gy.phase_plot, gy.raw_plot])
        lim = 1.15 * np.nanmax(np.abs(comp)); ax.set_ylim(-lim, lim)
        ax.set_title(f"{LABELS.get(sec, sec)}   (amp {a_c:.1f} Mkm²)", fontsize=10.5, pad=3)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if k % 3 == 0: ax.set_ylabel("anomaly (% of amplitude)", fontsize=9)
        if k >= 3: ax.set_xlabel(f"day of the {yr} cycle", fontsize=9)
        print(f"{yr} {LABELS.get(sec, sec):26s} phase {gy.phase_plot.min():+6.1f}..{gy.phase_plot.max():+6.1f}%  "
              f"amp {gy.amp_plot.min():+6.1f}..{gy.amp_plot.max():+6.1f}%")
    fig.legend(handles=leg, ncol=6, loc="lower center", frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Anatomy of the {yr} annual cycle by sector (components as % of each sector's amplitude)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, f"fig03_anatomy_{yr}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("wrote", out)
