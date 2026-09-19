"""
fig7a_final_HR.py — H&R Figure 7a, reproduced with the correct (mixed) scaling

KEY INSIGHT (from digitizing H&R's published figure at 400 DPI):
  * The invariant annual cycle (light blue) is plotted in raw Mkm², centered.
  * ALL OTHER components are plotted in NORMALIZED space x 100
    (i.e., percent of the year's amplitude):
        plotted_value = (component_Mkm2 / amplitude_year) * 100
  * This is why their phase reaches -8 while their own text calls the
    contributions "small": -8 'units' = -8% of amplitude = about -1.2 Mkm2.

Usage:
    python fig7a_final_HR.py --csv /path/to/daily_HR_v2.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {
    "invariant": "#A8D8EA",
    "trend":     "#1B7A1B",
    "amplitude": "#14148C",
    "phase":     "#E8140C",
    "raw_anom":  "#000000",
    "est_anom":  "#FFA500",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to daily_HR_v2.csv")
    ap.add_argument("--sector", default="SIE_circumpolar")
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--out", default="fig7a_FINAL.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["Date"] = pd.to_datetime(df["Date"])

    cycle_start = pd.Timestamp(f"{args.year}-02-21")
    cycle_end   = pd.Timestamp(f"{args.year + 1}-02-20")

    m = (df["Date"] >= cycle_start) & (df["Date"] <= cycle_end)
    if "sector" in df.columns:
        m &= df["sector"] == args.sector
    g = df[m].copy()
    if len(g) == 0:
        print("ERROR: no rows for that sector/cycle-year")
        sys.exit(1)

    g["cycle_day"] = (g["Date"] - cycle_start).dt.days
    g = g.sort_values("cycle_day").reset_index(drop=True)
    print(f"{len(g)} days, cycle days {g['cycle_day'].min()}-{g['cycle_day'].max()}")

    # ------------------------------------------------------------------
    # SCALING FIX 1 — undo the calendar-year normalization jump.
    # fitted_amp / fitted_apac were back-transformed with each row's own
    # CALENDAR-year min/amplitude, which switches on Jan 1 mid-cycle and
    # produces a cliff at cycle day ~315. Invert the back-transform to
    # recover the u-space predictions, then re-express the WHOLE cycle
    # with the cycle year's (e.g. 2016's) stats.
    # ------------------------------------------------------------------
    cyc = g[g["Year"] == args.year]
    amp_c = cyc["amplitude"].iloc[0]
    min_c = cyc["min_extent"].iloc[0]
    print(f"Cycle-year stats used throughout: amplitude={amp_c:.3f}, "
          f"min={min_c:.3f} Mkm²")

    # Recover u-space predictions (exact inversion, row by row)
    g["u_amp"]  = (g["fitted_amp"]  - g["min_extent"]) / g["amplitude"]
    g["u_apac"] = (g["fitted_apac"] - g["min_extent"]) / g["amplitude"]
    g["u_obs"]  = (g["Extent"]      - min_c) / amp_c   # observed, 2016 scale

    # Re-back-transform with consistent cycle-year stats
    g["famp_c"]  = g["u_amp"]  * amp_c + min_c
    g["fapac_c"] = g["u_apac"] * amp_c + min_c

    # ------------------------------------------------------------------
    # SCALING FIX 2 — H&R's mixed axis:
    # invariant in raw Mkm2 (centered); components as % of amplitude.
    # ------------------------------------------------------------------
    g["inv_plot"]   = g["invariant_component"] - g["invariant_component"].mean()
    g["trend_plot"] = g["trend_component"] / amp_c * 100.0
    g["amp_plot"]   = (g["famp_c"] - g["iac_notrend"] - g["trend_component"]) \
                      / amp_c * 100.0
    g["phase_plot"] = (g["u_apac"] - g["u_amp"]) * 100.0
    g["raw_plot"]   = (g["Extent"] - g["fapac_c"]) / amp_c * 100.0

    # Estimated anomaly = smoothed raw anomaly (H&R use GARCH/ARMA; a
    # centered rolling mean is a faithful visual stand-in)
    g["est_plot"] = g["raw_plot"].rolling(11, center=True, min_periods=1).mean()

    print("\nPlotted ranges (components in % of amplitude):")
    for col in ("inv_plot", "trend_plot", "amp_plot", "phase_plot", "raw_plot"):
        v = g[col].dropna()
        print(f"  {col:12} {v.min():7.2f} to {v.max():7.2f}")
    print("\nH&R digitized:  trend -1.8..+0.7 | amp -3.5..-1.0 | "
          "phase -7.8..+3.5 | raw ~-5..+2.5 | invariant -8.5..+7.0")

    # ------------------------------------------------------------------
    # PLOT (H&R style)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 7))
    x = g["cycle_day"].values

    ax.axhline(0, color="black", lw=0.8, linestyle="--", zorder=2)

    ax.plot(x, g["inv_plot"],  color=COLORS["invariant"], lw=3.5, zorder=3,
            solid_capstyle="round", label="Invariant annual cycle")
    ax.plot(x, g["trend_plot"], color=COLORS["trend"], lw=2.5, zorder=5,
            label="Trend component")
    ax.plot(x, g["amp_plot"],   color=COLORS["amplitude"], lw=2.5, zorder=5,
            label="Amplitude component")
    ax.plot(x, g["phase_plot"], color=COLORS["phase"], lw=2.8, zorder=6,
            label="Phase component")
    ax.plot(x, g["raw_plot"],  color=COLORS["raw_anom"], lw=1.0, zorder=4,
            label="Raw anomaly")
    ax.plot(x, g["est_plot"],  color=COLORS["est_anom"], lw=1.6, zorder=4,
            label="Estimated anomaly")

    ax.set_xlim(0, 365)
    ax.set_ylim(-10.5, 7.5)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel(f"Day of the {args.year} cycle", fontsize=13)
    ax.set_ylabel("Anomaly for sea ice extent", fontsize=13)
    sector_label = args.sector.replace("SIE_", "").replace("_", " ")
    ax.set_title(f"(a)  {sector_label} — {args.year}",
                 fontsize=13, fontweight="bold", loc="left")
    ax.tick_params(labelsize=10)
    ax.legend(loc="lower center", fontsize=9.5, frameon=False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
