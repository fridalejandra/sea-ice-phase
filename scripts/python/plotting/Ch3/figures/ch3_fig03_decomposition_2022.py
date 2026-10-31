#!/usr/bin/env python
"""ch3_fig03_decomposition_2022.py -- Fig. 3.3: the decomposition of one cycle
(circumpolar total, 21 Feb 2022 to 20 Feb 2023). Python port of
scripts/R/Ch3/plot_fig_decomposition.R (same maths: retreat-limb blend of the
year's and the next year's amplitude and minimum; components sum to the anomaly).
  (a) observed extent, invariant cycle, fully adjusted cycle
  (b) the extent anomaly split into Trend, Amplitude, Phase and the Raw Anomaly
Usage: python ch3_fig03_decomposition_2022.py [YEAR]"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch3_style as S
from ch3_config import DAILY_CSV

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2022
SECTOR, PERIOD, SMOOTH = "SIE_circumpolar", "FULL", 11
C = S.COMPONENT_COLORS


def components(d, sector, year):
    t0, t1 = pd.Timestamp(f"{year}-02-21"), pd.Timestamp(f"{year + 1}-02-20")
    g = d[(d.sector == sector) & (d.Date >= t0) & (d.Date <= t1)].sort_values("Date")
    if len(g) < 300:
        sys.exit(f"Only {len(g)} rows for {sector} {year}: cycle incomplete?")
    if g.Date.duplicated().any():
        sys.exit("Duplicate dates: period filter did not take.")
    cd = (g.Date - t0).dt.days.values.astype(float)
    ampY, minY = g.amplitude.iloc[0], g.min_extent.iloc[0]
    ampN, minN = g.amplitude.iloc[-1], g.min_extent.iloc[-1]
    dmax = cd[np.argmax(g.Extent.values)]
    w = np.clip((cd - dmax) / (365 - dmax), 0, 1)
    ampb, minb = ampY * (1 - w) + ampN * w, minY * (1 - w) + minN * w
    famp = g.u_amp.values * ampb + minb
    fapac = g.u_apac.values * ampb + minb
    tv = g.trend_component.values
    out = pd.DataFrame(dict(day=cd, observed=g.Extent.values, invariant=g.iac_notrend.values,
                            adjusted=fapac, trend=tv, amplitude=famp - g.iac_notrend.values - tv,
                            phase=fapac - famp, raw=g.Extent.values - fapac,
                            anomaly=g.Extent.values - g.iac_notrend.values))
    err = np.nanmax(np.abs(out[["trend", "amplitude", "phase", "raw"]].sum(axis=1) - out.anomaly))
    if err > 1e-6 * max(1, np.nanmax(np.abs(out.observed))):
        print(f"  WARNING: components do not sum to the anomaly (max difference {err:.3g})")
    if np.nanmax(out.observed) > 1000:
        out.iloc[:, 1:] *= 1e-6
    out["raw_smooth"] = out.raw.rolling(SMOOTH, center=True, min_periods=1).mean()
    months = pd.date_range(f"{year}-03-01", periods=12, freq="MS")
    return out, dmax, (months - t0).days.values, [m.strftime("%b") for m in months]


def main():
    d = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
    if "period" in d:
        d = d[d.period == PERIOD]
    x, dmax, ticks, labs = components(d, SECTOR, YEAR)

    fig, (a, b) = plt.subplots(2, 1, figsize=(S.TEXT_WIDTH_IN, 5.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[1, 1.4], hspace=0.12))
    a.plot(x.day, x.invariant, color=S.SKY, lw=4.0, solid_capstyle="round", label="invariant annual cycle")
    a.plot(x.day, x.adjusted, color=S.GREY, lw=1.2, ls="--", label=f"amplitude- and phase-adjusted cycle, {YEAR}")
    a.plot(x.day, x.observed, color=S.INK, lw=1.0, label=f"observed, {YEAR}")
    a.axvline(dmax, color=S.GREY, ls=":", lw=1)
    a.set_ylabel("Extent (10$^6$ km$^2$)")
    a.legend(loc="upper left")

    b.fill_between(x.day, x.raw, 0, color=S.GREY2, alpha=0.8, lw=0)
    b.axhline(0, color=S.GREY, lw=0.8)
    b.axvline(dmax, color=S.GREY, ls=":", lw=1)
    b.plot(x.day, x.raw_smooth, color=C["Raw"], lw=1.1)
    for k, col in (("trend", "Trend"), ("amplitude", "Amplitude"), ("phase", "Phase")):
        b.plot(x.day, x[k], color=C[col], lw=1.8)
    lo, hi = np.nanmin(x[["trend", "amplitude", "phase", "raw"]].values), np.nanmax(x[["trend", "amplitude", "phase", "raw"]].values)
    b.set_ylim(lo - 0.32 * (hi - lo), hi + 0.14 * (hi - lo))
    b.set_ylabel("Contribution to the\nextent anomaly (10$^6$ km$^2$)")
    b.legend(handles=[Line2D([], [], color=C["Trend"], lw=1.8, label="Trend"),
                      Line2D([], [], color=C["Amplitude"], lw=1.8, label="Amplitude"),
                      Line2D([], [], color=C["Phase"], lw=1.8, label="Phase"),
                      Patch(facecolor=S.GREY2, label="Raw Anomaly")],
             loc="lower left", ncol=4)
    b.set_xlim(0, 365); b.set_xticks(ticks[::2]); b.set_xticklabels(labs[::2])
    a.text(dmax, a.get_ylim()[1], "maximum", ha="center", va="bottom", fontsize=7.5, color=S.GREY)
    S.label_panels([a, b])
    S.save_figure(fig, f"ch3_fig03_decomposition_{YEAR}")

    print("mean |component| over the cycle, circumpolar (10^6 km^2):")
    for k in ("anomaly", "trend", "amplitude", "phase", "raw"):
        print(f"  {k:10s} {np.nanmean(np.abs(x[k])):.2f}")


if __name__ == "__main__":
    main()
