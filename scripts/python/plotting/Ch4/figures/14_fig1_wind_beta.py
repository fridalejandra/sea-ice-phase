#!/usr/bin/env python
"""
14_fig1_wind_beta.py -- Ch4 Figure 1.
Question: did the wind forcing, or the ice's response to it, change after 2016?

(a) change in sector-mean wind stress, 2016-2023 vs 1988-2015 (%)
(b) post-2016 change in the sensitivity of daily area tendency to wind stress,
    as % of the pre-2016 sensitivity, with 95% block-bootstrap interval (1988-2023).
    Hollow = poorly constrained (interval wider than +/-100%).
Rows: sector (bold, sector colour) x season.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_style as st  # noqa: E402  (applies house style on import)

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
BETA_CSV = f"{ROOT}/data/merged/analysis_results_clean/beta_shift_seasonal_fdr.csv"
WIND_CSV = f"{ROOT}/results/ch4/tables/opening_closing_change.csv"
OUT = f"{ROOT}/results/ch4/figures/fig1_wind_and_sensitivity.png"
SECTORS = ["WS", "KH", "EA", "RA", "ABS"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
SHORT = {"Weddell": "WS", "King Haakon VII": "KH", "East Antarctica": "EA",
         "Ross-Amundsen": "RA", "Amundsen-Bellingshausen": "ABS"}
XLIM_B = 150   # clip sensitivity axis; wider intervals get an arrow


def main():
    b = pd.read_csv(BETA_CSV)
    b["sector"] = b.sector.map(SHORT).fillna(b.sector)
    a = b.beta_pre.abs()
    # relative change in the MAGNITUDE of beta: + = stronger wind response, - = weaker
    p = b.beta_pre
    l, h = 100 * b.ci_low / p, 100 * b.ci_high / p
    b["shift"] = 100 * b.beta_shift / p
    b["lo"], b["hi"] = pd.concat([l, h], axis=1).min(axis=1), pd.concat([l, h], axis=1).max(axis=1)
    w = pd.read_csv(WIND_CSV)
    b, w = b.set_index(["sector", "season"]), w.set_index(["sector", "season"])

    # y positions: 4 seasons per sector, gap between sectors, top to bottom
    ypos, y = {}, 0.0
    for s in SECTORS:
        for se in SEASONS:
            ypos[(s, se)] = -y
            y += 1
        y += 0.8

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 6.4), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.08})
    for ax in (axa, axb):
        ax.axvline(0, color=st.INK, lw=0.8, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    for (s, se), yy in ypos.items():
        col = st.SECTOR_COLORS[s]
        # (a) wind
        wv, wp = w.loc[(s, se), "wind_stress_step_pct"], w.loc[(s, se), "wind_stress_step_p"]
        axa.plot(wv, yy, "o", ms=5.5, color=col, mec=col)
        if wp < 0.05:
            axa.text(wv + 0.6, yy, "*", va="center", ha="left", color=st.INK, fontsize=11)
        # (b) sensitivity change
        v, lo, hi, q = b.loc[(s, se), ["shift", "lo", "hi", "p_value_fdr"]]
        weak = (hi - lo) > 200
        loc, hic = max(lo, -XLIM_B), min(hi, XLIM_B)
        axb.plot([loc, hic], [yy, yy], color=col, lw=1.6, alpha=0.45 if weak else 0.9,
                 solid_capstyle="butt")
        for edge, clipped, sign in ((lo, lo < -XLIM_B, -1), (hi, hi > XLIM_B, 1)):
            if clipped:
                axb.plot(sign * XLIM_B, yy, marker="<" if sign < 0 else ">", ms=4,
                         color=col, alpha=0.45 if weak else 0.9)
        vv = float(np.clip(v, -XLIM_B, XLIM_B))
        axb.plot(vv, yy, "o", ms=5.5, color="white" if weak else col, mec=col, mew=1.3, zorder=3)
        if q < 0.10:
            axb.text(hic + 4, yy, "q = %.2f" % q, va="center", ha="left", fontsize=7, color=st.INK)

    # y labels: season ticks; sector names in bold, sector colour, left of the panel
    axa.set_yticks(list(ypos.values()))
    axa.set_yticklabels([se for (_, se) in ypos.keys()])
    bold = st.bold_font_properties(size=9)
    for s in SECTORS:
        mid = np.mean([ypos[(s, se)] for se in SEASONS])
        axa.text(-0.34, mid, st.SECTOR_NAMES[s], transform=axa.get_yaxis_transform(),
                 ha="right", va="center", color=st.SECTOR_COLORS[s], fontproperties=bold)

    axa.set_xlim(-6, 14)
    axa.set_xlabel("Change in wind stress (%)")
    axa.set_title("(a) Forcing", loc="left")
    axb.set_xlim(-XLIM_B - 10, XLIM_B + 40)
    axb.set_xlabel("Change in sensitivity of daily area tendency\n(% of 1988–2015 magnitude; + = stronger; 95% interval)")
    axb.set_title("(b) Response", loc="left")
    axb.text(0, max(ypos.values()) + 1.0, "no change", ha="center", va="bottom", fontsize=7.5, color=st.INK)
    axa.text(0, max(ypos.values()) + 1.0, "no change", ha="center", va="bottom", fontsize=7.5, color=st.INK)
    axb.text(1.0, 0.0, "hollow = poorly constrained (interval > ±100%)", transform=axb.transAxes,
             ha="right", va="bottom", fontsize=7, color=st.INK)
    axa.text(1.0, 0.0, "* p < 0.05", transform=axa.transAxes, ha="right", va="bottom",
             fontsize=7, color=st.INK)
    axa.set_ylim(min(ypos.values()) - 0.8, max(ypos.values()) + 1.8)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
