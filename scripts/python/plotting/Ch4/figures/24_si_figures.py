#!/usr/bin/env python
"""
24_si_figures.py -- two SI figures from tables that already exist (runs in seconds).

  FigS1a_drift_speed_step.png  Pan-Antarctic Sep-Oct NSIDC-0116 drift speed 1982-2024, all valid
                               cells and cells valid every year, with the 1987 sensor step and the
                               1988-2024 trend; lower panel: number of valid cells.
                               (from tables/motion_trend_diagnostic_SO.csv, script 23)
  FigS3_divergence_products.png  Trend in mean divergence rate by sector and season, NSIDC-0116 vs
                               OSI SAF OSI-455 (fixed coverage, satellite-only), 1991-2020, with the
                               correlation of their season-year series.
                               (from tables/osisaf_fixed_coverage_trends.csv, script 11b)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_style as st  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/user/geog/falejandraperez/sea-ice-phase"
TAB = f"{ROOT}/results/ch4/tables"
FIG = f"{ROOT}/results/ch4/figures"
SECTORS = ["WS", "KH", "EA", "RA", "ABS"]
LONG = {"WS": "Weddell", "KH": "King Haakon VII", "EA": "East Antarctica",
        "RA": "Ross–Amundsen", "ABS": "Amundsen–Bellingshausen"}


def fig_s1a():
    d = pd.read_csv(f"{TAB}/motion_trend_diagnostic_SO.csv")
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08))
    for a in (ax, axn):
        a.axvspan(1981.5, 1986.5, color="0.93", zorder=0, lw=0)
        a.axvline(1986.5, color=st.INK, lw=0.6, ls=(0, (2, 2)))
        a.spines[["top", "right"]].set_visible(False)
    ax.plot(d.year, d.pan_50, color=st.INK, lw=1.8, marker="o", ms=3, label="all valid cells")
    ax.plot(d.year, d.pan_fixed, color="0.6", lw=1.2, label="cells valid every year")
    m = d.year >= 1988
    lr = stats.linregress(d.year[m], d.pan_50[m])
    ax.plot(d.year[m], lr.intercept + lr.slope * d.year[m], color="#c0392b", lw=1.2)
    ptxt = "p < 0.001" if lr.pvalue < 0.001 else f"p = {lr.pvalue:.3f}"
    ax.text(0.98, 0.55, f"red: 1988–2024 trend, {lr.slope * 10:+.2f} cm s⁻¹ per decade ({ptxt})",
            transform=ax.transAxes, color="#c0392b", ha="right", va="center", fontsize=8)
    ax.text(1984, 0.97, "SMMR", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8, color=st.INK)
    ax.text(1986.8, 0.97, "SSM/I from Jul 1987", transform=ax.get_xaxis_transform(), ha="left",
            va="top", fontsize=8, color=st.INK)
    ax.set_ylabel("Mean drift speed (cm s⁻¹)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Pan-Antarctic sea-ice drift speed, September–October (NSIDC-0116)", loc="left")
    axn.plot(d.year, d.n_cells_50 / 1000, color=st.INK, lw=1.2)
    axn.set_ylabel("Valid cells\n(×1000)", fontsize=8)
    axn.set_ylim(0, None)
    axn.set_xlim(1981.5, 2024.5)
    fig.savefig(f"{FIG}/FigS1a_drift_speed_step.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {FIG}/FigS1a_drift_speed_step.png")


def fig_s3():
    t = pd.read_csv(f"{TAB}/osisaf_fixed_coverage_trends.csv")
    seasons = [s for s in ("JJA", "SO", "SON") if s in set(t.season)]
    rows, y = [], 0.0
    for sec in SECTORS:
        for s in seasons:
            rows.append((sec, s, -y))
            y += 1
        y += 0.6
    fig, ax = plt.subplots(figsize=(5.6, 0.36 * len(rows) + 1.2))
    for sec, s, yy in rows:
        r = t[(t.sector == sec) & (t.season == s)]
        if r.empty:
            continue
        r = r.iloc[0]
        col = st.SECTOR_COLORS[sec]
        for val, p, off, mk in ((r.nsidc_opening_pct_dec, r.p_nsidc, -0.15, "o"),
                                (r.osi_fixed_opening_pct_dec, r.p_open, 0.15, "s")):
            if np.isfinite(val):
                sig = np.isfinite(p) and p < 0.05
                ax.plot(val, yy + off, mk, ms=6, color=col if sig else "white", mec=col, mew=1.3, zorder=3)
        ax.plot([r.nsidc_opening_pct_dec, r.osi_fixed_opening_pct_dec], [yy - 0.15, yy + 0.15],
                color=col, lw=0.8, alpha=0.6, zorder=2)
        if np.isfinite(r.r_fixed_vs_nsidc):
            ax.text(1.01, yy, f"r = {r.r_fixed_vs_nsidc:+.2f}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7.5, color=st.INK)
    ax.axvline(0, color=st.INK, lw=0.8, zorder=0)
    ax.set_yticks([yy for _, _, yy in rows])
    ax.set_yticklabels([f"{LONG[sec]}  {s}" for sec, s, _ in rows], fontsize=8)
    for lab, (sec, _, _) in zip(ax.get_yticklabels(), rows):
        lab.set_color(st.SECTOR_COLORS[sec])
    ax.tick_params(axis="y", length=0)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_xlabel("Trend in mean divergence rate, 1991–2020 (% per decade)\n"
                  "● NSIDC-0116   ■ OSI SAF (fixed coverage)   filled: p < 0.05")
    ax.set_title("Drift products disagree on divergence trends", loc="left")
    fig.savefig(f"{FIG}/FigS3_divergence_products.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {FIG}/FigS3_divergence_products.png")


if __name__ == "__main__":
    fig_s1a()
    fig_s3()
