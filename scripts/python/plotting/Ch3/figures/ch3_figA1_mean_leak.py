#!/usr/bin/env python3
"""ch3_figA1_mean_leak.py -- Appendix Fig. A1: why a shift in the annual mean confined
to one or two years is credited to the fitted Amplitude component, and why the observed
amplitude (maximum minus minimum) does not show it. The Weddell Sea / Nino3.4 case.
  (a) schematic: a shift moves both ends of the cycle the same way
  (b) schematic: the smooth trend term cannot follow a single year's jump
  (c) COMPUTED here: detrended r between annual Nino3.4 and (i) the cycle-mean Amplitude
      component, (ii) the observed amplitude, Weddell Sea, 1979-2024. The appendix text
      quotes these two numbers; update it if the printout differs.
Reads results/ch3/tables/t32_attribution_by_cycle.csv, data/ch3/annual_params.csv,
      data/indices/nina34.data"""
import os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch3_style as S
from ch3_config import TABLES_DIR, ANNUAL_CSV, INDEX_DIR, INDEX_FILES


def nino_annual():
    rows = []
    for line in open(os.path.join(INDEX_DIR, INDEX_FILES["Nino34"])):
        t = line.split()
        if len(t) >= 13 and re.fullmatch(r"\d{4}", t[0]):
            v = np.array([float(x) for x in t[1:13]]); v[v <= -99] = np.nan
            rows.append((int(t[0]), np.nanmean(v) if np.isfinite(v).sum() == 12 else np.nan))
    return pd.Series(dict(rows))


def detr(s):
    s = s.dropna(); x = s.index.values.astype(float)
    return s - np.polyval(np.polyfit(x, s.values, 1), x)


def r_det(a, b, y0=1979, y1=2024):
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).loc[y0:y1].dropna()
    d = d.apply(detr)
    return np.corrcoef(d.a, d.b)[0, 1], len(d)


# ---- the two numbers -------------------------------------------------------
t32 = pd.read_csv(os.path.join(TABLES_DIR, "t32_attribution_by_cycle.csv"))
YR = "Year" if "Year" in t32.columns else "cycle"
amp_comp = t32[t32.sector == "SIE_Weddell"].set_index(YR)["amplitude_component"]
ann = pd.read_csv(ANNUAL_CSV); ann = ann[ann.period == "FULL"] if "period" in ann else ann
amp_obs = ann[ann.sector == "SIE_Weddell"].set_index("Year")["amplitude_raw_yr"]
nino = nino_annual()
r_comp, n1 = r_det(nino, amp_comp)
r_obs, n2 = r_det(nino, amp_obs)
print(f"Weddell, annual Nino3.4, detrended 1979-2024: Amplitude component r = {r_comp:+.2f} (n={n1}), "
      f"observed amplitude r = {r_obs:+.2f} (n={n2})")

# ---- figure ----------------------------------------------------------------
u = np.linspace(0, 1, 400)
cyc = lambda shift=0.0: shift + (1 - np.cos(2 * np.pi * u)) / 2
fig, (a, b, c) = plt.subplots(1, 3, figsize=(S.TEXT_WIDTH_IN, 2.5),
                              gridspec_kw=dict(width_ratios=[1, 1.15, 0.8], wspace=0.35))

base, shifted = cyc(), cyc(0.34)
a.plot(u, base, color=S.GREY, lw=1.6, ls=(0, (5, 3)), label="ordinary year")
a.plot(u, shifted, color=S.BLUE, lw=1.8, label="El Niño year")
for xx, y0, y1 in ((0.5, base.max(), shifted.max()), (0.02, base[0], shifted[0])):
    a.add_patch(FancyArrowPatch((xx, y0), (xx, y1), arrowstyle="-|>", mutation_scale=9,
                                color=S.INK, lw=1.0, shrinkA=0, shrinkB=0))
a.set_xlim(-0.03, 1.05); a.set_ylim(-0.1, 1.75)
a.set_xticks([0, 0.5, 1]); a.set_xticklabels(["Feb", "Aug", "Feb"])
a.set_yticks([]); a.spines["left"].set_visible(False)
a.set_ylabel("Extent")
a.legend(loc="upper right", fontsize=6.5)

years = np.arange(1988, 2011)
rng = np.random.default_rng(3)
trend = 0.15 * np.sin((years - 1988) / 22 * np.pi) + 0.02 * (years - 1999) / 11
mean = trend + rng.normal(0, 0.05, len(years))
i = int(np.where(years == 2003)[0][0]); mean[i] = trend[i] + 0.34
b.plot(years, trend, color=S.GREY, lw=1.6, label="trend term")
b.scatter(years, mean, s=12, color=S.CBGREY, zorder=3, label="annual mean")
b.scatter([years[i]], [mean[i]], s=34, color=S.BLUE, zorder=4, label="El Niño year")
b.add_patch(FancyArrowPatch((years[i] + 0.4, trend[i]), (years[i] + 0.4, mean[i]), arrowstyle="-|>",
                            mutation_scale=9, color=S.ORANGE, lw=1.2, shrinkA=0, shrinkB=2))
b.text(years[i] + 0.9, (trend[i] + mean[i]) / 2, "credited to\nAmplitude", color=S.ORANGE,
       fontsize=6.5, va="center")
b.set_xlabel("Year"); b.set_yticks([]); b.spines["left"].set_visible(False)
b.set_ylim(-0.3, 0.75)
b.legend(loc="upper left", fontsize=6.5)

vals = [r_comp, r_obs]
c.bar([0, 1], vals, color=[S.BLUE, S.GREY2], width=0.55)
for k, v in enumerate(vals):
    c.text(k, v + (0.02 if v >= 0 else -0.06), f"{v:+.2f}", ha="center", fontsize=7.5, color=S.INK)
c.axhline(0.29, color=S.GREY, lw=0.8, ls=(0, (4, 3)))
c.axhline(-0.29, color=S.GREY, lw=0.8, ls=(0, (4, 3)))
c.axhline(0, color=S.INK, lw=0.8)
c.set_xticks([0, 1]); c.set_xticklabels(["Amplitude\ncomponent", "Observed\namplitude"])
c.set_ylim(-0.4, 0.55)
c.set_ylabel("r with Niño3.4")
S.label_panels([a, b, c])
S.save_figure(fig, "ch3_figA1_mean_leak")
