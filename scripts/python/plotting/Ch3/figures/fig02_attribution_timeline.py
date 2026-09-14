"""
fig02_attribution_timeline.py — Handcock & Raphael Fig-7 attribution, every year, per sector.
=============================================================================================
For each sector-year, the mean over the year (or over a season) of each decomposition
component: trend, amplitude, phase, residual. They sum to the mean anomaly_from_iac
(Extent - invariant cycle) by construction, so the stacked bars add up to the black line.

This is the ONLY place the fitted components are used for a result (methods 2.2.3):
attribution of a year's anomaly, not inference.

    python fig02_attribution_timeline.py                # annual mean
    python fig02_attribution_timeline.py --season SON   # Sep-Nov mean (late maximum shows up here)
    python fig02_attribution_timeline.py --season DJF   # Dec-Feb (minimum side)

Outputs
    results/ch3/figures/fig02_attribution_<annual|SEASON>.png
    results/ch3/tables/t32_attribution_<annual|SEASON>.csv     sector, Year, trend, amplitude, phase, residual, anomaly
    results/ch3/tables/t32_attribution_<annual|SEASON>_shares.csv
        per sector: mean |component| / sum of mean |components|, full record and 2016+
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ch3_data as D
from ch3_config import (SECTORS, SECTOR_LABELS, COMPONENT_COLS, COMPONENT_COLORS,
                        TABLES_DIR, BREAK_YEAR, YEAR_MIN, YEAR_MAX)
from ch3_plot import sector_grid, mark_break, zero_line, year_axis, panel_letters, save

SEASON_MONTHS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11],
                 "ADV": [3, 4, 5, 6, 7, 8], "RET": [10, 11, 12, 1]}

season = None
if "--season" in sys.argv:
    season = sys.argv[sys.argv.index("--season") + 1].upper()
    assert season in SEASON_MONTHS, f"season must be one of {list(SEASON_MONTHS)}"
tag = season or "annual"
print(f"fig02 — attribution timeline ({tag})")

daily = D.load_daily()
D.summary(daily=daily)
daily = daily[daily["Year"].between(YEAR_MIN, YEAR_MAX)].copy()
daily["month"] = pd.to_datetime(daily["Date"]).dt.month

if season:
    months = SEASON_MONTHS[season]
    d = daily[daily["month"].isin(months)].copy()
    if 12 in months and 1 in months:                 # DJF / RET: Dec belongs to the following year's season
        d.loc[d["month"] == 12, "Year"] = d.loc[d["month"] == 12, "Year"] + 1
        d = d[d["Year"].between(YEAR_MIN, YEAR_MAX)]
else:
    d = daily

cols = dict(COMPONENT_COLS)  # Trend, Amplitude, Phase, Residual -> column names
att = (d.groupby(["sector", "Year"])
         .agg(trend=("trend_component", "mean"), amplitude=("amplitude_component", "mean"),
              phase=("phase_component", "mean"), residual=("residual_apac", "mean"),
              anomaly=("anomaly_from_iac", "mean"), n_days=("Date", "size"))
         .reset_index())
chk = (att[["trend", "amplitude", "phase", "residual"]].sum(axis=1) - att["anomaly"]).abs().max()
assert chk < 1e-9, f"components do not sum to the anomaly (max |err| {chk:.2e})"
att.to_csv(os.path.join(TABLES_DIR, f"t32_attribution_{tag}.csv"), index=False)

# shares of |contribution|, full record and post-2016
rows = []
for sec, g in att.groupby("sector"):
    for label, gg in (("1979-2023", g), (f"{BREAK_YEAR}-{YEAR_MAX}", g[g.Year >= BREAK_YEAR])):
        m = gg[["trend", "amplitude", "phase", "residual"]].abs().mean()
        rows.append(dict(sector=SECTOR_LABELS[sec], period=label, **(m / m.sum()).round(3).to_dict(),
                         mean_abs_anomaly=float(gg["anomaly"].abs().mean())))
shares = pd.DataFrame(rows)
shares.to_csv(os.path.join(TABLES_DIR, f"t32_attribution_{tag}_shares.csv"), index=False)
print(shares.to_string(index=False))

# ── figure ────────────────────────────────────────────────────────────────────
fig, axmap = sector_grid(2, 3, figsize=(16, 8.5), sharex=True, sharey=False)
order = [("Trend", "trend"), ("Amplitude", "amplitude"), ("Phase", "phase"), ("Residual", "residual")]
for sec in SECTORS:
    ax = axmap[sec]; g = att[att.sector == sec].sort_values("Year")
    yrs = g["Year"].values
    pos = np.zeros(len(g)); neg = np.zeros(len(g))
    for name, col in order:
        v = g[col].values
        up = np.where(v > 0, v, 0); dn = np.where(v < 0, v, 0)
        ax.bar(yrs, up, bottom=pos, color=COMPONENT_COLORS[name], width=0.82, lw=0, label=name)
        ax.bar(yrs, dn, bottom=neg, color=COMPONENT_COLORS[name], width=0.82, lw=0)
        pos += up; neg += dn
    ax.plot(yrs, g["anomaly"].values, color="#111111", lw=1.4, marker="o", ms=2.6, label="extent anomaly")
    zero_line(ax); mark_break(ax); year_axis(ax, YEAR_MIN, YEAR_MAX)
    ax.set_ylabel("contribution (10⁶ km²)", fontsize=9)
    sh = shares[(shares.sector == SECTOR_LABELS[sec]) & (shares.period != "1979-2023")].iloc[0]
    ax.text(0.02, 0.03, f"{BREAK_YEAR}+ |share|: T {sh.trend:.0%}  A {sh.amplitude:.0%}  P {sh.phase:.0%}  R {sh.residual:.0%}",
            transform=ax.transAxes, fontsize=7.5, color="#555555")
h, l = axmap[SECTORS[0]].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
panel_letters(list(axmap.values()))
ttl = "Attribution of the extent anomaly to trend, amplitude and phase" + (f" — {season} mean" if season else " — annual mean")
fig.suptitle(ttl, fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
save(fig, f"fig02_attribution_{tag}.png", sync=False)
