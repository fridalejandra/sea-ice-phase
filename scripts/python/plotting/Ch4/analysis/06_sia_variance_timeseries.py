#!/usr/bin/env python
"""
06_sia_variance_timeseries.py -- Ch4: is the winter/spring drop in var(dSIA)
a STEP at 2016, a gradual TREND, or a jump at a SENSOR change?

Per sector x season-year: within-season variance of
  raw  : delta_SIA_anomaly
  norm : delta_SIA_anomaly / SIA
  wind : wind_stress_anomaly
each expressed as a ratio to its own 1979-2015 mean (log axis, 1 = pre-2016 level).

Model comparison on log(raw variance) vs year, per sector x season:
  M0 constant | M1 linear trend | M2 step at 2016 | M3 trend + step
Lowest AIC wins. Models are nested, so dAIC vs next best tops out near 2
when the extra term adds nothing.
Also reports the step size in M3 (trend-adjusted) with p-value.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
IN_CSV = f"{ROOT}/data/merged/analysis_table_daily_anomaly_periodclim.csv"
OUT_FIG = f"{ROOT}/results/ch4/figures/sia_variance_timeseries_JJA_SON.png"
OUT_CSV = f"{ROOT}/results/ch4/tables/sia_variance_step_vs_trend.csv"

BREAK_YEAR = 2016
MIN_DAYS = 20
SEASONS = {"JJA": (6, 7, 8), "SON": (9, 10, 11)}
SECTORS = ["Weddell", "Ross-Amundsen", "East Antarctica", "King Haakon VII",
           "Amundsen-Bellingshausen"]
SHORT = {"Amundsen-Bellingshausen": "ABS", "Weddell": "WS", "King Haakon VII": "KH",
         "East Antarctica": "EA", "Ross-Amundsen": "RA"}
COLORS = {"Amundsen-Bellingshausen": "#2196F3", "Weddell": "#F44336",
          "King Haakon VII": "#FFC107", "East Antarctica": "#FF9800",
          "Ross-Amundsen": "#4CAF50"}
# Fill in the sensor-transition years for YOUR SIC product (check its documentation).
# They are drawn as dotted grey lines. Leave empty to skip.
SENSOR_YEARS = []


def yearly(g, months):
    g = g[g.date.dt.month.isin(months)]
    agg = g.groupby(g.date.dt.year).agg(
        raw=("delta_SIA_anomaly", "var"), norm=("_norm", "var"),
        wind=("wind_stress_anomaly", "var"), n=("delta_SIA_anomaly", "count"))
    return agg[agg.n >= MIN_DAYS]


def compare_models(t):
    y = np.log(t.raw.values)
    yr = t.index.values.astype(float)
    x = yr - yr.mean()
    step = (yr >= BREAK_YEAR).astype(float)
    designs = {"M0_const": np.ones((len(y), 1)),
               "M1_trend": np.column_stack([np.ones_like(x), x]),
               "M2_step": np.column_stack([np.ones_like(x), step]),
               "M3_trend+step": np.column_stack([np.ones_like(x), x, step])}
    fits = {k: sm.OLS(y, X).fit() for k, X in designs.items()}
    aic = {k: f.aic for k, f in fits.items()}
    best = min(aic, key=aic.get)
    second = sorted(aic.values())[1]
    m3, m2 = fits["M3_trend+step"], fits["M2_step"]
    return dict(best_model=best, dAIC_vs_next=second - aic[best],
                **{f"AIC_{k}": v for k, v in aic.items()},
                step_only_pct=(np.exp(m2.params[1]) - 1) * 100, step_only_p=m2.pvalues[1],
                step_trendadj_pct=(np.exp(m3.params[2]) - 1) * 100, step_trendadj_p=m3.pvalues[2],
                trend_pct_per_decade=(np.exp(fits["M1_trend"].params[1] * 10) - 1) * 100,
                trend_p=fits["M1_trend"].pvalues[1])


def main():
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df = df.dropna(subset=["delta_SIA_anomaly", "SIA"])
    df = df[df.SIA > 0].copy()
    df["_norm"] = df.delta_SIA_anomaly / df.SIA

    fig, axes = plt.subplots(len(SECTORS), 2, figsize=(12, 2.3 * len(SECTORS)),
                             sharex=True, sharey=True)
    rows = []
    for i, s in enumerate(SECTORS):
        g = df[df.sector == s]
        for j, (season, months) in enumerate(SEASONS.items()):
            ax = axes[i, j]
            t = yearly(g, months)
            pre = t.index < BREAK_YEAR
            for col, style, lab in (("raw", dict(color=COLORS[s], lw=2, marker="o", ms=3), "var(ΔSIA)"),
                                    ("norm", dict(color=COLORS[s], lw=1.2, ls="--"), "var(ΔSIA / SIA)"),
                                    ("wind", dict(color="0.45", lw=1), "var(wind stress)")):
                ratio = t[col] / t[col][pre].mean()
                ax.plot(t.index, ratio, label=lab, **style)
                if col == "raw":
                    for mask in (pre, ~pre):
                        m = np.exp(np.log(ratio[mask]).mean())
                        ax.hlines(m, t.index[mask].min(), t.index[mask].max(),
                                  color=COLORS[s], lw=3, alpha=0.35)
            ax.axhline(1, color="k", lw=0.5)
            ax.axvline(BREAK_YEAR - 0.5, color="k", lw=1, ls=":")
            for sy in SENSOR_YEARS:
                ax.axvline(sy, color="0.6", lw=0.8, ls=(0, (1, 2)))
            ax.set_yscale("log")
            ax.set_ylim(0.15, 5)
            ax.set_yticks([0.25, 0.5, 1, 2, 4])
            ax.set_yticklabels(["¼", "½", "1", "2", "4"])
            if i == 0:
                ax.set_title(season, fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(SHORT[s], fontsize=12, rotation=0, ha="right", va="center")
            r = compare_models(t)
            ax.text(0.01, 0.04, f"best: {r['best_model']} (ΔAIC {r['dAIC_vs_next']:.1f}); "
                    f"step|trend {r['step_trendadj_pct']:+.0f}% p={r['step_trendadj_p']:.2f}",
                    transform=ax.transAxes, fontsize=8)
            rows.append(dict(sector=SHORT[s], season=season,
                             n_years=len(t), n_post=int((~pre).sum()), **r))
    axes[0, 1].legend(fontsize=8, loc="upper right", frameon=False)
    fig.supylabel("Season-year variance / 1979–2015 mean", fontsize=11)
    fig.suptitle("Day-to-day variability of sea-ice area tendency, by season-year", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False, float_format="%.4g")
    print(f"wrote {OUT_CSV}\n")
    pd.set_option("display.width", 200)
    print(res[["sector", "season", "best_model", "dAIC_vs_next", "step_only_pct", "step_only_p",
               "step_trendadj_pct", "step_trendadj_p", "trend_pct_per_decade", "trend_p"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
