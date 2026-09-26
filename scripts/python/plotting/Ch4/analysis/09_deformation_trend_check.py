#!/usr/bin/env python
"""
09_deformation_trend_check.py -- Ch4: is the 1988-2023 rise in opening/closing
intensity a real gradual change, or steps at NSIDC-0116 input changes?

Per sector x season (JJA, SON), season-year mean opening (div_positive) and
closing (-div_negative), as a ratio to the 1988-2001 mean.
Model comparison on log(value) vs year:
  T  : linear trend
  S  : piecewise constant with steps at SENSOR_BREAKS only
  TS : trend + sensor steps
If S beats T  -> the "trend" is sensor steps (artifact).
If T beats S and the trend survives in TS -> gradual change (candidate real signal).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
DIV_CSV = f"{ROOT}/results/ch4/tables/ice_divergence_by_sector_season.csv"
OUT_FIG = f"{ROOT}/results/ch4/figures/deformation_trend_sensorcheck_JJA_SON.png"
OUT_CSV = f"{ROOT}/results/ch4/tables/deformation_trend_sensorcheck.csv"
START, END = 1988, 2024
# NSIDC-0116 input changes -- VERIFY against the v4 user guide and edit.
# Value = first season-year using the new input.
SENSOR_BREAKS = {"AVHRR ends": 2001, "AMSR-E starts": 2002, "SSMIS replaces SSM/I": 2007, "AMSR-E ends": 2012, "buoys+NCEP end": 2021}
SEASONS = {"JJA": (6, 7, 8), "SON": (9, 10, 11)}
SHORT = {"WED": "WS", "Weddell": "WS", "WS": "WS", "KHV": "KH", "King Haakon VII": "KH",
         "KH": "KH", "EA": "EA", "East Antarctica": "EA", "RA": "RA", "Ross-Amundsen": "RA",
         "ABS": "ABS", "Amundsen-Bellingshausen": "ABS"}
ORDER = ["WS", "RA", "EA", "KH", "ABS"]
COLORS = {"ABS": "#2196F3", "WS": "#F44336", "KH": "#FFC107", "EA": "#FF9800", "RA": "#4CAF50"}


def fit_models(yr, v):
    y = np.log(v)
    x = (yr - yr.mean()) / 10.0
    steps = np.column_stack([(yr >= b).astype(float) for b in sorted(set(SENSOR_BREAKS.values()))])
    one = np.ones_like(x)
    X = {"T": np.column_stack([one, x]),
         "S": np.column_stack([one, steps]),
         "TS": np.column_stack([one, x, steps])}
    f = {k: sm.OLS(y, m).fit() for k, m in X.items()}
    aic = {k: m.aic for k, m in f.items()}
    return dict(AIC_T=aic["T"], AIC_S=aic["S"], AIC_TS=aic["TS"],
                best=min(aic, key=aic.get),
                trend_pct_dec=(np.exp(f["T"].params[1]) - 1) * 100, trend_p=f["T"].pvalues[1],
                trend_given_sensor_pct_dec=(np.exp(f["TS"].params[1]) - 1) * 100,
                trend_given_sensor_p=f["TS"].pvalues[1])


def main():
    div = pd.read_csv(DIV_CSV, parse_dates=["date"])
    div["sec"] = div.sector.map(SHORT)
    div = div.groupby(["date", "sec"])[["div_positive", "div_negative"]].mean().reset_index()
    div["opening"], div["closing"] = div.div_positive, -div.div_negative
    div["sy"] = div.date.dt.year

    fig, axes = plt.subplots(len(ORDER), 2, figsize=(12, 2.2 * len(ORDER)), sharex=True, sharey=True)
    rows = []
    for i, s in enumerate(ORDER):
        for j, (season, months) in enumerate(SEASONS.items()):
            g = div[(div.sec == s) & div.date.dt.month.isin(months)]
            n = g.groupby("sy").size()
            y = g.groupby("sy")[["opening", "closing"]].mean()
            y = y[(n >= 60) & (y.index >= START) & (y.index <= END)]
            ax = axes[i, j]
            base_mask = y.index <= 2001
            for v, ls in (("opening", "-"), ("closing", "--")):
                r = y[v] / y.loc[base_mask, v].mean()
                ax.plot(y.index, r, ls, color=COLORS[s], lw=1.8, marker="o" if v == "opening" else None,
                        ms=3, label=v)
                res = fit_models(y.index.values.astype(float), y[v].values)
                rows.append(dict(sector=s, season=season, variable=v, n=len(y), **res))
                if v == "opening":
                    ax.text(0.01, 0.05, f"open: best {res['best']}, trend|sensor "
                            f"{res['trend_given_sensor_pct_dec']:+.1f}%/dec p={res['trend_given_sensor_p']:.3f}",
                            transform=ax.transAxes, fontsize=7.5)
            for lab, b in SENSOR_BREAKS.items():
                ax.axvline(b - 0.5, color="0.55", lw=0.9, ls=":")
            ax.axvline(2015.5, color="k", lw=0.9, ls=":")
            ax.axhline(1, color="k", lw=0.5)
            if i == 0:
                ax.set_title(season, fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(s, rotation=0, ha="right", va="center", fontsize=12)
    axes[0, 1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.supylabel("Season mean / 1988–2001 mean", fontsize=11)
    fig.suptitle("Opening and closing intensity (grey dotted = NSIDC-0116 input changes; black = 2016)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False, float_format="%.4g")
    print(f"wrote {OUT_CSV}\n")
    pd.set_option("display.width", 200)
    print(res[["sector", "season", "variable", "best", "AIC_T", "AIC_S",
               "trend_pct_dec", "trend_p", "trend_given_sensor_pct_dec", "trend_given_sensor_p"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
