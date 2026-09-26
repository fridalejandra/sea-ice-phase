#!/usr/bin/env python
"""
08_opening_closing_change.py -- Ch4 test 4 (a falsifiable consistency prediction):
wind stress increased AND the ice's sensitivity to wind (beta) did not change
  => the mean intensity of opening and closing should have increased,
     most where wind increased most.

Per sector x season, SSM/I era (1988-2023, to match the wind/SIA table):
  season-year means of opening (div_positive), closing (-div_negative), wind stress
  - % change 2016-2023 vs 1988-2015, Welch t across season-years
  - linear trend (% of 1988-2015 mean per decade), OLS p
Across the 20 sector-seasons: Spearman correlation of %d(wind) with %d(opening/closing).
BH-FDR across the 40 opening/closing step tests.
"""
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
SIA_CSV = f"{ROOT}/data/merged/analysis_table_daily_anomaly_periodclim.csv"
DIV_CSV = f"{ROOT}/results/ch4/tables/ice_divergence_by_sector_season.csv"
OUT_CSV = f"{ROOT}/results/ch4/tables/opening_closing_change.csv"
START, END = 1988, 2023
BREAK_YEAR = 2016
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SHORT = {"WED": "WS", "Weddell": "WS", "WS": "WS",
         "KHV": "KH", "King Haakon VII": "KH", "KH": "KH",
         "EA": "EA", "East Antarctica": "EA",
         "RA": "RA", "Ross-Amundsen": "RA",
         "ABS": "ABS", "Amundsen-Bellingshausen": "ABS"}


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    q = np.empty(n); q[o] = np.minimum(r, 1); return q


def main():
    div = pd.read_csv(DIV_CSV, parse_dates=["date"])
    div["sec"] = div.sector.map(SHORT)
    div = div.groupby(["date", "sec"])[["div_positive", "div_negative"]].mean().reset_index()
    div["closing"] = -div.div_negative
    div = div.rename(columns={"div_positive": "opening"})
    sia = pd.read_csv(SIA_CSV, parse_dates=["date"])
    sia["sec"] = sia.sector.map(SHORT)
    df = div[["date", "sec", "opening", "closing"]].merge(
        sia[["date", "sec", "wind_stress"]], on=["date", "sec"], how="inner")
    df["sy"] = df.date.dt.year + (df.date.dt.month == 12).astype(int)

    rows = []
    for s, g0 in df.groupby("sec"):
        for season, months in SEASONS.items():
            g = g0[g0.date.dt.month.isin(months)]
            y = g.groupby("sy")[["opening", "closing", "wind_stress"]].mean()
            y = y[(y.index >= START) & (y.index <= END)]
            n = g.groupby("sy").size().reindex(y.index)
            y = y[n >= 60]                                    # near-complete seasons only
            pre, post = y.index < BREAK_YEAR, y.index >= BREAK_YEAR
            row = dict(sector=s, season=season, n_pre=int(pre.sum()), n_post=int(post.sum()))
            for v in ("opening", "closing", "wind_stress"):
                base = y.loc[pre, v].mean()
                row[f"{v}_step_pct"] = 100 * (y.loc[post, v].mean() / base - 1)
                row[f"{v}_step_p"] = stats.ttest_ind(y.loc[post, v], y.loc[pre, v], equal_var=False).pvalue
                lr = stats.linregress(y.index, y[v])
                row[f"{v}_trend_pct_dec"] = 100 * lr.slope * 10 / base
                row[f"{v}_trend_p"] = lr.pvalue
            rows.append(row)
    res = pd.DataFrame(rows)
    q = bh(np.r_[res.opening_step_p, res.closing_step_p])
    res["opening_step_q"], res["closing_step_q"] = q[:len(res)], q[len(res):]
    res.to_csv(OUT_CSV, index=False, float_format="%.4g")
    print(f"wrote {OUT_CSV}  ({START}-{END}, break {BREAK_YEAR})\n")

    pd.set_option("display.width", 220)
    for col, r in (("wind_stress_step_pct", 1), ("wind_stress_step_p", 3),
                   ("opening_step_pct", 1), ("opening_step_q", 3),
                   ("closing_step_pct", 1), ("closing_step_q", 3),
                   ("opening_trend_pct_dec", 1), ("opening_trend_p", 3),
                   ("closing_trend_pct_dec", 1), ("closing_trend_p", 3)):
        print(f"-- {col}")
        print(res.pivot(index="sector", columns="season", values=col)[list(SEASONS)].round(r).to_string(), "\n")

    print("Across 20 sector-seasons: does opening/closing change track wind change?")
    for v in ("opening", "closing"):
        rho, p = stats.spearmanr(res.wind_stress_step_pct, res[f"{v}_step_pct"])
        print(f"  Spearman(dWind, d{v}) step:  rho={rho:+.2f}  p={p:.3f}")
        rho, p = stats.spearmanr(res.wind_stress_trend_pct_dec, res[f"{v}_trend_pct_dec"])
        print(f"  Spearman(dWind, d{v}) trend: rho={rho:+.2f}  p={p:.3f}")


if __name__ == "__main__":
    main()
