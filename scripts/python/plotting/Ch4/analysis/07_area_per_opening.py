#!/usr/bin/env python
"""
07_area_per_opening.py -- Ch4 test 1: did opening/closing stop translating into
area change after 2016?

Model, per sector x season, SSM/I era only (1988+):
  dSIA_anom = c + b_pos*POS + b_neg*NEG + d*post + g_pos*POS*post + g_neg*NEG*post + e
POS / NEG = deseasonalised sector-mean opening / closing (period-specific DOY climatology,
same convention as the SIA anomalies), scaled by their pre-2016 SD, so b is
"km^2/day of area change per 1 SD of opening (closing)".
g_* = change after 2016. Reported as % change in the coefficient.
SEs: Newey-West HAC (maxlags=HAC_LAGS) for daily autocorrelation.
Multiple testing: Benjamini-Hochberg across all g tests (20 fits x 2 terms).
Also R^2 of dSIA on POS+NEG, fitted separately pre and post.
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
SIA_CSV = f"{ROOT}/data/merged/analysis_table_daily_anomaly_periodclim.csv"
DIV_CSV = f"{ROOT}/results/ch4/tables/ice_divergence_by_sector_season.csv"
OUT_CSV = f"{ROOT}/results/ch4/tables/area_per_opening_prepost.csv"
START = "1988-01-01"
BREAK_YEAR = 2016
HAC_LAGS = 5
FDR_Q = 0.05
POS_COL = None   # set by hand if auto-detection fails, e.g. "div_positive"
NEG_COL = None   # e.g. "div_negative"
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SHORT = {"WED": "WS", "Weddell": "WS", "WS": "WS",
         "KHV": "KH", "King Haakon VII": "KH", "KH": "KH",
         "EA": "EA", "East Antarctica": "EA",
         "RA": "RA", "Ross-Amundsen": "RA",
         "ABS": "ABS", "Amundsen-Bellingshausen": "ABS"}


def pick(cols, keys, exclude=()):
    hits = [c for c in cols if any(k in c.lower() for k in keys)
            and not any(x in c.lower() for x in exclude)]
    return hits


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def main():
    div = pd.read_csv(DIV_CSV, parse_dates=["date"])
    print("divergence CSV columns:", div.columns.tolist())
    pos_c = [POS_COL] if POS_COL else pick(div.columns, ["pos", "open"])
    neg_c = [NEG_COL] if NEG_COL else pick(div.columns, ["neg", "clos", "conv"])
    if len(pos_c) != 1 or len(neg_c) != 1:
        print(f"Could not pick unique opening/closing columns: pos={pos_c} neg={neg_c}. "
              "Set POS_COL/NEG_COL by hand.")
        sys.exit(1)
    pos_c, neg_c = pos_c[0], neg_c[0]
    print(f"using opening='{pos_c}', closing='{neg_c}'")
    div["sec"] = div.sector.map(SHORT)
    if div.sec.isna().any():
        print("unmapped sectors:", div.sector[div.sec.isna()].unique()); sys.exit(1)
    div = div.groupby(["date", "sec"])[[pos_c, neg_c]].mean().reset_index()

    sia = pd.read_csv(SIA_CSV, parse_dates=["date"])
    sia["sec"] = sia.sector.map(SHORT)
    df = sia[["date", "sec", "delta_SIA_anomaly", "SIA"]].merge(div, on=["date", "sec"], how="inner")
    df = df[df.date >= START].dropna()
    df["post"] = (df.date.dt.year >= BREAK_YEAR).astype(int)
    df["doy"] = df.date.dt.dayofyear.clip(upper=365)
    for c in (pos_c, neg_c):   # period-specific DOY climatology, 15-day smoothed
        clim = (df.groupby(["sec", "post", "doy"])[c].mean()
                  .groupby(level=[0, 1]).transform(
                      lambda s: s.rolling(15, center=True, min_periods=1).mean()))
        df[c + "_a"] = df[c] - clim.reindex(pd.MultiIndex.from_frame(df[["sec", "post", "doy"]])).values
    print(f"merged rows: {len(df)}  {df.date.min().date()} -> {df.date.max().date()}  sectors {sorted(df.sec.unique())}")

    rows = []
    for s, g0 in df.groupby("sec"):
        for season, months in SEASONS.items():
            g = g0[g0.date.dt.month.isin(months)].sort_values("date").copy()
            pre = g.post == 0
            for c in (pos_c, neg_c):
                g[c + "_z"] = g[c + "_a"] / g.loc[pre, c + "_a"].std()
            P, N = g[pos_c + "_z"], g[neg_c + "_z"]
            X = pd.DataFrame({"const": 1.0, "pos": P, "neg": N, "post": g.post,
                              "pos_post": P * g.post, "neg_post": N * g.post})
            m = sm.OLS(g.delta_SIA_anomaly, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
            r2 = {}
            for lab, mask in (("pre", pre), ("post", ~pre)):
                r2[lab] = sm.OLS(g.delta_SIA_anomaly[mask],
                                 sm.add_constant(pd.concat([P[mask], N[mask]], axis=1))).fit().rsquared
            b = m.params
            rows.append(dict(
                sector=s, season=season, n_pre=int(pre.sum()), n_post=int((~pre).sum()),
                b_pos_pre=b["pos"], b_pos_post=b["pos"] + b["pos_post"],
                pos_change_pct=100 * b["pos_post"] / b["pos"], p_pos=m.pvalues["pos_post"],
                b_neg_pre=b["neg"], b_neg_post=b["neg"] + b["neg_post"],
                neg_change_pct=100 * b["neg_post"] / b["neg"], p_neg=m.pvalues["neg_post"],
                p_pos_pre=m.pvalues["pos"], p_neg_pre=m.pvalues["neg"],
                r2_pre=r2["pre"], r2_post=r2["post"]))
    res = pd.DataFrame(rows)
    q = bh(np.r_[res.p_pos.values, res.p_neg.values])
    res["q_pos"], res["q_neg"] = q[:len(res)], q[len(res):]
    res.to_csv(OUT_CSV, index=False, float_format="%.4g")
    print(f"wrote {OUT_CSV}\n")
    pd.set_option("display.width", 220)
    for col, rnd in (("b_pos_pre", 0), ("pos_change_pct", 0), ("q_pos", 3),
                     ("b_neg_pre", 0), ("neg_change_pct", 0), ("q_neg", 3),
                     ("r2_pre", 2), ("r2_post", 2)):
        print(f"-- {col}")
        print(res.pivot(index="sector", columns="season", values=col)[list(SEASONS)].round(rnd).to_string(), "\n")


if __name__ == "__main__":
    main()
