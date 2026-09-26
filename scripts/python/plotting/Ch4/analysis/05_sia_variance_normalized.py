#!/usr/bin/env python
"""
05_sia_variance_normalized.py -- Ch4: is the post-2016 "collapse" in var(dSIA)
more than a smaller ice pack would produce by itself?

Step 0  Reproduce poster Fig 3/4: plain var(delta_SIA_anomaly), all days, pre vs post.
Step 1  Per sector x season (+ ALL), within-season-year variance, pooled:
          raw   : delta_SIA_anomaly                  (km^2/day)
          norm  : delta_SIA_anomaly / SIA            (fraction of pack per day)
        Ratios post/pre for two baselines (1979-2015, 2003-2015),
        Welch t on log(season-year variance) for significance.
Step 2  Scaling exponent k = ln(var ratio raw) / ln(SIA ratio)
          k ~ 1   independent pieces of ice: var scales with area
          k ~ 2   fully coherent pack: var scales with area^2
          k > 2   variability dropped MORE than pack size explains  <- real signal
          k < 1   dropped less than even the weakest scaling predicts
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

IN_CSV = ("/user/geog/falejandraperez/sea-ice-phase/data/merged/"
          "analysis_table_daily_anomaly_periodclim.csv")
OUT_CSV = ("/user/geog/falejandraperez/sea-ice-phase/results/ch4/tables/"
           "sia_variance_normalized.csv")
RESP = "delta_SIA_anomaly"
SIA_CANDIDATES = ["SIA", "sia", "SIA_km2", "sea_ice_area", "sia_km2", "area", "SIA_raw"]
POSTER_EXCLUDE = [1978, 1987, 1991, 1995]   # poster's exclusions, used only in Step 0
BREAK_YEAR = 2016
PRE_RECENT_START = 2003
MIN_DAYS = 20
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8),
           "SON": (9, 10, 11), "ALL": tuple(range(1, 13))}
SHORT = {"Amundsen-Bellingshausen": "ABS", "Weddell": "WS", "King Haakon VII": "KH",
         "East Antarctica": "EA", "Ross-Amundsen": "RA"}


def welch_logvar(a, b):
    a, b = np.log(a[a > 0]), np.log(b[b > 0])
    if len(a) < 5 or len(b) < 4:
        return np.nan
    return stats.ttest_ind(b, a, equal_var=False).pvalue


def season_table(g, col, months):
    """Per season-year variance and n of column `col` within the given months."""
    g = g[g.date.dt.month.isin(months)].copy()
    g["sy"] = g.date.dt.year + ((g.date.dt.month == 12) & (len(months) == 3)).astype(int)
    out = g.groupby("sy")[col].agg(v="var", n="count")
    out["sia"] = g.groupby("sy")["_sia"].mean()
    return out[out.n >= MIN_DAYS]


def pooled(t):
    w = t.n - 1
    return (t.v * w).sum() / w.sum()


def main():
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    sia_col = next((c for c in SIA_CANDIDATES if c in df.columns), None)
    if sia_col is None:
        print("No raw SIA column found. Columns are:\n ", df.columns.tolist())
        print("Set SIA_CANDIDATES to the right name and re-run.")
        sys.exit(1)
    print(f"Using SIA column '{sia_col}', response '{RESP}'")
    print("Sectors:", df.sector.unique().tolist(), "|", df.date.min().date(), "->", df.date.max().date())
    df = df.dropna(subset=[RESP, sia_col])
    df = df[df[sia_col] > 0].copy()
    df["_sia"] = df[sia_col]
    df["_norm"] = df[RESP] / df[sia_col]
    df["_sector"] = df.sector.map(SHORT).fillna(df.sector)

    # ---- Step 0: reproduce poster (plain variance, all days, poster exclusions)
    print("\nStep 0 - poster reproduction, % change in var(dSIA) (poster: ABS -33, EA -35, KH -36, RA -71, WS -60)")
    p0 = df[~df.date.dt.year.isin(POSTER_EXCLUDE)]
    for s, g in p0.groupby("_sector"):
        pre, post = g[g.date.dt.year < BREAK_YEAR], g[g.date.dt.year >= BREAK_YEAR]
        raw = (post[RESP].var() / pre[RESP].var() - 1) * 100
        nrm = (post["_norm"].var() / pre["_norm"].var() - 1) * 100
        sia = (post._sia.mean() / pre._sia.mean() - 1) * 100
        print(f"  {s:4s} raw {raw:+5.0f}%   normalized {nrm:+5.0f}%   mean SIA {sia:+4.0f}%")

    # ---- Step 1-2
    rows = []
    for s, g in df.groupby("_sector"):
        for season, months in SEASONS.items():
            traw = season_table(g, RESP, months)
            tnrm = season_table(g, "_norm", months)
            for base, lo in (("full", 0), ("recent", PRE_RECENT_START)):
                pr = (traw.index >= lo) & (traw.index < BREAK_YEAR)
                po = traw.index >= BREAK_YEAR
                prn = (tnrm.index >= lo) & (tnrm.index < BREAK_YEAR)
                pon = tnrm.index >= BREAK_YEAR
                vr = pooled(traw[po]) / pooled(traw[pr])
                vn = pooled(tnrm[pon]) / pooled(tnrm[prn])
                sr = traw[po].sia.mean() / traw[pr].sia.mean()
                k = np.log(vr) / np.log(sr) if abs(np.log(sr)) > 0.05 else np.nan
                rows.append(dict(
                    sector=s, season=season, baseline=base,
                    pre_years=f"{traw.index[pr].min()}-{traw.index[pr].max()}",
                    n_pre=int(pr.sum()), n_post=int(po.sum()),
                    sia_change_pct=(sr - 1) * 100,
                    var_raw_change_pct=(vr - 1) * 100,
                    p_raw=welch_logvar(traw[pr].v.values, traw[po].v.values),
                    var_norm_change_pct=(vn - 1) * 100,
                    p_norm=welch_logvar(tnrm[prn].v.values, tnrm[pon].v.values),
                    k_scaling=k))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False, float_format="%.4g")
    print(f"\nwrote {OUT_CSV}")
    pd.set_option("display.width", 200)
    for base in ("full", "recent"):
        r = res[res.baseline == base]
        print(f"\n===== baseline: {base} ({r.pre_years.iloc[0]}) vs {BREAK_YEAR}+ =====")
        for col in ("sia_change_pct", "var_raw_change_pct", "var_norm_change_pct", "p_norm", "k_scaling"):
            print(f"\n-- {col}")
            print(r.pivot(index="sector", columns="season", values=col)
                  [list(SEASONS)].round(3 if col == "p_norm" else 1).to_string())


if __name__ == "__main__":
    main()
