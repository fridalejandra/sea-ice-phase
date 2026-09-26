import numpy as np, pandas as pd, statsmodels.api as sm
F = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_table_daily_anomaly_periodclim.csv"
d = pd.read_csv(F, parse_dates=["date"]).sort_values(["sector", "date"])
d = d[d.date.dt.year >= 1987]
out = []
for s, g in d.groupby("sector"):
    g = g.set_index("date").asfreq("D")                       # gaps become NaN, so no multi-day differences
    g["back"] = g.SIA - g.SIA.shift(1)
    g["cent"] = (g.SIA.shift(-1) - g.SIA.shift(1)) / 2
    g = g[g.index.year >= 1988]
    for season, mo in (("JJA", (6, 7, 8)), ("SON", (9, 10, 11))):
        h = g[g.index.month.isin(mo)]
        for col in ("back", "cent"):
            v = h.groupby(h.index.year)[col].var().dropna()
            yr = v.index.values.astype(float)
            X = np.column_stack([np.ones_like(yr), (yr - yr.mean()) / 10, (yr >= 2016).astype(float)])
            m = sm.OLS(np.log(v.values), X).fit()
            out.append(dict(sector=s, season=season, method=col,
                            step_pct=(np.exp(m.params[2]) - 1) * 100, p=m.pvalues[2]))
r = pd.DataFrame(out).pivot_table(index=["sector", "season"], columns="method", values=["step_pct", "p"])
print("2016 step in day-to-day variance of dSIA (trend-adjusted), backward vs central differences, 1988-2023")
print(r.round(3).to_string())
