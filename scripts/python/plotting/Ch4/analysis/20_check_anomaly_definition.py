import sys
import numpy as np, pandas as pd, statsmodels.api as sm
F = sys.argv[1] if len(sys.argv) > 1 else \
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_table_daily_anomaly_periodclim.csv"
d = pd.read_csv(F, parse_dates=["date"]).sort_values(["sector", "date"])
d = d[d.date.dt.year >= 1987]


def doy_clim(x, years):
    """smoothed (31-day circular) day-of-year mean of series x over the given years"""
    s = x[x.index.year.isin(years)]
    c = s.groupby(np.minimum(s.index.dayofyear.values, 365)).mean().reindex(range(1, 366))
    c = pd.concat([c.iloc[-15:], c, c.iloc[:15]]).rolling(31, center=True, min_periods=10).mean().iloc[15:-15]
    return c.reindex(np.minimum(x.index.dayofyear.values, 365)).values


rows = []
for sec, g in d.groupby("sector"):
    g = g.set_index("date").asfreq("D")
    ds = g.SIA - g.SIA.shift(1)
    pre, post = range(1988, 2016), range(2016, 2024)
    variants = {
        "A_raw": ds,
        "B_common_clim": ds - doy_clim(ds, pre),
        "C_period_clim_smoothed": pd.Series(np.where(ds.index.year >= 2016,
                                                     ds - doy_clim(ds, post), ds - doy_clim(ds, pre)), ds.index),
        "E_highpass_15d": ds - ds.rolling(15, center=True, min_periods=8).mean(),
    }
    if "delta_SIA_anomaly" in g:
        variants["F_table_anomaly"] = g.delta_SIA_anomaly
    for name, x in variants.items():
        x = x[x.index.year >= 1988]
        for season, mo in (("JJA", (6, 7, 8)), ("SON", (9, 10, 11))):
            h = x[x.index.month.isin(mo)]
            v = h.groupby(h.index.year).var().dropna()
            yr = v.index.values.astype(float)
            X = np.column_stack([np.ones_like(yr), (yr - yr.mean()) / 10, (yr >= 2016).astype(float)])
            m = sm.OLS(np.log(v.values), X).fit()
            rows.append(dict(sector=sec, season=season, variant=name,
                             step=f"{(np.exp(m.params[2]) - 1) * 100:+.0f}%{'*' if m.pvalues[2] < 0.05 else ''}"))
r = pd.DataFrame(rows).pivot_table(index=["sector", "season"], columns="variant", values="step", aggfunc="first")
pd.set_option("display.width", 220)
print("Trend-adjusted 2016 step in day-to-day variance of dSIA (1988-2023), * p < 0.05")
print(r.to_string())
