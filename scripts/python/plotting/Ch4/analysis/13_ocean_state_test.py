#!/usr/bin/env python
"""
13_ocean_state_test.py -- Ch4: does the ocean state explain how the remaining ice responds?

Ocean index O = season-year mean of sector SST anomaly (ERA5, sector-wide; see caveat).
Per sector x season (JJA, SON), 1988-2023:

A. Variability vs ocean (one value per season-year):
   log var(dSIA') ~ step2016 | ~ O | ~ O + step2016     (AIC; also the relative-tendency version)
   -> does ocean warmth track the collapse in area variability, and does it absorb the 2016 step?

B. Sensitivity conditioned on the ocean (daily, HAC SEs):
   y = b1*tau' + b2*O + b3*(tau' x O)        y = dSIA'      (km^2/day)
                                              y = dSIA'/SIA  (fraction of remaining ice per day)
   -> b3 != 0 means the response to wind depends on ocean state.
   O is standardised (per sector-season), so b3 = change in b1 per 1 SD of ocean warmth.

CAVEAT: the SST index is a whole-sector mean (mostly open ocean far from the ice),
so it is a broad Southern Ocean index, not the water at the ice edge.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
SIA_CSV = f"{ROOT}/data/merged/analysis_table_daily_anomaly_periodclim.csv"
SST_CSV = f"{ROOT}/scripts/python/scar_poster/sst_anomaly_by_sector_daily.csv"
OUT_A = f"{ROOT}/results/ch4/tables/ocean_variability_test.csv"
OUT_B = f"{ROOT}/results/ch4/tables/ocean_conditioned_sensitivity.csv"
START, END, BREAK = 1988, 2023, 2016
SEASONS = {"JJA": (6, 7, 8), "SON": (9, 10, 11)}
SHORT = {"Amundsen-Bellingshausen": "ABS", "Weddell": "WS", "King Haakon VII": "KH",
         "East Antarctica": "EA", "Ross-Amundsen": "RA"}
MIN_DAYS = 60
HAC_LAGS = 5


def ols_aic(y, X):
    m = sm.OLS(y, X).fit()
    return m


def main():
    sia = pd.read_csv(SIA_CSV, parse_dates=["date"])
    sst = pd.read_csv(SST_CSV, parse_dates=["date"])[["date", "sector", "sst_anom"]]
    df = sia.merge(sst, on=["date", "sector"], how="inner")
    df = df[(df.date.dt.year >= START) & (df.date.dt.year <= END)]
    df = df.dropna(subset=["delta_SIA_anomaly", "wind_stress_anomaly", "SIA", "sst_anom"])
    df = df[df.SIA > 0].copy()
    df["rel"] = df.delta_SIA_anomaly / df.SIA
    df["sec"] = df.sector.map(SHORT)
    df["yr"] = df.date.dt.year

    rowsA, rowsB = [], []
    for (s, season), g in [((s, se), df[(df.sec == s) & df.date.dt.month.isin(m)])
                           for s in SHORT.values() for se, m in SEASONS.items()]:
        y = g.groupby("yr").agg(var_raw=("delta_SIA_anomaly", "var"), var_rel=("rel", "var"),
                                O=("sst_anom", "mean"), n=("rel", "size"))
        y = y[y.n >= MIN_DAYS]
        step = (y.index >= BREAK).astype(float)
        Oz = (y.O - y.O.mean()) / y.O.std()
        rho_O_step = np.corrcoef(Oz, step)[0, 1]
        for resp in ("var_raw", "var_rel"):
            L = np.log(y[resp].values)
            one = np.ones(len(L))
            mS = ols_aic(L, np.column_stack([one, step]))
            mO = ols_aic(L, np.column_stack([one, Oz]))
            mOS = ols_aic(L, np.column_stack([one, Oz, step]))
            rowsA.append(dict(sector=s, season=season, response=resp, n_years=len(L),
                              corr_O_vs_step=rho_O_step,
                              O_slope_pct_per_sd=(np.exp(mO.params[1]) - 1) * 100, O_p=mO.pvalues[1],
                              r_O=np.corrcoef(Oz, L)[0, 1],
                              AIC_step=mS.aic, AIC_O=mO.aic, AIC_O_step=mOS.aic,
                              step_pct_given_O=(np.exp(mOS.params[2]) - 1) * 100, step_p_given_O=mOS.pvalues[2],
                              O_pct_given_step=(np.exp(mOS.params[1]) - 1) * 100, O_p_given_step=mOS.pvalues[1]))

        # B: daily sensitivity conditioned on season-year ocean state
        g = g.merge(Oz.rename("Oz"), left_on="yr", right_index=True)
        tau = (g.wind_stress_anomaly - g.wind_stress_anomaly.mean()) / g.wind_stress_anomaly.std()
        X = pd.DataFrame({"const": 1.0, "tau": tau.values, "O": g.Oz.values,
                          "tauO": (tau * g.Oz).values})
        for resp, col in (("dSIA", "delta_SIA_anomaly"), ("dSIA_over_SIA", "rel")):
            m = sm.OLS(g[col].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
            rowsB.append(dict(sector=s, season=season, response=resp,
                              b1_per_sd_tau=m.params["tau"], b1_p=m.pvalues["tau"],
                              b3_per_sd_tau_per_sd_O=m.params["tauO"], b3_p=m.pvalues["tauO"],
                              b3_over_b1_pct=100 * m.params["tauO"] / m.params["tau"]))

    A, B = pd.DataFrame(rowsA), pd.DataFrame(rowsB)
    A.to_csv(OUT_A, index=False, float_format="%.4g")
    B.to_csv(OUT_B, index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(f"wrote {OUT_A}\nwrote {OUT_B}\n")
    print("A. log var(dSIA') vs ocean index (per SD of season-mean SST anomaly)")
    print(A[["sector", "season", "response", "corr_O_vs_step", "O_slope_pct_per_sd", "O_p",
             "AIC_step", "AIC_O", "AIC_O_step", "step_pct_given_O", "step_p_given_O",
             "O_p_given_step"]].round(3).to_string(index=False))
    print("\nB. wind sensitivity conditioned on ocean state (b3 = change in b1 per SD of ocean warmth)")
    print(B.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
