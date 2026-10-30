#!/usr/bin/env python
# ch3_joint_regression.py -- do SST and the atmospheric modes act on the same part of each
# component, or on separate parts?  Joint OLS on z-scored detrended series, 1979-2024.
#   dR2   : drop in R2 when that predictor is removed (its unique share)
#   dcvR2 : drop in leave-one-year-out R2 when it is removed (>0 = adds out-of-sample skill)
# "+mean": amplitude and day of max with the annual mean added as a predictor.
# Seasons fixed in advance: annual mean <- annual; amplitude <- SON; day of max <- JJA;
# pre-specified relationships override the season for that mode only.
import numpy as np, pandas as pd, xarray as xr
from scipy.stats import t as tdist
from ch3_mapping_change import (ANNUAL, IDX, SST, OUT, SEC, LON,
                                rd_table, rd_asl, rd_zw3, seasonal, detr)

ANN = list(range(1, 13))
COMP = {"annual mean": "sie_annual", "amplitude": "amplitude_raw_yr", "day of max": "max_doy_raw"}
DEF = {"annual mean": ANN, "amplitude": [9, 10, 11], "day of max": [6, 7, 8]}
SPEC = {("WS", "amplitude"): {"SAM": [6, 7, 8]},
        ("EA", "day of max"): {"SAM": [10, 11, 12, 1]},
        ("ABS", "amplitude"): {"Nino34": [9, 10, 11], "ZW3": ANN},
        ("RA", "amplitude"): {"Nino34": ANN},
        ("RA", "day of max"): {"ZW3": [9, 10, 11]}}
MODES = ["SAM", "Nino34", "ZW3", "ASLp", "ASLlon"]


def rd_asl_lon(path):
    d = pd.read_csv(path, comment="#"); tc = [c for c in d if c.lower() in ("time", "date")][0]
    vc = [c for c in d if c.lower() == "lon"][0]; t = pd.to_datetime(d[tc])
    return pd.DataFrame({"year": t.dt.year, "month": t.dt.month, "v": d[vc].astype(float)})


def sst_monthly():
    ds = xr.open_dataset(SST); sst = ds["sst"].where(ds["sst"] > -2).sel(latitude=slice(-55, -65))
    out = {}
    for s, (lo, hi) in LON.items():
        m = ((ds.longitude >= lo) & (ds.longitude <= hi)) if lo < hi else ((ds.longitude >= lo) | (ds.longitude <= hi))
        ser = sst.where(m, drop=True).mean(("latitude", "longitude")).to_pandas()
        ser.index = pd.to_datetime(ser.index)
        out[s] = pd.DataFrame({"year": ser.index.year, "month": ser.index.month, "v": ser.values})
    return out


def ols(df, cols):
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols]); y = df.y.values
    b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    dof = len(y) - X.shape[1]; s2 = (e @ e) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    p = 2 * (1 - tdist.cdf(np.abs(b / se), dof))
    return b[1:], p[1:], r2


def cv_r2(df, cols):
    y = df.y.values; pred = np.empty(len(y))
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        b, *_ = np.linalg.lstsq(X[k], y[k], rcond=None); pred[i] = X[i] @ b
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def run(df, cols, tag):
    b, p, r2 = ols(df, cols); cv = cv_r2(df, cols)
    row = dict(n=len(df), R2=r2, cvR2=cv)
    for j, c in enumerate(cols):
        rest = [x for x in cols if x != c]
        row[f"{c}_beta"] = b[j]; row[f"{c}_p"] = p[j]
        row[f"{c}_dR2"] = r2 - ols(df, rest)[2]
        row[f"{c}_dcvR2"] = cv - cv_r2(df, rest)
    row["model"] = tag
    return row


def main():
    a = pd.read_csv(ANNUAL); a = a[a.period == "FULL"] if "period" in a else a
    g = {k: a[a.sector == v].set_index("Year").sort_index() for k, v in SEC.items()}
    idx = {"SAM": rd_table(f"{IDX}/marshall_sam_monthly.txt"), "Nino34": rd_table(f"{IDX}/nina34.data"),
           "ZW3": rd_zw3(f"{IDX}/ZW3_raphael_monthly.csv"),
           "ASLp": rd_asl(f"{IDX}/asli_era5_v3-latest.csv"), "ASLlon": rd_asl_lon(f"{IDX}/asli_era5_v3-latest.csv")}
    sm = sst_monthly(); pd.set_option("display.width", 250)

    print("\n=== Phase departures from the solar timing: interannual SD of observed dates (days), 1979-2024 ===")
    sd = pd.DataFrame({s: dict(day_of_max=g[s].loc[1979:2024, "max_doy_raw"].std(),
                               day_of_min=g[s].loc[1979:2024, "min_doy_raw"].std()) for s in SEC}).T
    print(sd.round(1).to_string())

    rows = []
    for s in LON:
        for comp, col in COMP.items():
            seas = {m: DEF[comp] for m in MODES}; seas.update(SPEC.get((s, comp), {}))
            parts = [g[s][col].rename("y"), seasonal(sm[s], DEF[comp]).rename("SST")]
            parts += [seasonal(idx[m], seas[m]).rename(m) for m in MODES]
            if comp != "annual mean":
                parts.append(g[s]["sie_annual"].rename("MEAN"))
            df = pd.concat(parts, axis=1).loc[1979:2024].dropna().apply(detr)
            df = (df - df.mean()) / df.std()
            cols = ["SST"] + MODES
            rows.append(dict(sector=s, component=comp, **run(df, cols, "SST+modes")))
            if comp != "annual mean":
                rows.append(dict(sector=s, component=comp, **run(df, cols + ["MEAN"], "+mean")))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/ch3_joint_regression.csv", index=False)

    for tag in ("SST+modes", "+mean"):
        T = R[R.model == tag]; preds = ["SST"] + MODES + (["MEAN"] if tag == "+mean" else [])
        print(f"\n=== {tag}: unique R2 share (dR2); * p<0.05, ** p<0.01 in the joint model ===")
        out = T[["sector", "component", "n", "R2", "cvR2"]].copy().round(2)
        for c in preds:
            out[c] = [f"{d:.2f}{'**' if p < 0.01 else '*' if p < 0.05 else ''}" for d, p in zip(T[f"{c}_dR2"], T[f"{c}_p"])]
        print(out.to_string(index=False))
        print(f"--- {tag}: drop in leave-one-year-out R2 when removed (>0 = adds skill)")
        out = T[["sector", "component"]].copy()
        for c in preds:
            out[c] = T[f"{c}_dcvR2"].round(2).values
        print(out.to_string(index=False))
    print(f"\nwrote {OUT}/ch3_joint_regression.csv")


if __name__ == "__main__":
    main()
