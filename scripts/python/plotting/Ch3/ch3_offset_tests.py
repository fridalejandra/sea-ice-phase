#!/usr/bin/env python
# ch3_offset_tests.py -- two tests of the "displaced annual cycle" claim, from the daily sector table.
#
# TEST 3  Shape of the change: mean daily cycle 2016-2024 minus mean daily cycle 1979-2015, by sector
#         and day of year, with a year-block bootstrap band.  Flat curve = pure offset (same loss every
#         day); deep in summer / shallow in winter = amplitude change.  -> fig_ch3_offset_doy.png
#
# TEST 1  Hobbs et al. (2024) statistics reproduced, then recomputed with the offset removed:
#         (a) std of DJF mean anomaly, 1979-2006 vs 2007-2022 (their numbers: 0.31 -> 0.76 for total SIA)
#         (b) lag-1/3/6-month autocorrelation of monthly anomalies in running 120-month windows
#         (c) correlation of SON with following DJF anomaly, 1979-2006 vs 2007-2022 (theirs: 0.14 -> 0.88)
#         Each statistic is computed twice: on the conventional anomaly (fixed 1979-2018 climatology,
#         as Hobbs) and on the anomaly after removing each year's own annual mean (the offset).
#         -> table ch3_hobbs_with_without_offset.csv and fig_ch3_hobbs_offset.png
#
# Input: a daily table with columns date, sector, and an extent/area column (default the Chapter 3/4
#        table data/merged/analysis_table_daily_anomaly_periodclim.csv, column SIA).  Change IN_CSV /
#        COL below if the circumpolar-total daily extent lives elsewhere.
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_figstyle as fs

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
IN_CSV = f"{ROOT}/data/merged/analysis_table_daily_anomaly_periodclim.csv"
COL = "SIA"                       # daily extent or area, same units for all sectors
OUT_TAB = f"{ROOT}/results/ch3/tables"; OUT_FIG = f"{ROOT}/results/ch3/figures"
os.makedirs(OUT_TAB, exist_ok=True); os.makedirs(OUT_FIG, exist_ok=True)
PRE = (1979, 2015); POST = (2016, 2024)
NBOOT, SEED = 2000, 42
LONG2CODE = {"Weddell": "WS", "King Haakon VII": "KH", "East Antarctica": "EA",
             "Ross-Amundsen": "RA", "Amundsen-Bellingshausen": "ABS"}


def load():
    df = pd.read_csv(IN_CSV, parse_dates=["date"]).dropna(subset=[COL])
    df["code"] = df.sector.map(LONG2CODE).fillna(df.sector)
    df["year"] = df.date.dt.year; df["doy"] = df.date.dt.dayofyear.clip(upper=365)
    df["month"] = df.date.dt.month
    tot = df.groupby("date")[COL].sum().reset_index()           # circumpolar total = sum of sectors
    tot["code"] = "Total"; tot["year"] = tot.date.dt.year; tot["doy"] = tot.date.dt.dayofyear.clip(upper=365)
    tot["month"] = tot.date.dt.month
    return pd.concat([df[["date", "code", "year", "doy", "month", COL]], tot[["date", "code", "year", "doy", "month", COL]]])


# ------------------------------------------------------------------ TEST 3: offset by day of year
def test3(df):
    rng = np.random.default_rng(SEED)
    rows = []
    codes = ["Total"] + fs.SECTOR_ORDER
    fig, axes = plt.subplots(2, 3, figsize=fs.size(1.0, 0.62), sharex=True)
    for ax, code in zip(axes.ravel(), codes):
        g = df[df.code == code]
        cyc = g.pivot_table(index="year", columns="doy", values=COL)      # year x doy
        cyc = cyc.interpolate(axis=1, limit_direction="both")
        pre = cyc.loc[PRE[0]:PRE[1]].dropna(how="all"); post = cyc.loc[POST[0]:POST[1]].dropna(how="all")
        diff = post.mean() - pre.mean()
        boots = np.empty((NBOOT, cyc.shape[1]))
        for b in range(NBOOT):
            bp = pre.iloc[rng.integers(0, len(pre), len(pre))].mean(); bq = post.iloc[rng.integers(0, len(post), len(post))].mean()
            boots[b] = (bq - bp).values
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        doy = diff.index.values
        c = fs.SECTOR_COLORS.get(code, fs.INK)
        ax.fill_between(doy, lo, hi, color=c, alpha=0.2, lw=0)
        ax.plot(doy, diff.values, color=c, lw=fs.lw(0.8))
        ax.axhline(diff.mean(), color=fs.INK, lw=0.7, ls="--")
        ax.axhline(0, color=fs.INK, lw=0.5)
        ax.set_title(fs.SECTOR_NAME.get(code, "Circumpolar total"), loc="left")
        ax.set_xticks([1, 91, 182, 274, 365]); ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Jan"])
        # summary numbers: mean offset, and the ratio of the seasonal range of the change to its mean
        rows.append(dict(sector=code, mean_change=diff.mean(), min_change=diff.min(), max_change=diff.max(),
                         doy_of_min=int(doy[diff.values.argmin()]), doy_of_max=int(doy[diff.values.argmax()]),
                         range_over_mean=(diff.max() - diff.min()) / abs(diff.mean()) if diff.mean() != 0 else np.nan))
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Change in daily {COL}\n(2016–2024 minus 1979–2015)")
    for ax in axes[-1]:
        ax.set_xlabel("Day of year")
    h = [Line2D([0], [0], color="#555", lw=fs.lw(0.8), label="post minus pre mean cycle"),
         Line2D([0], [0], color="#555", lw=6, alpha=0.2, label="95% year-block bootstrap"),
         Line2D([0], [0], color=fs.INK, lw=0.7, ls="--", label="annual-mean change (a pure offset would follow this line)")]
    fs.legend_below(fig, h, ncol=3, y=0.0)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    fs.FIG_ROOT = OUT_FIG
    fs.save(fig, "fig_ch3_offset_doy")
    tab = pd.DataFrame(rows); tab.to_csv(f"{OUT_TAB}/ch3_offset_by_doy.csv", index=False)
    print("\nTEST 3  change in the daily cycle, 2016-2024 minus 1979-2015  (range_over_mean ~0 = pure offset)")
    print(tab.round(3).to_string(index=False))


# ------------------------------------------------------------------ TEST 1: Hobbs statistics +/- offset
def monthly_anoms(df, code, remove_offset):
    g = df[df.code == code].copy()
    if remove_offset:                                   # subtract each year's own annual mean
        g[COL] = g[COL] - g.groupby("year")[COL].transform("mean")
    m = g.groupby(["year", "month"])[COL].mean().reset_index()
    clim = m[(m.year >= 1979) & (m.year <= 2018)].groupby("month")[COL].mean()   # Hobbs: 1979-2018 climatology
    m["anom"] = m[COL] - m.month.map(clim)
    m["t"] = pd.to_datetime(dict(year=m.year, month=m.month, day=15))
    return m.sort_values("t").reset_index(drop=True)


def hobbs_stats(m):
    # (a) DJF mean anomaly per summer (Dec of y-1, Jan, Feb of y)
    s = m.copy(); s["syear"] = s.year + (s.month == 12)
    djf = s[s.month.isin([12, 1, 2])].groupby("syear").anom.mean()
    son = s[s.month.isin([9, 10, 11])].groupby("year").anom.mean()
    sd_pre, sd_post = djf.loc[1979:2006].std(), djf.loc[2007:2022].std()
    # (c) SON -> following DJF correlation
    pair = pd.DataFrame({"son": son, "djf": djf.reindex(son.index + 1).values}, index=son.index).dropna()
    r_pre = pair.loc[1979:2006].corr().iloc[0, 1]; r_post = pair.loc[2007:2022].corr().iloc[0, 1]
    # (b) running 120-month autocorrelation at lags 1, 3, 6 (last window vs first)
    a = m.anom.values; out = {}
    for lag in (1, 3, 6):
        ac = []
        for i in range(0, len(a) - 120 + 1):
            w = a[i:i + 120]; ac.append(np.corrcoef(w[:-lag], w[lag:])[0, 1])
        ac = np.array(ac); out[f"acf{lag}_first"] = ac[:60].mean(); out[f"acf{lag}_last"] = ac[-60:].mean()
    return dict(sd_djf_1979_2006=sd_pre, sd_djf_2007_2022=sd_post, var_ratio=(sd_post / sd_pre) ** 2,
                r_son_djf_1979_2006=r_pre, r_son_djf_2007_2022=r_post, **out)


def test1(df):
    rows = []
    for code in ["Total"] + fs.SECTOR_ORDER:
        for remove in (False, True):
            st = hobbs_stats(monthly_anoms(df, code, remove))
            rows.append(dict(sector=code, offset_removed=remove, **st))
    tab = pd.DataFrame(rows); tab.to_csv(f"{OUT_TAB}/ch3_hobbs_with_without_offset.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nTEST 1  Hobbs et al. statistics on the conventional anomaly (offset_removed=False) and after removing each year's annual mean (True)")
    print("        Hobbs, total SIA: sd 0.31 -> 0.76 (var ratio ~6); SON->DJF r 0.14 -> 0.88; acf6 ~0 -> >0.7")
    print(tab.round(2).to_string(index=False))
    # figure: three bars per sector, with/without offset
    fig, axes = plt.subplots(1, 3, figsize=fs.size(1.0, 0.36))
    codes = ["Total"] + fs.SECTOR_ORDER; x = np.arange(len(codes))
    for ax, (key, lab) in zip(axes, (("var_ratio", "DJF variance ratio\n2007–22 / 1979–2006"),
                                     ("r_son_djf_2007_2022", "r(SON, next DJF)\n2007–2022"),
                                     ("acf6_last", "lag-6 autocorrelation\nlast 5 yr of windows"))):
        for k, remove in enumerate((False, True)):
            v = [tab[(tab.sector == c) & (tab.offset_removed == remove)][key].values[0] for c in codes]
            ax.bar(x + (k - 0.5) * 0.36, v, 0.36, color=[fs.SECTOR_COLORS.get(c, "#555") for c in codes],
                   alpha=1.0 if not remove else 0.4, edgecolor="none")
        ax.set_xticks(x); ax.set_xticklabels(["Tot"] + fs.SECTOR_ORDER); ax.set_title(lab, loc="left")
        ax.axhline(1 if key == "var_ratio" else 0, color=fs.INK, lw=0.5)
    h = [Line2D([0], [0], color="#555", lw=8, label="conventional anomaly (fixed climatology)"),
         Line2D([0], [0], color="#555", lw=8, alpha=0.4, label="offset removed (each year's own annual mean subtracted)")]
    fs.legend_below(fig, h, ncol=2, y=-0.02)
    fs.FIG_ROOT = OUT_FIG
    fs.save(fig, "fig_ch3_hobbs_offset")


if __name__ == "__main__":
    fs.set_mode("manuscript")
    df = load()
    print(f"loaded {IN_CSV}: {df.year.min()}-{df.year.max()}, sectors {sorted(df.code.unique())}")
    test3(df)
    test1(df)
