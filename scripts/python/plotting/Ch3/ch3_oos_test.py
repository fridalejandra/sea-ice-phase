#!/usr/bin/env python
# ch3_oos_test.py -- TEST 2: do the Table 3.3 atmosphere-cycle relationships still predict the post-2016 years?
#
# For each of the six relationships specified in advance (index, season, sector, cycle component), fit
#     y_t = a + b * x_t + c * year_t          on 1979-2015 only   (c removes the shared linear trend)
# then feed the OBSERVED 2016-2025 index values through the frozen fit and compare with what the ice did.
# Reported per relationship:
#     r_full        correlation over the full record (to check against Table 3.3; detrended)
#     r_pre, r_post correlation within each period (detrended within period)
#     rmse_pre_loo  leave-one-year-out prediction error on 1979-2015 (the fair in-sample yardstick)
#     rmse_post     out-of-sample prediction error on 2016-2025
#     p_rmse        fraction of 2000 draws of |pre residuals| (same count as post years) whose RMSE >= rmse_post
#                   (small p = the post years are predicted WORSE than the pre years; p ~ 0.5 = same skill)
#     bias_post     mean post residual (observed - predicted), in units of y, and in pre residual s.d.
#     n_outside     post years outside +/- 2 pre-residual s.d.
# Figure: fig_ch3_oos.png, one panel per relationship: observed (grey), predicted (colour), +/-2 s.d. band.
#
# Inputs (cluster paths): data/ch3/annual_params.csv (period FULL; amplitude_raw_yr, max_doy_raw),
#   data/indices/marshall_sam_monthly.txt, asli_era5_v3-latest.csv, nina34.data, ZW3_goyal_monthly.csv.
# The season-to-year convention for each index is chosen automatically from a short list of candidates
# as the one whose full-record r best matches Table 3.3; the choice is printed so it can be checked.
import os, sys, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_figstyle as fs

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
ANNUAL = f"{ROOT}/data/ch3/annual_params.csv"
IDX = f"{ROOT}/data/indices"
OUT_TAB = f"{ROOT}/results/ch3/tables"; OUT_FIG = f"{ROOT}/results/ch3/figures"
os.makedirs(OUT_TAB, exist_ok=True); os.makedirs(OUT_FIG, exist_ok=True)
PRE, POST = (1979, 2015), (2016, 2025)
NBOOT, SEED = 2000, 42
SEC = {"ABS": "SIE_Amundsen_Bellingshausen", "KH": "SIE_King_Haakon", "EA": "SIE_East_Antarctica",
       "WS": "SIE_Weddell", "RA": "SIE_Ross"}
# Table 3.3: (label, index, months, sector, y column, expected r over 1979-2025)
RELS = [("Nino3.4 (SON) -> ABS amplitude", "nino34", [9, 10, 11], "ABS", "amplitude_raw_yr", +0.42),
        ("SAM (JJA) -> WS amplitude", "sam", [6, 7, 8], "WS", "amplitude_raw_yr", -0.40),
        ("SAM (Oct-Jan) -> EA day of maximum", "sam", [10, 11, 12, 1], "EA", "max_doy_raw", +0.38),
        ("ZW3 (annual) -> ABS amplitude", "zw3", list(range(1, 13)), "ABS", "amplitude_raw_yr", -0.36),
        ("[search] Nino3.4 (annual) -> RA amplitude", "nino34", list(range(1, 13)), "RA", "amplitude_raw_yr", -0.42),
        ("[search] ZW3 (SON) -> RA day of maximum", "zw3", [9, 10, 11], "RA", "max_doy_raw", +0.40)]


# ----------------------------------------------------------------------------- index readers -> monthly series
def read_marshall(path):
    rows = []
    for line in open(path):
        t = line.split()
        if len(t) >= 13 and re.fullmatch(r"\d{4}", t[0]):
            rows += [(int(t[0]), m + 1, float(t[m + 1])) for m in range(12)]
    return pd.DataFrame(rows, columns=["year", "month", "v"])


def read_psl(path):            # NOAA PSL .data format (nina34.data)
    rows = []
    for line in open(path):
        t = line.split()
        if len(t) == 13 and re.fullmatch(r"\d{4}", t[0]):
            rows += [(int(t[0]), m + 1, float(t[m + 1])) for m in range(12)]
    df = pd.DataFrame(rows, columns=["year", "month", "v"]); df.loc[df.v <= -99, "v"] = np.nan
    return df


def read_asli(path):
    df = pd.read_csv(path, comment="#")
    tcol = [c for c in df.columns if c.lower() in ("time", "date")][0]
    vcol = [c for c in df.columns if "relcen" in c.lower().replace("_", "")]
    vcol = vcol[0] if vcol else [c for c in df.columns if "actcen" in c.lower().replace("_", "")][0]
    t = pd.to_datetime(df[tcol]); print(f"   ASL column used: {vcol}")
    return pd.DataFrame({"year": t.dt.year, "month": t.dt.month, "v": df[vcol].astype(float)})


def read_zw3(path):
    df = pd.read_csv(path, comment="#")
    print(f"   ZW3 file columns: {list(df.columns)}")
    cols = {c.lower(): c for c in df.columns}
    if "year" in cols and "month" in cols:
        y, m = df[cols["year"]].astype(int), df[cols["month"]].astype(int)
    else:
        tcol = [c for c in df.columns if c.lower() in ("time", "date")][0]; t = pd.to_datetime(df[tcol]); y, m = t.dt.year, t.dt.month
    vc = [c for c in df.columns if any(k in c.lower() for k in ("amp", "mag", "zw3_index", "index"))]
    vc = vc[0] if vc else [c for c in df.columns if c not in (cols.get("year"), cols.get("month")) and df[c].dtype != object][0]
    print(f"   ZW3 column used: {vc}")
    return pd.DataFrame({"year": y, "month": m, "v": df[vc].astype(float)})


def seasonal(monthly, months, lag):
    """Mean over `months`; months after December wrap into the NEXT calendar year.  lag=0: the season is
    assigned to the year in which it starts; lag=1: to the following year (i.e. the season precedes the ice year)."""
    d = monthly.copy()
    start = months[0]; wrap = [m for m in months if m < start]
    d["syear"] = d.year - d.month.isin(wrap).astype(int)       # Jan belonging to Oct-Jan goes with previous year
    d = d[d.month.isin(months)]
    s = d.groupby("syear").v.mean()
    s.index = s.index + lag
    return s


def detrend(y, x):
    ok = np.isfinite(y) & np.isfinite(x)
    return y - np.polyval(np.polyfit(x[ok], y[ok], 1), x)


def rcorr(a, b):
    ok = np.isfinite(a) & np.isfinite(b); return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 3 else np.nan


# ----------------------------------------------------------------------------- main
def main():
    fs.set_mode("manuscript"); fs.FIG_ROOT = OUT_FIG
    rng = np.random.default_rng(SEED)
    a = pd.read_csv(ANNUAL); a = a[a.period == "FULL"] if "period" in a.columns else a
    idx = {"sam": read_marshall(f"{IDX}/marshall_sam_monthly.txt"), "nino34": read_psl(f"{IDX}/nina34.data"),
           "asl": read_asli(f"{IDX}/asli_era5_v3-latest.csv"), "zw3": read_zw3(f"{IDX}/ZW3_raphael_monthly.csv")}
    for k, v in idx.items():
        print(f"   {k}: {v.year.min()}-{v.year.max()}, {len(v)} months")

    rows = []
    fig, axes = plt.subplots(3, 2, figsize=fs.size(1.0, 1.0), sharex=True)
    for ax, (label, iname, months, code, ycol, r_expected) in zip(axes.ravel(), RELS):
        g = a[a.sector == SEC[code]].set_index("Year").sort_index()
        y_all = g[ycol].astype(float)
        # choose the season-year convention that reproduces Table 3.3's full-record r
        best = None
        for lag in (0, 1):
            x_all = seasonal(idx[iname], months, lag).reindex(y_all.index)
            yrs = y_all.index.values.astype(float)
            r = rcorr(detrend(y_all.values, yrs), detrend(x_all.values, yrs))
            if best is None or abs(r - r_expected) < abs(best[1] - r_expected):
                best = (lag, r, x_all)
        lag, r_full, x_all = best
        yrs = y_all.index.values.astype(float)
        pre = (yrs >= PRE[0]) & (yrs <= PRE[1]); post = (yrs >= POST[0]) & (yrs <= POST[1])
        ok = np.isfinite(y_all.values) & np.isfinite(x_all.values)
        y, x = y_all.values, x_all.values
        # frozen fit on pre years: y = a + b x + c year
        Xp = np.column_stack([np.ones(pre.sum()), x[pre], yrs[pre]]); m = pre & ok
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), x[m], yrs[m]]), y[m], rcond=None)
        pred = beta[0] + beta[1] * x + beta[2] * np.minimum(yrs, PRE[1])   # trend held at its 2015 value after 2015
        res = y - pred
        # leave-one-year-out error in the pre period (fair in-sample yardstick)
        loo = []
        ipre = np.where(m)[0]
        for i in ipre:
            keep = ipre[ipre != i]
            b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(keep)), x[keep], yrs[keep]]), y[keep], rcond=None)
            loo.append(y[i] - (b[0] + b[1] * x[i] + b[2] * yrs[i]))
        loo = np.array(loo); rmse_pre = np.sqrt(np.mean(loo ** 2)); sd_pre = loo.std(ddof=1)
        q = post & ok; rp = res[q]; rmse_post = np.sqrt(np.mean(rp ** 2))
        draws = np.array([np.sqrt(np.mean(rng.choice(loo, len(rp)) ** 2)) for _ in range(NBOOT)])
        p_rmse = (draws >= rmse_post).mean()
        r_pre = rcorr(detrend(y[m], yrs[m]), detrend(x[m], yrs[m])); r_post = rcorr(detrend(y[q], yrs[q]), detrend(x[q], yrs[q]))
        rows.append(dict(relationship=label, season_lag=lag, r_full=r_full, r_table33=r_expected, r_pre=r_pre, r_post=r_post,
                         slope_pre=beta[1], rmse_pre_loo=rmse_pre, rmse_post=rmse_post, rmse_ratio=rmse_post / rmse_pre,
                         p_rmse=p_rmse, bias_post=rp.mean(), bias_post_sd=rp.mean() / sd_pre,
                         n_post=int(q.sum()), n_outside_2sd=int((np.abs(rp) > 2 * sd_pre).sum())))
        # panel
        c = fs.SECTOR_COLORS[code]
        ax.axvspan(POST[0] - 0.5, POST[1] + 0.5, color=fs.BAND, zorder=0)
        ax.fill_between(yrs, pred - 2 * sd_pre, pred + 2 * sd_pre, color=c, alpha=0.15, lw=0)
        ax.plot(yrs, pred, color=c, lw=fs.lw(0.8))
        ax.plot(yrs, y, "o-", color=fs.INK, lw=fs.lw(0.45), ms=fs.ms(0.4))
        ax.set_title(label, loc="left")
        ax.text(0.02, 0.04, f"r {r_full:+.2f} (Table 3.3 {r_expected:+.2f})   RMSE post/pre {rmse_post / rmse_pre:.2f}, p = {p_rmse:.2f}",
                transform=ax.transAxes, fontsize=plt.rcParams["legend.fontsize"] * 0.9, color=fs.INK)
        ax.set_ylabel("Amplitude (10$^6$ km$^2$)" if "amplitude" in ycol else "Day of maximum")
    for ax in axes[-1]:
        ax.set_xlabel("Year")
    h = [Line2D([0], [0], color=fs.INK, marker="o", ms=fs.ms(0.4), lw=fs.lw(0.45), label="observed"),
         Line2D([0], [0], color="#777", lw=fs.lw(0.8), label="predicted from the index (fit on 1979–2015 only)"),
         Line2D([0], [0], color="#777", lw=6, alpha=0.15, label="±2 s.d. of pre-2016 leave-one-out residuals")]
    fs.legend_below(fig, h, ncol=3, y=0.0)
    fig.subplots_adjust(hspace=0.45, wspace=0.3)
    fs.save(fig, "fig_ch3_oos")

    tab = pd.DataFrame(rows); tab.to_csv(f"{OUT_TAB}/ch3_oos_test.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nTEST 2  out-of-sample prediction of 2016-2025 from relationships fitted on 1979-2015")
    print("        rmse_ratio ~1 and p_rmse not small = the relationship predicts the new years as well as the old;")
    print("        bias_post_sd = mean post residual in pre s.d. (a systematic miss even if the scatter is unchanged)")
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
