#!/usr/bin/env python3
"""
test_episode_accumulation.py -- do residual episodes add up to a change in the
cycle?

The decomposition separates by timescale: sub-monthly departures stay in the
residual, anything sustained is booked as amplitude or phase. That says where
an accumulated effect SHOWS, not whether weather-scale episodes produced it.
This script uses the residual event list as the forcing and asks whether years
with more, or more one-signed, episodes have a different cycle.

For each sector and year:
    n_adv, n_ret        number of level events peaking in the advance
                        (Mar-Aug) and retreat (Oct-Jan, Jan counted with the
                        preceding year) seasons
    net_adv, net_ret    sum of signed integrated z over those events -- the
                        year's net episodic push, negative = edge below cycle
    abs_adv, abs_ret    sum of |integrated z| -- how eventful the season was
and correlates each, linearly detrended, with that year's observed amplitude
anomaly and day-of-maximum / day-of-minimum anomalies (annual_params.csv,
period == FULL). Pearson and Spearman both printed; n = 36 (1988-2023), so
|r| > 0.33 is p < 0.05 for a single test, and there are 6 predictors x 3
targets x 6 series, so read the pattern, not the stars.

Predictions if episodes accumulate into the cycle:
    net_adv < 0  ->  smaller amplitude, earlier maximum
    net_ret < 0  ->  earlier minimum (of the following February)
If nothing correlates, episodes do not accumulate through this channel, which
is the constraint the two-week e-folding time already implies.

Also prints: events per year, 2016 and 2023 against the distribution, and a
pre/post-2016 comparison of counts (is the record more eventful since 2016?).

Inputs   residual_events_level.csv  (find_residual_events.py; sector, start,
         end, peak_date, days, peak_z, integrated_z, sign)
         annual_params.csv          (period == FULL)
Outputs  results/ch3/tables/t36_episode_accumulation.csv   per sector-year
         results/ch3/tables/t36_episode_accumulation_corr.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import ANNUAL_CSV, SECTORS, SECTOR_LABELS, TABLES_DIR

EVENTS = os.environ.get("EVENTS_CSV", os.path.join(HERE, "residual_events_level.csv"))
Y0, Y1 = 1988, 2023
TARGETS = {"amplitude": "amplitude_raw_anom", "day of max": "max_doy_raw_anom",
           "day of min": "min_doy_raw_anom"}


def detrend(x):
    x = np.asarray(x, float); t = np.arange(len(x), dtype=float)
    ok = np.isfinite(x)
    b, a = np.polyfit(t[ok], x[ok], 1)
    return x - (a + b * t)


def corr_dt(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return (np.nan,) * 4
    xd, yd = detrend(x[ok]), detrend(y[ok])
    r, p = stats.pearsonr(xd, yd)
    rho, ps = stats.spearmanr(xd, yd)
    return r, p, rho, ps


# ── events -> season, year ───────────────────────────────────────────────────
ev = pd.read_csv(EVENTS, parse_dates=["start", "end", "peak_date"])
ev = ev[(ev["peak_date"].dt.year >= Y0) & (ev["end"] <= f"{Y1}-12-31")].copy()
m = ev["peak_date"].dt.month
ev["season"] = np.where(m.between(3, 8), "adv", np.where((m >= 10) | (m == 1), "ret", "other"))
ev["year"] = ev["peak_date"].dt.year - (m == 1).astype(int)   # Jan goes with the preceding retreat
ev["od"] = m.between(10, 12)   # October-December only: > 6 weeks before the next minimum,
                               # beyond the ice edge's own two-week memory
ev = ev[ev["season"] != "other"]

rows = []
for s in SECTORS:
    for y in range(Y0, Y1 + 1):
        e = ev[(ev["sector"] == s) & (ev["year"] == y)]
        rec = dict(sector=s, Year=y, n_all=len(e), abs_all=e["integrated_z"].abs().sum(),
                   net_all=e["integrated_z"].sum())
        for sea in ("adv", "ret"):
            q = e[e["season"] == sea]
            rec[f"n_{sea}"] = len(q)
            rec[f"net_{sea}"] = q["integrated_z"].sum()
            rec[f"abs_{sea}"] = q["integrated_z"].abs().sum()
        q = e[e["od"]]
        rec["net_od"] = q["integrated_z"].sum()          # Oct-Dec push only
        rows.append(rec)
per = pd.DataFrame(rows)

ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"].astype(str) == "FULL"]
# the raw maximum extent, if the file carries it: the cleanest "next winter" target,
# since amplitude Y+1 contains the Y+1 minimum that the retreat push sits next to
MAX_COL = next((c for c in ("max_extent_raw", "max_extent", "max_sie_raw", "max_sie", "maximum_raw")
                if c in ann.columns), None)
if MAX_COL is None:
    # derive it: the largest daily extent of each calendar year, per sector
    try:
        import ch3_data as D
        dd = D.load_daily(period="FULL")
        mx = (dd.groupby(["sector", "Year"])["Extent"].max().rename("max_extent_raw").reset_index())
        ann = ann.merge(mx, on=["sector", "Year"], how="left")
        MAX_COL = "max_extent_raw"
        print("raw maximum extent derived from daily_fitted.csv (max of Extent per sector-year)")
    except Exception as exc:
        print(f"could not derive the raw maximum from the daily file ({exc}); next-winter test uses day of max only")
if MAX_COL:
    TARGETS["max extent"] = MAX_COL
ann = ann[["sector", "Year"] + list(TARGETS.values())]
per = per.merge(ann, on=["sector", "Year"], how="left")
os.makedirs(TABLES_DIR, exist_ok=True)
per.to_csv(os.path.join(TABLES_DIR, "t36_episode_accumulation.csv"), index=False)

# ── how eventful were 2016 and 2023? ────────────────────────────────────────
print(f"level events per sector-year, {Y0}-{Y1} (advance + retreat seasons)")
print(f"{'sector':26s} {'mean':>5s} {'sd':>5s} {'max':>4s}   2016 (net adv/ret)   2023 (net adv/ret)   pre-2016 mean  2016+ mean  p")
for s in SECTORS:
    g = per[per["sector"] == s].set_index("Year")
    pre, post = g.loc[:2015, "n_all"], g.loc[2016:, "n_all"]
    p = stats.mannwhitneyu(pre, post).pvalue
    print(f"{SECTOR_LABELS[s]:26s} {g['n_all'].mean():5.1f} {g['n_all'].std():5.1f} {g['n_all'].max():4d}   "
          f"{g.loc[2016,'n_all']:2d} ({g.loc[2016,'net_adv']:+5.0f}/{g.loc[2016,'net_ret']:+5.0f})   "
          f"{g.loc[2023,'n_all']:2d} ({g.loc[2023,'net_adv']:+5.0f}/{g.loc[2023,'net_ret']:+5.0f})   "
          f"{pre.mean():6.1f}        {post.mean():5.1f}   {p:.2f}")

# ── do episodes accumulate into the cycle? ───────────────────────────────────
PRED = ["n_adv", "net_adv", "abs_adv", "n_ret", "net_ret", "abs_ret"]
out = []
print("\ndetrended correlation of episode statistics with the cycle, by sector (r / Spearman rho; * p<0.05)")
for tname, tcol in TARGETS.items():
    print(f"\n  target: {tname}")
    print(f"  {'sector':26s}" + "".join(f"{p:>16s}" for p in PRED))
    for s in SECTORS:
        g = per[per["sector"] == s].sort_values("Year")
        line = f"  {SECTOR_LABELS[s]:26s}"
        for pcol in PRED:
            r, p, rho, ps = corr_dt(g[pcol].values, g[tcol].values)
            line += f"{r:+6.2f}{'*' if p < 0.05 else ' '}/{rho:+5.2f}{'*' if ps < 0.05 else ' '}  "
            out.append(dict(target=tname, sector=SECTOR_LABELS[s], predictor=pcol, r=r, p=p, rho=rho, p_rho=ps))
        print(line)

# pooled across the five sectors (sector-years stacked, each sector detrended)
print("\npooled over the five sectors (each sector detrended separately, then stacked; n = 5 x 36)")
five = [s for s in SECTORS if "circumpolar" not in s]
for tname, tcol in TARGETS.items():
    line = f"  {tname:12s}"
    for pcol in PRED:
        xs, ys = [], []
        for s in five:
            g = per[per["sector"] == s].sort_values("Year")
            x, y = g[pcol].values.astype(float), g[tcol].values.astype(float)
            ok = np.isfinite(x) & np.isfinite(y)
            xs.append(detrend(x[ok]) / np.nanstd(x[ok])); ys.append(detrend(y[ok]) / np.nanstd(y[ok]))
        x, y = np.concatenate(xs), np.concatenate(ys)
        r, p = stats.pearsonr(x, y); rho, ps = stats.spearmanr(x, y)
        line += f"  {pcol}: {r:+.2f}{'*' if p < 0.05 else ' '}/{rho:+.2f}{'*' if ps < 0.05 else ' '}"
        out.append(dict(target=tname, sector="pooled", predictor=pcol, r=r, p=p, rho=rho, p_rho=ps))
    print(line)

# ── the ordered test: predictor strictly BEFORE target ──────────────────────
# Same-season pairs above are contaminated by leakage of the (shrunk) warp into
# the residual, and by construction that runs cycle -> residual. Here only pairs
# in which the episodes precede the cycle statistic are kept:
#   advance push of Y   -> day of max of Y, amplitude of Y      (same year; max follows the advance)
#   retreat push of Y   -> day of min of Y+1, amplitude of Y+1, day of max of Y+1   (through the summer)
#   advance push of Y   -> the same three of Y+1                (a full year's memory)
# The Y+1 rows are the ocean-memory hypothesis (Eabry et al. 2025: lingering
# upper-ocean warmth delaying the next growth season).
print("\n" + "=" * 100)
print("ORDERED TEST: episodes strictly before the cycle statistic (detrended r / Spearman rho; * p<0.05)")
print("=" * 100)
per = per.sort_values(["sector", "Year"])
for tcol in TARGETS.values():
    per[tcol + "_next"] = per.groupby("sector")[tcol].shift(-1)
ORDERED = [("net_adv", "max_doy_raw_anom",        "advance push Y  -> day of max Y"),
           ("net_adv", "amplitude_raw_anom",      "advance push Y  -> amplitude Y"),
           ("net_ret", "min_doy_raw_anom_next",   "retreat push Y  -> day of min Y+1"),
           ("net_ret", "amplitude_raw_anom_next", "retreat push Y  -> amplitude Y+1"),
           ("net_ret", "max_doy_raw_anom_next",   "retreat push Y  -> day of max Y+1"),
           ("net_adv", "min_doy_raw_anom_next",   "advance push Y  -> day of min Y+1"),
           ("net_adv", "amplitude_raw_anom_next", "advance push Y  -> amplitude Y+1"),
           ("abs_all", "amplitude_raw_anom_next", "eventfulness Y  -> amplitude Y+1"),
           # the memory test proper: Oct-Dec push only, > 6 weeks before the next minimum
           ("net_od",  "min_doy_raw_anom_next",   "Oct-Dec push Y  -> day of min Y+1"),
           ("net_od",  "amplitude_raw_anom_next", "Oct-Dec push Y  -> amplitude Y+1"),
           ("net_od",  "max_doy_raw_anom_next",   "Oct-Dec push Y  -> day of max Y+1")]
if MAX_COL:
    ORDERED += [("net_od",  MAX_COL + "_next", "Oct-Dec push Y  -> max extent Y+1"),
                ("net_ret", MAX_COL + "_next", "retreat push Y  -> max extent Y+1")]
ord_rows = []
print(f"{'pair':38s}" + "".join(f"{SECTOR_LABELS[s]:>15s}" for s in SECTORS) + f"{'pooled(5)':>15s}")
for pcol, tcol, label in ORDERED:
    line = f"{label:38s}"
    for s in SECTORS:
        g = per[per["sector"] == s]
        r, p, rho, ps = corr_dt(g[pcol].values.astype(float), g[tcol].values.astype(float))
        line += f"{r:+6.2f}{'*' if p < 0.05 else ' '}/{rho:+5.2f}{'*' if ps < 0.05 else ' '} "
        ord_rows.append(dict(pair=label, sector=SECTOR_LABELS[s], r=r, p=p, rho=rho, p_rho=ps))
    xs, ys = [], []
    for s in five:
        g = per[per["sector"] == s]
        x, y = g[pcol].values.astype(float), g[tcol].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        xs.append(detrend(x[ok]) / np.nanstd(x[ok])); ys.append(detrend(y[ok]) / np.nanstd(y[ok]))
    x, y = np.concatenate(xs), np.concatenate(ys)
    r, p = stats.pearsonr(x, y); rho, ps = stats.spearmanr(x, y)
    line += f"{r:+6.2f}{'*' if p < 0.05 else ' '}/{rho:+5.2f}{'*' if ps < 0.05 else ' '}"
    ord_rows.append(dict(pair=label, sector="pooled", r=r, p=p, rho=rho, p_rho=ps))
    print(line)
print("\nsigns if episodes accumulate: a negative push (edge below its cycle) -> smaller amplitude (r > 0),\n"
      "and, through the ocean, a later next minimum (r < 0) and a smaller next amplitude (r > 0).")
pd.DataFrame(ord_rows).to_csv(os.path.join(TABLES_DIR, "t36_episode_accumulation_ordered.csv"), index=False)

# ── robustness of the memory result: Oct-Dec push -> next winter, pooled ────
# (a) 1988-2015 only, so a common post-2016 shift cannot be what correlates;
# (b) leave-one-year-out over the full period, worst p;
# (c) partial out the same year's amplitude, in case a big-cycle year simply
#     has a negative spring residual AND is followed by another big year.
def pooled_xy(pcol, tcol, years):
    xs, ys = [], []
    for s in five:
        g = per[(per["sector"] == s) & per["Year"].isin(years)].sort_values("Year")
        x, y = g[pcol].values.astype(float), g[tcol].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8:
            continue
        xs.append(detrend(x[ok]) / np.nanstd(x[ok])); ys.append(detrend(y[ok]) / np.nanstd(y[ok]))
    return np.concatenate(xs), np.concatenate(ys)

print("\nROBUSTNESS of  Oct-Dec push Y -> next winter  (pooled, five sectors)")
mem_targets = [("amplitude_raw_anom_next", "amplitude Y+1")]
if MAX_COL:
    mem_targets.append((MAX_COL + "_next", "max extent Y+1"))
all_years = list(range(Y0, Y1 + 1))
for tcol, tlab in mem_targets:
    x, y = pooled_xy("net_od", tcol, all_years)
    r_all, p_all = stats.pearsonr(x, y)
    x, y = pooled_xy("net_od", tcol, [y_ for y_ in all_years if y_ <= 2015])
    r_pre, p_pre = stats.pearsonr(x, y)
    worst = 0.0
    for drop in all_years:
        x, y = pooled_xy("net_od", tcol, [y_ for y_ in all_years if y_ != drop])
        worst = max(worst, stats.pearsonr(x, y)[1])
    # partial: regress both on the same year's amplitude anomaly first
    xs, ys = [], []
    for s in five:
        g = per[per["sector"] == s].sort_values("Year")
        x, y, z = (g["net_od"].values.astype(float), g[tcol].values.astype(float),
                   g["amplitude_raw_anom"].values.astype(float))
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = detrend(x[ok]), detrend(y[ok]), detrend(z[ok])
        rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = y - np.polyval(np.polyfit(z, y, 1), z)
        xs.append(rx / rx.std()); ys.append(ry / ry.std())
    r_part, p_part = stats.pearsonr(np.concatenate(xs), np.concatenate(ys))
    # (d) block bootstrap over YEARS: resample the 36 years with replacement,
    #     keeping all five sectors of a year together, so cross-sector dependence
    #     within a year is respected. 95 % interval of the pooled r; the pooled
    #     p above treats 180 sector-years as independent and is optimistic.
    rng = np.random.default_rng(7)
    byy = {}
    for s in five:
        g = per[per["sector"] == s].sort_values("Year")
        x, y = g["net_od"].values.astype(float), g[tcol].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        xd, yd = detrend(x[ok]) / np.nanstd(x[ok]), detrend(y[ok]) / np.nanstd(y[ok])
        for yr, xi, yi in zip(g["Year"].values[ok], xd, yd):
            byy.setdefault(int(yr), []).append((xi, yi))
    yrs = np.array(sorted(byy))
    boots = []
    for _ in range(2000):
        pick = rng.choice(yrs, len(yrs), replace=True)
        pts = np.array([p_ for yr in pick for p_ in byy[yr]])
        boots.append(np.corrcoef(pts[:, 0], pts[:, 1])[0, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    frac_le0 = float(np.mean(np.array(boots) <= 0))
    print(f"  -> {tlab:16s} full {r_all:+.2f} (p={p_all:.3f})   1988-2015 only {r_pre:+.2f} (p={p_pre:.3f})   "
          f"LOYO worst p={worst:.3f}   partial on same-year amplitude {r_part:+.2f} (p={p_part:.3f})")
    print(f"     year-block bootstrap (n_years = {len(yrs)}): r = {r_all:+.2f} [{lo:+.2f}, {hi:+.2f}]; "
          f"fraction of resamples with r <= 0: {frac_le0:.3f}")

# ── is this more than the known summer-to-winter persistence? ───────────────
# Extent anomalies are known to persist / re-emerge from spring into the next
# winter (Stammerjohn et al. 2012; Holland et al. 2013; Meehl et al. 2019;
# Libera et al. 2022). The question that is NOT known is whether the episodic
# structure of the spring -- how far and how often the edge crossed a threshold
# -- carries information beyond the seasonal-mean state. So: Oct-Dec seasonal
# mean of the SIE anomaly from the invariant cycle (the persistence literature's
# variable) and of the residual, per sector-year, and
#   (1) each alone against next winter,
#   (2) the episode push with the seasonal SIE anomaly partialled out,
#   (3) a two-predictor fit, standardised betas.
print("\n" + "=" * 100)
print("BEYOND PERSISTENCE: does the Oct-Dec episode push add to the Oct-Dec seasonal extent anomaly?")
print("=" * 100)
try:
    dd
except NameError:
    import ch3_data as D
    dd = D.load_daily(period="FULL")
dd["Date"] = pd.to_datetime(dd["Date"])
od = dd[dd["Date"].dt.month.between(10, 12)]
sea = (od.groupby(["sector", "Year"])
         .agg(sie_od=("anomaly_from_iac", "mean"), res_od=("residual_apac", "mean"))
         .reset_index())
per = per.merge(sea, on=["sector", "Year"], how="left")


def partial_r(x, y, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return stats.pearsonr(rx, ry)


def stacked(cols, tcol, sectors):
    X, Y = [], []
    for s in sectors:
        g = per[per["sector"] == s].sort_values("Year")
        M = g[cols + [tcol]].values.astype(float)
        ok = np.isfinite(M).all(axis=1)
        M = M[ok]
        M = np.column_stack([detrend(M[:, j]) / np.nanstd(M[:, j]) for j in range(M.shape[1])])
        X.append(M[:, :-1]); Y.append(M[:, -1])
    return np.vstack(X), np.concatenate(Y)


for tcol, tlab in mem_targets:
    print(f"\n  target: {tlab}  (pooled, five sectors, each detrended and standardised)")
    X, y = stacked(["net_od", "sie_od", "res_od"], tcol, five)
    r_ev, p_ev = stats.pearsonr(X[:, 0], y)
    r_sie, p_sie = stats.pearsonr(X[:, 1], y)
    r_res, p_res = stats.pearsonr(X[:, 2], y)
    r_ev_sie, p_ev_sie = partial_r(X[:, 0], y, X[:, 1])          # episodes | seasonal SIE anomaly
    r_sie_ev, p_sie_ev = partial_r(X[:, 1], y, X[:, 0])          # seasonal SIE anomaly | episodes
    r_ev_res, p_ev_res = partial_r(X[:, 0], y, X[:, 2])          # episodes | seasonal-mean residual
    A = np.column_stack([np.ones(len(y)), X[:, 0], X[:, 1]])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"    alone:   episode push {r_ev:+.2f} (p={p_ev:.3f})   seasonal SIE anomaly {r_sie:+.2f} (p={p_sie:.3f})   "
          f"seasonal-mean residual {r_res:+.2f} (p={p_res:.3f})")
    print(f"    partial: episode push | SIE anomaly {r_ev_sie:+.2f} (p={p_ev_sie:.3f})   "
          f"SIE anomaly | episode push {r_sie_ev:+.2f} (p={p_sie_ev:.3f})   "
          f"episode push | mean residual {r_ev_res:+.2f} (p={p_ev_res:.3f})")
    print(f"    joint fit, standardised betas: episode push {beta[1]:+.2f}, SIE anomaly {beta[2]:+.2f}")
    print(f"    corr(episode push, SIE anomaly) = {stats.pearsonr(X[:, 0], X[:, 1])[0]:+.2f}")
    print("    per sector, episode push | SIE anomaly:", end="")
    for s in SECTORS:
        Xs, ys = stacked(["net_od", "sie_od"], tcol, [s])
        r_, p_ = partial_r(Xs[:, 0], ys, Xs[:, 1])
        print(f"  {SECTOR_LABELS[s]} {r_:+.2f}{'*' if p_ < 0.05 else ''}", end="")
    print()
print("\n  reading: if 'episode push | SIE anomaly' stays near its unconditional value, the episodes carry\n"
      "  information the seasonal-mean state does not (preconditioning, Eabry et al. 2025), which is new.\n"
      "  If it collapses while 'SIE anomaly | episode push' holds, the result is the known persistence of\n"
      "  extent anomalies seen through the residual, and the contribution is the framing, not a predictor.")

pd.DataFrame(out).to_csv(os.path.join(TABLES_DIR, "t36_episode_accumulation_corr.csv"), index=False)
print("\nwrote t36_episode_accumulation.csv, t36_episode_accumulation_corr.csv")
print("\nreading guide: net_adv vs amplitude and day of max is the accumulation test; net_ret vs day of min "
      "the retreat-side one. A season's net episodic push predicting that year's cycle would mean episodes "
      "add up; no relationship means they decay before they can, which is what a two-week e-folding time implies.")