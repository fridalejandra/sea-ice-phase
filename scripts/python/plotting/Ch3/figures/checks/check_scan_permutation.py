#!/usr/bin/env python3
"""
check_scan_permutation.py -- is "19 of 210 raw-anomaly cells at p < 0.05"
more than chance, once the overlap between the 210 tests is respected?

Why: the exploratory-search table compares each count with "about 10 by
chance" (0.05 x 210). That assumes 210 independent tests. They aren't: the
annual, Mar-Aug and Oct-Jan windows contain the standard seasons, and the
five indices are correlated with each other, so one relationship (real or
chance) gets counted several times. The real chance spread is wider than
binomial, and the right question is how often a count this high turns up
when there is no relationship at all.

Method: keep every ice series exactly as observed, and break the link to
the atmosphere by re-assigning the index YEARS -- the same reassignment for
every index and season at once, so the overlap between seasons and the
correlation between indices are kept intact. Then redo all 4 x 210
correlations and count p < 0.05, exactly as the scan does.
  1. random shuffle of years   (N_PERM times)  -- standard permutation null
  2. circular shift of years   (every shift)   -- also keeps any year-to-year
     persistence in the indices; only n-1 possible nulls, so its p is coarse
For each measure: observed count, the null's mean and 95th percentile, and
p = (1 + number of nulls with a count >= observed) / (1 + number of nulls).

Series are built the way the scan builds them (fig_relationships_heatmap.py's
series(): seasonal means of the daily anomaly for departure and raw anomaly,
detrended; the annual observed scalars for amplitude and day of maximum;
indices already detrended). The OBSERVED counts printed here should match
the 48 / 21 / 7 / 19 in the exploratory table -- if they differ by one or two,
that's the known scan-vs-rebuild mismatch, and the comparison with the null
is still like for like, because both use this script's own series.

Reads   DAILY_CSV, ANNUAL_CSV, INDEX_CSV, t37b_component_comparison_scan.csv (for the 210 cells)
Writes  results/ch3/tables/t_scan_permutation.csv
Run     python checks/check_scan_permutation.py        (from the figures folder; ~1-2 min)
"""
import os
import sys
import numpy as np; np.seterr(invalid="ignore", divide="ignore")
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # ch3_config lives one folder up
sys.path.insert(0, HERE)                           # ...or here, if run from the figures folder
from ch3_config import TABLES_DIR, DAILY_CSV, ANNUAL_CSV, INDEX_CSV, SECTORS_COMPUTE, YEAR_MIN, YEAR_MAX

N_PERM = 1000
SEED = 20261004
ALPHA = 0.05
MEAS = ["sie_anom", "amplitude", "max_doy", "raw_anom"]
MNAME = {"sie_anom": "departure (SIE anomaly)", "amplitude": "amplitude",
         "max_doy": "phase (day of max)", "raw_anom": "raw anomaly"}
MONTHS = {"annual": set(range(1, 13)), "DJF": {12, 1, 2}, "MAM": {3, 4, 5}, "JJA": {6, 7, 8},
          "SON": {9, 10, 11}, "ADV": set(range(3, 9)), "RET": {10, 11, 12, 1}}
KEY = {"SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "ABS", "Amundsen-Bellingshausen": "ABS",
       "Amundsen_Bellingshausen": "ABS", "SIE_Ross": "Ross", "SIE_East_Antarctica": "East Antarctica",
       "East_Antarctica": "East Antarctica", "SIE_King_Haakon": "King Haakon",
       "King_Haakon": "King Haakon", "SIE_circumpolar": "circumpolar", "Circumpolar": "circumpolar"}

# ---- the 210 cells, taken from the scan table itself ------------------------
scan = pd.read_csv(os.path.join(TABLES_DIR, "t37b_component_comparison_scan.csv"))
scan["scan_label"] = scan["sector"]
scan["sector"] = scan["sector"].map(lambda s: KEY.get(s, s))
LABEL = dict(zip(scan["sector"], scan["scan_label"]))
cells = scan[["sector", "index_base", "season"]].drop_duplicates().values.tolist()
print(f"{len(cells)} cells x {len(MEAS)} measures, years {YEAR_MIN}-{YEAR_MAX}")

# ---- data -------------------------------------------------------------------
daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
if "period" in daily.columns:
    daily = daily[daily["period"] == "FULL"]
daily = daily[daily["Date"].dt.year.between(YEAR_MIN, YEAR_MAX + 1)].copy()
daily["Year"] = daily["Date"].dt.year
daily["month"] = daily["Date"].dt.month
_dl = {str(x).lower(): x for x in daily["sector"].unique()}
idx = pd.read_csv(INDEX_CSV)
idx = idx[idx["Year"].between(YEAR_MIN, YEAR_MAX)].set_index("Year")
ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"] == "FULL"]
INV = {v: k for k, v in SECTORS_COMPUTE.items()}
YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1)


def detrend(x):
    x = np.asarray(x, float); ok = ~np.isnan(x)
    if ok.sum() < 3:
        return x
    tt = np.arange(len(x), dtype=float); m, b = np.polyfit(tt[ok], x[ok], 1)
    out = x.copy(); out[ok] = x[ok] - (m * tt[ok] + b)
    return out


def daily_for(lab):
    c = [lab, INV.get(lab, ""), lab.replace(" ", "_"), "SIE_" + lab.replace(" ", "_")]
    if "circ" in lab.lower():
        c += ["circumpolar", "Circumpolar", "SIE_circumpolar", "SIE_Circumpolar", "total"]
    for x in c:
        if x and x.lower() in _dl:
            return daily[daily["sector"] == _dl[x.lower()]]
    return daily.iloc[0:0]


def ice_series(sec, season, m):
    """yearly ice series for one cell and measure, as a Series indexed by Year"""
    lab = LABEL[sec]
    if m in ("sie_anom", "raw_anom"):
        d = daily_for(lab)
        d = d[d["month"].isin(MONTHS[season])].copy()
        d["sy"] = np.where((season == "RET") & (d["month"] == 1), d["Year"] - 1,
                           np.where((season == "DJF") & (d["month"] == 12), d["Year"] + 1, d["Year"]))
        g = d.groupby("sy")
        v = "anomaly_from_iac" if m == "sie_anom" else "residual_apac"
        sm = pd.DataFrame({"Year": g.size().index, "y": g[v].mean().values, "nd": g.size().values})
        sm = sm[(sm["nd"] >= 0.35 * sm["nd"].median()) & sm.Year.between(YEAR_MIN, YEAR_MAX)]
        return pd.Series(detrend(sm["y"].values), index=sm["Year"].values)
    a = ann[ann["sector"] == lab]
    if a.empty:
        a = ann[ann["sector"] == INV.get(lab, lab)]
    v = "amplitude_raw_anom" if m == "amplitude" else "max_doy_raw_anom"
    a = a[a.Year.between(YEAR_MIN, YEAR_MAX)]
    a = a.sort_values("Year"); return pd.Series(detrend(a[v].values.astype(float)), index=a["Year"].values)


# Precompute: for every cell x measure, the ice values on the full year grid
# (NaN where missing) and which index column it pairs with.
ICE = {m: [] for m in MEAS}      # list of arrays aligned to YEARS
COLS = []
for sec, base, season in cells:
    COLS.append(f"{base}_{season}")
    for m in MEAS:
        s = ice_series(sec, season, m)
        ICE[m].append(s.reindex(YEARS).values.astype(float))
ICE = {m: np.vstack(v) for m, v in ICE.items()}                     # (cells, years)
missing_cols = sorted({c for c in COLS if c not in idx.columns})
if missing_cols:
    sys.exit(f"index columns missing from INDEX_CSV: {missing_cols}")
IDX = np.vstack([idx[c].reindex(YEARS).values.astype(float) for c in COLS])   # (cells, years)


def count_hits(order):
    """number of cells with p < ALPHA per measure, with the index years taken in `order`"""
    X = IDX[:, order]
    out = {}
    for m in MEAS:
        Y = ICE[m]
        ok = ~(np.isnan(X) | np.isnan(Y))
        n = ok.sum(1)
        xm = np.where(ok, X, 0); ym = np.where(ok, Y, 0)
        mx = xm.sum(1) / n; my = ym.sum(1) / n
        dx = np.where(ok, X - mx[:, None], 0); dy = np.where(ok, Y - my[:, None], 0)
        r = (dx * dy).sum(1) / np.sqrt((dx ** 2).sum(1) * (dy ** 2).sum(1))
        tstat = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
        p = 2 * stats.t.sf(np.abs(tstat), n - 2)
        out[m] = int(np.sum((p < ALPHA) & (n >= 8)))
    return out


identity = np.arange(len(YEARS))
obs = count_hits(identity)
rng = np.random.default_rng(SEED)
perm = [count_hits(rng.permutation(len(YEARS))) for _ in range(N_PERM)]
shift = [count_hits(np.roll(identity, k)) for k in range(1, len(YEARS))]

rows = []
print(f"\nExpected by the binomial (independent tests): {ALPHA * len(cells):.1f}\n")
print(f"{'measure':26s} {'observed':>8s}   {'shuffle: mean  95th   p':>26s}   {'circular shift: mean  95th   p':>32s}")
for m in MEAS:
    a = np.array([d[m] for d in perm]); b = np.array([d[m] for d in shift])
    p_a = (1 + (a >= obs[m]).sum()) / (1 + len(a))
    p_b = (1 + (b >= obs[m]).sum()) / (1 + len(b))
    rows.append(dict(measure=MNAME[m], observed=obs[m],
                     shuffle_mean=a.mean(), shuffle_p95=np.percentile(a, 95), shuffle_p=p_a, n_shuffle=len(a),
                     shift_mean=b.mean(), shift_p95=np.percentile(b, 95), shift_p=p_b, n_shift=len(b)))
    print(f"{MNAME[m]:26s} {obs[m]:8d}   {a.mean():12.1f} {np.percentile(a, 95):5.0f} {p_a:6.3f}"
          f"   {b.mean():18.1f} {np.percentile(b, 95):5.0f} {p_b:6.3f}")
out = os.path.join(TABLES_DIR, "t_scan_permutation.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nwrote {out}")
print("Read it as: p < 0.05 means the count is higher than the overlapping tests would produce by chance.\n"
      "The circular-shift p can't go below 1/47 = 0.021, so treat it as a robustness check on the shuffle p.")
