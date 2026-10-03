#!/usr/bin/env python3
"""
check_asl_ross_departure.py -- DRAFT diagnostic, not a numbered chapter script.

Two checks, both following from the fig_04_relationships.py double-hit
finding (2026-10-03) that ASL's correlation with Ross amplitude collapses
(partial r +0.38 -> +0.04) once you control for Ross's overall departure
from the invariant cycle -- i.e. the "amplitude" framing in Table 3 / Fig. 6
may really be a departure/overall-anomaly effect wearing an amplitude label.

CHECK 1 -- does the pre/post-2001 nonstationarity (the actual 3.3 finding)
survive if you rebuild it on DEPARTURE instead of AMPLITUDE? Mirrors
ch3_stats.py's own Ross~ASL block (its t36_ross_asl_detail.csv, lines
420-440) exactly -- same pear()/zshift() machinery, same split years, same
seasons -- swapping only the y-variable, so the two tables are directly
comparable. If departure shows a comparably sharp break, the reframe from
"amplitude" to "departure" is clean. If it doesn't, that's a more
complicated conversation before touching the 3.3 prose.

CHECK 2 -- Frida's question: are the GAM's own trend/amplitude/phase/
residual components (ch3_config.py's canonical decomposition, used in
Fig. 3's per-cycle attribution) actually uncorrelated with each other
across years, or just constructed via sequential subtraction (which
doesn't guarantee empirical independence -- see Check 1's own collapse,
which was departure vs. amplitude, not even phase)? This reads the already-
computed t32_attribution_by_cycle.csv (from fig03_attribution_annual.py)
and reports the pairwise correlation of all four components, per sector.

NOTE on fitted vs. observed (the other half of Frida's question): NOT
rebuilt here on the fitted trend/amplitude/phase/residual_apac terms, on
purpose. ch3_stats.py's own docstring (methods 2.2.3) already settled this:
every correlation in the chapter deliberately uses the OBSERVED scalars
(amplitude_raw_anom, max_doy_raw_anom, etc.), specifically because
tS1_fitted_vs_raw.csv (Fig. S1) showed fitted and observed amplitude agree
at r=1.00 but fitted and observed day-of-max only agree at r=0.11-0.55 --
i.e. the fitted phase term is a poor proxy for the real, observed timing.
Switching this check to fitted components would cut against that existing,
evidenced decision for no benefit (amplitude: no difference either way;
phase: fitted is the less trustworthy of the two). Check 2 above is the
correct version of "are these really separate" -- component-vs-component,
not fitted-vs-observed.

Reads   results/ch3/tables/t36_ross_asl_detail.csv            (existing, amplitude-based -- for the side-by-side)
        results/ch3/data/ (DAILY_CSV, INDEX_CSV, via ch3_config)
        results/ch3/tables/t32_attribution_by_cycle.csv       (from fig03_attribution_annual.py)
Writes  results/ch3/tables/t36_ross_asl_detail_departure.csv  (Check 1, same shape as the existing t36 file)
        results/ch3/tables/t_component_correlation_matrix.csv (Check 2)
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR, DAILY_CSV, INDEX_CSV, SPLIT_YEAR, SEASONS

ALPHA = 0.05
MONTHS = {"annual": set(range(1, 13)), "DJF": {12, 1, 2}, "MAM": {3, 4, 5}, "JJA": {6, 7, 8},
          "SON": {9, 10, 11}, "ADV": set(range(3, 9)), "RET": {10, 11, 12, 1}}

# -- same detrend/Fisher-z machinery as ch3_stats.py, copied verbatim so the
# two tables are built the exact same way and stay directly comparable ----
def detrend(x):
    x = np.asarray(x, float); ok = np.isfinite(x)
    if ok.sum() < 3:
        return x
    t = np.arange(len(x), dtype=float); b, a = np.polyfit(t[ok], x[ok], 1)
    out = x.copy(); out[ok] = x[ok] - (a + b * t[ok]); return out


def fz(r): return 0.5 * np.log((1 + r) / (1 - r))


def pear(x, y, dt=True):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if dt:
        x, y = detrend(x), detrend(y)
    ok = np.isfinite(x) & np.isfinite(y); n = int(ok.sum())
    if n < 6:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x[ok], y[ok]); return float(r), float(p), n


def zshift(r1, n1, r2, n2):
    if not (np.isfinite(r1) and np.isfinite(r2)) or n1 < 4 or n2 < 4:
        return np.nan, np.nan
    z = (fz(r1) - fz(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(z), float(2 * stats.norm.sf(abs(z)))


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 1 -- Ross departure vs. ASL, same shape as t36_ross_asl_detail.csv
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 78)
print("CHECK 1: does the pre/post-2001 ASL~Ross break survive on DEPARTURE")
print("         instead of AMPLITUDE?")
print("=" * 78)

if not (os.path.exists(DAILY_CSV) and os.path.exists(INDEX_CSV)):
    sys.exit(f"Missing DAILY_CSV or INDEX_CSV -- check ch3_config.py paths.")

daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
if "period" in daily.columns:
    daily = daily[daily["period"] == "FULL"]
daily["month"] = daily["Date"].dt.month

# find the Ross rows the same fuzzy way fig_04_relationships.py's daily_for() does
_dl = {str(x).lower(): x for x in daily["sector"].unique()}
_ross_candidates = ["ross", "sie_ross"]
_ross_key = next((c for c in _ross_candidates if c in _dl), None)
if _ross_key is None:
    sys.exit(f"Could not find a Ross sector in DAILY_CSV's sector column "
              f"(saw: {sorted(daily['sector'].unique())}) -- check the name and add it above.")
dross = daily[daily["sector"] == _dl[_ross_key]].copy()

idx = pd.read_csv(INDEX_CSV)

detail = []
for s in SEASONS:
    col = f"ASL_{s}"
    if col not in idx.columns:
        continue
    d = dross[dross["month"].isin(MONTHS[s])].copy()
    d["sy"] = np.where((s == "RET") & (d["month"] == 1), d["Year"] - 1,
                       np.where((s == "DJF") & (d["month"] == 12), d["Year"] + 1, d["Year"]))
    g = d.groupby("sy")
    sm = pd.DataFrame({"Year": g.size().index, "y": g["anomaly_from_iac"].mean().values,
                        "nd": g.size().values})
    sm = sm[sm["nd"] >= 0.35 * sm["nd"].median()]
    sm = sm.merge(idx[["Year", col]], on="Year").dropna()
    a1, a2 = sm[sm.Year < SPLIT_YEAR], sm[sm.Year >= SPLIT_YEAR]
    r1, _, n1 = pear(a1[col], a1["y"]); r2, _, n2 = pear(a2[col], a2["y"])
    _, p = zshift(r1, n1, r2, n2)
    detail.append(dict(test="season", key=s, r_pre=r1, r_post=r2, p_shift=p, n_pre=n1, n_post=n2))

# full annual series, for the split-year sweep and LOO (mirrors ch3_stats.py lines 431-440)
d = dross.copy()
g = d.groupby("Year")
ann_dep = pd.DataFrame({"Year": g.size().index, "y": g["anomaly_from_iac"].mean().values,
                         "nd": g.size().values})
ann_dep = ann_dep[ann_dep["nd"] >= 0.35 * ann_dep["nd"].median()]
ann_dep = ann_dep.merge(idx[["Year", "ASL_annual"]], on="Year").dropna()

for yr in (1996, 2001, 2006, 2011, 2016):
    a1, a2 = ann_dep[ann_dep.Year < yr], ann_dep[ann_dep.Year >= yr]
    r1, _, n1 = pear(a1["ASL_annual"], a1["y"]); r2, _, n2 = pear(a2["ASL_annual"], a2["y"])
    _, p = zshift(r1, n1, r2, n2)
    detail.append(dict(test="split_sweep", key=yr, r_pre=r1, r_post=r2, p_shift=p, n_pre=n1, n_post=n2))

ps = []
for y in ann_dep.Year:
    b = ann_dep[ann_dep.Year != y]
    a1, a2 = b[b.Year < SPLIT_YEAR], b[b.Year >= SPLIT_YEAR]
    r1, _, n1 = pear(a1["ASL_annual"], a1["y"]); r2, _, n2 = pear(a2["ASL_annual"], a2["y"])
    ps.append(zshift(r1, n1, r2, n2)[1])
detail.append(dict(test="loo_shift", key="worst p", r_pre=np.nan, r_post=np.nan,
                    p_shift=max(ps) if ps else np.nan, n_pre=np.nan, n_post=np.nan))

DEP = pd.DataFrame(detail)
out1 = os.path.join(TABLES_DIR, "t36_ross_asl_detail_departure.csv")
DEP.to_csv(out1, index=False)
print(f"\nwrote {out1}")

ann_row = DEP[(DEP.test == "season") & (DEP.key == "annual")]
if len(ann_row):
    r1d, r2d = ann_row.iloc[0]["r_pre"], ann_row.iloc[0]["r_post"]
    pd_ = ann_row.iloc[0]["p_shift"]
    print(f"\nASL_annual ~ Ross DEPARTURE:  {r1d:+.2f} (pre-{SPLIT_YEAR}) -> {r2d:+.2f} (post), shift p={pd_:.3g}")

existing = os.path.join(TABLES_DIR, "t36_ross_asl_detail.csv")
if os.path.exists(existing):
    AMPT = pd.read_csv(existing)
    amp_row = AMPT[(AMPT.test == "season") & (AMPT.key == "annual")]
    if len(amp_row):
        r1a, r2a = amp_row.iloc[0]["r_pre"], amp_row.iloc[0]["r_post"]
        pa_ = amp_row.iloc[0]["p_shift"]
        print(f"ASL_annual ~ Ross AMPLITUDE: {r1a:+.2f} (pre-{SPLIT_YEAR}) -> {r2a:+.2f} (post), shift p={pa_:.3g}"
              "   [existing t36_ross_asl_detail.csv]")
        print("\n-> if these two lines tell a similar story (sharp break, similar sign and rough size),"
              "\n   the departure reframe is clean. If departure's break is much weaker or absent,"
              "\n   the pre/post-2001 story may be more specific to amplitude than today's partial-"
              "\n   correlation check implied -- worth a closer look before reframing 3.3/Fig. 6.")
else:
    print(f"\n(no existing {existing} found to compare against -- run ch3_stats.py first for the side-by-side)")


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 2 -- are the GAM's own trend/amplitude/phase/residual components
# empirically correlated with each other, sector by sector?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("CHECK 2: correlation between the GAM's own trend/amplitude/phase/")
print("         residual components, per sector (t32_attribution_by_cycle.csv)")
print("=" * 78)

t32_path = os.path.join(TABLES_DIR, "t32_attribution_by_cycle.csv")
if not os.path.exists(t32_path):
    print(f"\n{t32_path} not found -- run fig03_attribution_annual.py first. Skipping Check 2.")
else:
    t32 = pd.read_csv(t32_path)
    COMPS = ["trend_component", "amplitude_component", "phase_component", "residual_apac"]
    missing = [c for c in COMPS if c not in t32.columns]
    if missing:
        print(f"\nt32_attribution_by_cycle.csv is missing {missing} -- can't run Check 2 as written.")
    else:
        rows = []
        for sec, g in t32.groupby("sector"):
            for c1, c2 in itertools.combinations(COMPS, 2):
                r, p, n = pear(g[c1], g[c2], dt=False)  # already per-cycle values, not a time series to detrend
                rows.append(dict(sector=sec, pair=f"{c1}~{c2}", r=r, p=p, n=n))
        CC = pd.DataFrame(rows)
        out2 = os.path.join(TABLES_DIR, "t_component_correlation_matrix.csv")
        CC.to_csv(out2, index=False)
        print(f"\nwrote {out2}")
        print("\n(|r| > 0.3 flagged -- these pairs are not empirically independent, despite being")
        print(" built as separate terms in the GAM fit)")
        for sec, g in CC.groupby("sector"):
            flagged = g[g["r"].abs() > 0.3]
            line = "  ".join(f"{row.pair}: {row.r:+.2f}{'*' if abs(row.r) > 0.3 else ''}"
                             for row in g.itertuples())
            print(f"   {sec:30s} {line}")
        if (CC["r"].abs() > 0.3).any():
            print("\n-> at least one component pair is non-trivially correlated in at least one sector.")
            print("   Components being 'statistically separate' by construction (sequential subtraction")
            print("   in the GAM fit) does not guarantee they're empirically independent across years --")
            print("   same lesson as Check 1 / today's departure-vs-amplitude finding, one level down.")
        else:
            print("\n-> no component pair exceeds |r|=0.3 in any sector -- the four components look")
            print("   empirically close to independent here, which is reassuring for reading Fig. 4's")
            print("   per-component correlations at face value.")
