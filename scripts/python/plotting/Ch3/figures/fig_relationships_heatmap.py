#!/usr/bin/env python3
"""
fig_04_relationships.py -- Fig. 4 and Table S2 from one scan table.
Replaces fig_04_component_comparison_heatmap.py.

Fig. 4, two blocks of rows, four columns (departure from the invariant cycle,
amplitude, phase = day of maximum, raw anomaly), 1979-2023, detrended:
  top     the seven relationships from the literature (Table 3), at their Table 3 season
  bottom  relationships FOUND IN THE SEARCH that pass the filter, each shown
          at its strongest season

The filter (all must hold):
  1. not a Table 3 pair: the sector and index are not one of the seven (those are
     the top block seen in another season, not new relationships)
  2. repeats: in one measure, p < 0.05 with the same sign in two seasons that share
     no months (MAM and DJF, Mar-Aug and Oct-Jan, ...). Overlapping seasons (annual
     with anything, Mar-Aug with MAM or JJA, Oct-Jan with SON or DJF) are largely the
     same data and do not count twice.
  4. survives removing any single year: recomputed from the yearly series, the
     correlation stays at p < 0.05 with each year left out in turn, in both seasons
     that satisfied rule 2.
  5. circumpolar rows are dropped when a sector with the same index is already in
     the figure (the total is the sum of the sectors, so it is that sector showing through).
(Rule 3, "or p < 0.01 in a single season", was considered and rejected: it lets
through about as many chance results as it adds.)
Every candidate that fails a rule is printed with the reason.

Table S2: every combination (sector x index x season) with r and p for all four
measures, and whether it appears in Table 3 or Fig. 4.

NOTE (flagged 2026-10-02, not yet addressed): every correlation here is
contemporaneous -- the index and the ice measure are matched to the same
season in the same year. No lag/lead structure is tested (e.g. does last
winter's ASL predict this year's Ross amplitude better than the same-year
ASL does). If a real relationship operates with a lag, this scan and Fig. 4
would under- or mis-state it. Worth a dedicated lagged-correlation pass
before treating an absence here as a genuine null.

Reads  results/ch3/tables/t37b_component_comparison_scan.csv (compute_fig05_component_comparison.py)
       DAILY_CSV, ANNUAL_CSV, INDEX_CSV (for rule 4, same as the compute script)
Writes results/ch3/figures/fig04_relationships.png
       results/ch3/figures/fig04_relationships_presentation.png (transparent bg, slides)
       results/ch3/tables/tS2_all_combinations.csv
       results/ch3/tables/t_fig04_rows.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR, OUTPUT_DIR, DAILY_CSV, ANNUAL_CSV, INDEX_CSV, SECTORS_COMPUTE
import ch3_style

# ---- diverging colormap for the heatmap, via the cblind package -----------
# cblind (https://pypi.org/project/cblind/) -- install with `pip install
# cblind`. Not installable in the Claude sandbox this edit was made in (no
# network access to that index), so this falls back to matplotlib's RdBu_r
# here; on Frida's own machine, once cblind is installed, this picks up
# cb.rbscale, cblind's colorblind-safe red-blue diverging scale -- the direct
# replacement for RdBu_r, same semantics (negative = blue, positive = red).
#
# cb.cbmap() itself is now deprecated in favour of matplotlib's own
# get_cmap (confirmed from the DeprecationWarning text, 2026-10: "Please use
# matplotlib.colormaps.get_cmap instead, or matplotlib.pyplot.get_cmap if
# you need to specify nbin"). We try that path first and only fall back to
# the deprecated cbmap() call if the name isn't registered with matplotlib
# under this cblind version -- so this works either way and stops warning.
try:
    import cblind as cb  # noqa: F401  (importing it registers its colormaps with matplotlib)
    try:
        HEATMAP_CMAP = plt.get_cmap("cb.rbscale", 9)
        _CMAP_SOURCE = "cblind (cb.rbscale, via matplotlib.pyplot.get_cmap)"
    except (ValueError, KeyError):
        HEATMAP_CMAP = cb.cbmap("cb.rbscale", nbin=9)
        _CMAP_SOURCE = "cblind (cb.rbscale, via deprecated cb.cbmap -- update cblind to drop this)"
except ImportError:
    HEATMAP_CMAP = "RdBu_r"
    _CMAP_SOURCE = "matplotlib RdBu_r (cblind not installed in this environment)"
print(f"  [fig_04_relationships] heatmap colormap source: {_CMAP_SOURCE}")

# ---- definitions -----------------------------------------------------------
TABLE3 = [("Nino34", "SON", "ABS", "amplitude"),
          ("Nino34", "annual", "King Haakon", "amplitude"),
          ("SAM", "RET", "East Antarctica", "max_doy"),
          ("SAM", "JJA", "Weddell", "amplitude"),
          ("ASL", "annual", "Ross", "amplitude"),
          ("ZW3R", "SON", "King Haakon", "max_doy"),
          ("ZW3R", "annual", "ABS", "amplitude")]
T3_PAIRS = {(b, k) for b, _, k, _ in TABLE3}
MEAS = ["sie_anom", "amplitude", "max_doy", "raw_anom"]
COL_LAB = ["Extent anomaly", "Amplitude", "Phase\n(day of max.)", "Raw anomaly"]
MONTHS = {"annual": set(range(1, 13)), "DJF": {12, 1, 2}, "MAM": {3, 4, 5}, "JJA": {6, 7, 8},
          "SON": {9, 10, 11}, "ADV": set(range(3, 9)), "RET": {10, 11, 12, 1}}
SEASON_NAME = {"annual": "annual", "DJF": "DJF", "MAM": "MAM", "JJA": "JJA", "SON": "SON",
               "ADV": "Mar–Aug", "RET": "Oct–Jan"}
INDEX_NAME = {"Nino34": "Niño3.4", "SAM": "SAM", "ASL": "ASL", "ZW3R": "ZW3", "ZW3G": "ZW3 (Goyal)"}
KEY = {"SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "ABS", "Amundsen-Bellingshausen": "ABS",
       "Amundsen_Bellingshausen": "ABS", "SIE_Ross": "Ross", "SIE_East_Antarctica": "East Antarctica",
       "East_Antarctica": "East Antarctica", "SIE_King_Haakon": "King Haakon",
       "King_Haakon": "King Haakon", "SIE_circumpolar": "circumpolar", "Circumpolar": "circumpolar"}
SHORT = {"Weddell": "Weddell", "ABS": "A-B", "Ross": "Ross", "East Antarctica": "EA",
         "King Haakon": "KH", "circumpolar": "Circumpolar"}
MNAME = {"sie_anom": "departure", "amplitude": "amplitude", "max_doy": "phase", "raw_anom": "raw anomaly"}
ALPHA, VMAX, INK = 0.05, 0.8, "0.35"


SCAN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TABLES_DIR, "t37b_component_comparison_scan.csv")
if not os.path.exists(SCAN):
    sys.exit(f"{SCAN} not found; run compute_fig05_component_comparison.py first")
t = pd.read_csv(SCAN)
t["scan_label"] = t["sector"]
t["sector"] = t["sector"].map(lambda s: KEY.get(s, s))
LABEL = dict(zip(t["sector"], t["scan_label"]))       # canonical -> label used by the data files
t = t.set_index(["sector", "index_base", "season"]).sort_index()


def disjoint_pairs(seasons):
    """all pairs of these seasons that share no months"""
    s = list(seasons)
    return [(s[i], s[j]) for i in range(len(s)) for j in range(i + 1, len(s))
            if not (MONTHS[s[i]] & MONTHS[s[j]])]


# ---- yearly series, built exactly as compute_fig05_component_comparison.py does
YEAR_LO, YEAR_HI = 1979, 2023
SEASON_MONTHS = {k: sorted(v) for k, v in MONTHS.items()}
daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
if "period" in daily.columns:
    daily = daily[daily["period"] == "FULL"]
daily = daily[(daily["Year"] >= YEAR_LO) & (daily["Year"] <= YEAR_HI)].copy()
daily["month"] = daily["Date"].dt.month
_dl = {str(x).lower(): x for x in daily["sector"].unique()}
idx = pd.read_csv(INDEX_CSV)
idx = idx[(idx["Year"] >= YEAR_LO) & (idx["Year"] <= YEAR_HI)]
ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"] == "FULL"]
INV = {v: k for k, v in SECTORS_COMPUTE.items()}


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


def series(sec, base, season, m):
    """(years, index, ice) as used for the scan correlation"""
    lab, col = LABEL[sec], f"{base}_{season}"
    src = idx[["Year", col]]
    if m in ("sie_anom", "raw_anom"):
        d = daily_for(lab)
        d = d[d["month"].isin(SEASON_MONTHS[season])].copy()
        d["sy"] = np.where((season == "RET") & (d["month"] == 1), d["Year"] - 1,
                           np.where((season == "DJF") & (d["month"] == 12), d["Year"] + 1, d["Year"]))
        g = d.groupby("sy")
        v = "anomaly_from_iac" if m == "sie_anom" else "residual_apac"
        sm = pd.DataFrame({"Year": g.size().index, "y": g[v].mean().values, "nd": g.size().values})
        sm = sm[(sm["nd"] >= 0.35 * sm["nd"].median()) & sm.Year.between(YEAR_LO, YEAR_HI)]
        sm = sm.merge(src, on="Year")
        y = detrend(sm["y"].values)
    else:
        a = ann[ann["sector"] == lab]
        if a.empty:
            a = ann[ann["sector"] == INV.get(lab, lab)]
        v = "amplitude_raw_anom" if m == "amplitude" else "max_doy_raw_anom"
        sm = a[a.Year.between(YEAR_LO, YEAR_HI)][["Year", v]].rename(columns={v: "y"}).merge(src, on="Year")
        y = sm["y"].values.astype(float)
    x = sm[col].values.astype(float)
    ok = ~(np.isnan(x) | np.isnan(y))
    return sm["Year"].values[ok], x[ok], y[ok]


def worst_drop_one(sec, base, season, m):
    """r on all years, and the largest p (and which year) with one year left out"""
    yrs, x, y = series(sec, base, season, m)
    if len(x) < 8:
        return np.nan, np.nan, None
    r = stats.pearsonr(x, y)[0]
    worst, wy = 0.0, None
    for i in range(len(x)):
        k = np.arange(len(x)) != i
        rr, pp = stats.pearsonr(x[k], y[k])
        if np.sign(rr) != np.sign(r):
            pp = 1.0                       # sign flips without this year
        if pp > worst:
            worst, wy = pp, int(yrs[i])
    return r, worst, wy


# ---- the filter ------------------------------------------------------------
found, dropped, MISMATCH = [], [], []
for (sec, bse), g in t.groupby(level=[0, 1]):
    if (bse, sec) in T3_PAIRS:
        continue                                                      # rule 1
    g = g.droplevel([0, 1])
    options = []
    for m in MEAS:
        for sign in (1, -1):
            sig = g[(g[f"p_{m}"] < ALPHA) & (np.sign(g[f"r_{m}"]) == sign)]
            if len(sig) >= 2 and not disjoint_pairs(sig.index):
                dropped.append(f"rule 2  {SHORT[sec]} · {INDEX_NAME[bse]} {MNAME[m]}: significant in "
                               f"{', '.join(SEASON_NAME[s] for s in sig.index)}, but these overlap")
            for s1, s2 in disjoint_pairs(sig.index):
                options.append((max(sig.loc[s1, f"p_{m}"], sig.loc[s2, f"p_{m}"]), m, s1, s2,
                                sig[f"p_{m}"].idxmin()))
    if not options:
        continue
    kept = None
    for _, m, s1, s2, show in sorted(options):                       # best evidence first
        checks = [worst_drop_one(sec, bse, s, m) for s in (s1, s2)]
        for s, c in zip((s1, s2), checks):                            # rebuilt series must match the scan
            if abs(c[0] - g.loc[s, f"r_{m}"]) > 0.02:
                MISMATCH.append(f"{SHORT[sec]} · {INDEX_NAME[bse]} {MNAME[m]} ({SEASON_NAME[s]}): "
                                f"scan r = {g.loc[s, f'r_{m}']:+.2f}, rebuilt r = {c[0]:+.2f}")
        bad = [(s, c) for s, c in zip((s1, s2), checks) if not (c[1] < ALPHA)]
        if not bad:
            kept = dict(sector=sec, index_base=bse, season=show, measure=MNAME[m],
                        evidence=f"{SEASON_NAME[s1]} and {SEASON_NAME[s2]}",
                        worst_p_one_year_out=max(c[1] for c in checks))
            break
        s, c = bad[0]
        dropped.append(f"rule 4  {SHORT[sec]} · {INDEX_NAME[bse]} {MNAME[m]} ({SEASON_NAME[s]}): "
                       + (f"sign flips without {c[2]}" if c[1] >= 1 else f"p = {c[1]:.2f} without {c[2]}"))
    if kept:
        found.append(kept)

# rule 5: circumpolar rows explained by a sector already in the figure
in_fig = T3_PAIRS | {(f["index_base"], f["sector"]) for f in found}
for f in list(found):
    if f["sector"] == "circumpolar":
        who = [SHORT[k] for b, k in in_fig if b == f["index_base"] and k != "circumpolar"]
        if who:
            found.remove(f)
            dropped.append(f"rule 5  Circumpolar · {INDEX_NAME[f['index_base']]}: "
                           f"{', '.join(sorted(set(who)))} with the same index is already in the figure")

# ---- rows of the figure ----------------------------------------------------
rows = []
for b, se, k, m in TABLE3:
    rows.append(dict(block="From the literature (Table 3)", sector=k, index_base=b, season=se, measure=MNAME[m]))
for f in sorted(found, key=lambda f: (list(SHORT).index(f["sector"]), f["index_base"])):
    rows.append(dict(block="Found in the search", **f))
R = pd.DataFrame(rows)
for m in MEAS:
    R[f"r_{m}"] = [t.loc[(k, b, s), f"r_{m}"] for k, b, s in zip(R.sector, R.index_base, R.season)]
    R[f"p_{m}"] = [t.loc[(k, b, s), f"p_{m}"] for k, b, s in zip(R.sector, R.index_base, R.season)]
R["label"] = [f"{SHORT[k]}  ·  {INDEX_NAME[b]} ({SEASON_NAME[s]})"
              for k, b, s in zip(R.sector, R.index_base, R.season)]
R.to_csv(os.path.join(TABLES_DIR, "t_fig04_rows.csv"), index=False)

# ---- figure: one panel; search rows follow the seven, separated by a slightly wider gap
n1 = (R.block == R.block.iloc[0]).sum()
bold = ch3_style.bold_font_properties(size=9)
fig, ax = plt.subplots(figsize=(6.8, 0.46 * len(R) + 1.2))
M = R[[f"r_{m}" for m in MEAS]].values.astype(float)
P = R[[f"p_{m}" for m in MEAS]].values.astype(float)

# ---- diagnostic: rows where this index hits >=2 components at once --------
# Added 2026-10-03, re Frida's question about what it means when one index
# is significant on two measures (e.g. amplitude AND phase) in the same row.
# trend/amplitude/phase/raw-anomaly sum to the total by construction, but
# that doesn't make them statistically independent of each other across
# years in practice. For every such row: compute how correlated the two
# measures' yearly series are with each other, and the partial correlation
# of the index with each measure controlling for the other -- if both
# partial correlations survive, it's a genuine joint effect; if one
# collapses toward zero, that one was riding along with the other and
# isn't really a separate finding.
import itertools


def partial_r(r_xy, r_xz, r_yz):
    """Partial correlation of x,y controlling for z, from the three pairwise r's."""
    den = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    return (r_xy - r_xz * r_yz) / den if den > 1e-12 else np.nan


double_hits = []
for _, row in R.iterrows():
    sig_m = [m for m in MEAS if row[f"p_{m}"] < ALPHA]
    if len(sig_m) < 2:
        continue
    for m1, m2 in itertools.combinations(sig_m, 2):
        yrs1, x1, y1 = series(row.sector, row.index_base, row.season, m1)
        yrs2, x2, y2 = series(row.sector, row.index_base, row.season, m2)
        common = np.intersect1d(yrs1, yrs2)
        if len(common) < 8:
            continue
        i1, i2 = np.isin(yrs1, common), np.isin(yrs2, common)
        x, y_m1, y_m2 = x1[i1], y1[i1], y2[i2]     # same index series either way, same years
        r_xy1 = stats.pearsonr(x, y_m1)[0]
        r_xy2 = stats.pearsonr(x, y_m2)[0]
        r_y1y2 = stats.pearsonr(y_m1, y_m2)[0]
        double_hits.append(dict(
            sector=SHORT[row.sector], driver=INDEX_NAME[row.index_base], season=SEASON_NAME[row.season],
            measure_1=MNAME[m1], measure_2=MNAME[m2],
            r_index_m1=r_xy1, r_index_m2=r_xy2, r_m1_m2=r_y1y2,
            partial_r_index_m1_given_m2=partial_r(r_xy1, r_xy2, r_y1y2),
            partial_r_index_m2_given_m1=partial_r(r_xy2, r_xy1, r_y1y2),
            n=len(x)))

if double_hits:
    # NOTE: the "driver" column is deliberately not called "index" -- on a
    # Series (what .iterrows() hands you per row), ".index" is the row's own
    # axis-label attribute, already defined by pandas, so a column actually
    # named "index" gets silently shadowed by it instead of raising -- that's
    # the "unsupported format string passed to Index.__format__" crash from
    # the first run. Fixed 2026-10-03 by renaming the column, not by using
    # d["index"] bracket access, so nothing downstream can hit the same trap.
    DH = pd.DataFrame(double_hits)
    DH.to_csv(os.path.join(TABLES_DIR, "t_fig04_double_hits.csv"), index=False)
    print(f"\nwrote {os.path.join(TABLES_DIR, 't_fig04_double_hits.csv')} ({len(DH)} double-hit pairs)")
    print("Rows where one index is significant on >=2 measures at once -- partial correlations:")
    for _, d in DH.iterrows():
        print(f"   {d.sector:12s} {d.driver:10s} {d.season:6s}  {d.measure_1} vs {d.measure_2}:"
              f"  r(index,{d.measure_1})={d.r_index_m1:+.2f}  r(index,{d.measure_2})={d.r_index_m2:+.2f}"
              f"  r({d.measure_1},{d.measure_2})={d.r_m1_m2:+.2f}"
              f"  | partial r(index,{d.measure_1}|{d.measure_2})={d.partial_r_index_m1_given_m2:+.2f}"
              f"  partial r(index,{d.measure_2}|{d.measure_1})={d.partial_r_index_m2_given_m1:+.2f}")
else:
    print("\nNo row in Fig. 4 has the same index significant on two measures at once "
          "(nothing to partial out).")

# ---- structural-risk check: proactively test amplitude/phase rows against -
# departure in sectors where the two are entangled in general, even when
# departure didn't independently clear ALPHA in THIS row's own season.
# Added 2026-10-03, after check_observed_measure_correlation.py showed
# departure~amplitude is flagged (|r|>0.3) in ABS, Ross, and East Antarctica
# -- the double-hit block above only catches a conflation when departure
# ALSO happens to be significant in that exact row; a real confound can sit
# just under that threshold without disappearing (n~45-47 isn't a lot of
# power), so this checks every amplitude/phase row in a structurally-
# flagged sector regardless, and skips rows the double-hit block already
# covered so nothing prints twice.
obs_corr_path = os.path.join(TABLES_DIR, "t_observed_measure_correlation_matrix.csv")
if not os.path.exists(obs_corr_path):
    print(f"\n({obs_corr_path} not found -- run check_observed_measure_correlation.py first "
          f"for the structural-risk check.)")
else:
    OC = pd.read_csv(obs_corr_path)
    STRUCT_FLAG = {}
    for d in OC.itertuples():
        if d.pair == "departure~amplitude" and abs(d.r) > 0.3:
            STRUCT_FLAG[(d.sector, "amplitude")] = d.r
        elif d.pair == "departure~phase" and abs(d.r) > 0.3:
            STRUCT_FLAG[(d.sector, "phase")] = d.r

    struct_rows = []
    for _, row in R.iterrows():
        if row.measure not in ("amplitude", "phase"):
            continue
        key = (row.sector, row.measure)
        if key not in STRUCT_FLAG:
            continue
        m_code = {"amplitude": "amplitude", "phase": "max_doy"}[row.measure]
        if row[f"p_{m_code}"] < ALPHA and row["p_sie_anom"] < ALPHA:
            continue  # already covered by the double-hit block above
        yrs1, x1, y1 = series(row.sector, row.index_base, row.season, m_code)
        yrs2, x2, y2 = series(row.sector, row.index_base, row.season, "sie_anom")
        common = np.intersect1d(yrs1, yrs2)
        if len(common) < 8:
            continue
        i1, i2 = np.isin(yrs1, common), np.isin(yrs2, common)
        x, y_m, y_dep = x1[i1], y1[i1], y2[i2]
        r_xm = stats.pearsonr(x, y_m)[0]
        r_xd = stats.pearsonr(x, y_dep)[0]
        r_md = stats.pearsonr(y_m, y_dep)[0]
        struct_rows.append(dict(
            sector=SHORT[row.sector], driver=INDEX_NAME[row.index_base], season=SEASON_NAME[row.season],
            measure=row.measure, sector_structural_r_departure_measure=STRUCT_FLAG[key],
            r_index_measure=r_xm, r_index_departure=r_xd, r_measure_departure=r_md,
            partial_r_index_measure_given_departure=partial_r(r_xm, r_xd, r_md),
            departure_independently_significant=bool(row["p_sie_anom"] < ALPHA)))

    if struct_rows:
        SR = pd.DataFrame(struct_rows)
        SR.to_csv(os.path.join(TABLES_DIR, "t_fig04_structural_risk_partials.csv"), index=False)
        print(f"\nwrote {os.path.join(TABLES_DIR, 't_fig04_structural_risk_partials.csv')} ({len(SR)} rows)")
        print("Rows in a structurally-entangled sector, checked proactively even though departure")
        print("wasn't independently significant in this row's own season:")
        for d in SR.itertuples():
            print(f"   {d.sector:12s} {d.driver:10s} {d.season:6s} {d.measure:10s}"
                  f"  r(index,{d.measure})={d.r_index_measure:+.2f}"
                  f"  partial r(index,{d.measure}|departure)={d.partial_r_index_measure_given_departure:+.2f}"
                  f"  [sector's departure~{d.measure} r={d.sector_structural_r_departure_measure:+.2f}]")
    else:
        print("\nNo additional rows to check structurally beyond the double-hit block above.")

im = ax.imshow(M, cmap=HEATMAP_CMAP, vmin=-VMAX, vmax=VMAX, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        r, p = M[i, j], P[i, j]
        if np.isnan(r):
            ax.text(j, i, "n/a", ha="center", va="center", color=INK, fontsize=8.5)
            continue
        ax.text(j, i, f"{r:+.2f}".replace("-", "\u2212"), ha="center", va="center", fontsize=8.5,
                color="white" if abs(r) > 0.45 else "0.15",
                fontproperties=bold if p < ALPHA else None)
ax.set_yticks(range(len(R)))
ax.set_yticklabels(R.label, fontsize=8.5)
ax.set_xticks(range(len(MEAS)))
ax.set_xticklabels(COL_LAB, fontproperties=bold)
ax.xaxis.tick_top()
ax.tick_params(length=0, colors=INK)
for sp in ax.spines.values():
    sp.set_visible(False)
for i in range(1, len(R)):
    ax.axhline(i - 0.5, color="white", lw=4 if i == n1 else 1.5)
for j in range(1, len(MEAS)):
    ax.axvline(j - 0.5, color="white", lw=1.5)
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("correlation", color=INK, fontsize=9)
cbar.ax.tick_params(labelsize=8, colors=INK, length=3)
cbar.outline.set_visible(False)
fig04_stem = os.path.join(OUTPUT_DIR, "fig04_relationships")
if hasattr(ch3_style, "savefig_pair"):
    out, out_presentation = ch3_style.savefig_pair(fig, fig04_stem, dpi=220)
else:
    # Local ch3_style.py here predates savefig_pair (added to the version
    # in Frida's cloud copy on 2026-09-30) -- fall back to writing both
    # files directly so this script doesn't hard-depend on that helper.
    print("  [fig_04_relationships] ch3_style.savefig_pair not found -- "
          "your local ch3_style.py may be behind the synced copy; "
          "writing both PNGs directly instead.")
    out = f"{fig04_stem}.png"
    out_presentation = f"{fig04_stem}_presentation.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(out_presentation, dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)

# ---- Table S2 --------------------------------------------------------------
S = t.reset_index()
t3set = {(k, b, s) for b, s, k, _ in TABLE3}
f4set = {(f["sector"], f["index_base"], f["season"]) for f in found}
S["where"] = ["Table 3" if (k, b, s) in t3set else ("Fig. 4" if (k, b, s) in f4set else "")
              for k, b, s in zip(S.sector, S.index_base, S.season)]
S["sector"] = S.sector.map(lambda k: SHORT.get(k, k))
S["index"] = S.index_base.map(lambda b: INDEX_NAME.get(b, b))
S["season"] = S.season.map(lambda s: SEASON_NAME.get(s, s))
for m in MEAS:
    S[MNAME[m]] = [f"{r:+.2f}{'*' if p < ALPHA else ''}" if np.isfinite(r) else ""
                   for r, p in zip(S[f"r_{m}"], S[f"p_{m}"])]
S2 = S[["sector", "index", "season"] + [MNAME[m] for m in MEAS] + ["where"]]
S2.to_csv(os.path.join(TABLES_DIR, "tS2_all_combinations.csv"), index=False)

# ---- report ----------------------------------------------------------------
print(f"\nwrote {out}")
print(f"wrote {out_presentation}")
print(f"wrote {os.path.join(TABLES_DIR, 'tS2_all_combinations.csv')} ({len(S2)} rows)")
print("\nFound in the search and kept:")
for f in found:
    print(f"   {SHORT[f['sector']]:12s} {INDEX_NAME[f['index_base']]:12s} {f['measure']:12s}"
          f" shown at {SEASON_NAME[f['season']]}; significant in {f['evidence']};"
          f" worst p with one year out {f['worst_p_one_year_out']:.3f}")
print("\nDropped, and why:")
for d in dict.fromkeys(dropped):
    print("   " + d)
if MISMATCH:
    print("\nWARNING: the rebuilt yearly series do not reproduce the scan for these, so rule 4 is not"
          "\nreliable for them (check detrending / season definitions against the compute script):")
    for x in dict.fromkeys(MISMATCH):
        print("   " + x)
else:
    print("\nrule 4 check: every rebuilt series reproduces the scan correlation (within 0.02).")
for m in MEAS:
    n = (t[f"p_{m}"] < ALPHA).sum()
    print(f"{MNAME[m]:12s}: {n} of {len(t)} significant, about {ALPHA * len(t):.0f} by chance")