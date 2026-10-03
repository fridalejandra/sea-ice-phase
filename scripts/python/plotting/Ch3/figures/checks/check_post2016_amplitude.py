#!/usr/bin/env python3
"""
check_post2016_amplitude.py -- the two checks Sect. 3.4.1 needs before it can say
the post-2016 drop in amplitude variability is "a constraint the atmosphere does
not set".

  A. Did the atmosphere itself get quieter after 2016? For every index-season in
     INDEX_CSV (already detrended): SD after / SD before, with an F-test and a
     Brown-Forsythe (median-centred Levene) test. If the indices tied to the
     cycle's size lost variance too, quieter forcing could explain the smaller
     amplitude spread.
  B. East Antarctica's Table 3 relationship is with timing, not amplitude, so
     "unchanged relationships" isn't tested for its amplitude. For each series:
     its strongest index relationship with amplitude over the full record,
     before 2016 and after (n ~ 10, descriptive), and how many of the 35
     index-seasons reach p < 0.05 in the full record (~1.75 expected by chance).

Conventions as in ch3_stats.py: observed amplitude (amplitude_raw_anom),
linear detrending within each period before correlating, break at BREAK_YEAR.

Reads   ANNUAL_CSV, INDEX_CSV (via ch3_config)
Writes  results/ch3/tables/t_post2016_index_variance.csv
        results/ch3/tables/t_post2016_amplitude_links.csv
Run     python checks/check_post2016_amplitude.py     (from the figures folder)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from ch3_config import ANNUAL_CSV, INDEX_CSV, TABLES_DIR, YEAR_MIN, YEAR_MAX, BREAK_YEAR, SECTOR_LABELS

AMP = "amplitude_raw_anom"
SIZE_LINKED = ["Nino34_SON", "Nino34_annual", "ASL_annual", "SAM_JJA", "ZW3R_annual"]  # Table 3 amplitude indices


def detrend(x):
    x = np.asarray(x, float); ok = np.isfinite(x)
    if ok.sum() < 3: return x
    t = np.arange(len(x), dtype=float); b, a = np.polyfit(t[ok], x[ok], 1)
    out = x.copy(); out[ok] = x[ok] - (a + b * t[ok]); return out


def pear(x, y):
    x, y = detrend(x), detrend(y)
    ok = np.isfinite(x) & np.isfinite(y); n = int(ok.sum())
    if n < 6: return np.nan, np.nan, n
    r, p = stats.pearsonr(x[ok], y[ok]); return float(r), float(p), n


idx = pd.read_csv(INDEX_CSV)
idx = idx[idx.Year.between(YEAR_MIN, YEAR_MAX)].sort_values("Year")
ICOLS = [c for c in idx.columns if c != "Year"]
pre_i, post_i = idx[idx.Year < BREAK_YEAR], idx[idx.Year >= BREAK_YEAR]

# ---- A. index variance before / after ------------------------------------
rows = []
for c in ICOLS:
    a, b = pre_i[c].dropna().values, post_i[c].dropna().values
    if len(a) < 5 or len(b) < 5:
        continue
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    F = (s2 / s1) ** 2
    pF = 2 * min(stats.f.cdf(F, len(b) - 1, len(a) - 1), stats.f.sf(F, len(b) - 1, len(a) - 1))
    pBF = stats.levene(a, b, center="median").pvalue
    rows.append(dict(index=c, sd_pre=s1, sd_post=s2, ratio=s2 / s1, p_F=pF, p_BF=pBF,
                     n_pre=len(a), n_post=len(b), size_linked=c in SIZE_LINKED))
V = pd.DataFrame(rows)
V.to_csv(os.path.join(TABLES_DIR, "t_post2016_index_variance.csv"), index=False)

print(f"A. Index variability after {BREAK_YEAR} vs before (SD ratio; <1 = quieter after)\n")
print("   Indices tied to the cycle's size in Table 3:")
for r in V[V.size_linked].itertuples():
    print(f"     {r.index:16s} ratio {r.ratio:4.2f}   F p = {r.p_F:.3f}   Brown-Forsythe p = {r.p_BF:.3f}   (n {r.n_pre}/{r.n_post})")
sig = V[(V.p_F < 0.05) | (V.p_BF < 0.05)]
print(f"\n   All {len(V)} index-seasons: median ratio {V.ratio.median():.2f}; "
      f"{(V.ratio < 1).sum()} quieter, {(V.ratio > 1).sum()} more variable; "
      f"{len(sig)} with p < 0.05 on either test (~{0.05 * len(V):.1f} expected by chance):")
for r in sig.sort_values("p_F").itertuples():
    print(f"     {r.index:16s} ratio {r.ratio:4.2f}   F p = {r.p_F:.3f}   BF p = {r.p_BF:.3f}")

# ---- B. amplitude's own atmospheric links, by period -------------------------
ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"] == "FULL"]
ann = ann[ann.Year.between(YEAR_MIN, YEAR_MAX)]
ai = ann.merge(idx, on="Year", how="left").sort_values("Year")

out = []
print(f"\nB. Amplitude vs all {len(ICOLS)} index-seasons, by series (strongest |r|; p<0.05 count is full record)\n")
for sec in sorted(ai.sector.unique()):
    g = ai[ai.sector == sec]
    lab = SECTOR_LABELS.get(sec, sec)
    sd_ratio = g[g.Year >= BREAK_YEAR][AMP].std() / g[g.Year < BREAK_YEAR][AMP].std()
    res = {}
    for name, sub in (("full", g), ("pre", g[g.Year < BREAK_YEAR]), ("post", g[g.Year >= BREAK_YEAR])):
        rs = [(c, *pear(sub[c].values, sub[AMP].values)) for c in ICOLS]
        rs = [x for x in rs if np.isfinite(x[1])]
        best = max(rs, key=lambda x: abs(x[1])) if rs else (None, np.nan, np.nan, 0)
        res[name] = dict(best=best, n_sig=sum(1 for x in rs if x[2] < 0.05))
    fb, pb, qb = res["full"]["best"], res["pre"]["best"], res["post"]["best"]
    out.append(dict(series=lab, amp_sd_ratio_post_pre=sd_ratio,
                    full_best_index=fb[0], full_best_r=fb[1], full_best_p=fb[2], full_n_sig=res["full"]["n_sig"],
                    pre_best_index=pb[0], pre_best_r=pb[1], post_best_index=qb[0], post_best_r=qb[1], post_n=qb[3]))
    print(f"   {lab:16s} amplitude SD ratio {sd_ratio:4.2f} | full: {fb[0]} r={fb[1]:+.2f} (p={fb[2]:.3f}), "
          f"{res['full']['n_sig']}/{len(ICOLS)} at p<0.05 | pre: {pb[0]} {pb[1]:+.2f} | post (n={qb[3]}): {qb[0]} {qb[1]:+.2f}")
pd.DataFrame(out).to_csv(os.path.join(TABLES_DIR, "t_post2016_amplitude_links.csv"), index=False)

print(f"\nwrote t_post2016_index_variance.csv, t_post2016_amplitude_links.csv to {TABLES_DIR}")
print("Read A: if the size-linked indices are NOT quieter after 2016, the amplitude drop isn't explained by\n"
      "quieter forcing. Read B: if a series' amplitude relates to no index (few p<0.05, close to the ~1.75\n"
      "expected by chance), its variance can't have fallen through a change in those relationships.")
