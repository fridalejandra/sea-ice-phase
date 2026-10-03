#!/usr/bin/env python3
"""
check_asl_ross_moved.py -- Sect. 3.3: did the ASL-Ross relationship MOVE
(from the winter maximum to the summer minimum, i.e. from amplitude to
level) after 2001, or just break?

Why: the candidate Sect. 3.3 story rests on ch3_numbers.md's "which extent
carries it" table (2026-09-14 ledger, 1979-2023):
    ASL_annual ~ sie_SON   +0.73 -> -0.06      ASL_annual ~ sie_DJF  +0.24 -> +0.48
    ASL_annual ~ sie_JJA   +0.55 -> +0.06      ASL_annual ~ sie_MAM  +0.12 -> +0.55
That table printed r only -- no p, no shift test, and it's from a ledger
two years short of the current 1979-2025 record. This script gives every
row its own p in each period, a Fisher-z shift test, and a leave-one-year-
out check on the post-2001 correlations, so "it moved" is only written if
the numbers actually carry it.

What would SUPPORT "it moved":
  - SON/JJA (maximum seasons): significant before, not after, shift p < 0.05
  - DJF/MAM (minimum seasons): not significant before, significant after,
    shift p < 0.05, and the post-2001 r survives leave-one-year-out
  - amplitude falls while sie_annual (the level) holds up after 2001
What would mean "it just broke": the DJF/MAM post-2001 r's don't clear
p < 0.05, or don't survive LOO, or their shift p isn't significant.

There are 2 indices x 6 targets = 12 rows here; print-out flags which rows
would survive a x12 Bonferroni, so it's clear which claims are strong.

Helpers (detrend, fz, pear, zshift) are copied verbatim from ch3_stats.py
so every number is directly comparable with Table 3 and the ledger.

Reads   ANNUAL_CSV, INDEX_CSV (via ch3_config)
Writes  results/ch3/tables/t36_ross_asl_moved.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ch3_config import ANNUAL_CSV, INDEX_CSV, TABLES_DIR, YEAR_MIN, YEAR_MAX, SPLIT_YEAR

SECTOR = "SIE_Ross"
INDICES = ["ASL_annual", "ASL_RET"]   # the two ASL seasons that carry the break in the ledger
TARGETS = [  # column, plain-language label
    ("sie_SON", "spring extent (SON)"),
    ("sie_JJA", "winter extent (JJA)"),
    ("sie_DJF", "summer extent (DJF)"),
    ("sie_MAM", "autumn extent (MAM)"),
    ("sie_annual", "annual-mean extent (level)"),
    ("amplitude_raw_anom", "amplitude"),
]
ALPHA = 0.05


# ── helpers, verbatim from ch3_stats.py ─────────────────────────────────────
def detrend(x):
    x = np.asarray(x, float); ok = np.isfinite(x)
    if ok.sum() < 3: return x
    t = np.arange(len(x), dtype=float); b, a = np.polyfit(t[ok], x[ok], 1)
    out = x.copy(); out[ok] = x[ok] - (a + b * t[ok]); return out

def fz(r): return 0.5 * np.log((1 + r) / (1 - r))

def pear(x, y, dt=True):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if dt: x, y = detrend(x), detrend(y)
    ok = np.isfinite(x) & np.isfinite(y); n = int(ok.sum())
    if n < 6: return np.nan, np.nan, n
    r, p = stats.pearsonr(x[ok], y[ok]); return float(r), float(p), n

def zshift(r1, n1, r2, n2):
    if not (np.isfinite(r1) and np.isfinite(r2)) or n1 < 4 or n2 < 4: return np.nan, np.nan
    z = (fz(r1) - fz(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(z), float(2 * stats.norm.sf(abs(z)))


def loo_worst_p(a, ic, tv):
    """worst p across leave-one-year-out refits (detrending redone each time,
    same as ch3_stats.py's own LOO)"""
    worst = 0.0
    for yr in a.Year.values:
        b = a[a.Year != yr]
        _, p, _ = pear(b[ic], b[tv])
        worst = max(worst, p) if np.isfinite(p) else worst
    return worst


# ── load ────────────────────────────────────────────────────────────────────
ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:          # ch3_data convention: FULL vs HR2018 copies
    ann = ann[ann["period"] == "FULL"]
ann = ann[ann.Year.between(YEAR_MIN, YEAR_MAX)]
idx = pd.read_csv(INDEX_CSV)
ai = ann.merge(idx, how="left", on="Year")
ross = ai[ai.sector == SECTOR].sort_values("Year")
if ross.empty:
    sys.exit(f"No rows for {SECTOR} in ANNUAL_CSV (sectors seen: {sorted(ai.sector.unique())})")

missing = [c for c, _ in TARGETS if c not in ross.columns] + [c for c in INDICES if c not in ross.columns]
if missing:
    sys.exit(f"Missing columns {missing}. sie_* columns available: "
             f"{sorted(c for c in ross.columns if c.startswith('sie'))}")

pre = ross[ross.Year < SPLIT_YEAR]
post = ross[ross.Year >= SPLIT_YEAR]
print(f"Ross: {len(ross)} years {ross.Year.min()}-{ross.Year.max()}  |  "
      f"pre {pre.Year.min()}-{pre.Year.max()} (n={len(pre)})  post {post.Year.min()}-{post.Year.max()} (n={len(post)})\n")

rows = []
n_tests = len(INDICES) * len(TARGETS)
for ic in INDICES:
    for tv, label in TARGETS:
        r1, p1, n1 = pear(pre[ic], pre[tv])
        r2, p2, n2 = pear(post[ic], post[tv])
        z, ps = zshift(r1, n1, r2, n2)
        rows.append(dict(index=ic, target=tv, label=label,
                         r_pre=r1, p_pre=p1, n_pre=n1, r_post=r2, p_post=p2, n_post=n2,
                         z_shift=z, p_shift=ps, p_shift_bonf=min(1.0, ps * n_tests),
                         loo_worst_p_pre=loo_worst_p(pre, ic, tv),
                         loo_worst_p_post=loo_worst_p(post, ic, tv)))
T = pd.DataFrame(rows)
out = os.path.join(TABLES_DIR, "t36_ross_asl_moved.csv")
T.to_csv(out, index=False)


def star(p):
    return "*" if p < ALPHA else " "


for ic in INDICES:
    print(f"=== {ic} vs Ross ===")
    print(f"  {'target':28s} {'pre r (p)':>16s} {'post r (p)':>16s} {'shift p':>9s} {'x' + str(n_tests):>6s}  LOO worst p pre/post")
    for d in T[T["index"] == ic].itertuples():
        print(f"  {d.label:28s} {d.r_pre:+.2f} ({d.p_pre:.3f}){star(d.p_pre)} "
              f"{d.r_post:+.2f} ({d.p_post:.3f}){star(d.p_post)} {d.p_shift:9.4f} "
              f"{'yes' if d.p_shift_bonf < ALPHA else 'no':>6s}  {d.loo_worst_p_pre:.3f} / {d.loo_worst_p_post:.3f}")
    print()

# ── verdict on the "it moved" story, ASL_annual (the Table 3 index) ─────────
a = T[T["index"] == "ASL_annual"].set_index("target")
mx_lost = all(a.loc[s, "p_pre"] < ALPHA and a.loc[s, "p_post"] >= ALPHA and a.loc[s, "p_shift"] < ALPHA
              for s in ("sie_SON", "sie_JJA"))
mn_gained = [s for s in ("sie_DJF", "sie_MAM")
             if a.loc[s, "p_post"] < ALPHA and a.loc[s, "loo_worst_p_post"] < ALPHA and a.loc[s, "p_shift"] < ALPHA]
print("Verdict (ASL_annual):")
print(f"  maximum seasons (SON, JJA) significant before, gone after, shift significant: {'YES' if mx_lost else 'NO'}")
print(f"  minimum seasons gained a robust post-2001 relationship with a significant shift: "
      f"{', '.join(mn_gained) if mn_gained else 'NONE'}")
if mx_lost and len(mn_gained) == 2:
    print("  -> supports 'it moved' (maximum -> minimum, amplitude -> level).")
elif mx_lost and mn_gained:
    print("  -> mostly supports 'it moved': it left the maximum; one minimum season clearly picks it up.")
elif mn_gained:
    print("  -> partial: a robust post-2001 relationship appears at the minimum, but the loss at the\n"
          "     maximum isn't significant in both seasons (n~22/25 per half -- low power for a shift\n"
          "     test). 'It moved' can be written, but say 'appears at the minimum' rather than\n"
          "     'moved from the maximum'.")
elif mx_lost:
    print("  -> it left the maximum and nothing picks it up at the minimum: use the 'it broke' framing.")
else:
    print("  -> neither half of the 'it moved' story clears the bar: use the 'it broke' framing.")
print(f"\nwrote {out}")
