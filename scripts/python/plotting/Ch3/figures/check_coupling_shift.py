"""
t34_coupling_shift_test.py — §3.3 follow-up: is the post-2016 rise in the
observed timing-amplitude correlation (Fig. 9 / fig03_rolling_phase_amp_corr.py)
a real, cross-sector pattern, or five noisy numbers that happen to lean the
same way?

Motivation: the whole-era Spearman rho's already printed by
fig03_rolling_phase_amp_corr.py show the day-of-maximum / amplitude
correlation rising after 2016 in four of the five sectors (Weddell, ABS,
Ross, King Haakon), flat in East Antarctica. None of those five post-2016
rho's individually clears significance at n~9 (|r|~0.66 needed). This script
asks the question properly: not "is each rho significant" (it can't be, at
this n) but "is the SHIFT in rho, combined across the five independent
sectors, bigger than sampling noise would produce."

Two tests, deliberately different in their assumptions:

  1. Fisher z-difference, per sector: is pre-2016 rho significantly
     different from post-2016 rho, given both n's? Then Stouffer-combine the
     five sector z-statistics into one cross-sector z. This uses the
     magnitude of each shift, and assumes the five sectors are independent
     draws — which is optimistic (they share circumpolar-scale forcing) but
     is the standard combination and the one worth reporting first.
  2. A sign test on the direction of the shift alone (binomial, k of 5
     positive), which throws away magnitude but does not assume normality
     or independence of magnitude — a cruder, more robust cross-check.

Both are run for BOTH timing metrics (day of max, day of min), not just the
one that motivated this script, so the min-date column is an honest
out-of-sample comparison rather than a result being fished for.

Circumpolar is excluded from both combined tests (SECTORS_ONLY): it is the
sum of the five sectors, not an independent sixth data point, and including
it would double-count.

Inputs
    ANNUAL_CSV (period == "FULL")   via ch3_data.load_annual
Outputs
    results/ch3/tables/t34_coupling_shift_test.csv   (per-sector detail)
    console: per-sector Fisher-z shift test, Stouffer combination, sign test
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import ch3_data as D
from ch3_config import SECTORS_ONLY, SECTOR_LABELS, TABLES_DIR, BREAK_YEAR

PAIRS = [
    ("max_doy_raw_anom", "day of maximum"),
    ("min_doy_raw_anom", "day of minimum"),
]
AMP = "amplitude_raw_anom"

print("t34 — is the post-2016 timing-amplitude coupling shift real, combined across sectors?")
annual = D.load_annual(period="FULL")
D.summary(annual=annual)
annual = annual[annual.sector.isin(SECTORS_ONLY)]


def fisher_shift(r_pre, n_pre, r_post, n_post):
    """Two-sided z-test for a difference between two independent
    correlations, via the Fisher z transform. Returns (z_stat, p_two)."""
    z_pre, z_post = np.arctanh(r_pre), np.arctanh(r_post)
    se = np.sqrt(1 / (n_pre - 3) + 1 / (n_post - 3))
    z = (z_post - z_pre) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


rows = []
for col, label in PAIRS:
    for sec in SECTORS_ONLY:
        g = annual[annual.sector == sec]
        pre  = g[g.Year <  BREAK_YEAR]
        post = g[g.Year >= BREAK_YEAR]
        r_pre  = pre[col].corr(pre[AMP],  method="spearman")
        r_post = post[col].corr(post[AMP], method="spearman")
        z, p = fisher_shift(r_pre, len(pre), r_post, len(post))
        rows.append(dict(pair=label, sector=SECTOR_LABELS[sec],
                         r_pre=r_pre, n_pre=len(pre),
                         r_post=r_post, n_post=len(post),
                         shift=r_post - r_pre, z_shift=z, p_shift=p))
res = pd.DataFrame(rows)
os.makedirs(TABLES_DIR, exist_ok=True)
res.to_csv(os.path.join(TABLES_DIR, "t34_coupling_shift_test.csv"), index=False)

pd.set_option("display.width", 160)
for label in [l for _, l in PAIRS]:
    x = res[res.pair == label]
    print("\n" + "=" * 100)
    print(f"{label} vs amplitude — per-sector shift (Fisher z, independent-correlations test)")
    print("=" * 100)
    print(x[["sector", "r_pre", "n_pre", "r_post", "n_post", "shift", "z_shift", "p_shift"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    k = len(SECTORS_ONLY)

    # 1. Stouffer combination of the five per-sector z's — tests whether the
    #    shift, signed and summed, is bigger than 5 independent zero-mean
    #    draws would produce. One-sided in the a-priori direction (rho rose).
    Z = x["z_shift"].sum() / np.sqrt(k)
    p_one = 1 - stats.norm.cdf(Z)
    p_two = 2 * (1 - stats.norm.cdf(abs(Z)))
    print(f"\nStouffer-combined shift across {k} independent sectors: Z = {Z:+.2f}"
          f"   one-sided p (rho rose) = {p_one:.3f}   two-sided p = {p_two:.3f}")

    # 2. Sign test — direction only, no magnitude, no normality assumption.
    n_pos = int((x["shift"] > 0).sum())
    bt = stats.binomtest(n_pos, k, 0.5, alternative="greater")
    print(f"Sign test: {n_pos} of {k} sectors shifted positive "
          f"(one-sided binomial p = {bt.pvalue:.3f})")

print("\n" + "=" * 100)
print("Caveat: Stouffer assumes the five sectors are independent draws. They")
print("are not fully — they share circumpolar-scale atmospheric forcing — so")
print("this combined p is optimistic, a ceiling on the evidence, not a")
print("conservative estimate of it. Report both tests, and report the")
print("caveat with them.")
print("=" * 100)
print("\nwrote t34_coupling_shift_test.csv")
