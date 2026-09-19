#!/usr/bin/env python3
"""
test_component_dominance.py -- for each sector, answers two distinct forms of
"which component dominates," since they aren't the same question and can
disagree with each other.

  (A) MAGNITUDE dominance -- which component is typically BIGGEST, cycle to
      cycle? Friedman test (nonparametric repeated-measures omnibus across
      the four paired components -- trend/amplitude/phase/residual, paired
      by Year within sector) on |contribution|, followed by pairwise
      Wilcoxon signed-rank tests (all 6 pairs) with Holm-Bonferroni
      correction, to rank components against each other. Run on BOTH NET
      and GROSS magnitudes (|trend_component| etc. for NET; the already-abs
      GROSS columns), matching this chapter's convention of reporting both.
      The Friedman omnibus p-value is also Bonferroni-corrected across the
      five independent sectors (Circumpolar excluded -- it's the sum of the
      other five, not an independent sixth draw, same convention as
      PRIMARY_PAIRS / t34_coupling_shift_test.py / test_trend_share_over_time.py).

  (B) VARIANCE-share dominance -- which component actually DRIVES year-to-
      year movement in the annual anomaly? Because NET's four components sum
      EXACTLY to anomaly_from_iac (verified elsewhere to ~1e-16), the
      variance identity

          Var(anomaly) = Sum_i Cov(component_i, anomaly)

      holds exactly, so each component's "share of variance" =
      Cov(component_i, anomaly) / Var(anomaly), and these four shares sum to
      exactly 1 -- not clipped, not renormalized. A component with a
      NEGATIVE share is a real, meaningful result: it means that component
      moves opposite the net anomaly more often than with it, i.e. it damps
      variability rather than driving it. This uses ONLY the NET table
      (t32_attribution_by_cycle.csv) -- GROSS's abs-valued components don't
      sum to anything, so this identity does not apply to GROSS at all and
      is not computed there.

  This is a descriptive exact decomposition over the observed record (like
  the NET/GROSS era-dominance tables in fig_04_attribution_annual.py), not
  itself a significance-tested quantity -- there is no p-value attached to
  a variance share, only to the magnitude-dominance ranking in (A).

  A component can easily be the (A) magnitude leader without being the (B)
  variance leader, or vice versa: e.g. a component could be large every
  single year but nearly CONSTANT in size (wins on magnitude, loses on
  variance -- something that barely varies can't explain year-to-year
  movement no matter how big it typically is), or a component could usually
  be small but have its rare large excursions line up tightly with the
  anomaly's own big swings (loses on magnitude, wins on variance). This
  script flags every sector where the two rankings disagree on the #1
  component, since that's the case worth a sentence in the paper rather
  than collapsing "dominant" into one number.

Inputs
    results/ch3/tables/t32_attribution_by_cycle.csv        (NET; needs anomaly_from_iac)
    results/ch3/tables/t32_attribution_by_cycle_gross.csv  (GROSS; magnitude test only)
Outputs
    results/ch3/tables/t32_component_dominance_magnitude.csv
    results/ch3/tables/t32_component_dominance_variance.csv
    console: per-sector Friedman + Holm-Bonferroni pairwise Wilcoxon ranking (NET & GROSS),
             per-sector variance-share decomposition (NET only), and a flagged
             disagreement summary at the end

Paste the full console output back.
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR, SECTORS, SECTORS_ONLY, SECTOR_LABELS

COMPS = [("trend_component", "trend"), ("amplitude_component", "amplitude"),
         ("phase_component", "phase"), ("residual_apac", "residual")]
COMP_COLS = [c for c, _ in COMPS]
COMP_LABS = [l for _, l in COMPS]
PAIRS = list(itertools.combinations(COMP_LABS, 2))


def holm_bonferroni(pvals):
    """Standard Holm step-down correction. Returns adjusted p-values in the
    same order as the input."""
    pvals = np.asarray(pvals, float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj


def magnitude_dominance(tab, metric_name):
    """Friedman omnibus + Holm-Bonferroni pairwise Wilcoxon, per sector, on
    |component| magnitudes. Returns a list of per-pair result dicts."""
    rows = []
    print(f"\n{'=' * 100}")
    print(f"(A) MAGNITUDE DOMINANCE -- {metric_name.upper()} -- "
          f"Friedman + pairwise Wilcoxon (Holm-Bonferroni), per sector")
    print(f"{'=' * 100}")
    for sec in SECTORS:
        a = tab[tab.sector == sec].sort_values("Year")
        if len(a) < 3:
            continue
        mags = {lab: a[col].abs().values for col, lab in COMPS}
        stat, p_fried = stats.friedmanchisquare(*[mags[l] for l in COMP_LABS])
        indep = sec in SECTORS_ONLY
        p_fried_bonf5 = min(1.0, p_fried * len(SECTORS_ONLY)) if indep else np.nan
        means = {l: np.mean(mags[l]) for l in COMP_LABS}
        ranked = sorted(COMP_LABS, key=lambda l: -means[l])
        print(f"\n  -- {SECTOR_LABELS[sec]} ({metric_name}) -- "
              f"Friedman chi2={stat:.3f}, p={p_fried:.4g}"
              + (f", p_bonf5={p_fried_bonf5:.4g}" if indep else " (Circumpolar, not corrected/pooled)")
              + f", n={len(a)}")
        print(f"     mean |contribution|: " + ", ".join(f"{l}={means[l]:.4g}" for l in COMP_LABS))
        print(f"     ranked (largest first): {' > '.join(ranked)}")
        pair_rows = []
        pvals = []
        for l1, l2 in PAIRS:
            try:
                w_stat, w_p = stats.wilcoxon(mags[l1], mags[l2])
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            pair_rows.append((l1, l2, w_stat, w_p))
            pvals.append(w_p if np.isfinite(w_p) else 1.0)
        adj = holm_bonferroni(pvals)
        print("     pairwise Wilcoxon (Holm-Bonferroni over 6 comparisons):")
        sector_rows = []
        for (l1, l2, w_stat, w_p), p_adj in zip(pair_rows, adj):
            sig = bool(p_adj < 0.05)
            bigger = l1 if means[l1] > means[l2] else l2
            flag = "*" if sig else " "
            print(f"       {l1:10s} vs {l2:10s}: W={w_stat:8.1f}  p={w_p:.4g}  "
                  f"p_holm={p_adj:.4g} {flag}  ({bigger} bigger)")
            sector_rows.append(dict(metric=metric_name, sector=SECTOR_LABELS[sec], comp1=l1, comp2=l2,
                                     mean1=means[l1], mean2=means[l2], W=w_stat, p=w_p, p_holm=p_adj,
                                     bigger=bigger, sig=sig))
        leader = ranked[0]
        clear = all(r["sig"] and r["bigger"] == leader
                    for r in sector_rows if leader in (r["comp1"], r["comp2"]))
        print(f"     magnitude leader: {leader}"
              f"{'  (significantly bigger than all 3 others)' if clear else '  (NOT clearly separated from all others)'}")
        for r in sector_rows:
            r["friedman_chi2"] = stat
            r["friedman_p"] = p_fried
            r["friedman_p_bonf5"] = p_fried_bonf5
            r["independent"] = indep
            r["leader"] = leader
            r["leader_clear"] = clear
        rows.extend(sector_rows)
    return rows


def variance_share(tab):
    """Exact Cov(component, anomaly)/Var(anomaly) decomposition, NET only."""
    rows = []
    print(f"\n{'=' * 100}")
    print("(B) VARIANCE-SHARE DOMINANCE -- NET only -- Cov(component, anomaly) / Var(anomaly), per sector")
    print("(shares sum to exactly 1 by construction; a negative share means that component DAMPENS variability)")
    print(f"{'=' * 100}")
    for sec in SECTORS:
        a = tab[tab.sector == sec].sort_values("Year")
        if len(a) < 3:
            continue
        if "anomaly_from_iac" not in a.columns:
            print(f"\n  -- {SECTOR_LABELS[sec]}: SKIPPED, no anomaly_from_iac column in this table")
            continue
        anomaly = a["anomaly_from_iac"].values
        var_anom = np.var(anomaly, ddof=1)
        recon = a[COMP_COLS].sum(axis=1).values
        max_resid = np.max(np.abs(recon - anomaly))
        shares = {}
        for col, lab in COMPS:
            cov = np.cov(a[col].values, anomaly, ddof=1)[0, 1]
            shares[lab] = cov / var_anom
        total = sum(shares.values())
        leader = max(shares, key=shares.get)
        print(f"\n  -- {SECTOR_LABELS[sec]} -- Var(anomaly)={var_anom:.4g}  "
              f"(sum-check: max|recon - anomaly| = {max_resid:.2e}, should be ~0)")
        print("     " + ", ".join(f"{l}={shares[l]:+.3f}" for l in COMP_LABS) + f"   (sum={total:.4f})")
        print(f"     variance leader: {leader}")
        rows.append(dict(sector=SECTOR_LABELS[sec], var_anomaly=var_anom, n=len(a),
                          sum_check_max_abs_err=max_resid, leader=leader,
                          **{f"share_{l}": shares[l] for l in COMP_LABS}))
    return rows


def main():
    net_path = os.path.join(TABLES_DIR, "t32_attribution_by_cycle.csv")
    gross_path = os.path.join(TABLES_DIR, "t32_attribution_by_cycle_gross.csv")
    if not os.path.exists(net_path):
        raise FileNotFoundError(f"{net_path} not found -- run fig_04_attribution_annual.py first.")
    net_tab = pd.read_csv(net_path)

    mag_rows = magnitude_dominance(net_tab, "net")
    if os.path.exists(gross_path):
        gross_tab = pd.read_csv(gross_path)
        mag_rows += magnitude_dominance(gross_tab, "gross")
    else:
        print(f"\n  ({gross_path} not found -- skipping GROSS magnitude test)")

    var_rows = variance_share(net_tab)

    mag_df = pd.DataFrame(mag_rows)
    var_df = pd.DataFrame(var_rows)
    mag_out = os.path.join(TABLES_DIR, "t32_component_dominance_magnitude.csv")
    var_out = os.path.join(TABLES_DIR, "t32_component_dominance_variance.csv")
    mag_df.to_csv(mag_out, index=False)
    var_df.to_csv(var_out, index=False)
    print(f"\nwrote {mag_out}  ({len(mag_df)} rows)")
    print(f"wrote {var_out}  ({len(var_df)} rows)")

    print(f"\n{'=' * 100}")
    print("DISAGREEMENT FLAGS -- sectors where the magnitude leader (NET) != the variance leader")
    print(f"{'=' * 100}")
    mag_leaders = (mag_df[mag_df.metric == "net"][["sector", "leader", "leader_clear"]]
                   .drop_duplicates().reset_index(drop=True))
    var_leaders = var_df[["sector", "leader"]].rename(columns={"leader": "var_leader"})
    merged = mag_leaders.merge(var_leaders, on="sector", how="inner")
    any_flag = False
    for _, row in merged.iterrows():
        if row["leader"] != row["var_leader"]:
            any_flag = True
            print(f"  {row['sector']:20s}  magnitude leader (NET) = {row['leader']:10s}"
                  f"{'  (clear)' if row['leader_clear'] else '  (not clearly separated)'}"
                  f"   vs   variance leader = {row['var_leader']}   <-- DISAGREE")
        else:
            print(f"  {row['sector']:20s}  agree: {row['leader']}")
    if not any_flag:
        print("\n  (no sector disagrees between magnitude and variance dominance)")


if __name__ == "__main__":
    main()