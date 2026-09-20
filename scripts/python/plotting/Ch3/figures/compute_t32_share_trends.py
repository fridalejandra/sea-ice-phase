#!/usr/bin/env python3
"""
test_trend_share_over_time.py -- tests whether each component's share of the
cycle's total |contribution| has a monotonic trend across the WHOLE record,
rather than assuming any particular boundary (2016 or otherwise). This is
deliberately boundary-free: it correlates each component's per-cycle share
directly against Year, so it doesn't inherit the "only true if you split
exactly here" fragility that several other results in this chapter already
carry a caveat about.

Primary test: Pearson r(share_trend, Year) per sector, both NET and GROSS.
Spearman rho is reported alongside as the chapter's standard robustness
companion (matches the convention used for the phase-amplitude coupling
test in S3.2: Fisher's z on Pearson AND Spearman's rank on the same
question).

Multiple comparisons: five sectors are independent (Weddell, ABS, Ross, East
Antarctica, King Haakon); Circumpolar is their sum, not an independent sixth
draw (same exclusion PRIMARY_PAIRS/t34_coupling_shift_test.py already use),
so it is reported separately rather than folded into the correction or the
pooled estimate. Bonferroni over the five independent sectors.

Pooling: the five independent sectors' Pearson correlations are combined by
the same random-effects (DerSimonian-Laird) meta-analysis already used for
the phase-amplitude pooled coupling estimate (t35_pooled_meta.csv), giving
one combined r with a CI, p, and I^2 (heterogeneity) rather than reporting
five separate tests as if that settles it.

Every component is tested, not just trend -- since shares sum to 1, if
trend's share is rising, something else's share must be falling; this shows
which one(s) carry the offsetting decline, per sector and pooled.

Inputs
    results/ch3/tables/t32_attribution_by_cycle.csv        (NET)
    results/ch3/tables/t32_attribution_by_cycle_gross.csv  (GROSS)
Outputs
    results/ch3/tables/t32_trend_share_over_time.csv
    console: per-sector tests, Bonferroni-corrected p, pooled estimate, per metric

Paste the full console output back.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR, SECTORS_ONLY, SECTOR_LABELS, SECTORS

COMPS = [("trend_component", "trend"), ("amplitude_component", "amplitude"),
         ("phase_component", "phase"), ("residual_apac", "residual")]
COMP_COLS = [c for c, _ in COMPS]


def fz(r):
    return 0.5 * np.log((1 + r) / (1 - r))


def fz_inv(z):
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def meta(rs, ns):
    """DerSimonian-Laird random-effects pooling of Fisher-z correlations --
    identical method to t35_pooled_meta.csv, reused here rather than
    reimplemented differently."""
    rs, ns = np.asarray(rs, float), np.asarray(ns, float)
    ok = np.isfinite(rs) & (ns > 3)
    rs, ns = rs[ok], ns[ok]
    k = len(rs)
    if k < 2:
        return None
    z = fz(rs)
    v = 1 / (ns - 3)
    w = 1 / v
    z_fe = np.sum(w * z) / np.sum(w)
    Q = np.sum(w * (z - z_fe) ** 2)
    df = k - 1
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_re = 1 / (v + tau2)
    z_re = np.sum(w_re * z) / np.sum(w_re)
    se = np.sqrt(1 / np.sum(w_re))
    return dict(k=k, r_re=fz_inv(z_re), lo=fz_inv(z_re - 1.96 * se), hi=fz_inv(z_re + 1.96 * se),
                p=2 * stats.norm.sf(abs(z_re / se)), I2=I2, p_Q=stats.chi2.sf(Q, df) if df > 0 else np.nan)


def load_shares(metric):
    name = "t32_attribution_by_cycle_gross.csv" if metric == "gross" else "t32_attribution_by_cycle.csv"
    path = os.path.join(TABLES_DIR, name)
    tab = pd.read_csv(path)
    for c in COMP_COLS:
        tab[c] = tab[c].abs()
    tab["total"] = tab[COMP_COLS].sum(axis=1)
    for c, lab in COMPS:
        tab[f"share_{lab}"] = tab[c] / tab["total"]
    return tab


all_rows = []
for metric in ("net", "gross"):
    tab = load_shares(metric)
    print("\n" + "=" * 100)
    print(f"METRIC = {metric.upper()}  -- Pearson/Spearman r(component share, Year), per sector, no boundary assumed")
    print("=" * 100)
    for _, comp_lab in COMPS:
        print(f"\n  -- share_{comp_lab} vs Year --")
        rows = []
        for sec in SECTORS:  # includes circumpolar, reported but excluded from pooling/correction
            a = tab[tab.sector == sec].sort_values("Year")
            x, y = a["Year"].values, a[f"share_{comp_lab}"].values
            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
            rows.append(dict(metric=metric, component=comp_lab, sector=SECTOR_LABELS[sec],
                             n=len(a), r_pearson=r_p, p_pearson=p_p, r_spearman=r_s, p_spearman=p_s,
                             independent=sec in SECTORS_ONLY))
        d = pd.DataFrame(rows)
        d["p_pearson_bonf5"] = np.where(d["independent"], np.minimum(1, d["p_pearson"] * len(SECTORS_ONLY)), np.nan)
        print(d[["sector", "n", "r_pearson", "p_pearson", "p_pearson_bonf5", "r_spearman", "p_spearman"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        all_rows.append(d)

        indep = d[d.independent]
        m = meta(indep["r_pearson"].values, indep["n"].values)
        if m:
            print(f"  pooled (5 independent sectors, DerSimonian-Laird): "
                  f"r = {m['r_re']:+.3f}  [{m['lo']:+.3f}, {m['hi']:+.3f}]  "
                  f"p = {m['p']:.4g}   I2 = {m['I2']:.0f}%  (p_Q = {m['p_Q']:.3g})")

out = pd.concat(all_rows, ignore_index=True)
out_path = os.path.join(TABLES_DIR, "t32_trend_share_over_time.csv")
out.to_csv(out_path, index=False)
print(f"\nwrote {out_path}  ({len(out)} rows)")
