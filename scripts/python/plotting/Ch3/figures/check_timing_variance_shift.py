"""
check_timing_variance_shift.py — §3.3 cross-check: did timing variability
change after 2016, sector by sector, and does the OBSERVED timing agree with
the FITTED phase component about where?

Renamed 2026-09-18 from t33_variance_prepost.py -- logic unchanged, only the
filename (the "t33" table-number prefix didn't say anything about what this
actually checks). Its output table is still named t33_variance_prepost.csv
and the supplementary figure fig_s33_variance_ratio.png -- those names are
unchanged and tracked in _provenance_audit.csv, so they're left alone even
though the script's own name no longer matches them exactly.

Motivation (from §3.2): the fitted phase component's cycle-mean magnitude
grew after 2016 in the Weddell (x1.6) and Ross (x1.4) and shrank in the other
three sectors. That is a model quantity extracted after amplitude. This
script asks whether the directly observed timing scalars — day of maximum,
day of minimum — show the same thing, and puts the amplitude scalar beside
them for comparison.

Per sector, pre-2016 (1979–2015) vs 2016+:
  observed scalars  SD ratio (post/pre), F-test two-sided p, 95% CI on the
                    ratio, Brown–Forsythe p (robust to non-normality)
  fitted phase      ratio of mean |cycle-mean phase| (net) and of mean
                    within-cycle |phase| (gross), Mann–Whitney p, bootstrap CI

This is a CHECK, not a test of coupling. n = 9–10 after 2016; report the
effect sizes and CIs, and do not read a p-value below 0.05 in one sector out
of five as a discovery.

Inputs
    ANNUAL_CSV (period == "FULL")            from ch3_config
    results/ch3/tables/t32_attribution_by_cycle.csv
    results/ch3/tables/t32_attribution_by_cycle_gross.csv
Outputs
    results/ch3/tables/t33_variance_prepost.csv
    results/ch3/figures/fig_s33_variance_ratio.png   (small; optional)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from ch3_config import SECTORS, SECTOR_LABELS, TABLES_DIR, ANNUAL_CSV
from ch3_plot import save
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure

BREAK = 2016
rng = np.random.default_rng(0)

# ── observed scalars ────────────────────────────────────────────────────────
a = pd.read_csv(ANNUAL_CSV)
if "period" in a.columns:
    a = a[a["period"] == "FULL"]
OBS = [("max_doy_raw_anom", "day of maximum (obs)"),
       ("min_doy_raw_anom", "day of minimum (obs)"),
       ("amplitude_raw_anom", "amplitude (obs)")]
miss = [c for c, _ in OBS if c not in a.columns]
if miss:
    sys.exit(f"annual file lacks {miss}; columns are {sorted(a.columns)}")
a = a[a.sector.isin(SECTORS)]
print(f"annual scalars: {a.Year.min()}–{a.Year.max()}, "
      f"{a.Year.nunique()} years, {a.sector.nunique()} sectors")

# ── fitted phase, per cycle ─────────────────────────────────────────────────
def load_cycle(name):
    p = os.path.join(TABLES_DIR, name)
    if not os.path.exists(p):
        sys.exit(f"missing {p} — run fig_04_attribution_annual.py first")
    x = pd.read_csv(p)
    return x[x.sector.isin(SECTORS)]
net   = load_cycle("t32_attribution_by_cycle.csv")
gross = load_cycle("t32_attribution_by_cycle_gross.csv")
FIT = [(net,   "phase_component", "phase, |cycle mean| (fit)"),
       (gross, "phase_component", "phase, within-cycle |x| (fit)")]

# ── stats helpers ───────────────────────────────────────────────────────────
def var_ratio(pre, post):
    """SD ratio post/pre with two-sided F p, 95% CI, Brown–Forsythe p."""
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    pre, post = pre[~np.isnan(pre)], post[~np.isnan(post)]
    d1, d2 = len(post) - 1, len(pre) - 1
    F = np.var(post, ddof=1) / np.var(pre, ddof=1)
    p = 2 * min(stats.f.cdf(F, d1, d2), stats.f.sf(F, d1, d2))
    lo = F / stats.f.ppf(0.975, d1, d2)
    hi = F / stats.f.ppf(0.025, d1, d2)
    bf = stats.levene(pre, post, center="median").pvalue
    return dict(pre_sd=np.std(pre, ddof=1), post_sd=np.std(post, ddof=1),
                ratio=np.sqrt(F), ci_lo=np.sqrt(lo), ci_hi=np.sqrt(hi),
                p_F=p, p_BF=bf, n_pre=len(pre), n_post=len(post))

def mag_ratio(pre, post, B=2000):
    """Ratio of mean |x| post/pre, Mann–Whitney p, bootstrap 95% CI."""
    pre, post = np.abs(np.asarray(pre, float)), np.abs(np.asarray(post, float))
    pre, post = pre[~np.isnan(pre)], post[~np.isnan(post)]
    r = post.mean() / pre.mean()
    p = stats.mannwhitneyu(pre, post, alternative="two-sided").pvalue
    bs = [rng.choice(post, len(post)).mean() / rng.choice(pre, len(pre)).mean()
          for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(pre_sd=pre.mean(), post_sd=post.mean(),   # columns reused: these are means
                ratio=r, ci_lo=lo, ci_hi=hi, p_F=p, p_BF=np.nan,
                n_pre=len(pre), n_post=len(post))

# ── run ─────────────────────────────────────────────────────────────────────
rows = []
for sec in SECTORS:
    s = a[a.sector == sec]
    for col, lab in OBS:
        r = var_ratio(s.loc[s.Year < BREAK, col], s.loc[s.Year >= BREAK, col])
        rows.append(dict(sector=SECTOR_LABELS[sec], quantity=lab, kind="observed", **r))
    for tab, col, lab in FIT:
        t = tab[tab.sector == sec]
        r = mag_ratio(t.loc[t.Year < BREAK, col], t.loc[t.Year >= BREAK, col])
        rows.append(dict(sector=SECTOR_LABELS[sec], quantity=lab, kind="fitted", **r))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(TABLES_DIR, "t33_variance_prepost.csv"), index=False)

pd.set_option("display.width", 160)
print("\n" + "=" * 110)
print(f"post-{BREAK} / pre-{BREAK}. Observed rows: SD ratio, F-test p, Brown–Forsythe p. "
      f"Fitted rows: ratio of mean |x|, Mann–Whitney p.")
print("=" * 110)
for lab in [l for _, l in OBS] + [l for _, _, l in FIT]:
    x = res[res.quantity == lab]
    print(f"\n-- {lab} --")
    print(x[["sector", "pre_sd", "post_sd", "ratio", "ci_lo", "ci_hi", "p_F", "p_BF", "n_pre", "n_post"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# the question, answered in one table
piv = res.pivot(index="sector", columns="quantity", values="ratio")
piv = piv[[l for _, l in OBS] + [l for _, _, l in FIT]]
print("\n" + "=" * 110)
print(f"post/pre ratios side by side (>1 = more variable after {BREAK})")
print("=" * 110)
print(piv.to_string(float_format=lambda v: f"{v:.2f}"))

# ── small figure: ratios with CIs, per sector ──────────────────────────────
quants = [l for _, l in OBS] + [l for _, _, l in FIT]
cols   = ["#2c7fb8", "#7fb3d5", "#5b6b7a", "#e8703a", "#f2a77a"]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
x = np.arange(len(SECTORS)); w = 0.15
for j, (q, c) in enumerate(zip(quants, cols)):
    t = res[res.quantity == q].set_index("sector").loc[[SECTOR_LABELS[s] for s in SECTORS]]
    xx = x + (j - 2) * w
    ax.errorbar(xx, t.ratio, yerr=[t.ratio - t.ci_lo, t.ci_hi - t.ratio],
                fmt="o", ms=4, color=c, ecolor=c, elinewidth=1, capsize=2, label=q)
ax.axhline(1, color="k", lw=0.8)
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels([SECTOR_LABELS[s] for s in SECTORS], fontsize=8.5)
ax.set_ylabel(f"post-{BREAK} / pre-{BREAK}", fontsize=9)
ax.tick_params(labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
fig.tight_layout()
save(fig, "fig_s33_variance_ratio.png", sync=False)
print("\nwrote t33_variance_prepost.csv, fig_s33_variance_ratio.png")