#!/usr/bin/env python3
"""
ch3_stats.py — every statistic quoted in Chapter 3, from two inputs.
=====================================================================
Inputs  (paths from ch3_config.py):
    ANNUAL_CSV  data/ch3/annual_params_E.csv          (R/ch3/01_fit_apac.R)
    INDEX_CSV   data/ch3/master_index_detrended.csv   (processing/compute_atmospheric_correlations.py)
    INDEX_DIR/asli_era5_v3-latest.csv                 (Hosking ASL v3: lon, lat, RelCenPres)

Outputs (results/ch3/tables/):
    ch3_numbers.csv                     the ledger: one row per number in the text
    t33_phase_amp_by_sector.csv         full-record r(phase, amplitude) per sector
    t33_phase_amp_splits.csv            pooled pre/post estimates, both splits
    t33_phase_amp_loo2016.csv           leave-one-year-out on the post-2016 pooled r
    t33_phase_amp_rolling10.csv         10-yr rolling Spearman, per sector-year
    t34_variance_ratio_2016.csv         post/pre-2016 variance ratios (F-test)
    t35_index_scan_raw.csv              6 sectors x 35 index cols x 2 targets, BH-FDR
    t35_primary_pairs.csv               pre-specified pairs: r, p, LOO, seasons-consistent
    t35_pooled_meta.csv                 random-effects pooling across sectors (I^2)
    t36_stationarity_2001.csv           pre/post-2001 r and Fisher-z shift, primary pairs
    t36_ross_asl_detail.csv             Ross~ASL by season, split sweep, LOO, lon/lat
    tS1_fitted_vs_raw.csv               r(fitted, observed) per sector for timing and amplitude

RULES (methods 2.2.3): every test below uses the OBSERVED scalars
(max_doy_raw_anom, min_doy_raw_anom, amplitude_raw_anom). Fitted components
appear only in tS1, as the evidence for that rule. Series are linearly
detrended before correlation unless the row says otherwise. Indices in
INDEX_CSV are already detrended.

Run:  python ch3_stats.py            (from scripts/python/plotting/Ch3/figures)
"""
import os, sys, datetime
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if len(sys.argv) < 5:
    from ch3_config import (ANNUAL_CSV, INDEX_CSV, INDEX_DIR, INDEX_FILES, TABLES_DIR,
                            NUMBERS_CSV, SECTORS, SECTOR_LABELS, SECTORS_ONLY,
                            YEAR_MIN, YEAR_MAX, BREAK_YEAR, SPLIT_YEAR, ROLL_SHORT,
                            SEASONS, PRIMARY_PAIRS)
else:  # standalone use: python ch3_stats.py annual.csv index.csv asl.csv outdir
    ANNUAL_CSV, INDEX_CSV, ASL_RAW, TABLES_DIR = sys.argv[1:5]
    INDEX_DIR, INDEX_FILES = os.path.dirname(ASL_RAW), {"ASL": os.path.basename(ASL_RAW)}
    NUMBERS_CSV = os.path.join(TABLES_DIR, "ch3_numbers.csv")
    SECTORS = ["SIE_Weddell", "SIE_Amundsen_Bellingshausen", "SIE_Ross",
               "SIE_East_Antarctica", "SIE_King_Haakon", "SIE_circumpolar"]
    SECTOR_LABELS = {s: s.replace("SIE_", "").replace("_", " ") for s in SECTORS}
    SECTORS_ONLY = SECTORS[:-1]
    YEAR_MIN, YEAR_MAX, BREAK_YEAR, SPLIT_YEAR, ROLL_SHORT = 1979, 2023, 2016, 2001, 10
    SEASONS = ["annual", "DJF", "MAM", "JJA", "SON", "ADV", "RET"]
    PRIMARY_PAIRS = [
        ("SIE_Weddell", "amplitude_raw_anom", "Nino34_SON", "ENSO dipole"),
        ("SIE_King_Haakon", "amplitude_raw_anom", "Nino34_annual", "ENSO dipole"),
        ("SIE_Ross", "amplitude_raw_anom", "ASL_annual", "ASL-Ross"),
        ("SIE_Amundsen_Bellingshausen", "amplitude_raw_anom", "SAM_JJA", "SAM-ABS"),
        ("SIE_East_Antarctica", "max_doy_raw_anom", "SAM_RET", "SAM-EA retreat"),
        ("SIE_King_Haakon", "max_doy_raw_anom", "ZW3R_SON", "ZW3"),
        ("SIE_Weddell", "amplitude_raw_anom", "ZW3R_annual", "ZW3"),
    ]
os.makedirs(TABLES_DIR, exist_ok=True)

PH, AMP, EXT = "max_doy_raw_anom", "amplitude_raw_anom", "sie_annual"
SCRIPT = "ch3_stats.py"
STAMP = datetime.date.today().isoformat()
LEDGER = []


def note(section, quantity, value, sector="all", n=np.nan, p=np.nan, extra=""):
    LEDGER.append(dict(section=section, quantity=quantity, sector=sector,
                       value=value, n=n, p=p, extra=extra, source=SCRIPT, date=STAMP))


# ── helpers ───────────────────────────────────────────────────────────────────
def detrend(x):
    x = np.asarray(x, float); ok = np.isfinite(x)
    if ok.sum() < 3: return x
    t = np.arange(len(x), dtype=float); b, a = np.polyfit(t[ok], x[ok], 1)
    out = x.copy(); out[ok] = x[ok] - (a + b * t[ok]); return out

def fz(r): return 0.5 * np.log((1 + r) / (1 - r))
def fz_inv(z): return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def pear(x, y, dt=True):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if dt: x, y = detrend(x), detrend(y)
    ok = np.isfinite(x) & np.isfinite(y); n = int(ok.sum())
    if n < 6: return np.nan, np.nan, n
    r, p = stats.pearsonr(x[ok], y[ok]); return float(r), float(p), n

def spear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); n = int(ok.sum())
    if n < 6: return np.nan, np.nan, n
    r, p = stats.spearmanr(x[ok], y[ok]); return float(r), float(p), n

def zshift(r1, n1, r2, n2):
    if not (np.isfinite(r1) and np.isfinite(r2)) or n1 < 4 or n2 < 4: return np.nan, np.nan
    z = (fz(r1) - fz(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(z), float(2 * stats.norm.sf(abs(z)))

def meta(rs, ns):
    """DerSimonian-Laird random-effects pooling of Fisher-z correlations."""
    rs, ns = np.asarray(rs, float), np.asarray(ns, float)
    ok = np.isfinite(rs) & (ns > 3); rs, ns = rs[ok], ns[ok]; k = len(rs)
    if k < 2: return None
    z = fz(rs); v = 1 / (ns - 3); w = 1 / v
    z_fe = np.sum(w * z) / np.sum(w); Q = np.sum(w * (z - z_fe) ** 2); df = k - 1
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w); tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_re = 1 / (v + tau2); z_re = np.sum(w_re * z) / np.sum(w_re); se = np.sqrt(1 / np.sum(w_re))
    return dict(k=k, r_re=fz_inv(z_re), lo=fz_inv(z_re - 1.96 * se), hi=fz_inv(z_re + 1.96 * se),
                se_z=se, p=2 * stats.norm.sf(abs(z_re / se)), I2=I2, Q=Q, p_Q=stats.chi2.sf(Q, df))

def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / (np.arange(m) + 1))[::-1])[::-1]
    return np.minimum(q, 1)

def sec_df(ann, sec, lo=None, hi=None):
    a = ann[ann.sector == sec].sort_values("Year")
    if lo is not None: a = a[a.Year >= lo]
    if hi is not None: a = a[a.Year <= hi]
    return a

def lab(s): return SECTOR_LABELS.get(s, s)


# ── load ──────────────────────────────────────────────────────────────────────
ann = pd.read_csv(ANNUAL_CSV)
ann = ann[ann.Year.between(YEAR_MIN, YEAR_MAX)].copy()
assert ann.Year.min() == YEAR_MIN, f"annual_params starts {ann.Year.min()}, expected {YEAR_MIN}"
idx = pd.read_csv(INDEX_CSV)
ICOLS = [c for c in idx.columns if c != "Year"]
ai = ann.merge(idx, "left", "Year")
n_years = ann.Year.nunique()
print(f"annual_params: {len(ann)} rows, {n_years} years {ann.Year.min()}-{ann.Year.max()}, "
      f"{ann.sector.nunique()} sectors | index cols: {len(ICOLS)}")
note("data", "years", n_years, extra=f"{ann.Year.min()}-{ann.Year.max()}")


# ── S1  fitted vs observed scalars (why the rule exists) ─────────────────────
rows = []
for sec in SECTORS:
    a = sec_df(ann, sec)
    for fit, raw, name in [("max_doy_anom", PH, "timing"), ("amplitude_anom", AMP, "amplitude")]:
        r, p, n = pear(a[fit], a[raw]); rows.append(dict(sector=lab(sec), quantity=name, r_fitted_vs_observed=r, p=p, n=n))
        note("S1/methods", f"r(fitted, observed) {name}", round(r, 3), lab(sec), n, p)
pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "tS1_fitted_vs_raw.csv"), index=False)


# ── 3.3  phase ~ amplitude ────────────────────────────────────────────────────
rows = []
for sec in SECTORS:
    a = sec_df(ann, sec)
    r, p, n = pear(a[PH], a[AMP]); rho, ps, _ = spear(a[PH], a[AMP])
    rows.append(dict(sector=lab(sec), r_pearson_detrended=r, p_pearson=p, rho_spearman=rho, p_spearman=ps, n=n))
    note("3.3", "r(phase, amplitude) full record, Pearson detrended", round(r, 3), lab(sec), n, p)
pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "t33_phase_amp_by_sector.csv"), index=False)

def pooled(lo, hi, method="pearson"):
    rs, ns = [], []
    for sec in SECTORS:
        a = sec_df(ann, sec, lo, hi)
        r, _, n = pear(a[PH], a[AMP]) if method == "pearson" else spear(a[PH], a[AMP])
        rs.append(r); ns.append(n)
    return meta(rs, ns), dict(zip([lab(s) for s in SECTORS], rs)), ns

rows = []
for split in (SPLIT_YEAR, BREAK_YEAR):
    for method in ("pearson", "spearman"):
        m1, r1, n1 = pooled(YEAR_MIN, split - 1, method); m2, r2, n2 = pooled(split, YEAR_MAX, method)
        z, p = zshift(m1["r_re"], sum(n1), m2["r_re"], sum(n2))
        # shift test on pooled estimates uses the meta SEs
        zd = (fz(m1["r_re"]) - fz(m2["r_re"])) / np.sqrt(m1["se_z"] ** 2 + m2["se_z"] ** 2)
        p_shift = 2 * stats.norm.sf(abs(zd))
        rows.append(dict(split=split, method=method,
                         r_pre=m1["r_re"], lo_pre=m1["lo"], hi_pre=m1["hi"], p_pre=m1["p"], I2_pre=m1["I2"],
                         r_post=m2["r_re"], lo_post=m2["lo"], hi_post=m2["hi"], p_post=m2["p"], I2_post=m2["I2"],
                         p_shift=p_shift, **{f"post_{k}": v for k, v in r2.items()}))
        note("3.3", f"pooled r(phase,amp) pre-{split} ({method})", round(m1["r_re"], 3), "pooled", sum(n1), m1["p"], f"I2={m1['I2']:.0f}%")
        note("3.3", f"pooled r(phase,amp) {split}+ ({method})", round(m2["r_re"], 3), "pooled", sum(n2), m2["p"], f"I2={m2['I2']:.0f}%")
        note("3.3", f"shift pre/post {split} ({method})", round(p_shift, 3), "pooled", extra="p of Fisher-z difference")
        if split == BREAK_YEAR and method == "pearson":
            for s, r in r2.items(): note("3.3", f"r(phase,amp) {split}-{YEAR_MAX}", round(r, 2), s, n2[0])
pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "t33_phase_amp_splits.csv"), index=False)

# leave-one-year-out on the post-2016 pooled estimate
rows = []
for y in range(BREAK_YEAR, YEAR_MAX + 1):
    sub = ann[ann.Year != y]
    rs, ns = [], []
    for sec in SECTORS:
        a = sub[(sub.sector == sec) & (sub.Year >= BREAK_YEAR)].sort_values("Year")
        r, _, n = pear(a[PH], a[AMP]); rs.append(r); ns.append(n)
    m = meta(rs, ns); rows.append(dict(dropped=y, r_re=m["r_re"], p=m["p"]))
loo = pd.DataFrame(rows); loo.to_csv(os.path.join(TABLES_DIR, "t33_phase_amp_loo2016.csv"), index=False)
note("3.3", f"LOO on {BREAK_YEAR}+ pooled r: worst p", round(loo.p.max(), 3), "pooled",
     extra=f"dropping {int(loo.loc[loo.p.idxmax(), 'dropped'])}; r range {loo.r_re.min():.2f}-{loo.r_re.max():.2f}")

# 10-yr rolling Spearman (matches the fig03 statistic) and the 2007-15 vs 2016-23 contrast
rows, summ = [], []
for sec in SECTORS:
    a = sec_df(ann, sec).reset_index(drop=True)
    for i in range(ROLL_SHORT - 1, len(a)):
        w = a.iloc[i - ROLL_SHORT + 1:i + 1]; rho, _, _ = spear(w[PH], w[AMP])
        rows.append(dict(sector=lab(sec), end_year=int(w.Year.iloc[-1]), rho=rho))
    rr = pd.DataFrame([r for r in rows if r["sector"] == lab(sec)])
    pre, post = rr[rr.end_year < BREAK_YEAR].rho.mean(), rr[rr.end_year >= BREAK_YEAR].rho.mean()
    a1 = sec_df(ann, sec, 2007, 2015); a2 = sec_df(ann, sec, BREAK_YEAR, YEAR_MAX)
    r1, _, n1 = spear(a1[PH], a1[AMP]); r2, _, n2 = spear(a2[PH], a2[AMP]); _, p = zshift(r1, n1, r2, n2)
    summ.append(dict(sector=lab(sec), roll10_mean_pre2016=pre, roll10_mean_post2016=post,
                     rho_2007_2015=r1, rho_2016_2023=r2, p_contrast=p))
    note("3.3", "10yr rolling Spearman mean, windows ending <2016 / >=2016", f"{pre:+.2f} / {post:+.2f}", lab(sec))
pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "t33_phase_amp_rolling10.csv"), index=False)
pd.DataFrame(summ).to_csv(os.path.join(TABLES_DIR, "t33_phase_amp_rolling10_summary.csv"), index=False)


# ── 3.3b  min-date as the other phase marker (descriptive) ───────────────────
# The chapter's phase is day of max. Day of min is the sharper extremum (fitted
# == observed, r 0.99-1.00; SD 2-4x smaller) and is nearly independent of day
# of max (r ~ 0). Reported alongside so the two ends of the cycle can be compared.
MN = "min_doy_raw_anom"
rows = []
for sec in SECTORS:
    a = sec_df(ann, sec); pre, post = a[a.Year < BREAK_YEAR], a[a.Year >= BREAK_YEAR]
    r_full, p_full, n = pear(a[MN], a[AMP]); r_pre, _, n1 = pear(pre[MN], pre[AMP]); r_post, _, n2 = pear(post[MN], post[AMP])
    _, p_sh = zshift(r_pre, n1, r_post, n2)
    rx_pre, _, _ = pear(pre[PH], pre[AMP]); rx_post, _, _ = pear(post[PH], post[AMP])
    r_mm, p_mm, _ = pear(a[MN], a[PH])
    rows.append(dict(sector=lab(sec), r_min_amp_full=r_full, p_full=p_full, r_min_amp_pre2016=r_pre, r_min_amp_post2016=r_post,
                     p_shift=p_sh, r_max_amp_pre2016=rx_pre, r_max_amp_post2016=rx_post, r_min_vs_max=r_mm, p_min_vs_max=p_mm,
                     sd_min_date=float(np.nanstd(a[MN], ddof=1)), sd_max_date=float(np.nanstd(a[PH], ddof=1))))
    note("3.3b", f"r(min-date, amplitude) pre/post {BREAK_YEAR}", f"{r_pre:+.2f} -> {r_post:+.2f}", lab(sec), p=p_sh)
mn = pd.DataFrame(rows); mn.to_csv(os.path.join(TABLES_DIR, "t33b_mindate_vs_amplitude.csv"), index=False)
rs, ns = [], []
for sec in SECTORS:
    a = sec_df(ann, sec, BREAK_YEAR, YEAR_MAX); r, _, n = pear(a[MN], a[AMP]); rs.append(r); ns.append(n)
m_post = meta(rs, ns); rs, ns = [], []
for sec in SECTORS:
    a = sec_df(ann, sec, YEAR_MIN, BREAK_YEAR - 1); r, _, n = pear(a[MN], a[AMP]); rs.append(r); ns.append(n)
m_pre = meta(rs, ns)
zd = (fz(m_pre["r_re"]) - fz(m_post["r_re"])) / np.sqrt(m_pre["se_z"] ** 2 + m_post["se_z"] ** 2)
note("3.3b", f"pooled r(min-date, amplitude) pre-{BREAK_YEAR}", round(m_pre["r_re"], 3), "pooled", p=m_pre["p"], extra=f"I2={m_pre['I2']:.0f}%")
note("3.3b", f"pooled r(min-date, amplitude) {BREAK_YEAR}+", round(m_post["r_re"], 3), "pooled", p=m_post["p"], extra=f"I2={m_post['I2']:.0f}%")
note("3.3b", f"shift pre/post {BREAK_YEAR} (min-date)", round(2 * stats.norm.sf(abs(zd)), 3), "pooled")


# ── 3.3c  growth-season length (day of max − day of min) vs amplitude ────────
# Unifies the max-date and min-date results: an early minimum and a late maximum
# both lengthen the growth season. Tested per sector, pre/post-2016.
ann["growth_len"] = ann["max_doy_raw"] - ann["min_doy_raw"]
ai["growth_len"] = ai["max_doy_raw"] - ai["min_doy_raw"]
GL = "growth_len"; rows = []
for sec in SECTORS:
    a = sec_df(ann, sec); pre, post = a[a.Year < BREAK_YEAR], a[a.Year >= BREAK_YEAR]
    rf, pf, n = pear(a[GL], a[AMP]); r1, _, n1 = pear(pre[GL], pre[AMP]); r2, _, n2 = pear(post[GL], post[AMP])
    _, psh = zshift(r1, n1, r2, n2)
    rows.append(dict(sector=lab(sec), mean_len_days=float(a[GL].mean()), sd_len_pre=float(pre[GL].std(ddof=1)),
                     sd_len_post=float(post[GL].std(ddof=1)), r_full=rf, p_full=pf, r_pre2016=r1, r_post2016=r2, p_shift=psh,
                     r2_post2016=r2 ** 2))
    note("3.3c", f"r(growth-season length, amplitude) pre/post {BREAK_YEAR}", f"{r1:+.2f} -> {r2:+.2f}", lab(sec), n2, psh)
gl = pd.DataFrame(rows); gl.to_csv(os.path.join(TABLES_DIR, "t33c_growth_length_vs_amplitude.csv"), index=False)
# Weddell year table for the text
w = ann[(ann.sector == "SIE_Weddell") & (ann.Year >= BREAK_YEAR)][["Year", "min_doy_raw", "max_doy_raw", GL, "amplitude_raw_yr"]]
w.to_csv(os.path.join(TABLES_DIR, "t33c_weddell_post2016_years.csv"), index=False)
# does any index set Weddell growth-season length? (H5 atmospheric test, full record, all index cols)
rows = []
a = sec_df(ai, "SIE_Weddell")
for ic in ICOLS:
    r, p, n = pear(a[ic], a[GL]); rows.append(dict(index=ic, r=r, p=p, n=n))
gw = pd.DataFrame(rows).sort_values("p"); gw["q_bh"] = bh(gw.p.values)
gw.to_csv(os.path.join(TABLES_DIR, "t35_H5_weddell_growth_length_vs_indices.csv"), index=False)
note("3.5/H5", "Weddell growth length ~ indices: best", f"{gw.iloc[0]['index']} r={gw.iloc[0].r:+.2f}", "Weddell", p=gw.iloc[0].p,
     extra=f"BH q={gw.iloc[0].q_bh:.2f}; {int((gw.p < 0.05).sum())}/{len(gw)} at p<0.05")


# ── 3.3d  breakpoint sensitivity for the recent-period results ───────────────
# Where the pre/post boundary is drawn matters for the pooled coupling and not
# for the Weddell growth-season result; the table makes that visible.
rows = []
for B in range(2013, 2018):
    rs, ns = [], []
    for sec in SECTORS:
        a = sec_df(ann, sec, B, YEAR_MAX); r, _, n = pear(a[PH], a[AMP]); rs.append(r); ns.append(n)
    m2 = meta(rs, ns); rs, ns = [], []
    for sec in SECTORS:
        a = sec_df(ann, sec, YEAR_MIN, B - 1); r, _, n = pear(a[PH], a[AMP]); rs.append(r); ns.append(n)
    m1 = meta(rs, ns)
    zd = (fz(m1["r_re"]) - fz(m2["r_re"])) / np.sqrt(m1["se_z"] ** 2 + m2["se_z"] ** 2)
    w_ = sec_df(ann, "SIE_Weddell"); pre, post = w_[w_.Year < B], w_[w_.Year >= B]
    r1, _, n1 = pear(pre[GL], pre[AMP]); r2, _, n2 = pear(post[GL], post[AMP]); _, pw = zshift(r1, n1, r2, n2)
    ea = sec_df(ann, "SIE_East_Antarctica"); vr = np.nanvar(ea[ea.Year >= B][AMP], ddof=1) / np.nanvar(ea[ea.Year < B][AMP], ddof=1)
    rows.append(dict(breakpoint=B, n_post=YEAR_MAX - B + 1, pooled_r_pre=m1["r_re"], pooled_r_post=m2["r_re"],
                     p_post=m2["p"], p_shift=2 * stats.norm.sf(abs(zd)),
                     weddell_growth_r_pre=r1, weddell_growth_r_post=r2, weddell_growth_p_shift=pw,
                     east_antarctica_amp_var_ratio=vr))
bp = pd.DataFrame(rows); bp.to_csv(os.path.join(TABLES_DIR, "t33d_breakpoint_sensitivity.csv"), index=False)
note("3.3d", "pooled coupling significant only for break =", ", ".join(str(int(b)) for b in bp[bp.p_post < 0.05].breakpoint), "pooled")
note("3.3d", "Weddell growth-length shift p<0.05 for break in", ", ".join(str(int(b)) for b in bp[bp.weddell_growth_p_shift < 0.05].breakpoint), "Weddell")


# ── 3.4  variance of the components, post/pre-2016 ───────────────────────────
rows = []
for sec in SECTORS:
    a = sec_df(ann, sec)
    pre, post = a.Year < BREAK_YEAR, a.Year >= BREAK_YEAR
    for v, name in [(PH, "phase (day of max)"), (AMP, "amplitude"), (EXT, "annual extent")]:
        for dt in (False, True):
            x = detrend(a[v].values) if dt else a[v].values.astype(float)
            vr = np.nanvar(x[post], ddof=1) / np.nanvar(x[pre], ddof=1)
            d1, d2 = post.sum() - 1, pre.sum() - 1
            p = 2 * min(stats.f.cdf(vr, d1, d2), stats.f.sf(vr, d1, d2))
            rows.append(dict(sector=lab(sec), variable=name, detrended=dt, var_ratio_post_pre=vr, p_F=p,
                             sd_pre=np.nanstd(x[pre], ddof=1), sd_post=np.nanstd(x[post], ddof=1),
                             n_pre=int(pre.sum()), n_post=int(post.sum())))
            if not dt: note("3.4", f"variance ratio {BREAK_YEAR}+/pre, {name}", round(vr, 2), lab(sec), int(post.sum()), p)
pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "t34_variance_ratio_2016.csv"), index=False)


# ── 3.5  atmosphere: exploratory scan (raw targets), then pre-specified pairs ─
rows = []
for sec in SECTORS:
    a = sec_df(ai, sec)
    for tv in (PH, AMP):
        y = detrend(a[tv].values)
        for ic in ICOLS:
            x = a[ic].values.astype(float); ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10: continue
            r, p = stats.pearsonr(x[ok], y[ok])
            rows.append(dict(sector=lab(sec), target=tv, index=ic, n=int(ok.sum()), r=r, p=p))
scan = pd.DataFrame(rows); scan["q_bh"] = bh(scan.p.values)
scan.sort_values("p").to_csv(os.path.join(TABLES_DIR, "t35_index_scan_raw.csv"), index=False)
note("3.5", "exploratory scan: tests", len(scan)); note("3.5", "exploratory scan: expected p<0.05 by chance", round(0.05 * len(scan)))
note("3.5", "exploratory scan: observed p<0.05", int((scan.p < 0.05).sum()))
note("3.5", "exploratory scan: BH q<0.05", int((scan.q_bh < 0.05).sum())); note("3.5", "exploratory scan: BH q<0.10", int((scan.q_bh < 0.10).sum()))

def loo_worst_p(a, tv, ic):
    ps = []
    for y in a.Year:
        b = a[a.Year != y]; r, p, _ = pear(b[ic], b[tv]); ps.append(p)
    return max(ps)

def seasons_same_sign(sec, tv, ic):
    base = ic.rsplit("_", 1)[0]; a = sec_df(ai, sec)
    r0, _, _ = pear(a[ic], a[tv]); k = 0
    for s in SEASONS:
        c = f"{base}_{s}"
        if c in a.columns:
            r, p, _ = pear(a[c], a[tv])
            if p < 0.05 and np.sign(r) == np.sign(r0): k += 1
    return k

rows = []
for sec, tv, ic, basis in PRIMARY_PAIRS:
    a = sec_df(ai, sec); r, p, n = pear(a[ic], a[tv])
    rows.append(dict(sector=lab(sec), target=tv, index=ic, basis=basis, n=n, r=r, p=p,
                     p_bonf7=min(1, p * len(PRIMARY_PAIRS)), loo_worst_p=loo_worst_p(a, tv, ic),
                     seasons_p05_same_sign=seasons_same_sign(sec, tv, ic)))
    note("3.5", f"{ic} ~ {tv}", round(r, 2), lab(sec), n, p, f"LOO worst p={rows[-1]['loo_worst_p']:.3f}")
pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "t35_primary_pairs.csv"), index=False)

# pooled across sectors (the circumpolar test): report I^2 as the finding
rows = []
for base in sorted({ic.rsplit("_", 1)[0] for _, _, ic, _ in PRIMARY_PAIRS}):
    for s in ("annual", "SON", "RET"):
        ic = f"{base}_{s}"
        if ic not in ai.columns: continue
        for tv in (PH, AMP):
            rs, ns = [], []
            for sec in SECTORS_ONLY:
                a = sec_df(ai, sec); r, _, n = pear(a[ic], a[tv]); rs.append(r); ns.append(n)
            m = meta(rs, ns)
            rows.append(dict(index=ic, target=tv, k=m["k"], r_pooled=m["r_re"], p_pooled=m["p"], I2=m["I2"], p_Q=m["p_Q"]))
pool = pd.DataFrame(rows); pool.to_csv(os.path.join(TABLES_DIR, "t35_pooled_meta.csv"), index=False)
for _, r in pool.iterrows():
    if r["index"].startswith("Nino34") and r.target == AMP:
        note("3.5", f"pooled {r['index']} ~ amplitude (5 sectors)", round(r.r_pooled, 2), "pooled", p=r.p_pooled, extra=f"I2={r.I2:.0f}% (dipole)")


# ── 3.6  stationarity of the primary pairs, split at 2001 ────────────────────
rows = []
for sec, tv, ic, basis in PRIMARY_PAIRS:
    a1 = sec_df(ai, sec, YEAR_MIN, SPLIT_YEAR - 1); a2 = sec_df(ai, sec, SPLIT_YEAR, YEAR_MAX)
    r1, _, n1 = pear(a1[ic], a1[tv]); r2, _, n2 = pear(a2[ic], a2[tv]); z, p = zshift(r1, n1, r2, n2)
    rows.append(dict(sector=lab(sec), target=tv, index=ic, r_pre=r1, n_pre=n1, r_post=r2, n_post=n2, z=z, p_shift=p,
                     p_shift_bonf7=min(1, p * len(PRIMARY_PAIRS))))
    note("3.6", f"{ic} ~ {tv}: pre/post {SPLIT_YEAR}", f"{r1:+.2f} -> {r2:+.2f}", lab(sec), p=p)
pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "t36_stationarity_2001.csv"), index=False)

# Ross ~ ASL detail: seasons, split sweep, LOO on the shift, and ASL position
ross = sec_df(ai, "SIE_Ross"); detail = []
for s in SEASONS:
    ic = f"ASL_{s}"
    if ic not in ross.columns: continue
    a1, a2 = ross[ross.Year < SPLIT_YEAR], ross[ross.Year >= SPLIT_YEAR]
    r1, _, n1 = pear(a1[ic], a1[AMP]); r2, _, n2 = pear(a2[ic], a2[AMP]); _, p = zshift(r1, n1, r2, n2)
    detail.append(dict(test="season", key=s, r_pre=r1, r_post=r2, p_shift=p))
    for ext in ("sie_SON", "sie_JJA", "sie_DJF", "sie_MAM"):   # which extreme carries it
        e1, _, _ = pear(a1[ic], a1[ext]); e2, _, _ = pear(a2[ic], a2[ext])
        detail.append(dict(test="season_vs_extent", key=f"{s}~{ext}", r_pre=e1, r_post=e2, p_shift=np.nan))
for yr in (1996, 2001, 2006, 2011, 2016):
    a1, a2 = ross[ross.Year < yr], ross[ross.Year >= yr]
    r1, _, n1 = pear(a1["ASL_annual"], a1[AMP]); r2, _, n2 = pear(a2["ASL_annual"], a2[AMP]); _, p = zshift(r1, n1, r2, n2)
    detail.append(dict(test="split_sweep", key=yr, r_pre=r1, r_post=r2, p_shift=p))
ps = []
for y in ross.Year:
    b = ross[ross.Year != y]; a1, a2 = b[b.Year < SPLIT_YEAR], b[b.Year >= SPLIT_YEAR]
    r1, _, n1 = pear(a1["ASL_annual"], a1[AMP]); r2, _, n2 = pear(a2["ASL_annual"], a2[AMP]); ps.append(zshift(r1, n1, r2, n2)[1])
detail.append(dict(test="loo_shift", key="worst p", r_pre=np.nan, r_post=np.nan, p_shift=max(ps)))
note("3.6", "Ross amplitude ~ ASL_annual: LOO worst shift p", round(max(ps), 4), "Ross", extra=f"{sum(p > 0.05 for p in ps)}/{len(ps)} drops lose significance")

# ASL position from the raw Hosking file
asl_path = os.path.join(INDEX_DIR, INDEX_FILES["ASL"])
if os.path.exists(asl_path):
    asl = pd.read_csv(asl_path, comment="#"); asl["time"] = pd.to_datetime(asl.time)
    asl["year"], asl["month"] = asl.time.dt.year, asl.time.dt.month
    def seas_mean(col, months=None, jan_prev=False):
        d = asl if months is None else asl[asl.month.isin(months)].copy()
        if jan_prev: d.loc[d.month == 1, "year"] -= 1
        return d.groupby("year")[col].mean()
    yrs = ross.Year.values
    for col in ("lon", "lat", "ActCenPres", "RelCenPres"):
        for s, months, jp in (("annual", None, False), ("SON", [9, 10, 11], False), ("RET", [10, 11, 12, 1], True)):
            x = seas_mean(col, months, jp).reindex(yrs).values
            y = ross[AMP].values
            m1, m2 = yrs < SPLIT_YEAR, yrs >= SPLIT_YEAR
            r1, _, n1 = pear(x[m1], y[m1]); r2, _, n2 = pear(x[m2], y[m2]); _, p = zshift(r1, n1, r2, n2)
            detail.append(dict(test=f"asl_property:{col}", key=s, r_pre=r1, r_post=r2, p_shift=p))
            if col == "lon" and s == "annual":
                note("3.6", "ASL longitude ~ Ross amplitude pre/post 2001", f"{r1:+.2f} -> {r2:+.2f}", "Ross", p=p,
                     extra=f"mean lon {np.nanmean(x[m1]):.1f}E -> {np.nanmean(x[m2]):.1f}E")
pd.DataFrame(detail).to_csv(os.path.join(TABLES_DIR, "t36_ross_asl_detail.csv"), index=False)


# ── 3.5  hypothesis tests ─────────────────────────────────────────────────────
# H2: the post-2016 amplitude-variance drop is not an atmospheric change — the
#     atmosphere~amplitude relationships are unchanged across 2016 (descriptive,
#     n=8 after) while amplitude variance fell.
rows = []
for sec, tv, ic, basis in PRIMARY_PAIRS:
    if tv != AMP: continue
    a = sec_df(ai, sec); pre, post = a[a.Year < BREAK_YEAR], a[a.Year >= BREAK_YEAR]
    r1, _, n1 = pear(pre[ic], pre[tv]); r2, _, n2 = pear(post[ic], post[tv]); _, p = zshift(r1, n1, r2, n2)
    vr = np.nanvar(post[tv], ddof=1) / np.nanvar(pre[tv], ddof=1)
    rows.append(dict(sector=lab(sec), index=ic, r_pre2016=r1, r_post2016=r2, n_post=n2, p_shift=p, amp_var_ratio_post_pre=vr))
    note("3.5/H2", f"{ic} ~ amplitude pre/post {BREAK_YEAR}", f"{r1:+.2f} -> {r2:+.2f}", lab(sec), n2, p, f"amp var ratio {vr:.2f}")
pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "t35_H2_amplitude_relationships_2016.csv"), index=False)

# H3: the ASL–Ross break tracks the ENSO→ASL teleconnection (IPO shift ~1999), not 2016.
rows = []
yrs_all = sorted(idx.Year.unique())
for lo, hi, label in ((YEAR_MIN, SPLIT_YEAR - 1, f"{YEAR_MIN}-{SPLIT_YEAR-1}"), (SPLIT_YEAR, YEAR_MAX, f"{SPLIT_YEAR}-{YEAR_MAX}")):
    sub = idx[idx.Year.between(lo, hi)]
    for s_asl in ("annual", "SON", "RET"):
        for s_n in ("annual", "SON", "RET"):
            ca, cn = f"ASL_{s_asl}", f"Nino34_{s_n}"
            if ca in sub.columns and cn in sub.columns:
                r, p, n = pear(sub[ca], sub[cn], dt=False)
                rows.append(dict(period=label, asl=s_asl, nino=s_n, r=r, p=p, n=n))
    # and the ASL's relationship to SAM in the same periods/seasons
    for s_asl in ("annual", "SON", "RET"):
        ca, cs = f"ASL_{s_asl}", f"SAM_{s_asl}"
        if ca in sub.columns and cs in sub.columns:
            r, p, n = pear(sub[ca], sub[cs], dt=False)
            rows.append(dict(period=label, asl=s_asl, nino="SAM_" + s_asl, r=r, p=p, n=n))
h3 = pd.DataFrame(rows); h3.to_csv(os.path.join(TABLES_DIR, "t35_H3_asl_nino_teleconnection_2001.csv"), index=False)
for s_asl in ("SON", "RET"):
    g = h3[(h3.asl == s_asl) & (h3.nino == "SAM_" + s_asl)]
    note("3.5/H3", f"ASL_{s_asl} ~ SAM_{s_asl}: pre/post {SPLIT_YEAR}", f"{g.r.iloc[0]:+.2f} -> {g.r.iloc[1]:+.2f}", "indices", extra=f"p {g.p.iloc[0]:.3f} -> {g.p.iloc[1]:.3f}")
for (sa, sn), g in h3.groupby(["asl", "nino"]):
    if sa == sn and not str(sn).startswith("SAM"):
        r1 = g[g.period.str.startswith(str(YEAR_MIN))].r.iloc[0]; r2 = g[~g.period.str.startswith(str(YEAR_MIN))].r.iloc[0]
        note("3.5/H3", f"ASL_{sa} ~ Nino34_{sn}: pre/post {SPLIT_YEAR}", f"{r1:+.2f} -> {r2:+.2f}", "indices")
# partial: Ross amplitude ~ ASL controlling for Nino34, by period
def partial(x, y, z):
    x, y, z = detrend(x), detrend(y), detrend(z); ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rxy = np.corrcoef(x[ok], y[ok])[0, 1]; rxz = np.corrcoef(x[ok], z[ok])[0, 1]; ryz = np.corrcoef(y[ok], z[ok])[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
ross = sec_df(ai, "SIE_Ross"); rows = []
for lo, hi in ((YEAR_MIN, SPLIT_YEAR - 1), (SPLIT_YEAR, YEAR_MAX)):
    b = ross[ross.Year.between(lo, hi)]
    rows.append(dict(period=f"{lo}-{hi}", r_asl_amp=pear(b["ASL_annual"], b[AMP])[0],
                     r_asl_amp_given_nino=partial(b["ASL_annual"], b[AMP], b["Nino34_annual"]),
                     r_nino_amp=pear(b["Nino34_annual"], b[AMP])[0]))
pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "t35_H3_ross_asl_partial_nino.csv"), index=False)
for r in rows: note("3.5/H3", "Ross amp ~ ASL | Nino34 (partial)", round(r["r_asl_amp_given_nino"], 2), "Ross", extra=r["period"])


# ── 3.4c  volatility (from R/ch3/05_volatility_gamlss.R, if present) ──────────
_v = os.path.join(TABLES_DIR, "t34c_volatility_gamlss_post2016.csv")
if os.path.exists(_v):
    v = pd.read_csv(_v)
    for _, r in v.iterrows():
        ci = f"[{r.boot_lo:.2f}, {r.boot_hi:.2f}] n_boot={int(r.n_boot)}" if np.isfinite(r.get("boot_lo", np.nan)) else "no bootstrap yet"
        note("3.4c", f"day-to-day volatility post/pre-{BREAK_YEAR} ({r.response}), season+sensor fixed",
             round(r.vol_ratio_post2016, 3), lab(r.sector), extra=f"sensor factor {r.vol_ratio_ssmis_vs_ssmi:.2f}; {ci}; "
             f"by season DJF {r.ratio_DJF:.2f} MAM {r.ratio_MAM:.2f} JJA {r.ratio_JJA:.2f} SON {r.ratio_SON:.2f}")


# ── ledger ────────────────────────────────────────────────────────────────────
led = pd.DataFrame(LEDGER); led.to_csv(NUMBERS_CSV, index=False)
print(f"\nwrote {len(led)} ledger rows -> {NUMBERS_CSV}")
print("tables ->", TABLES_DIR)
print("\nHeadline numbers:")
for _, r in led[led.section.isin(["3.3", "3.6"]) & led.quantity.str.contains("pooled|LOO|Ross amplitude ~ ASL_annual|pre/post 2001")].iterrows():
    print(f"  [{r.section}] {r.quantity:55s} {str(r.value):>14s}  {r.sector:10s} p={r.p if isinstance(r.p, str) else (f'{r.p:.3f}' if np.isfinite(r.p) else '')}")


# ── markdown digest (regenerated every run; do not edit by hand) ─────────────

def _plain_md(df, index=True, **kw):
    """Markdown table without the tabulate dependency."""
    if True:
        d = df.reset_index() if index else df
        cols = [str(c) for c in d.columns]
        fmt = lambda v: "" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.2f}" if isinstance(v, float) else str(v))
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for _, r in d.iterrows():
            lines.append("| " + " | ".join(fmt(v) for v in r.values) + " |")
        return "\n".join(lines)

try:
    import tabulate  # noqa: F401  (pandas.to_markdown needs it)
except ImportError:
    pd.DataFrame.to_markdown = _plain_md   # fall back to the plain writer

def _tomd(df, index=True):
    return df.to_markdown(index=index)

def _md():
    T = lambda name: pd.read_csv(os.path.join(TABLES_DIR, name))
    L = []
    w = L.append
    w(f"# Chapter 3 numbers\n\nGenerated by `{SCRIPT}` on {STAMP} from `{os.path.basename(ANNUAL_CSV)}` "
      f"({ann.Year.min()}–{ann.Year.max()}, fit from 1979-01-01) and `{os.path.basename(INDEX_CSV)}`. "
      f"All tests on OBSERVED scalars, linearly detrended. Full tables in `results/ch3/tables/`; "
      f"ledger in `ch3_numbers.csv`. Do not edit by hand — rerun `python ch3_stats.py`.\n")

    w("## S1 · Why fitted timing is not used (methods)\n")
    t = T("tS1_fitted_vs_raw.csv").pivot(index="sector", columns="quantity", values="r_fitted_vs_observed").round(2)
    w(_tomd(t) + "\n\nr(fitted, observed): amplitude is the same quantity; timing is not.\n")

    w("## 3.3 · Phase–amplitude coupling\n")
    t = T("t33_phase_amp_by_sector.csv").round(2)
    w("**Full record 1979–2023, per sector**\n\n" + _tomd(t, index=False) + "\n")
    s = T("t33_phase_amp_splits.csv")
    w("**Pooled across six sectors (random-effects), pre vs post split**\n")
    w("| split | method | r pre [95% CI] | r post [95% CI] | I² post | p post | p shift |\n|---|---|---|---|---|---|---|")
    for _, r in s.iterrows():
        w(f"| {int(r.split)} | {r.method} | {r.r_pre:+.2f} [{r.lo_pre:+.2f}, {r.hi_pre:+.2f}] | "
          f"{r.r_post:+.2f} [{r.lo_post:+.2f}, {r.hi_post:+.2f}] | {r.I2_post:.0f}% | {r.p_post:.3f} | {r.p_shift:.3f} |")
    r16 = s[(s.split == BREAK_YEAR) & (s.method == "pearson")].iloc[0]
    w(f"\n**Per sector, {BREAK_YEAR}–{YEAR_MAX} (n=8, Pearson detrended):** " +
      ", ".join(f"{c.replace('post_', '')} {r16[c]:+.2f}" for c in s.columns if c.startswith("post_")) + "\n")
    loo = T("t33_phase_amp_loo2016.csv")
    w(f"**Leave-one-year-out on the {BREAK_YEAR}+ pooled r:** worst p = {loo.p.max():.3f} (dropping "
      f"{int(loo.loc[loo.p.idxmax(), 'dropped'])}); r ranges {loo.r_re.min():+.2f} to {loo.r_re.max():+.2f}; "
      f"{int((loo.p > 0.05).sum())} of {len(loo)} drops lose p<0.05. → call it *emerging*.\n")
    rs = T("t33_phase_amp_rolling10_summary.csv").round(2)
    w("**10-yr rolling Spearman (fig03 statistic): mean of windows ending before/after 2016, and the clean 2007–15 vs 2016–23 contrast**\n\n"
      + _tomd(rs, index=False) + "\n\nABS 'decoupling' is a 2007–15 feature that reverses in 2016–23; East Antarctica is negative in both.\n")

    w("## 3.3b · Day of minimum as the other phase marker (descriptive)\n")
    mn = T("t33b_mindate_vs_amplitude.csv").round(2)
    w(mn[["sector", "sd_max_date", "sd_min_date", "r_min_vs_max", "r_max_amp_pre2016", "r_max_amp_post2016",
          "r_min_amp_pre2016", "r_min_amp_post2016", "p_shift"]].to_markdown(index=False) + "\n")
    l33b = led[led.section == "3.3b"]
    w("Pooled (six sectors): " + "; ".join(f"{q}: {v}" for q, v in zip(l33b[l33b.sector == "pooled"].quantity, l33b[l33b.sector == "pooled"].value)) +
      ". Min-date is the sharper extremum (fitted == observed) and is independent of max-date; the two ends of the cycle need not couple to amplitude the same way.\n")

    w("## 3.3c · Growth-season length (day of max − day of min) vs amplitude\n")
    g = T("t33c_growth_length_vs_amplitude.csv").round(2)
    w(_tomd(g[["sector", "mean_len_days", "sd_len_pre", "sd_len_post", "r_pre2016", "r_post2016", "p_shift"]], index=False) + "\n")
    w("Weddell 2016–2023 by year:\n\n" + _tomd(T("t33c_weddell_post2016_years.csv").round(2), index=False) + "\n")

    w("## 3.3d · Breakpoint sensitivity (2013–2017)\n")
    w(_tomd(T("t33d_breakpoint_sensitivity.csv").round(3), index=False) +
      "\n\nThe pooled coupling exists for a 2016 break only; the Weddell growth-season result holds for 2014–2016; the East Antarctic amplitude-variance drop for 2014–2017.\n")

    w("## 3.4 · Component variance, post/pre-2016 (F-test, raw anomalies)\n")
    v = T("t34_variance_ratio_2016.csv").query("not detrended").pivot(index="sector", columns="variable", values="var_ratio_post_pre").round(2)
    pv = T("t34_variance_ratio_2016.csv").query("not detrended").pivot(index="sector", columns="variable", values="p_F").round(2)
    w(_tomd(v) + "\n\np (F-test):\n\n" + _tomd(pv) +
      "\n\nTiming flat, amplitude down (circumpolar 0.23, p=0.05), extent up (n.s. at n=8). Ties to Ch.2: phase variance unchanged.\n")

    w("## 3.5 · Atmospheric modes by component and sector\n")
    sc = T("t35_index_scan_raw.csv")
    w(f"**Exploratory scan (observed targets):** {len(sc)} tests; {int((sc.p < 0.05).sum())} at p<0.05 vs "
      f"{0.05 * len(sc):.0f} expected by chance; {int((sc.q_bh < 0.05).sum())} survive BH q<0.05. "
      f"Hits cluster by sector × mode with consistent sign across seasons:\n")
    pp = T("t35_primary_pairs.csv").round(3)
    w(pp[["sector", "target", "index", "r", "p", "p_bonf7", "loo_worst_p", "seasons_p05_same_sign", "basis"]].to_markdown(index=False) + "\n")
    w("Bonferroni over 7 is post-hoc; the defence is mechanism + LOO + seasonal consistency. Weddell~ZW3R is the weakest (LOO 0.07) → supplement.\n")
    pm = T("t35_pooled_meta.csv").round(2)
    w("**Pooled across the five sectors (the circumpolar test) — I² is the result:**\n\n" + _tomd(pm, index=False) +
      "\n\nHigh I² for ENSO~amplitude = dipole (Weddell +, King Haakon −); nothing pools to a circumpolar effect.\n")

    if os.path.exists(os.path.join(TABLES_DIR, "t34c_volatility_gamlss_post2016.csv")):
        w("## 3.4c · Day-to-day volatility, post/pre-2016 (gamlss; R/ch3/05_volatility_gamlss.R)\n")
        v = T("t34c_volatility_gamlss_post2016.csv").round(3)
        v["sector"] = v["sector"].map(lambda x: SECTOR_LABELS.get(x, x))
        w(_tomd(v[["sector", "response", "vol_ratio_post2016", "boot_lo", "boot_hi", "vol_ratio_ssmis_vs_ssmi",
                   "ratio_DJF", "ratio_MAM", "ratio_JJA", "ratio_SON"]], index=False) + "\n")

    w("## 3.5 · Hypothesis tests\n")
    w("**H2 — amplitude relationships across 2016 (descriptive, n=8 after) vs amplitude variance ratio**\n\n" +
      _tomd(T("t35_H2_amplitude_relationships_2016.csv").round(2), index=False) + "\n")
    w("**H3 — ENSO→ASL teleconnection, pre/post 2001 (index-only, detrended indices)**\n\n" +
      _tomd(T("t35_H3_asl_nino_teleconnection_2001.csv").query("asl == nino or nino.str.startswith('SAM')", engine="python").round(2), index=False) + "\n")
    w("Ross amplitude ~ ASL, with and without Niño3.4 partialled out:\n\n" +
      _tomd(T("t35_H3_ross_asl_partial_nino.csv").round(2), index=False) + "\n")
    w("**H5 — does any index set Weddell growth-season length? (full record, all 35 index-seasons, BH)**\n\n" +
      _tomd(T("t35_H5_weddell_growth_length_vs_indices.csv").head(8).round(3), index=False) + "\n")

    w("## 3.6 · Stationarity, 1979–2000 vs 2001–2023\n")
    st = T("t36_stationarity_2001.csv").round(3)
    w(st[["sector", "index", "target", "r_pre", "r_post", "p_shift", "p_shift_bonf7"]].to_markdown(index=False) + "\n")
    d = T("t36_ross_asl_detail.csv")
    w("**Ross amplitude ~ ASL (relative central pressure; negative = deeper)**\n")
    w("By season:\n\n" + d[d.test == "season"][["key", "r_pre", "r_post", "p_shift"]].round(3).to_markdown(index=False) + "\n")
    w("Which extent carries it (ASL_annual, ASL_RET vs seasonal extent):\n\n" +
      d[(d.test == "season_vs_extent") & d.key.str.startswith(("annual", "RET"))][["key", "r_pre", "r_post"]].round(2).to_markdown(index=False) + "\n")
    w("Split-year sweep (ASL_annual):\n\n" + d[d.test == "split_sweep"][["key", "r_pre", "r_post", "p_shift"]].round(3).to_markdown(index=False) + "\n")
    w(f"Leave-one-year-out on the 2001 shift: worst p = {float(d[d.test == 'loo_shift'].p_shift.iloc[0]):.4f}.\n")
    w("ASL position (Hosking file), by period:\n\n" +
      d[d.test.str.startswith("asl_property")][["test", "key", "r_pre", "r_post", "p_shift"]].round(2).to_markdown(index=False) +
      "\n\nLongitude and latitude do not shift and never correlate with Ross amplitude: the change is in the depth–amplitude coupling, not position. No mechanism claimed.\n")

    out = os.path.join(TABLES_DIR, "ch3_numbers.md")
    open(out, "w").write("\n".join(L)); print("wrote", out)

_md()