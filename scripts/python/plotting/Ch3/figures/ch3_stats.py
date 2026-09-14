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


# ── ledger ────────────────────────────────────────────────────────────────────
led = pd.DataFrame(LEDGER); led.to_csv(NUMBERS_CSV, index=False)
print(f"\nwrote {len(led)} ledger rows -> {NUMBERS_CSV}")
print("tables ->", TABLES_DIR)
print("\nHeadline numbers:")
for _, r in led[led.section.isin(["3.3", "3.6"]) & led.quantity.str.contains("pooled|LOO|Ross amplitude ~ ASL_annual|pre/post 2001")].iterrows():
    print(f"  [{r.section}] {r.quantity:55s} {str(r.value):>14s}  {r.sector:10s} p={r.p if isinstance(r.p, str) else (f'{r.p:.3f}' if np.isfinite(r.p) else '')}")
