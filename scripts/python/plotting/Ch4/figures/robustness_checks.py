#!/usr/bin/env python
"""
robustness_checks.py -- four checks on the Chapter 4 drift results, from files that already exist.

  1. FDR (Benjamini-Hochberg, q = 0.05) across the sector-season tests of
       excess (fixed-cell, all-cell NSIDC, all-cell OSI), alpha change, |C| change, |dC|.
     p-values: for |dC| from the saved bootstrap; for the others a two-sided normal
     approximation from the bootstrap CI (p = 2*[1 - Phi(|est| / ((hi-lo)/3.92))]) -- stated as such.
  2. Wind distribution: mean and std of daily wind speed over fixed cells, pre vs post,
     with year-bootstrap CI on the ratio (a change in variance with the mean fixed would move C).
  3. Fixed-cell coverage: mean number of valid cells contributing per day, pre vs post.
  4. Sea-ice concentration on the same fixed cells, pre vs post, against dC
     (direct test of the "loosened pack" reading of the residual dipole).

Inputs : results/ch4/tables/bootstrap_main_fm_1992-2015_vs_2016-2020.csv
         results/ch4/tables/windfactor_prepost.csv (+ _dc_boots.npz)
         results/ch4/tables/uv_daily_sector_fixedcells_1992-2024.csv
         data/merged/merged_bootstrap_SH_latest.nc   (daily SIC on the analysis grid)
         results/NSIDC0116/NSIDC0116_uv_season_year.nc (fixed-cell definition)
Outputs: results/ch4/tables/robustness_fdr.csv, robustness_wind_dist.csv, robustness_ncells.csv,
         robustness_sic_vs_dc.csv, sic_season_year_fixedcells.csv
         results/ch4/figures/{manuscript,slides}/fig09_sic_vs_dc.*
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_figstyle as fs  # noqa: E402

ROOT = fs.ROOT
TAB = f"{ROOT}/results/ch4/tables"
BOOT_TAB = f"{TAB}/bootstrap_main_fm_1992-2015_vs_2016-2020.csv"
WF_TAB = f"{TAB}/windfactor_prepost.csv"
UV_CSV = f"{TAB}/uv_daily_sector_fixedcells_1992-2024.csv"
SIC_NC = f"{ROOT}/data/merged/merged_bootstrap_SH_latest.nc"
SECTOR_NC = f"{ROOT}/data/canonical_sectors.nc"
DRIFT_NC = f"{ROOT}/results/NSIDC0116/NSIDC0116_uv_season_year.nc"
SIC_SERIES = f"{TAB}/sic_season_year_fixedcells.csv"
BASE0, BASE1, POST0, POST1 = 1992, 2015, 2016, 2020
SEASONS = ["MAM", "JJA", "SON"]
N_BOOT, SEED, Q = 2000, 42, 0.05
rng = np.random.default_rng(SEED)
pd.set_option("display.width", 220)


def bh(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    n, order = len(p), np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def p_from_ci(est, lo, hi):
    se = (hi - lo) / 3.92
    return 2 * (1 - norm.cdf(np.abs(est) / np.where(se > 0, se, np.nan)))


# ----------------------------------------------------------------------------
def check_fdr():
    b = pd.read_csv(BOOT_TAB)
    w = pd.read_csv(WF_TAB)
    boots = dict(np.load(WF_TAB.replace(".csv", "_dc_boots.npz")))
    rows = []
    for prod, label in (("NSIDC-0116 fixed", "excess_fixed"), ("NSIDC-0116", "excess_allcell_nsidc"),
                        ("OSI-455", "excess_allcell_osi")):
        t = b[b["product"] == prod]
        rows += [dict(test=label, sector=r.sector, season=r.season, est=r.excess_pp,
                      p=p_from_ci(r.excess_pp, r.excess_lo, r.excess_hi)) for _, r in t.iterrows()]
    rows += [dict(test="alpha_diff", sector=r.sector, season=r.season, est=r.alpha_diff,
                  p=p_from_ci(r.alpha_diff, r.alpha_diff_lo, r.alpha_diff_hi)) for _, r in w.iterrows()]
    rows += [dict(test="resid_diff", sector=r.sector, season=r.season, est=r.resid_diff,
                  p=p_from_ci(r.resid_diff, r.resid_diff_lo, r.resid_diff_hi)) for _, r in w.iterrows()]
    # |dC|: exact bootstrap p that the change vector magnitude is consistent with zero =
    #        fraction of replicates whose dC projects negatively onto the estimate's heading, doubled
    for _, r in w.iterrows():
        k = f"{r.sector}_{r.season}"
        if k in boots:
            dc = complex(r.dc_e, r.dc_n)
            proj = (boots[k] * np.conj(dc)).real
            rows.append(dict(test="dc_vector", sector=r.sector, season=r.season, est=r.dc_mag,
                             p=min(1.0, 2 * np.mean(proj <= 0))))
    t = pd.DataFrame(rows)
    t["q"] = np.nan
    for test, g in t.groupby("test"):
        t.loc[g.index, "q"] = bh(g.p.values)
    t["sig_raw"] = t.p < 0.05
    t["sig_fdr"] = t.q < Q
    t.to_csv(f"{TAB}/robustness_fdr.csv", index=False)
    print("\n=== 1. FDR (q = 0.05) per family of 15 tests. p for all but dc_vector is a normal approx from the CI ===")
    summ = t.groupby("test").agg(n=("p", "size"), raw_sig=("sig_raw", "sum"), fdr_sig=("sig_fdr", "sum"))
    print(summ.to_string())
    print("\nRows surviving FDR:")
    print(t[t.sig_fdr][["test", "sector", "season", "est", "p", "q"]].round(4).to_string(index=False))
    return t


# ----------------------------------------------------------------------------
def load_uv():
    df = pd.read_csv(UV_CSV, parse_dates=["date"]).dropna()
    df["sector"] = df["sector"].map(fs.SECTOR_ID)
    df["year"] = df.date.dt.year + (df.date.dt.month == 12).astype(int)
    df["wspd"] = np.hypot(df.uw_e, df.uw_n)
    return df


def boot_ratio(pre, post, yrs_pre, yrs_post, stat):
    """Bootstrap (years) CI on stat(post)/stat(pre) for daily series grouped by year."""
    bp = {y: pre[yrs_pre == y] for y in np.unique(yrs_pre)}
    bq = {y: post[yrs_post == y] for y in np.unique(yrs_post)}
    kp, kq = list(bp), list(bq)
    r = np.empty(N_BOOT)
    for i in range(N_BOOT):
        a = np.concatenate([bp[y] for y in rng.choice(kp, len(kp))])
        c = np.concatenate([bq[y] for y in rng.choice(kq, len(kq))])
        r[i] = stat(c) / stat(a)
    return np.percentile(r, [2.5, 97.5])


def check_wind_dist(df):
    rows = []
    for (sec, sea), g in df[df.season.isin(SEASONS)].groupby(["sector", "season"]):
        pre, post = g[(g.year >= BASE0) & (g.year <= BASE1)], g[(g.year >= POST0) & (g.year <= POST1)]
        row = dict(sector=sec, season=sea, mean_pre=pre.wspd.mean(), mean_post=post.wspd.mean(),
                   std_pre=pre.wspd.std(), std_post=post.wspd.std())
        row["mean_ratio"] = row["mean_post"] / row["mean_pre"]
        row["std_ratio"] = row["std_post"] / row["std_pre"]
        row["mean_ratio_lo"], row["mean_ratio_hi"] = boot_ratio(pre.wspd.values, post.wspd.values,
                                                                pre.year.values, post.year.values, np.mean)
        row["std_ratio_lo"], row["std_ratio_hi"] = boot_ratio(pre.wspd.values, post.wspd.values,
                                                              pre.year.values, post.year.values, np.std)
        row["std_sig"] = (row["std_ratio_lo"] > 1) or (row["std_ratio_hi"] < 1)
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(f"{TAB}/robustness_wind_dist.csv", index=False)
    print("\n=== 2. Daily wind speed over fixed cells: mean and std, post/pre ratios with 95% CI ===")
    print(t[["sector", "season", "mean_pre", "mean_post", "mean_ratio", "mean_ratio_lo", "mean_ratio_hi",
             "std_pre", "std_post", "std_ratio", "std_ratio_lo", "std_ratio_hi", "std_sig"]].round(3).to_string(index=False))
    return t


def check_ncells(df):
    rows = []
    for (sec, sea), g in df[df.season.isin(SEASONS)].groupby(["sector", "season"]):
        pre, post = g[(g.year >= BASE0) & (g.year <= BASE1)], g[(g.year >= POST0) & (g.year <= POST1)]
        rows.append(dict(sector=sec, season=sea, ncells_pre=pre.n_cells.mean(), ncells_post=post.n_cells.mean(),
                         ratio=post.n_cells.mean() / pre.n_cells.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(f"{TAB}/robustness_ncells.csv", index=False)
    print("\n=== 3. Valid cells contributing to the fixed-cell daily mean, post/pre ===")
    print(t.round(3).to_string(index=False))
    return t


# ----------------------------------------------------------------------------
def sic_series():
    if os.path.exists(SIC_SERIES):
        return pd.read_csv(SIC_SERIES)
    sector = xr.open_dataset(SECTOR_NC)["sector_id"].values
    d = xr.open_dataset(DRIFT_NC)["speed"]
    dpre, dpost = d.sel(year=slice(BASE0, BASE1)), d.sel(year=slice(POST0, POST1))
    fixed = ((dpre.mean("year") > 0.03) & (np.isfinite(dpre).sum("year") >= 15)
             & (np.isfinite(dpost).sum("year") >= 3))
    ds = xr.open_dataset(SIC_NC)
    var = next(v for v in ds.data_vars if ds[v].ndim == 3 and "time" in ds[v].dims)
    sic = ds[var]
    scale = 100.0 if float(sic.isel(time=slice(0, 30)).max()) <= 1.5 else 1.0   # fraction -> percent
    print(f"SIC variable '{var}', dims {sic.dims}, scale x{scale}")
    rows = []
    years = sorted(set(int(y) for y in sic.time.dt.year.values))
    for yr in years:
        if yr < BASE0 - 1 or yr > 2024:
            continue
        a = sic.sel(time=slice(f"{yr}-01-01", f"{yr}-12-31")).load() * scale
        m = a.time.dt.month
        for sea, months in (("MAM", [3, 4, 5]), ("JJA", [6, 7, 8]), ("SON", [9, 10, 11]), ("DJF", [1, 2])):
            sel = a.isel(time=m.isin(months).values)
            if sel.sizes["time"] == 0:
                continue
            mean = sel.mean("time").values
            fm = fixed.sel(season=sea).values
            for sid, code in fs.SECTOR_ID.items():
                mk = fm & (sector == sid) & np.isfinite(mean)
                if mk.sum():
                    rows.append(dict(sector=code, season=sea, syear=yr, sic=float(mean[mk].mean()), n=int(mk.sum())))
        print(f"  {yr}", end="", flush=True)
    print()
    t = pd.DataFrame(rows)
    t.to_csv(SIC_SERIES, index=False)
    return t


def check_sic(wf):
    s = sic_series()
    rows = []
    for (sec, sea), g in s[s.season.isin(SEASONS)].groupby(["sector", "season"]):
        pre, post = g[(g.syear >= BASE0) & (g.syear <= BASE1)], g[(g.syear >= POST0) & (g.syear <= POST1)]
        if len(pre) < 10 or len(post) < 3:
            continue
        d = post.sic.mean() - pre.sic.mean()
        db = np.array([post.sic.values[rng.integers(0, len(post), len(post))].mean()
                       - pre.sic.values[rng.integers(0, len(pre), len(pre))].mean() for _ in range(N_BOOT)])
        w = wf[(wf.sector == sec) & (wf.season == sea)].iloc[0]
        rows.append(dict(sector=sec, season=sea, sic_pre=pre.sic.mean(), sic_post=post.sic.mean(), dsic=d,
                         dsic_lo=np.percentile(db, 2.5), dsic_hi=np.percentile(db, 97.5),
                         dsic_sig=(np.percentile(db, 2.5) > 0) or (np.percentile(db, 97.5) < 0),
                         dc_north=w.dc_n * 100, dc_east=w.dc_e * 100, dc_mag=w.dc_mag * 100, resid_sig=bool(w.resid_sig),
                         alpha_diff=w.alpha_diff, alpha_sig=bool(w.alpha_sig)))
    t = pd.DataFrame(rows)
    t.to_csv(f"{TAB}/robustness_sic_vs_dc.csv", index=False)
    print("\n=== 4. Fixed-cell sea-ice concentration change (percentage points) against the residual change (cm/s) ===")
    print(t.round(2).to_string(index=False))
    from scipy.stats import spearmanr
    rho, p = spearmanr(t.dsic, t.dc_north)
    print(f"\nSpearman rho(dSIC, dC_north) over 15 sector-seasons = {rho:.2f}, p = {p:.3f}")
    print("Read: the 'loosened pack' reading predicts dSIC < 0 where dC_north > 0 (Pacific) and dSIC >= 0 where\n"
          "      dC_north < 0 (Atlantic-Indian): i.e. a NEGATIVE correlation.")
    return t


def fig09(t):
    fig, ax = plt.subplots(figsize=fs.size(0.6, 0.85) if fs.mode() == "manuscript" else fs.size(0.45, 0.8))
    ax.axhline(0, color=fs.INK, lw=0.6)
    ax.axvline(0, color=fs.INK, lw=0.6)
    for _, r in t.iterrows():
        c = fs.SECTOR_COLORS[r.sector]
        a = 1.0 if r.resid_sig else 0.45
        ax.plot([r.dsic_lo, r.dsic_hi], [r.dc_north, r.dc_north], color=c, lw=fs.lw(0.5), alpha=0.5 * a)
        ax.plot(r.dsic, r.dc_north, marker=fs.SEASON_MARKERS[r.season], ls="none", color=c, ms=fs.ms(0.8),
                markeredgecolor="white", markeredgewidth=0.8, alpha=a)
    ax.set_xlabel(f"Change in sea-ice concentration on fixed cells, {POST0}–{POST1} vs {BASE0}–{BASE1} (pp)")
    ax.set_ylabel("Equatorward change in wind-independent drift, ΔC$_N$ (cm s$^{-1}$)")
    h = [Line2D([0], [0], marker="o", ls="none", color=fs.SECTOR_COLORS[s], ms=fs.ms(0.7), label=fs.SECTOR_NAME[s])
         for s in fs.SECTOR_ORDER] + \
        [Line2D([0], [0], marker=fs.SEASON_MARKERS[se], ls="none", color="#666", ms=fs.ms(0.7), label=se) for se in SEASONS]
    ax.legend(handles=h, loc="best", ncol=2)
    fs.save(fig, "fig09_sic_vs_dc")


if __name__ == "__main__":
    check_fdr()
    df = load_uv()
    check_wind_dist(df)
    check_ncells(df)
    wf = pd.read_csv(WF_TAB)
    t = check_sic(wf)
    for m in ("manuscript", "slide"):
        fs.set_mode(m)
        fig09(t)
