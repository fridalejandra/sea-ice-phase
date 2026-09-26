#!/usr/bin/env python
"""
25b_field_significance.py -- is the number of bin-months with a 'significant' beta change in
Fig 4c more than expected by chance, given that neighbouring bins and months are correlated?

Permutation test (Livezey & Chen 1983 style): the 8 'post' years are re-drawn at random from all
years (same draw for every bin-month, so spatial/seasonal correlation is preserved); each draw
gives a change in beta for all 432 cells. Per-cell permutation p-values, and the null distribution
of 'how many cells reach p < 0.05'. Also: Wilks (2016) FDR at alpha_FDR = 0.10, and where the
nominal cells are (season x sector, weaker vs stronger).
usage: python 25b_field_significance.py [tau_y|tau_mag]
"""
import importlib.util
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("lc", os.path.join(HERE, "25_local_coupling.py"))
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)
N_PERM = 2000
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def sector(lon):
    lon = lon % 360
    for s, (lo, hi) in {"WS": (300, 20), "KH": (20, 90), "EA": (90, 160), "RA": (160, 230), "ABS": (230, 300)}.items():
        if (lo > hi and (lon >= lo or lon < hi)) or (lo < hi and lo <= lon < hi):
            return s


def main(wvar):
    from statsmodels.stats.multitest import multipletests
    d = pd.read_csv(lc.DAILY_CSV, parse_dates=["date"])
    years = np.arange(lc.START, lc.END + 1)
    cells, S = [], []
    for lb, g in d.groupby("lonbin"):
        g = g.set_index("date").sort_index().asfreq("D")
        dsia = g.SIA - g.SIA.shift(1)
        g, dsia = g[g.index.year >= lc.START], dsia[dsia.index.year >= lc.START]
        last = g.index.year.max()
        ya, xa = lc.anom(dsia, last), lc.anom(g[wvar], last)
        for m in range(1, 13):
            sel = (ya.index.month == m) & (ya.index.year <= lc.END)
            yrs = ya.index.year.values[sel]
            x, y = xa.values[sel], ya.values[sel]
            ok = np.isfinite(x) & np.isfinite(y)
            df = pd.DataFrame(dict(yr=yrs[ok], x=x[ok], y=y[ok]))
            df["xx"], df["xy"], df["yy"] = df.x ** 2, df.x * df.y, df.y ** 2
            st = df.groupby("yr").agg(n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"), sxx=("xx", "sum"),
                                      sxy=("xy", "sum"), syy=("yy", "sum")).reindex(years).fillna(0)
            st.loc[st.n < 10] = 0
            cells.append((lb, m))
            S.append(st.values)
    S = np.array(S)                                  # (cells, years, 6)
    post_true = years >= lc.BREAK
    npost = post_true.sum()

    def dbeta(mask):                                 # mask: (..., years) bool
        post = np.einsum("...y,cyk->...ck", mask.astype(float), S)
        pre = np.einsum("...y,cyk->...ck", (~mask).astype(float), S)
        return lc.beta_r(post)[0] - lc.beta_r(pre)[0]

    obs = dbeta(post_true)                            # (cells,)
    rng = np.random.default_rng(7)
    perm = np.zeros((N_PERM, len(years)), bool)
    for k in range(N_PERM):
        perm[k, rng.choice(len(years), npost, replace=False)] = True
    null = dbeta(perm)                                # (perm, cells)
    valid = np.isfinite(obs) & (np.isfinite(null).mean(0) > 0.9)
    a_null = np.abs(null)
    p_cell = (1 + (a_null >= np.abs(obs)).sum(0)) / (N_PERM + 1)
    thr = np.nanpercentile(a_null, 95, axis=0)
    n_obs = int(((p_cell < 0.05) & valid).sum())
    n_null = ((a_null > thr) & valid).sum(1)
    field_p = (1 + (n_null >= n_obs).sum()) / (N_PERM + 1)
    q10 = multipletests(p_cell[valid], alpha=0.10, method="fdr_bh")[0].sum()
    print(f"\n{wvar}: {valid.sum()} cells; permutation p<0.05 in {n_obs} cells; "
          f"null count median {np.median(n_null):.0f}, 95th pct {np.percentile(n_null, 95):.0f}; "
          f"FIELD SIGNIFICANCE p = {field_p:.3f}")
    print(f"   Wilks (2016) FDR at alpha_FDR = 0.10: {q10} cells")
    res = pd.DataFrame(dict(lonbin=[c[0] for c in cells], month=[c[1] for c in cells],
                            dbeta=obs, p_perm=p_cell, valid=valid))
    res = res.merge(pd.read_csv(f"{lc.TAB}/lonbin_month_coupling_{wvar}.csv")[["lonbin", "month", "beta_pre"]],
                    on=["lonbin", "month"], how="left")
    res["weaker"] = np.sign(res.dbeta) != np.sign(res.beta_pre)
    res["season"], res["sector"] = res.month.map(SEASON), res.lonbin.map(sector)
    sig = res[(res.p_perm < 0.05) & res.valid]
    tab = sig.groupby(["season", "sector"]).weaker.agg(n="size", n_weaker="sum").unstack("sector").fillna(0).astype(int)
    print("   nominal cells by season x sector  (n / of which weaker coupling):")
    for s in ["DJF", "MAM", "JJA", "SON"]:
        line = "   " + s + "  " + "  ".join(
            f"{sec}:{int(tab.loc[s, ('n', sec)]) if (s in tab.index and ('n', sec) in tab.columns) else 0}"
            f"/{int(tab.loc[s, ('n_weaker', sec)]) if (s in tab.index and ('n_weaker', sec) in tab.columns) else 0}"
            for sec in ["WS", "KH", "EA", "RA", "ABS"])
        print(line)
    res.to_csv(f"{lc.TAB}/lonbin_month_fieldsig_{wvar}.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tau_y")
