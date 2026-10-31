#!/usr/bin/env python
# ch3_sst_partition.py -- how SST partitions among the three cycle components, by sector
# and by season, and how it has changed around 2016.  Builds on ch3_quick_ocean_test.py.
#
# For each sector in {WS, KH, EA, RA, ABS}:
#   for each cycle component in {annual mean, amplitude, day of max}:
#     for each SST season in {annual, DJF, MAM, JJA, SON, advance Mar-Aug, retreat Oct-Jan}:
#       detrended Pearson r over 1979-2015 (pre), 2016-2024 (post), and 1979-2024 (full)
#       Fisher's z test on pre vs post change
#   compare the strongest-SST pairing with the pre-specified atmospheric mode in Table 3.3
#
# Outputs:
#   results/ch3/tables/ch3_sst_partition_full.csv       # (sector, component, season, r_full)
#   results/ch3/tables/ch3_sst_partition_prepost.csv   # (sector, component, season, r_pre, r_post, delta_z, p)
#   results/ch3/tables/ch3_sst_vs_atmos.csv            # (sector, component, best_SST_season, r_SST, atmos_mode, r_atmos)
#   (figure: Ch3/figures/ch3_fig06_sst_components.py reads ch3_sst_partition_full.csv)

import os, sys
import numpy as np, pandas as pd, xarray as xr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "figures"))     # ch3_config lives in Ch3/figures/
from ch3_config import ROOT, ANNUAL_CSV as ANNUAL, TABLES_DIR as OUT_TAB
LOC = os.path.join(ROOT, "data", "ocean", "HadISST_sst.nc")
os.makedirs(OUT_TAB, exist_ok=True)
PRE, POST = (1979, 2015), (2016, 2024)
LAT = (-65, -55)
SECTORS = {"WS":(-70,-14,"SIE_Weddell"), "KH":(-14,71,"SIE_King_Haakon"), "EA":(71,162,"SIE_East_Antarctica"), "RA":(162,-110,"SIE_Ross"), "ABS":(-110,-70,"SIE_Amundsen_Bellingshausen")}   # canonical_sectors.nc bounds
SEASONS = {"annual": list(range(1,13)), "DJF":[12,1,2], "MAM":[3,4,5],
           "JJA":[6,7,8], "SON":[9,10,11], "advance":[3,4,5,6,7,8], "retreat":[10,11,12,1]}
QUANTS = [("sie_annual","annual mean"), ("amplitude_raw_yr","amplitude"),
          ("max_doy_raw","day of max")]
# Table 3.3 pre-specified relationships (atmospheric comparison)
ATMOS = {("ABS","amplitude"):("Nino3.4 SON",     +0.42),
         ("WS","amplitude"):("SAM JJA",          -0.40),
         ("EA","day of max"):("SAM Oct-Jan",     +0.38),
         ("RA","amplitude"):("[s] Nino3.4 annual", -0.42),
         ("RA","day of max"):("[s] ZW3 SON",     +0.40)}


def sst_sector_month(ds):
    sst = ds["sst"].where(ds["sst"] > -2)
    sub = sst.sel(latitude=slice(LAT[1], LAT[0]))
    out = {}
    for code, (lo, hi, _) in SECTORS.items():
        s = sub.where(((ds.longitude >= lo) & (ds.longitude <= hi)) if lo < hi
                      else ((ds.longitude >= lo) | (ds.longitude <= hi)), drop=True)
        m = s.mean(("latitude","longitude"), skipna=True).to_pandas()
        m.index = pd.to_datetime(m.index)
        out[code] = m
    return pd.DataFrame(out)


def seasonal(df_monthly, months):
    wrap = [m for m in months if m < months[0]]
    syear = df_monthly.index.year - df_monthly.index.month.isin(wrap).astype(int)
    sel = df_monthly[df_monthly.index.month.isin(months)]
    sel = sel.copy(); sel["syear"] = syear[df_monthly.index.month.isin(months)]
    return sel.groupby("syear").mean(numeric_only=True)


def detrend_period(y):
    y = y.dropna(); x = y.index.values.astype(float)
    return y - np.polyval(np.polyfit(x, y.values, 1), x)


def rcorr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 5 else np.nan


def fisher_z(r1, n1, r2, n2):
    """two-tailed p for r1 != r2"""
    if any(pd.isna([r1, r2])) or n1 < 4 or n2 < 4: return np.nan
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    from scipy.stats import norm
    return 2 * (1 - norm.cdf(abs(z1-z2)/se))


def main():
    ds = xr.open_dataset(LOC)
    sst_m = sst_sector_month(ds)
    a = pd.read_csv(ANNUAL)
    if "period" in a.columns: a = a[a.period == "FULL"]

    # --- Table 1: full-record r for every sector x component x season
    full_rows = []
    pp_rows = []
    for code, (_, _, full) in SECTORS.items():
        g = a[a.sector == full].set_index("Year").sort_index()
        for q, qlab in QUANTS:
            if q not in g.columns: continue
            for sname, months in SEASONS.items():
                s = seasonal(sst_m[[code]], months)[code]
                # full
                common = s.index.intersection(g.index).intersection(range(1979, 2025))
                x = detrend_period(s.reindex(common))
                y = detrend_period(g.loc[common, q])
                r_full = rcorr(x.values, y.values)
                full_rows.append(dict(sector=code, component=qlab, season=sname,
                                      n=len(common), r=r_full))
                # pre / post with within-period detrend
                def period_r(lo, hi):
                    idx = [t for t in common if lo <= t <= hi]
                    if len(idx) < 6: return np.nan, len(idx)
                    xp = detrend_period(s.reindex(idx)); yp = detrend_period(g.loc[idx, q])
                    return rcorr(xp.values, yp.values), len(idx)
                r_pre, n_pre = period_r(*PRE); r_post, n_post = period_r(*POST)
                p = fisher_z(r_pre, n_pre, r_post, n_post)
                pp_rows.append(dict(sector=code, component=qlab, season=sname,
                                    r_pre=r_pre, r_post=r_post, delta=r_post-r_pre,
                                    n_pre=n_pre, n_post=n_post, p_change=p))
    full = pd.DataFrame(full_rows); full.to_csv(f"{OUT_TAB}/ch3_sst_partition_full.csv", index=False)
    pp = pd.DataFrame(pp_rows); pp.to_csv(f"{OUT_TAB}/ch3_sst_partition_prepost.csv", index=False)

    # --- Table 2: for each (sector, component), strongest SST season vs atmospheric mode
    vs_rows = []
    for code in SECTORS:
        for q, qlab in QUANTS:
            sub = full[(full.sector == code) & (full.component == qlab)]
            if sub.empty: continue
            idx = sub.r.abs().idxmax()
            best = sub.loc[idx]
            atmos = ATMOS.get((code, qlab), (None, None))
            vs_rows.append(dict(sector=code, component=qlab,
                                best_SST_season=best.season, r_SST=round(best.r, 2),
                                atmos_mode=atmos[0], r_atmos=atmos[1]))
    vs = pd.DataFrame(vs_rows); vs.to_csv(f"{OUT_TAB}/ch3_sst_vs_atmos.csv", index=False)

    # --- print ---
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    print("\n=== 1.  Full-record detrended r, SST (55-65 S band) vs each cycle component ===")
    piv_full = full.pivot_table(index=["sector","component"], columns="season", values="r")[list(SEASONS)]
    print(piv_full.round(2).to_string())

    print("\n=== 2.  Pre (1979-2015) vs Post (2016-2024), within-period detrend, Fisher's z p ===")
    print("    (only rows where |r_pre| or |r_post| >= 0.3 shown)")
    tight = pp[(pp.r_pre.abs() >= 0.3) | (pp.r_post.abs() >= 0.3)].copy()
    tight = tight.sort_values(["sector","component","season"])
    print(tight[["sector","component","season","r_pre","r_post","delta","p_change"]]
          .round(2).to_string(index=False))

    print("\n=== 3.  Strongest SST season vs the atmospheric mode from Table 3.3 ===")
    print(vs.to_string(index=False))

    print(f"\nwrote tables to {OUT_TAB}")

if __name__ == "__main__": main()
