"""
wind_divergence_coupling_test.py

Ch4 Layer 2 / SCAR poster: has the coupling between wind stress and sea ice
DIVERGENCE changed across the 2016 regime shift?

WHY DIVERGENCE RATHER THAN AREA CHANGE
Daily SIA change conflates wind-driven redistribution with thermodynamic melt
and growth. Divergence (del . u) is the component that mechanically opens leads
and changes ice area, so regressing wind stress on divergence isolates the
mechanical pathway in a way that regressing on dSIA cannot. This is the test
the SCAR wind-sensitivity analysis could not do.

POSITIONING
Holland & Kwok (2012) established wind-divergence coupling for the pre-2010
expansion era; de Jager & Vichi (2025) examined rotational coupling through 2020
but stopped short of divergence. Neither asks whether wind->divergence coupling
inflected at 2016.

MODELS
  Test A (binary):     div ~ wind + post + wind:post
  Test B (continuous): div ~ wind + ocean_state + wind:ocean_state

Both fitted per sector x season (20 tests each), with stationary block bootstrap
CIs and Benjamini-Hochberg FDR correction across the 20-test family.

CIRCULARITY CAVEAT
NSIDC-0116 blends NCEP/NCAR reanalysis winds into its optimal interpolation, and
the Southern Hemisphere has no IABP buoy constraint. Cells whose nearest input
vector exceeded 1250 km were already excluded upstream (~1.3% of vectors), but
that threshold is lenient: passing it does not mean a cell was well constrained
by observations. Interpret a POSITIVE coupling result cautiously -- some of it
may be built in by construction. A NULL result is not affected by this concern,
since built-in coupling would bias toward finding coupling, not away from it.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# ---------------- CONFIG ----------------
DIVERGENCE_PATH = "ice_divergence_by_sector_season.csv"
# from compute_ice_divergence_nsidc0116.py
# columns: date, sector, div_net, div_positive, div_negative, n_valid_cells,
#          year, month, season

WIND_PATH = ("/user/geog/falejandraperez/sea-ice-phase/data/merged/"
             "analysis_table_daily_anomaly_periodclim.csv")
# columns: date, sector, wind_stress, SIA, delta_SIA, doy, SIA_climatology,
#          SIA_anomaly, delta_SIA_anomaly, wind_stress_climatology,
#          wind_stress_anomaly
# wind_stress_anomaly is ALREADY deseasonalized against period-specific
# climatologies, so it is used directly and not deseasonalized again here.
# Using the same file as the SIA sensitivity test keeps the two analyses
# directly comparable -- identical wind data, identical anomaly construction.
WIND_COL = "wind_stress_anomaly"

OCEAN_STATE_PATH = "/user/geog/falejandraperez/sea-ice-phase/scripts/python/scar_poster/sst_anomaly_by_sector_daily.csv"

# optional: path to per-(year, sector) ocean state, e.g. Ch3 amplitude anomaly
# expected columns: year, sector, ocean_state
# set to None to skip Test B

DIV_VARIABLE = "div_net"          # div_net | div_positive | div_negative
RUN_ALL_DIV_VARIANTS = True       # also run positive/negative separately

SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]   # partial year + record gaps (as Ch2)

SECTOR_RENAME = {
    "ABS": "Amundsen-Bellingshausen",
    "WED": "Weddell",
    "KHV": "King Haakon VII",
    "EA":  "East Antarctica",
    "RA":  "Ross-Amundsen",
}
# compute_ice_divergence_nsidc0116.py writes short codes; the existing analysis
# tables use full names. Mapped here rather than re-running the divergence
# computation.

SECTORS = ["Amundsen-Bellingshausen", "Weddell", "King Haakon VII",
           "East Antarctica", "Ross-Amundsen"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]

N_BOOT = 2000
BLOCK_YEARS = 3
FDR_Q = 0.05
MIN_N = 30
RNG_SEED = 42

SEC_PER_DAY = 86400.0             # s^-1 -> day^-1 for readable coefficients

OUT_BINARY = "wind_divergence_binary_test.csv"
OUT_CONTINUOUS = "wind_divergence_oceanstate_test.csv"
# -----------------------------------------


def load_and_merge():
    div = pd.read_csv(DIVERGENCE_PATH, parse_dates=["date"])
    wind = pd.read_csv(WIND_PATH, parse_dates=["date"])

    div["sector"] = div["sector"].replace(SECTOR_RENAME)
    unmapped = set(div["sector"]) - set(SECTORS)
    if unmapped:
        print(f"[warn] divergence sectors not in SECTORS list: {unmapped}")

    # drop padded/empty days (1978 pre-November, record gaps)
    if "n_valid_cells" in div.columns:
        before = len(div)
        div = div[div["n_valid_cells"] > 0]
        print(f"Dropped {before - len(div):,} rows with no valid cells")

    if WIND_COL not in wind.columns:
        raise KeyError(f"{WIND_COL!r} not in {WIND_PATH}. "
                       f"Available: {list(wind.columns)}")

    wind = wind[["date", "sector", WIND_COL]].rename(
        columns={WIND_COL: "wind_anom"})

    df = div.merge(wind, on=["date", "sector"], how="inner")
    print(f"Merged: {len(df):,} sector-days "
          f"({df['date'].min().date()} to {df['date'].max().date()})")

    if df.empty:
        raise ValueError(
            "Merge produced no rows. Check that sector labels match between "
            "the two files (e.g. 'ABS' vs 'A-B', 'WED' vs 'Weddell')."
        )

    df = df[~df["year"].isin(EXCLUDE_YEARS)]
    df["post"] = (df["year"] >= SPLIT_YEAR).astype(int)

    n_pre = (df["post"] == 0).sum()
    n_post = (df["post"] == 1).sum()
    print(f"Pre-{SPLIT_YEAR}: {n_pre:,} sector-days | "
          f"Post-{SPLIT_YEAR}: {n_post:,} sector-days")
    return df


def deseasonalize(df, columns):
    """Remove period-specific day-of-year climatologies.

    Period-specific (not full-record) climatologies are essential here: a single
    climatology leaves the post-2016 mean state shift inside the anomalies,
    which would then be picked up by the interaction term as if it were a change
    in the forcing-response relationship. Same reasoning as the SCAR
    deseasonalization fix.
    """
    df = df.copy()
    df["doy"] = df["date"].dt.dayofyear

    for col in columns:
        anom = np.full(len(df), np.nan)
        for sector in df["sector"].unique():
            for post in (0, 1):
                m = (df["sector"] == sector) & (df["post"] == post)
                sub = df.loc[m]
                if sub.empty:
                    continue
                clim = sub.groupby("doy")[col].transform("mean")
                anom[np.where(m)[0]] = (sub[col] - clim).values
        df[f"{col}_anom"] = anom
    return df


def fit(df, formula):
    return smf.ols(formula, data=df).fit()


def block_bootstrap_ci(df, formula, term, n_boot=N_BOOT,
                       block_years=BLOCK_YEARS, seed=RNG_SEED):
    """Stationary block bootstrap over contiguous year-blocks.

    Resampling whole multi-year blocks preserves both within-year
    autocorrelation (strong in daily sea ice and wind series) and between-year
    dependence. Resampling individual days would badly understate uncertainty.
    """
    rng = np.random.default_rng(seed)
    years = np.sort(df["year"].unique())
    n_years = len(years)
    if n_years < block_years * 2:
        return np.nan, np.nan

    by_year = {y: df[df["year"] == y] for y in years}
    coefs = []

    for _ in range(n_boot):
        n_blocks = int(np.ceil(n_years / block_years))
        starts = rng.integers(0, n_years - block_years + 1, size=n_blocks)
        sampled = []
        for s in starts:
            sampled.extend(years[s:s + block_years])
        sampled = sampled[:n_years]

        boot = pd.concat([by_year[y] for y in sampled], ignore_index=True)
        # need both periods present for an interaction to be estimable
        if "post" in formula and boot["post"].nunique() < 2:
            continue
        try:
            coefs.append(fit(boot, formula).params.get(term, np.nan))
        except Exception:
            continue

    coefs = np.array([c for c in coefs if np.isfinite(c)])
    if coefs.size < n_boot * 0.5:
        print(f"      [warn] only {coefs.size}/{n_boot} bootstrap fits succeeded")
    if coefs.size == 0:
        return np.nan, np.nan
    return np.percentile(coefs, 2.5), np.percentile(coefs, 97.5)


def run_test(df, div_col, moderator, out_path, label):
    """moderator: 'post' (binary) or 'ocean_state' (continuous)."""
    y = f"{div_col}_anom"
    formula = f"{y} ~ wind_anom * {moderator}"
    term = f"wind_anom:{moderator}"

    print(f"\n=== {label}: {div_col} ~ wind x {moderator} ===")
    rows = []

    for sector in SECTORS:
        for season in SEASONS:
            sub = df[(df["sector"] == sector) & (df["season"] == season)]
            sub = sub.dropna(subset=[y, "wind_anom", moderator])
            if len(sub) < MIN_N or sub[moderator].nunique() < 2:
                print(f"  [skip] {sector}-{season}: n={len(sub)}")
                continue

            model = fit(sub, formula)
            lo, hi = block_bootstrap_ci(sub, formula, term)

            rows.append({
                "sector": sector,
                "season": season,
                "n": len(sub),
                "wind_coef": model.params.get("wind_anom", np.nan) * SEC_PER_DAY,
                "interaction_coef": model.params.get(term, np.nan) * SEC_PER_DAY,
                "p_value": model.pvalues.get(term, np.nan),
                "boot_ci_lo": lo * SEC_PER_DAY if np.isfinite(lo) else np.nan,
                "boot_ci_hi": hi * SEC_PER_DAY if np.isfinite(hi) else np.nan,
                "r_squared": model.rsquared,
            })

    res = pd.DataFrame(rows)
    if res.empty:
        print("  no estimable tests")
        return res

    ok = res["p_value"].notna()
    reject, p_adj, _, _ = multipletests(res.loc[ok, "p_value"],
                                        alpha=FDR_Q, method="fdr_bh")
    res.loc[ok, "p_fdr"] = p_adj
    res.loc[ok, "significant_fdr"] = reject
    res["significant_fdr"] = res["significant_fdr"].fillna(False)

    res.to_csv(out_path, index=False)
    print(res[["sector", "season", "n", "wind_coef", "interaction_coef",
               "p_value", "p_fdr", "significant_fdr"]].to_string(index=False))

    n_sig = int(res["significant_fdr"].sum())
    print(f"\n  {n_sig}/{len(res)} significant after FDR (q={FDR_Q})")
    if n_sig:
        sig = res[res["significant_fdr"]]
        for _, r in sig.iterrows():
            direction = "stronger" if r["interaction_coef"] > 0 else "weaker"
            print(f"    {r['sector']}-{r['season']}: coupling {direction} "
                  f"(coef={r['interaction_coef']:.3e} /day, "
                  f"CI [{r['boot_ci_lo']:.3e}, {r['boot_ci_hi']:.3e}])")
    print(f"  -> {out_path}")
    return res


def main():
    df = load_and_merge()

    div_cols = ([DIV_VARIABLE] if not RUN_ALL_DIV_VARIANTS
                else ["div_net", "div_positive", "div_negative"])
    div_cols = [c for c in div_cols if c in df.columns]

    # Only divergence needs deseasonalizing -- wind arrives as an anomaly
    # already computed against period-specific climatologies. Deseasonalizing
    # it a second time would remove structure that is no longer there and
    # distort the regression.
    df = deseasonalize(df, div_cols)

    for col in div_cols:
        suffix = "" if col == "div_net" else f"_{col}"
        run_test(df, col, "post",
                 OUT_BINARY.replace(".csv", f"{suffix}.csv"),
                 "Test A (binary pre/post)")

    if OCEAN_STATE_PATH:
        ocean = pd.read_csv(OCEAN_STATE_PATH)
        df = df.merge(ocean[["year", "sector", "ocean_state"]],
                      on=["year", "sector"], how="left")
        n_miss = df["ocean_state"].isna().sum()
        if n_miss:
            print(f"\n[warn] {n_miss:,} rows lack an ocean_state value")
        for col in div_cols:
            suffix = "" if col == "div_net" else f"_{col}"
            run_test(df, col, "ocean_state",
                     OUT_CONTINUOUS.replace(".csv", f"{suffix}.csv"),
                     "Test B (continuous ocean state)")
    else:
        print("\n[info] OCEAN_STATE_PATH not set; skipping Test B")

    print("\nCoefficients are in day^-1 per unit wind stress. Interpretation: "
          "a positive interaction means a given wind stress produces MORE "
          "divergence post-2016 (or at higher ocean-state values); negative "
          "means less. Sign conventions differ by sector, so compare the "
          "magnitude of the marginal effect, not the raw sign, across sectors.")


if __name__ == "__main__":
    main()