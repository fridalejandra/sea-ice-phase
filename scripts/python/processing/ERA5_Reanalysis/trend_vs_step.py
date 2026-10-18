# =====================================================
# Trend vs step: is the post-2016 drift increase a continuation of a
# multi-decadal trend, or an additional step on top of it?
# Also: wind factor (drift/wind) time series by sector, JJA & SON.
# =====================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

WIND_CSV  = "/user/geog/falejandraperez/sea-ice-phase/results/ERA5/ERA5_windSpeed_daily_sector_v2.csv"
DRIFT_CSV = "/user/geog/falejandraperez/sea-ice-phase/results/NSIDC0116/NSIDC0116_driftSpeed_daily_sector.csv"
OUT_DIR   = "/user/geog/falejandraperez/sea-ice-phase/results/wind_drift_norm"
PERIOD_SPLIT = 2016
TREND_START = 1992   # Kwok-style window; set 1980 to use everything. SMMR era (<1988) is coarsest.

SECTOR_LABELS = {1: "Weddell", 2: "Amundsen-Bellingshausen", 3: "Ross",
                 4: "East Antarctica", 5: "King Haakon VII"}
SECTOR_COLORS = {1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 4: "#D55E00", 5: "#CC79A7"}
os.makedirs(OUT_DIR, exist_ok=True)

def seasonal_means(path, value_col):
    df = pd.read_csv(path, parse_dates=["date"])
    m = df["date"].dt.month
    df["season"] = np.select([m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])],
                             ["DJF", "MAM", "JJA"], default="SON")
    df["syear"] = df["date"].dt.year + (m == 12).astype(int)
    return (df.groupby(["sector", "season", "syear"])[value_col]
              .mean().rename("value").reset_index())

wind  = seasonal_means(WIND_CSV,  "wind_speed_mps")
drift = seasonal_means(DRIFT_CSV, "drift_speed_mps")
both = wind.merge(drift, on=["sector", "season", "syear"], suffixes=("_w", "_d")).dropna()
both = both[(both["syear"] >= 1980) & (both["syear"] <= 2024)]
both["wf"] = both["value_d"] / both["value_w"]   # wind factor (dimensionless)

# =====================================================
# Trend vs step table
# =====================================================
rows = []
for (sid, season), g in both.groupby(["sector", "season"]):
    g = g.sort_values("syear")
    pre = g[(g["syear"] >= TREND_START) & (g["syear"] < PERIOD_SPLIT)]
    post = g[g["syear"] >= PERIOD_SPLIT]
    out = dict(season=season, sector=sid)
    for var, lab in [("value_d", "drift"), ("value_w", "wind"), ("wf", "wf")]:
        # Theil-Sen slope + Mann-Kendall on pre-2016 window
        slope, intercept, lo, hi = stats.theilslopes(pre[var].values, pre["syear"].values)
        tau, p = stats.kendalltau(pre["syear"].values, pre[var].values)
        pre_mean = pre[var].mean()
        # extrapolate pre trend into post years, compare with observed
        extrap = intercept + slope * post["syear"].values
        obs = post[var].values
        step_pp = 100.0 * (obs.mean() - extrap.mean()) / pre_mean   # residual step, % of pre mean
        trend_pp_per_decade = 100.0 * slope * 10 / pre_mean
        out[f"{lab}_trend_%/dec"] = trend_pp_per_decade
        out[f"{lab}_MK_p"] = p
        out[f"{lab}_step_above_trend_%"] = step_pp
    rows.append(out)

tab = pd.DataFrame(rows)
tab["season"] = pd.Categorical(tab["season"], ["DJF", "MAM", "JJA", "SON"])
tab = tab.sort_values(["season", "sector"])
tab.to_csv(os.path.join(OUT_DIR, "trend_vs_step.csv"), index=False)

pd.set_option("display.width", 220)
print(f"\nPre-2016 trends ({TREND_START}-2015, Theil-Sen, % of pre-mean per decade; Mann-Kendall p)")
print("and the post-2016 'step above trend' = observed post mean minus the extrapolated pre trend, % of pre mean.")
print("  step ~ 0  -> post-2016 is just the trend continuing")
print("  step >> 0 -> something extra happened after 2016\n")
cols = ["season", "sector",
        "drift_trend_%/dec", "drift_MK_p", "drift_step_above_trend_%",
        "wind_trend_%/dec", "wind_MK_p", "wind_step_above_trend_%",
        "wf_trend_%/dec", "wf_MK_p", "wf_step_above_trend_%"]
print(tab[cols].round(2).to_string(index=False))

# =====================================================
# Wind factor time series, JJA and SON, with pre-2016 trend + extrapolation
# =====================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), dpi=130, sharex=True)
for ax, season in zip(axes, ["JJA", "SON"]):
    for sid in sorted(SECTOR_COLORS):
        g = both[(both["sector"] == sid) & (both["season"] == season)].sort_values("syear")
        c = SECTOR_COLORS[sid]
        ax.plot(g["syear"], g["wf"] * 100, lw=2, color=c, label=SECTOR_LABELS[sid])
        pre = g[(g["syear"] >= TREND_START) & (g["syear"] < PERIOD_SPLIT)]
        slope, intercept, _, _ = stats.theilslopes(pre["wf"].values, pre["syear"].values)
        yrs = np.arange(TREND_START, 2025)
        ax.plot(yrs, (intercept + slope * yrs) * 100, lw=1, ls="--", color=c, alpha=0.7)
    ax.axvline(PERIOD_SPLIT - 0.5, color="#999", lw=0.8, ls="--")
    ax.set_ylabel(f"{season} wind factor (drift/wind, %)")
    ax.grid(True, color="#eee", lw=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
axes[1].set_xlabel("Season year")
fig.suptitle(f"Wind factor by sector. Dashed = Theil-Sen trend fit on {TREND_START}-2015, extrapolated.\n"
             "If post-2016 points sit on the dashed line, it's trend continuation; above it, an extra step.",
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "wind_factor_trend_vs_step.png"), bbox_inches="tight")
print("\nSaved: wind_factor_trend_vs_step.png")
