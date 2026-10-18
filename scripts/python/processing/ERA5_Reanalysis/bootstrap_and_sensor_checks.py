# =====================================================
# (1) Block bootstrap (resample YEARS) -> CIs on % change of wind, drift, excess
# (2) 2012 sensor check: annual JJA/SON drift series + post-2016 change vs an
#     AMSR2-era-only baseline (2012-2015)
# Inputs: the two long-format sector-daily CSVs.
# =====================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WIND_CSV  = "/user/geog/falejandraperez/sea-ice-phase/results/ERA5/ERA5_windSpeed_daily_sector_v2.csv"
DRIFT_CSV = "/user/geog/falejandraperez/sea-ice-phase/results/NSIDC0116/NSIDC0116_driftSpeed_daily_sector.csv"
OUT_DIR   = "/user/geog/falejandraperez/sea-ice-phase/results/wind_drift_norm"
PERIOD_SPLIT = 2016
N_BOOT = 1000
SEED = 42

SECTOR_LABELS = {1: "Weddell", 2: "Amundsen-Bellingshausen", 3: "Ross",
                 4: "East Antarctica", 5: "King Haakon VII"}
SECTOR_COLORS = {1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 4: "#D55E00", 5: "#CC79A7"}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
SEASON_MARKERS = {"DJF": "o", "MAM": "s", "JJA": "^", "SON": "D"}

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# -----------------------------------------------------
# Load -> one row per (sector, season, season_year): seasonal mean
# DJF is assigned to the year of its Jan/Feb (Dec belongs to the following year)
# -----------------------------------------------------
def seasonal_means(path, value_col):
    df = pd.read_csv(path, parse_dates=["date"])
    m = df["date"].dt.month
    df["season"] = np.select([m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])],
                             ["DJF", "MAM", "JJA"], default="SON")
    df["syear"] = df["date"].dt.year + (m == 12).astype(int)
    g = (df.groupby(["sector", "season", "syear"])[value_col]
           .mean().rename("value").reset_index())
    return g

wind  = seasonal_means(WIND_CSV,  "wind_speed_mps")
drift = seasonal_means(DRIFT_CSV, "drift_speed_mps")
both = wind.merge(drift, on=["sector", "season", "syear"], suffixes=("_w", "_d")).dropna()

# drop incomplete edge seasons (first DJF has no Dec; last may be partial)
both = both[(both["syear"] >= 1980) & (both["syear"] <= 2024)]

# =====================================================
# (1) BLOCK BOOTSTRAP OVER YEARS
# =====================================================
def pct(post, pre):
    return 100.0 * (post.mean() - pre.mean()) / pre.mean()

rows = []
for (sid, season), g in both.groupby(["sector", "season"]):
    g = g.sort_values("syear")
    pre  = g[g["syear"] <  PERIOD_SPLIT]
    post = g[g["syear"] >= PERIOD_SPLIT]
    n_pre, n_post = len(pre), len(post)
    if n_pre < 5 or n_post < 3:
        continue

    w_obs = pct(post["value_w"], pre["value_w"])
    d_obs = pct(post["value_d"], pre["value_d"])

    w_b = np.empty(N_BOOT); d_b = np.empty(N_BOOT)
    pre_w, pre_d = pre["value_w"].values, pre["value_d"].values
    post_w, post_d = post["value_w"].values, post["value_d"].values
    for b in range(N_BOOT):
        ip = rng.integers(0, n_pre,  n_pre)    # same resampled years for wind & drift
        iq = rng.integers(0, n_post, n_post)
        w_b[b] = 100.0 * (post_w[iq].mean() - pre_w[ip].mean()) / pre_w[ip].mean()
        d_b[b] = 100.0 * (post_d[iq].mean() - pre_d[ip].mean()) / pre_d[ip].mean()
    e_b = d_b - w_b

    def ci(a): return np.percentile(a, [2.5, 97.5])
    wlo, whi = ci(w_b); dlo, dhi = ci(d_b); elo, ehi = ci(e_b)
    rows.append(dict(sector=sid, season=season, n_pre=n_pre, n_post=n_post,
                     wind_pct=w_obs, wind_lo=wlo, wind_hi=whi,
                     drift_pct=d_obs, drift_lo=dlo, drift_hi=dhi,
                     excess_pp=d_obs - w_obs, excess_lo=elo, excess_hi=ehi,
                     excess_excludes_zero=(elo > 0) or (ehi < 0)))

boot = pd.DataFrame(rows)
boot["season"] = pd.Categorical(boot["season"], SEASON_ORDER)
boot = boot.sort_values(["season", "sector"])
boot.to_csv(os.path.join(OUT_DIR, "bootstrap_pct_change_CIs.csv"), index=False)

pd.set_option("display.width", 200)
print(f"\nBlock bootstrap over years, {N_BOOT} reps, 95% CIs:")
print(boot[["season", "sector", "n_pre", "n_post",
            "wind_pct", "wind_lo", "wind_hi",
            "drift_pct", "drift_lo", "drift_hi",
            "excess_pp", "excess_lo", "excess_hi", "excess_excludes_zero"]]
      .round(1).to_string(index=False))

# ---- scatter with error bars
fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=140)
lim = max(20.0, float(np.nanmax(np.abs(boot[["wind_hi", "wind_lo", "drift_hi", "drift_lo"]].values))) * 1.05)
ax.fill_between([-lim, lim], [-lim, lim], [lim, lim], color="#F5EEF8", zorder=0)
ax.plot([-lim, lim], [-lim, lim], color="#999", lw=1.2, ls="--", zorder=1)
ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
for _, r in boot.iterrows():
    c = SECTOR_COLORS[int(r.sector)]
    ax.errorbar(r.wind_pct, r.drift_pct,
                xerr=[[r.wind_pct - r.wind_lo], [r.wind_hi - r.wind_pct]],
                yerr=[[r.drift_pct - r.drift_lo], [r.drift_hi - r.drift_pct]],
                fmt="none", ecolor=c, elinewidth=1.2, alpha=0.6, capsize=0, zorder=2)
    ax.scatter(r.wind_pct, r.drift_pct, s=110, c=c, marker=SEASON_MARKERS[r.season],
               edgecolors="white", linewidths=1.5, zorder=3,
               alpha=1.0 if r.excess_excludes_zero else 0.45)
ax.text(0.5 * lim, 0.95 * lim, "drift > wind\n(free-drift signature)", ha="center", va="top",
        fontsize=9, color="#5B2C6F")
sector_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=SECTOR_COLORS[s],
                         markersize=10, label=SECTOR_LABELS[s]) for s in sorted(SECTOR_COLORS)]
season_handles = [Line2D([0], [0], marker=SEASON_MARKERS[s], color="w", markerfacecolor="#555",
                         markersize=10, label=s) for s in SEASON_ORDER]
leg1 = ax.legend(handles=sector_handles, title="Sector", loc="upper left", frameon=False, fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=season_handles, title="Season (faded = excess CI spans 0)",
          loc="lower right", frameon=False, fontsize=9, title_fontsize=8)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_xlabel("Wind speed change post-2016 (%)"); ax.set_ylabel("Drift speed change post-2016 (%)")
ax.set_title("Wind vs drift change with 95% bootstrap CIs (years resampled)", fontsize=12, pad=12)
ax.grid(True, color="#eee", lw=0.6)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "wind_vs_drift_pct_change_CI.png"), bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "wind_vs_drift_pct_change_CI.pdf"), bbox_inches="tight")
print("Saved: wind_vs_drift_pct_change_CI.png/.pdf")

# =====================================================
# (2) 2012 SENSOR CHECK
# =====================================================
# (a) annual series of sector-mean drift, JJA and SON, with sensor-era markers
eras = {1987.5: "SMMR→SSM/I", 2002.5: "AMSR-E on", 2011.8: "AMSR-E off", 2012.5: "AMSR2 on", 2016: "2016"}
fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=130, sharex=True)
for ax, season in zip(axes, ["JJA", "SON"]):
    for sid in sorted(SECTOR_COLORS):
        g = both[(both["sector"] == sid) & (both["season"] == season)].sort_values("syear")
        ax.plot(g["syear"], g["value_d"] * 100, lw=2, color=SECTOR_COLORS[sid],
                label=SECTOR_LABELS[sid])
    for yr, lab in eras.items():
        ax.axvline(yr, color="#999", lw=0.8, ls=":" if yr != 2016 else "--")
        ax.text(yr + 0.2, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 0 else 1, lab,
                fontsize=7, color="#666", va="top", rotation=90)
    ax.set_ylabel(f"{season} drift speed (cm/s)")
    ax.grid(True, color="#eee", lw=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].legend(frameon=False, fontsize=8, ncol=5, loc="lower left")
axes[1].set_xlabel("Season year")
fig.suptitle("Sector-mean drift speed by year — look for steps at sensor transitions vs 2016", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "drift_annual_series_sensor_eras.png"), bbox_inches="tight")
print("Saved: drift_annual_series_sensor_eras.png")

# (b) post-2016 change vs an AMSR2-only baseline (2012-2015) vs the full baseline
print("\n2012 sensor check: post-2016 drift % change vs two baselines")
print("  full = 1980-2015 (mixed sensors);  amsr2 = 2012-2015 only (same sensor family as post-2016)")
print("  If 'amsr2' is still strongly positive, the signal is not an AMSR2 step.\n")
print(f"{'season':>6} {'sector':>6} {'vs_full%':>9} {'vs_amsr2%':>10} {'2008-11→2012-15%':>17}")
for (sid, season), g in both.groupby(["sector", "season"]):
    if season not in ("JJA", "SON", "MAM"):
        continue
    d = g.set_index("syear")["value_d"]
    full  = d[(d.index >= 1980) & (d.index <= 2015)].mean()
    amsr2 = d[(d.index >= 2012) & (d.index <= 2015)].mean()
    pre12 = d[(d.index >= 2008) & (d.index <= 2011)].mean()
    post  = d[d.index >= 2016].mean()
    print(f"{season:>6} {sid:>6} {100*(post-full)/full:>9.1f} {100*(post-amsr2)/amsr2:>10.1f} "
          f"{100*(amsr2-pre12)/pre12:>17.1f}")
print("\nLast column = step across the 2012 AMSR2 transition *within* the pre-2016 period.")
print("A large value there, comparable to the post-2016 change, would be a red flag.")
