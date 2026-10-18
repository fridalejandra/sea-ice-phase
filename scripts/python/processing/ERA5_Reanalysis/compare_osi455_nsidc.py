# =====================================================
# Cross-check: does OSI-455 show the same multi-decadal rise in drift speed
# and wind factor as NSIDC-0116?  (1992-2015 Theil-Sen trends, per sector-season)
# =====================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats, signal

WIND_CSV  = "/user/geog/falejandraperez/sea-ice-phase/results/ERA5/ERA5_windSpeed_daily_sector_v2.csv"
NSIDC_CSV = "/user/geog/falejandraperez/sea-ice-phase/results/NSIDC0116/NSIDC0116_driftSpeed_daily_sector.csv"
OSI_CSV   = "/user/geog/falejandraperez/sea-ice-phase/results/OSI455/OSI455_driftSpeed_daily_sector.csv"
OUT_DIR   = "/user/geog/falejandraperez/sea-ice-phase/results/wind_drift_norm"
T0, T1 = 1992, 2015          # common trend window (both products, post-SMMR)
SEASONS = ["MAM", "JJA", "SON"]

SECTOR_LABELS = {1: "Weddell", 2: "Amundsen-Bellingshausen", 3: "Ross",
                 4: "East Antarctica", 5: "King Haakon VII"}
SECTOR_COLORS = {1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 4: "#D55E00", 5: "#CC79A7"}
SEASON_MARKERS = {"MAM": "s", "JJA": "^", "SON": "D"}
os.makedirs(OUT_DIR, exist_ok=True)

# Use only the calendar months OSI-455 was downloaded for, in ALL three datasets,
# so a season means the same months in every product (e.g. SON = Sep-Oct if Nov wasn't fetched).
OSI_MONTHS = sorted(pd.read_csv(OSI_CSV, parse_dates=["date"])["date"].dt.month.unique())
print(f"Months available in OSI-455 CSV: {OSI_MONTHS}  (all products restricted to these)")

def seasonal(path, col, name):
    df = pd.read_csv(path, parse_dates=["date"])
    m = df["date"].dt.month
    df = df[m.isin(OSI_MONTHS)]
    m = df["date"].dt.month
    df["season"] = np.select([m.isin([3, 4, 5]), m.isin([6, 7, 8]), m.isin([9, 10, 11])],
                             ["MAM", "JJA", "SON"], default="DJF")
    df["syear"] = df["date"].dt.year
    return (df[df["season"].isin(SEASONS)]
              .groupby(["sector", "season", "syear"])[col].mean().rename(name).reset_index())

w = seasonal(WIND_CSV,  "wind_speed_mps",  "wind")
n = seasonal(NSIDC_CSV, "drift_speed_mps", "nsidc")
o = seasonal(OSI_CSV,   "drift_speed_mps", "osi")
df = w.merge(n, on=["sector", "season", "syear"]).merge(o, on=["sector", "season", "syear"]).dropna()
df["wf_nsidc"] = df["nsidc"] / df["wind"]
df["wf_osi"]   = df["osi"]   / df["wind"]

# ---- trends 1992-2015
rows = []
for (sid, season), g in df.groupby(["sector", "season"]):
    g = g[(g["syear"] >= T0) & (g["syear"] <= T1)].sort_values("syear")
    if len(g) < 15:
        continue
    r = dict(sector=sid, season=season, n=len(g))
    for var in ["nsidc", "osi", "wf_nsidc", "wf_osi"]:
        slope, intercept, lo, hi = stats.theilslopes(g[var].values, g["syear"].values)
        tau, p = stats.kendalltau(g["syear"].values, g[var].values)
        mean = g[var].mean()
        r[f"{var}_trend_pct_dec"] = 100 * slope * 10 / mean
        r[f"{var}_lo"] = 100 * lo * 10 / mean
        r[f"{var}_hi"] = 100 * hi * 10 / mean
        r[f"{var}_p"] = p
    # correlation of the two products' interannual series (detrended)
    r["corr_detrended"] = np.corrcoef(
        signal.detrend(g["nsidc"].values), signal.detrend(g["osi"].values))[0, 1]
    # post-2016 (2016-2020 only, OSI span) vs 1992-2015, both products
    gp = df[(df["sector"] == sid) & (df["season"] == season)]
    pre, post = gp[(gp["syear"] >= T0) & (gp["syear"] <= T1)], gp[gp["syear"] >= 2016]
    if len(post) >= 3:
        r["nsidc_post_pct"] = 100 * (post["nsidc"].mean() - pre["nsidc"].mean()) / pre["nsidc"].mean()
        r["osi_post_pct"]   = 100 * (post["osi"].mean()   - pre["osi"].mean())   / pre["osi"].mean()
    rows.append(r)

tab = pd.DataFrame(rows)
tab["season"] = pd.Categorical(tab["season"], SEASONS)
tab = tab.sort_values(["season", "sector"])
tab.to_csv(os.path.join(OUT_DIR, "osi455_vs_nsidc_trends.csv"), index=False)

pd.set_option("display.width", 220)
print(f"\nDrift-speed trend {T0}-{T1} (% of mean per decade), NSIDC vs OSI-455, and the wind-factor trends:")
print(tab[["season", "sector", "n",
           "nsidc_trend_pct_dec", "nsidc_p", "osi_trend_pct_dec", "osi_p",
           "wf_nsidc_trend_pct_dec", "wf_osi_trend_pct_dec",
           "corr_detrended", "nsidc_post_pct", "osi_post_pct"]].round(2).to_string(index=False))
print("\nRead: if osi_trend ~ nsidc_trend and corr_detrended is high, the ramp is physical.")
print("      If osi_trend ~ 0 while nsidc_trend is positive, the NSIDC ramp is a product property.")

# ---- figure 1: trend vs trend scatter (wind factor)
fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=140)
lim = max(8, float(np.nanmax(np.abs(tab[["wf_nsidc_trend_pct_dec", "wf_osi_trend_pct_dec"]].values))) * 1.2)
ax.plot([-lim, lim], [-lim, lim], ls="--", color="#999", lw=1.2)
ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
for _, r in tab.iterrows():
    c = SECTOR_COLORS[int(r.sector)]
    ax.errorbar(r.wf_nsidc_trend_pct_dec, r.wf_osi_trend_pct_dec,
                xerr=[[r.wf_nsidc_trend_pct_dec - r.wf_nsidc_lo], [r.wf_nsidc_hi - r.wf_nsidc_trend_pct_dec]],
                yerr=[[r.wf_osi_trend_pct_dec - r.wf_osi_lo], [r.wf_osi_hi - r.wf_osi_trend_pct_dec]],
                fmt="none", ecolor=c, alpha=0.5, lw=1)
    ax.scatter(r.wf_nsidc_trend_pct_dec, r.wf_osi_trend_pct_dec, s=110, c=c,
               marker=SEASON_MARKERS[r.season], edgecolors="white", linewidths=1.5, zorder=3)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_xlabel(f"NSIDC-0116 wind-factor trend {T0}-{T1} (%/decade)")
ax.set_ylabel(f"OSI-455 wind-factor trend {T0}-{T1} (%/decade)")
ax.set_title("Do two independent drift products agree on the wind-factor trend?", fontsize=11)
h1 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=SECTOR_COLORS[s], markersize=9,
             label=SECTOR_LABELS[s]) for s in sorted(SECTOR_COLORS)]
h2 = [Line2D([0], [0], marker=SEASON_MARKERS[s], color="w", markerfacecolor="#555", markersize=9, label=s)
      for s in SEASONS]
l1 = ax.legend(handles=h1, loc="upper left", frameon=False, fontsize=8); ax.add_artist(l1)
ax.legend(handles=h2, loc="lower right", frameon=False, fontsize=8)
ax.grid(True, color="#eee", lw=0.6)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "osi455_vs_nsidc_windfactor_trends.png"), bbox_inches="tight")

# ---- figure 2: time series, both products, normalized to their 1992-2015 mean, JJA & SON
fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=130, sharex=True)
for ax, season in zip(axes, ["JJA", "SON"]):
    for sid in sorted(SECTOR_COLORS):
        g = df[(df["sector"] == sid) & (df["season"] == season)].sort_values("syear")
        base = g[(g["syear"] >= T0) & (g["syear"] <= T1)]
        c = SECTOR_COLORS[sid]
        ax.plot(g["syear"], 100 * g["nsidc"] / base["nsidc"].mean(), lw=2, color=c, label=SECTOR_LABELS[sid])
        ax.plot(g["syear"], 100 * g["osi"]   / base["osi"].mean(),   lw=1.4, color=c, ls=":")
    ax.axhline(100, color="#bbb", lw=0.8)
    ax.axvline(2015.5, color="#999", lw=0.8, ls="--")
    ax.set_ylabel(f"{season} drift speed\n(% of {T0}-{T1} mean)")
    ax.grid(True, color="#eee", lw=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
axes[0].text(0.99, 0.04, "solid = NSIDC-0116   dotted = OSI-455", transform=axes[0].transAxes,
             ha="right", fontsize=8, color="#555")
axes[1].set_xlabel("Year")
fig.suptitle("Two drift products, same sectors, normalized to their own 1992-2015 mean", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "osi455_vs_nsidc_timeseries.png"), bbox_inches="tight")
print("\nSaved: osi455_vs_nsidc_windfactor_trends.png, osi455_vs_nsidc_timeseries.png")
