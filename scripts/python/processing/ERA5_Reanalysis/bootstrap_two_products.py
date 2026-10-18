import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WIND_CSV  = "/user/geog/falejandraperez/sea-ice-phase/results/ERA5/ERA5_windSpeed_daily_sector_v2.csv"
DRIFT = {
    "NSIDC-0116": "/user/geog/falejandraperez/sea-ice-phase/results/NSIDC0116/NSIDC0116_driftSpeed_daily_sector.csv",
    "OSI-455":    "/user/geog/falejandraperez/sea-ice-phase/results/OSI455/OSI455_driftSpeed_daily_sector.csv",
}
OUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/results/wind_drift_norm"
BASELINE_START, PERIOD_SPLIT = 1992, 2016
POST_WINDOWS = {"2016-2020": (2016, 2020), "2016-2024": (2016, 2024)}
N_BOOT, SEED = 2000, 42
SEASONS = ["MAM", "JJA", "SON"]

SECTOR_LABELS = {1: "Weddell", 2: "Amundsen-Bellingshausen", 3: "Ross",
                 4: "East Antarctica", 5: "King Haakon VII"}
SECTOR_COLORS = {1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 4: "#D55E00", 5: "#CC79A7"}
SEASON_MARKERS = {"MAM": "s", "JJA": "^", "SON": "D"}
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

OSI_MONTHS = sorted(pd.read_csv(DRIFT["OSI-455"], parse_dates=["date"])["date"].dt.month.unique())
print(f"Months used in all products: {[int(m) for m in OSI_MONTHS]}")

def seasonal(path, col):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"].dt.month.isin(OSI_MONTHS)]
    m = df["date"].dt.month
    df["season"] = np.select([m.isin([3, 4, 5]), m.isin([6, 7, 8]), m.isin([9, 10, 11])],
                             ["MAM", "JJA", "SON"], default="DJF")
    df["syear"] = df["date"].dt.year
    return (df[df["season"].isin(SEASONS)]
              .groupby(["sector", "season", "syear"])[col].mean().rename("value").reset_index())

wind = seasonal(WIND_CSV, "wind_speed_mps")

def boot_pct(pre_w, post_w, pre_d, post_d):
    n_pre, n_post = len(pre_w), len(post_w)
    w = np.empty(N_BOOT); d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ip = rng.integers(0, n_pre, n_pre); iq = rng.integers(0, n_post, n_post)
        w[b] = 100 * (post_w[iq].mean() - pre_w[ip].mean()) / pre_w[ip].mean()
        d[b] = 100 * (post_d[iq].mean() - pre_d[ip].mean()) / pre_d[ip].mean()
    return w, d, d - w

rows = []
for product, path in DRIFT.items():
    drift = seasonal(path, "drift_speed_mps")
    both = wind.merge(drift, on=["sector", "season", "syear"], suffixes=("_w", "_d")).dropna()
    for win_name, (p0, p1) in POST_WINDOWS.items():
        if product == "OSI-455" and p1 > 2020:
            continue
        for (sid, season), g in both.groupby(["sector", "season"]):
            pre  = g[(g["syear"] >= BASELINE_START) & (g["syear"] < PERIOD_SPLIT)]
            post = g[(g["syear"] >= p0) & (g["syear"] <= p1)]
            if len(pre) < 10 or len(post) < 3:
                continue
            w_obs = 100 * (post["value_w"].mean() - pre["value_w"].mean()) / pre["value_w"].mean()
            d_obs = 100 * (post["value_d"].mean() - pre["value_d"].mean()) / pre["value_d"].mean()
            wb, db, eb = boot_pct(pre["value_w"].values, post["value_w"].values,
                                  pre["value_d"].values, post["value_d"].values)
            ci = lambda a: np.percentile(a, [2.5, 97.5])
            (wlo, whi), (dlo, dhi), (elo, ehi) = ci(wb), ci(db), ci(eb)
            rows.append(dict(product=product, post=win_name, season=season, sector=sid,
                             n_pre=len(pre), n_post=len(post),
                             wind_pct=w_obs, wind_lo=wlo, wind_hi=whi,
                             drift_pct=d_obs, drift_lo=dlo, drift_hi=dhi,
                             excess_pp=d_obs - w_obs, excess_lo=elo, excess_hi=ehi,
                             sig=(elo > 0) or (ehi < 0)))

tab = pd.DataFrame(rows)
tab["season"] = pd.Categorical(tab["season"], SEASONS)
tab = tab.sort_values(["post", "season", "sector", "product"])
tab.to_csv(os.path.join(OUT_DIR, f"bootstrap_two_products_baseline{BASELINE_START}.csv"), index=False)

pd.set_option("display.width", 230)
print(f"\nBaseline {BASELINE_START}-2015. {N_BOOT} bootstrap reps. 'sig' = excess CI excludes 0.\n")
for win in POST_WINDOWS:
    t = tab[tab["post"] == win]
    if t.empty: continue
    print(f"=== post window {win} ===")
    print(t[["season", "sector", "product", "n_pre", "n_post",
             "wind_pct", "drift_pct", "drift_lo", "drift_hi",
             "excess_pp", "excess_lo", "excess_hi", "sig"]].round(1).to_string(index=False))
    print()

t = tab[tab["post"] == "2016-2020"]
fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=140)
lim = max(15, float(np.nanmax(np.abs(t[["wind_lo", "wind_hi", "drift_lo", "drift_hi"]].values))) * 1.05)
ax.fill_between([-lim, lim], [-lim, lim], [lim, lim], color="#F5EEF8", zorder=0)
ax.plot([-lim, lim], [-lim, lim], ls="--", color="#999", lw=1.2, zorder=1)
ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
for _, r in t.iterrows():
    c = SECTOR_COLORS[int(r.sector)]
    hollow = r["product"] == "OSI-455"
    ax.errorbar(r.wind_pct, r.drift_pct,
                xerr=[[r.wind_pct - r.wind_lo], [r.wind_hi - r.wind_pct]],
                yerr=[[r.drift_pct - r.drift_lo], [r.drift_hi - r.drift_pct]],
                fmt="none", ecolor=c, alpha=0.45, lw=1, zorder=2)
    ax.scatter(r.wind_pct, r.drift_pct, s=110, marker=SEASON_MARKERS[r.season],
               facecolors="white" if hollow else c, edgecolors=c, linewidths=1.8,
               alpha=1.0 if r.sig else 0.5, zorder=3)
ax.text(0.5 * lim, 0.95 * lim, "drift > wind", ha="center", va="top", fontsize=9, color="#5B2C6F")
h1 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=SECTOR_COLORS[s], markersize=9,
             label=SECTOR_LABELS[s]) for s in sorted(SECTOR_COLORS)]
h2 = [Line2D([0], [0], marker=SEASON_MARKERS[s], color="w", markerfacecolor="#555", markersize=9, label=s)
      for s in SEASONS]
h3 = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", markersize=9, label="NSIDC-0116"),
      Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#555",
             markeredgewidth=1.8, markersize=9, label="OSI-455")]
l1 = ax.legend(handles=h1, loc="upper left", frameon=False, fontsize=8, title="Sector"); ax.add_artist(l1)
l2 = ax.legend(handles=h2, loc="lower right", frameon=False, fontsize=8, title="Season"); ax.add_artist(l2)
ax.legend(handles=h3, loc="lower left", frameon=False, fontsize=8, title="Product (faded = CI spans 0)")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_xlabel("Wind speed change, 2016-2020 vs 1992-2015 (%)")
ax.set_ylabel("Drift speed change, 2016-2020 vs 1992-2015 (%)")
ax.set_title("Post-2016 drift vs wind change, two drift products, 95% bootstrap CIs", fontsize=11)
ax.grid(True, color="#eee", lw=0.6)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout()
out = os.path.join(OUT_DIR, f"wind_vs_drift_two_products_baseline{BASELINE_START}.png")
plt.savefig(out, bbox_inches="tight"); plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved: {out}")
