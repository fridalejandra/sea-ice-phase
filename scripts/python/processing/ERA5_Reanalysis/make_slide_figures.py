import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "/user/geog/falejandraperez/sea-ice-phase/results"
WIND  = f"{R}/ERA5/ERA5_windSpeed_daily_sector_v2.csv"
NSIDC = f"{R}/NSIDC0116/NSIDC0116_driftSpeed_daily_sector.csv"
OSI   = f"{R}/OSI455/OSI455_driftSpeed_daily_sector.csv"
BOOT  = f"{R}/wind_drift_norm/bootstrap_two_products_baseline1992.csv"
OUT   = f"{R}/wind_drift_norm"
T0, T1 = 1992, 2015

LAB = {1: "Weddell", 2: "Amundsen–Bellingshausen", 3: "Ross", 4: "East Antarctica", 5: "King Haakon VII"}
COL = {1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 4: "#D55E00", 5: "#CC79A7"}
MK  = {"MAM": "s", "JJA": "^", "SON": "D"}
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})

months = sorted(pd.read_csv(OSI, parse_dates=["date"])["date"].dt.month.unique())
def seasonal(path, col, name):
    d = pd.read_csv(path, parse_dates=["date"]); d = d[d["date"].dt.month.isin(months)]
    m = d["date"].dt.month
    d["season"] = np.select([m.isin([3,4,5]), m.isin([6,7,8]), m.isin([9,10,11])], ["MAM","JJA","SON"], "DJF")
    d["year"] = d["date"].dt.year
    return d.groupby(["sector","season","year"])[col].mean().rename(name).reset_index()
w = seasonal(WIND, "wind_speed_mps", "wind"); n = seasonal(NSIDC, "drift_speed_mps", "nsidc"); o = seasonal(OSI, "drift_speed_mps", "osi")
ts = w.merge(n, on=["sector","season","year"], how="left").merge(o, on=["sector","season","year"], how="left")
def norm(g, c):
    base = g[(g.year >= T0) & (g.year <= T1)][c].mean(); return 100 * g[c] / base

boot = pd.read_csv(BOOT); boot = boot[boot["post"] == "2016-2020"]

def scatter(ax, products, emphasize_products=True):
    t = boot[boot["product"].isin(products)]
    lim = 24
    ax.fill_between([-lim, lim], [-lim, lim], [lim, lim], color="#F5EEF8", zorder=0)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#999", lw=1.2, zorder=1)
    ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
    for _, r in t.iterrows():
        c = COL[int(r.sector)]; hollow = r["product"] == "OSI-455"
        ax.errorbar(r.wind_pct, r.drift_pct, xerr=[[r.wind_pct-r.wind_lo],[r.wind_hi-r.wind_pct]],
                    yerr=[[r.drift_pct-r.drift_lo],[r.drift_hi-r.drift_pct]], fmt="none", ecolor=c,
                    alpha=0.35 if r.sig else 0.18, lw=1, zorder=2)
        ax.scatter(r.wind_pct, r.drift_pct, s=120 if r.sig else 70, marker=MK[r.season],
                   facecolors="white" if hollow else c, edgecolors=c, linewidths=1.8,
                   alpha=1 if r.sig else 0.45, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Wind-speed change, 2016–20 vs 1992–2015 (%)")
    ax.set_ylabel("Ice-drift-speed change, 2016–20 vs 1992–2015 (%)")
    ax.text(0.45*lim, 0.95*lim, "drift ↑ more than wind", ha="center", va="top", fontsize=9, color="#5B2C6F")
    ax.grid(True, color="#eee", lw=0.6)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    h1 = [Line2D([0],[0], marker="o", color="w", markerfacecolor=COL[s], markersize=8, label=LAB[s]) for s in COL]
    h2 = [Line2D([0],[0], marker=MK[s], color="w", markerfacecolor="#555", markersize=8, label=s) for s in MK]
    l1 = ax.legend(handles=h1, loc="upper left", frameon=False, fontsize=8); ax.add_artist(l1)
    l2 = ax.legend(handles=h2, loc="lower right", frameon=False, fontsize=8); ax.add_artist(l2)
    if emphasize_products and len(products) == 2:
        h3 = [Line2D([0],[0], marker="o", color="w", markerfacecolor="#555", markersize=8, label="NSIDC-0116"),
              Line2D([0],[0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#555", markeredgewidth=1.8, markersize=8, label="OSI-455")]
        ax.legend(handles=h3, loc="lower left", frameon=False, fontsize=8, title="Drift product", title_fontsize=8)
    ax.text(0.99, 0.01, "bold = 95% CI excludes 0 (years bootstrapped)", transform=ax.transAxes, ha="right", fontsize=7, color="#777")

# ============ SLIDE A: product difference ============
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=150, gridspec_kw={"width_ratios": [1, 1.25]})
scatter(a1, ["NSIDC-0116", "OSI-455"])
a1.set_title("Same test, two independent drift products", loc="left")
for sid in COL:
    g = ts[(ts.sector == sid) & (ts.season == "JJA")].sort_values("year")
    a2.plot(g.year, norm(g, "nsidc"), lw=2, color=COL[sid], label=LAB[sid])
    a2.plot(g.year, norm(g, "osi"), lw=1.6, ls=":", color=COL[sid])
a2.axhline(100, color="#bbb", lw=0.8); a2.axvspan(1991, 1995.5, color="#f3f3f3", zorder=0)
a2.text(1993.2, 123, "early record:\nproducts diverge", ha="center", fontsize=8.5, color="#666")
a2.axvline(2015.5, color="#999", lw=0.8, ls="--")
a2.set_xlim(1990.5, 2024.5); a2.set_ylim(60, 128)
a2.set_ylabel("JJA drift speed (% of 1992–2015 mean)"); a2.set_xlabel("Year")
a2.set_title("Products agree on variability, not on the trend", loc="left")
a2.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
a2.text(0.01, 0.03, "solid NSIDC-0116  ·  dotted OSI-455", transform=a2.transAxes, fontsize=8, color="#555")
a2.grid(True, color="#eee", lw=0.6)
for s in ("top","right"): a2.spines[s].set_visible(False)
fig.suptitle("Wind–ice-drift coupling after 2016: what two drift records can and cannot constrain", x=0.01, ha="left", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{OUT}/slide_products.png", bbox_inches="tight"); plt.savefig(f"{OUT}/slide_products.pdf", bbox_inches="tight")

# ============ SLIDE B: physical takeaway ============
fig, (b1, b2) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=150, gridspec_kw={"width_ratios": [1, 1.25]})
scatter(b1, ["NSIDC-0116", "OSI-455"], emphasize_products=False)
b1.set_title("Post-2016 change in drift vs wind, by sector & season", loc="left")
b1.text(0.99, 0.05, "filled NSIDC-0116 · hollow OSI-455", transform=b1.transAxes, ha="right", fontsize=7.5, color="#777")
g = ts[(ts.sector == 2) & (ts.season == "SON")].sort_values("year")
b2.axvspan(2015.5, 2020.5, color="#EAF2FB", zorder=0)
b2.plot(g.year, norm(g, "wind"), lw=2, color="#777", label="ERA5 wind speed")
b2.plot(g.year, norm(g, "nsidc"), lw=2.2, color=COL[2], label="Ice drift speed (NSIDC-0116)")
b2.plot(g.year, norm(g, "osi"), lw=1.8, ls=":", color=COL[2], label="Ice drift speed (OSI-455)")
b2.axhline(100, color="#bbb", lw=0.8)
b2.set_xlim(1991.5, 2024.5); b2.set_ylim(70, 140)
b2.set_ylabel("Spring (SON) value, % of 1992–2015 mean"); b2.set_xlabel("Year")
b2.set_title("Amundsen–Bellingshausen, spring: a brief surge, then relaxation", loc="left")
b2.text(2018, 134, "2016–20: drift +10–14%,\nwind +1%", ha="center", fontsize=9, color="#1f4e79")
b2.legend(frameon=False, fontsize=8.5, loc="lower left")
b2.grid(True, color="#eee", lw=0.6)
for s in ("top","right"): b2.spines[s].set_visible(False)
fig.suptitle("After 2016, Antarctic sea ice did not become more responsive to wind — except briefly, where the ice was lost", x=0.01, ha="left", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{OUT}/slide_physics.png", bbox_inches="tight"); plt.savefig(f"{OUT}/slide_physics.pdf", bbox_inches="tight")
print("Saved slide_products.png and slide_physics.png in", OUT)
