import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager

R = "/user/geog/falejandraperez/sea-ice-phase/results"
WIND, NSIDC, OSI = f"{R}/ERA5/ERA5_windSpeed_daily_sector_v2.csv", f"{R}/NSIDC0116/NSIDC0116_driftSpeed_daily_sector.csv", f"{R}/OSI455/OSI455_driftSpeed_daily_sector.csv"
BOOT = f"{R}/wind_drift_norm/bootstrap_two_products_baseline1992.csv"
OUT = f"{R}/wind_drift_norm/slide_physics_clean"
T0, T1 = 1992, 2015
PANEL_TITLES = True
LAB = {1: "Weddell", 2: "Amundsen–Bellingshausen", 3: "Ross", 4: "East Antarctica", 5: "King Haakon VII"}
MK  = {"MAM": "s", "JJA": "^", "SON": "D"}
GREY, INK = "#6e6e6e", "#222"

# ---- style: Helvetica if present, cblind qualitative palette, transparent
for f in ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"]:
    if any(f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 10.5,
                     "axes.edgecolor": "#999", "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                     "axes.labelcolor": INK, "legend.fontsize": 9})
try:
    import cblind as cb
    pal, _ = cb.Colorplots().cblind(5); print("palette: cblind")
except Exception as e:
    pal = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]; print(f"cblind unavailable ({e}); Okabe-Ito")
COL = {s: pal[i] for i, s in enumerate([1, 2, 3, 4, 5])}

# ---- data
months = sorted(pd.read_csv(OSI, parse_dates=["date"])["date"].dt.month.unique())
def seasonal(path, col, name):
    d = pd.read_csv(path, parse_dates=["date"]); d = d[d["date"].dt.month.isin(months)]
    m = d["date"].dt.month
    d["season"] = np.select([m.isin([3,4,5]), m.isin([6,7,8]), m.isin([9,10,11])], ["MAM","JJA","SON"], "DJF")
    d["year"] = d["date"].dt.year
    return d.groupby(["sector","season","year"])[col].mean().rename(name).reset_index()
ts = (seasonal(WIND, "wind_speed_mps", "wind")
      .merge(seasonal(NSIDC, "drift_speed_mps", "nsidc"), on=["sector","season","year"], how="left")
      .merge(seasonal(OSI, "drift_speed_mps", "osi"), on=["sector","season","year"], how="left"))
def norm(g, c): return 100 * g[c] / g[(g.year >= T0) & (g.year <= T1)][c].mean()
boot = pd.read_csv(BOOT); boot = boot[boot["post"] == "2016-2020"]

fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.0), dpi=200, gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.28})
fig.patch.set_alpha(0)
for ax in (a, b):
    ax.patch.set_alpha(0); ax.grid(True, color="#e6e6e6", lw=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# ---- left: scatter
lim = 24
a.fill_between([-lim, lim], [-lim, lim], [lim, lim], color="#5B2C6F", alpha=0.06, zorder=0, lw=0)
a.plot([-lim, lim], [-lim, lim], ls="--", color="#999", lw=1.1, zorder=1)
a.axhline(0, color="#ccc", lw=0.8); a.axvline(0, color="#ccc", lw=0.8)
for _, r in boot.iterrows():
    c = COL[int(r.sector)]; hollow = r["product"] == "OSI-455"
    a.errorbar(r.wind_pct, r.drift_pct, xerr=[[r.wind_pct-r.wind_lo],[r.wind_hi-r.wind_pct]],
               yerr=[[r.drift_pct-r.drift_lo],[r.drift_hi-r.drift_pct]], fmt="none", ecolor=c,
               alpha=0.35 if r.sig else 0.15, lw=1, zorder=2)
    a.scatter(r.wind_pct, r.drift_pct, s=120 if r.sig else 65, marker=MK[r.season],
              facecolors="none" if hollow else c, edgecolors=c, linewidths=1.8, alpha=1 if r.sig else 0.45, zorder=3)
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
a.set_xlabel("Wind-speed change, 2016–20 vs 1992–2015 (%)")
a.set_ylabel("Ice-drift-speed change, 2016–20 vs 1992–2015 (%)")
a.text(0.55*lim, 0.92*lim, "drift ↑ more than wind", ha="center", fontsize=9, color="#5B2C6F")
h1 = [Line2D([0],[0], marker="o", ls="", markerfacecolor=COL[s], markeredgecolor=COL[s], markersize=8, label=LAB[s]) for s in COL]
h2 = [Line2D([0],[0], marker=MK[s], ls="", markerfacecolor=GREY, markeredgecolor=GREY, markersize=8, label=s) for s in MK]
h3 = [Line2D([0],[0], marker="o", ls="", markerfacecolor=GREY, markeredgecolor=GREY, markersize=8, label="NSIDC-0116"),
      Line2D([0],[0], marker="o", ls="", markerfacecolor="none", markeredgecolor=GREY, markeredgewidth=1.8, markersize=8, label="OSI-455")]
l1 = a.legend(handles=h1, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, handletextpad=0.3); a.add_artist(l1)
l2 = a.legend(handles=h2, loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=False, handletextpad=0.3); a.add_artist(l2)
a.legend(handles=h3, loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, handletextpad=0.3)
a.text(0.5, -0.16, "bold = 95% CI excludes zero (years bootstrapped); faded = not significant",
       transform=a.transAxes, ha="center", fontsize=8, color=GREY)
if PANEL_TITLES: a.set_title("Post-2016 change in drift vs wind, by sector and season", loc="left")

# ---- right: Amundsen–Bellingshausen spring time series
g = ts[(ts.sector == 2) & (ts.season == "SON")].sort_values("year")
c2 = COL[2]
b.axvspan(2015.5, 2020.5, color=c2, alpha=0.10, lw=0, zorder=0)
b.plot(g.year, norm(g, "wind"), lw=2.0, color=GREY, label="ERA5 10 m wind speed")
b.plot(g.year, norm(g, "nsidc"), lw=2.4, color=c2, label="Ice-drift speed, NSIDC-0116")
b.plot(g.year, norm(g, "osi"), lw=2.0, ls=(0, (1.5, 1.5)), color=c2, label="Ice-drift speed, OSI-455")
b.axhline(100, color="#bbb", lw=0.8)
b.set_xlim(1991.5, 2024.5); b.set_ylim(70, 140)
b.set_ylabel("Spring (SON) value, % of 1992–2015 mean"); b.set_xlabel("Year")
b.text(2018, 133, "2016–20: drift +10–14 %,\nwind +1 %", ha="center", fontsize=9.5, color=c2)
b.legend(frameon=False, loc="lower left")
if PANEL_TITLES: b.set_title("Amundsen–Bellingshausen, spring: a brief surge, then relaxation", loc="left")

plt.savefig(f"{OUT}.png", bbox_inches="tight", transparent=True)
plt.savefig(f"{OUT}.pdf", bbox_inches="tight", transparent=True)
print("saved", f"{OUT}.png")
