import numpy as np, pandas as pd, matplotlib.pyplot as plt
R = "/user/geog/falejandraperez/sea-ice-phase/results"
OUT = f"{R}/wind_drift_norm/slide_products_backup.png"
def circ(path):
    d = pd.read_csv(path, parse_dates=["date"]); d = d[d["date"].dt.month.isin([6,7,8])]
    d["year"] = d["date"].dt.year
    s = d.groupby("year")["drift_speed_mps"].mean()
    return 100 * s / s.loc[1992:2015].mean()
n = circ(f"{R}/NSIDC0116/NSIDC0116_driftSpeed_daily_sector.csv")
o = circ(f"{R}/OSI455/OSI455_driftSpeed_daily_sector.csv")
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
ax.axvspan(1978.5, 1995.5, color="#f3f3f3", zorder=0)
ax.plot(n.index, n.values, lw=2.4, color="#1f77b4", label="NSIDC-0116")
ax.plot(o.index, o.values, lw=2.4, color="#d62728", ls="--", label="OSI-455")
ax.axhline(100, color="#bbb", lw=0.8); ax.axvline(2015.5, color="#999", lw=0.8, ls=":")
ax.text(1987, 118, "pre-1996: products diverge\n(early satellite record)", ha="center", fontsize=9.5, color="#555")
ax.set_ylim(60, 125); ax.set_xlim(1978.5, 2024.5)
ax.set_ylabel("Circumpolar winter (JJA) ice-drift speed\n(% of 1992–2015 mean)"); ax.set_xlabel("Year")
ax.set_title("Two independent drift products agree on variability since the mid-1990s — not on the long-term trend", loc="left", fontsize=11.5)
ax.legend(frameon=False, loc="lower right"); ax.grid(True, color="#eee", lw=0.6)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT, bbox_inches="tight"); print("saved", OUT)
