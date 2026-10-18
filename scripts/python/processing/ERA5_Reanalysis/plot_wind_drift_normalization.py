# =====================================================
# Star plot: wind vs drift % change (pre-2016 -> post-2016)
# Each point = one sector-season.
# Points ABOVE the 1:1 line = drift increased more than wind
#                            = free-drift signature.
# =====================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =====================================================
# PATHS + CONFIG
# =====================================================
WIND_CSV  = "/user/geog/falejandraperez/sea-ice-phase/results/ERA5/ERA5_windSpeed_daily_sector_v2.csv"
DRIFT_CSV = "/user/geog/falejandraperez/sea-ice-phase/results/NSIDC0116/NSIDC0116_driftSpeed_daily_sector.csv"
OUT_DIR   = "/user/geog/falejandraperez/sea-ice-phase/results/wind_drift_norm"
PERIOD_SPLIT = 2016  # years >= this are "post"

# Must match SECTORS dict in compute_ERA5_winds_daily_sector.py
SECTORS = {
    1: "Weddell",
    2: "Amundsen_Bellingshausen",
    3: "Ross",
    4: "East_Antarctica",
    5: "King_Haakon",
}
# Nice display labels for the legend
SECTOR_LABELS = {
    1: "Weddell",
    2: "Amundsen-Bellingshausen",
    3: "Ross",
    4: "East Antarctica",
    5: "King Haakon VII",
}

# Okabe-Ito, colorblind-safe
SECTOR_COLORS = {
    1: "#E69F00",  # orange
    2: "#56B4E9",  # sky blue
    3: "#009E73",  # bluish green
    4: "#D55E00",  # vermillion
    5: "#CC79A7",  # reddish purple
}

SEASON_ORDER   = ["DJF", "MAM", "JJA", "SON"]
SEASON_MARKERS = {"DJF": "o", "MAM": "s", "JJA": "^", "SON": "D"}

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# LOADERS
# =====================================================
def season_from_month(m):
    if m in (12, 1, 2):   return "DJF"
    if m in (3, 4, 5):    return "MAM"
    if m in (6, 7, 8):    return "JJA"
    return "SON"

def load_wind(path):
    """Long format v2: date, sector, wind_speed_mps."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"]   = df["date"].dt.year
    df["month"]  = df["date"].dt.month
    df["season"] = df["month"].map(season_from_month)
    df["period"] = np.where(df["year"] >= PERIOD_SPLIT, "post", "pre")
    return df.rename(columns={"wind_speed_mps": "value"})[
        ["sector", "season", "period", "value"]
    ].dropna()

def load_drift(path):
    """Long format already: date, sector, drift_speed_mps."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"]   = df["date"].dt.year
    df["month"]  = df["date"].dt.month
    df["season"] = df["month"].map(season_from_month)
    df["period"] = np.where(df["year"] >= PERIOD_SPLIT, "post", "pre")
    return df.rename(columns={"drift_speed_mps": "value"})[
        ["sector", "season", "period", "value"]
    ].dropna()

wind  = load_wind(WIND_CSV)
drift = load_drift(DRIFT_CSV)

# Sanity: positive wind values
if (wind["value"] < 0).any():
    neg = (wind["value"] < 0).sum()
    print(f"WARNING: {neg} negative wind values found — did you re-run the wind script?")

# =====================================================
# PRE/POST SEASONAL MEANS -> % CHANGE
# =====================================================
def pct_change(df, label):
    means = (df.groupby(["sector", "season", "period"])["value"]
               .mean().unstack("period"))
    out = (100.0 * (means["post"] - means["pre"]) / means["pre"]).rename(f"{label}_pct")
    return out

wind_pct  = pct_change(wind,  "wind")
drift_pct = pct_change(drift, "drift")

both = pd.concat([wind_pct, drift_pct], axis=1).dropna().reset_index()
print("\nPercent change by sector × season:")
print(both.to_string(index=False))
both.to_csv(os.path.join(OUT_DIR, "pct_change_by_sector_season.csv"), index=False)

# =====================================================
# PLOT
# =====================================================
fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=140)

lim = max(20.0, np.nanmax(np.abs(both[["wind_pct", "drift_pct"]].values)) * 1.1)

# Shade free-drift region (drift > wind, above diagonal)
ax.fill_between([-lim, lim], [-lim, lim], [lim, lim],
                color="#F5EEF8", zorder=0)
ax.text(0.5 * lim, 0.95 * lim, "drift > wind\n(free-drift signature)",
        ha="center", va="top", fontsize=9, color="#5B2C6F", alpha=0.9)
ax.text(0.95 * lim, -0.5 * lim, "wind > drift",
        ha="right", va="bottom", fontsize=9, color="#555", alpha=0.9)

# 1:1 line and zero axes
ax.plot([-lim, lim], [-lim, lim], color="#999", lw=1.2, ls="--", zorder=1)
ax.axhline(0, color="#ccc", lw=0.8, zorder=0)
ax.axvline(0, color="#ccc", lw=0.8, zorder=0)

# Points: color = sector, marker = season
for _, r in both.iterrows():
    sid = int(r["sector"])
    ax.scatter(
        r["wind_pct"], r["drift_pct"],
        s=120,
        c=SECTOR_COLORS.get(sid, "#333"),
        marker=SEASON_MARKERS.get(r["season"], "o"),
        edgecolors="white", linewidths=1.5,
        zorder=3,
    )

# Legends
sector_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=SECTOR_COLORS[sid], markersize=10,
           label=SECTOR_LABELS[sid])
    for sid in sorted(SECTOR_COLORS) if sid in both["sector"].unique()
]
season_handles = [
    Line2D([0], [0], marker=SEASON_MARKERS[s], color="w",
           markerfacecolor="#555", markersize=10, label=s)
    for s in SEASON_ORDER if s in both["season"].unique()
]

leg1 = ax.legend(handles=sector_handles, title="Sector",
                 loc="upper left", frameon=False, fontsize=9, title_fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=season_handles, title="Season",
          loc="upper right", frameon=False, fontsize=9, title_fontsize=9)

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("Wind speed change post-2016 (%)", fontsize=11)
ax.set_ylabel("Drift speed change post-2016 (%)", fontsize=11)
ax.set_title("Did ice drift respond more than wind did?", fontsize=12, pad=12)
ax.set_aspect("equal")
ax.grid(True, color="#eee", lw=0.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#999")
ax.spines["bottom"].set_color("#999")

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "wind_vs_drift_pct_change.png")
out_pdf = os.path.join(OUT_DIR, "wind_vs_drift_pct_change.pdf")
plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
