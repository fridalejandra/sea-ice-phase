"""
fig_timeseries_poster.py

Poster figure 1: SIA anomaly + wind stress anomaly time series.
Faint sector lines, bold circumpolar total. Sector colors match
the sector map (fig_sector_map_poster.py).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = ("/user/geog/falejandraperez/sea-ice-phase/data/merged/"
          "analysis_table_daily_anomaly_periodclim.csv")
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
SPLIT_YEAR = 2016
SMOOTH_DAYS = 90

# match fig_sector_map_poster.py colors exactly
SECTOR_COLORS = {
    "Weddell":                  "#F44336",
    "King Haakon VII":          "#FFC107",
    "East Antarctica":          "#FF9800",
    "Ross-Amundsen":            "#4CAF50",
    "Amundsen-Bellingshausen":  "#2196F3",
}
SECTOR_ORDER = ["Weddell", "King Haakon VII", "East Antarctica",
                "Ross-Amundsen", "Amundsen-Bellingshausen"]
SECTOR_ABBREV = {
    "Weddell": "WS", "King Haakon VII": "KH", "East Antarctica": "EA",
    "Ross-Amundsen": "RS", "Amundsen-Bellingshausen": "ABS",
}

OUT = "fig_timeseries_poster.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def smooth(series, window=SMOOTH_DAYS):
    return series.rolling(window, center=True, min_periods=window // 2).mean()


def main():
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df = df[~df.date.dt.year.isin(EXCLUDE_YEARS)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for row, (var, ylabel, title) in enumerate([
        ("SIA_anomaly", "SIA anomaly (10⁴ km²)", "Sea ice area anomaly by sector"),
        ("wind_stress_anomaly", "Wind stress anomaly (N/m²)", "Wind stress anomaly by sector"),
    ]):
        ax = axes[row]

        # faint sector lines
        for sector in SECTOR_ORDER:
            sub = df[df.sector == sector].sort_values("date")
            ax.plot(sub.date, smooth(sub[var]),
                    color=SECTOR_COLORS[sector], alpha=0.35, lw=0.8,
                    label=SECTOR_ABBREV[sector])

        # bold circumpolar total
        circ = df.groupby("date")[var].sum().reset_index()
        circ = circ.sort_values("date")
        ax.plot(circ.date, smooth(circ[var]),
                color="black", lw=2.0, alpha=0.9, label="Total", zorder=5)

        # 2016 line
        ax.axvline(pd.Timestamp(f"{SPLIT_YEAR}-01-01"), color="black",
                   ls="--", lw=1.2, alpha=0.7, zorder=4)

        # shade post-2016
        ax.axvspan(pd.Timestamp(f"{SPLIT_YEAR}-01-01"), df.date.max(),
                   color="gray", alpha=0.06, zorder=0)

        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.axhline(0, color="gray", lw=0.4, ls="-")

        if row == 0:
            # legend: sectors + total, horizontal, above the panel
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=9,
                      frameon=False)

    axes[1].set_xlabel("Year", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    # upload
    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()