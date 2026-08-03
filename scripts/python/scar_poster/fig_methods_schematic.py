"""
fig_methods_schematic.py

Methods figure for the poster: real data from one sector-season (ABS-JJA)
showing the wind-vs-ΔSIA scatter pre and post-2016 with fitted slopes.
The visual point: the two slopes are indistinguishable. β didn't shift.
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

SECTOR = "Amundsen-Bellingshausen"
SEASON = "SON"

OUT = "fig_methods_schematic.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def main():
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df = df[~df.date.dt.year.isin(EXCLUDE_YEARS)]
    df["post"] = (df.date.dt.year >= SPLIT_YEAR).astype(int)

    # assign season
    month_to_season = {12: "DJF", 1: "DJF", 2: "DJF",
                       3: "MAM", 4: "MAM", 5: "MAM",
                       6: "JJA", 7: "JJA", 8: "JJA",
                       9: "SON", 10: "SON", 11: "SON"}
    df["season"] = df.date.dt.month.map(month_to_season)

    sub = df[(df.sector == SECTOR) & (df.season == SEASON)].dropna(
        subset=["wind_stress_anomaly", "delta_SIA_anomaly"])

    pre = sub[sub.post == 0]
    post = sub[sub.post == 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax, data, label, color in [
        (axes[0], pre, f"pre-{SPLIT_YEAR}  (n={len(pre):,})", "steelblue"),
        (axes[1], post, f"post-{SPLIT_YEAR}  (n={len(post):,})", "coral"),
    ]:
        x = data["wind_stress_anomaly"].values
        y = data["delta_SIA_anomaly"].values

        ax.scatter(x, y, s=3, alpha=0.15, color=color, rasterized=True)

        # fit and plot slope
        mask = np.isfinite(x) & np.isfinite(y)
        coef = np.polyfit(x[mask], y[mask], 1)
        xline = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 100)
        ax.plot(xline, np.polyval(coef, xline), color="black", lw=2.5,
                label=f"β = {coef[0]:.0f}")

        ax.set_xlabel("wind stress anomaly (N/m²)", fontsize=12)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=12, loc="upper left")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")

    axes[0].set_ylabel("ΔSIA anomaly (km²)", fontsize=12)

    fig.suptitle(f"{SECTOR}, {SEASON}\n"
                 f"Did the slope change?  →  No.",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    # print slopes for reference
    for label, data in [("pre", pre), ("post", post)]:
        x = data["wind_stress_anomaly"].values
        y = data["delta_SIA_anomaly"].values
        mask = np.isfinite(x) & np.isfinite(y)
        coef = np.polyfit(x[mask], y[mask], 1)
        print(f"  {label}: β = {coef[0]:.1f} km²/(N/m²)")

    # upload to Google Drive
    print(f"\nUploading to {RCLONE_REMOTE}...")
    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("done.")


if __name__ == "__main__":
    main()