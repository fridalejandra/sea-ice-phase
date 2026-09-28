"""
fig_variance_decomposition.py

Poster figure: wind's share of daily area-change variance, pre vs post-2016,
by sector. Sector colors match fig_sector_map_poster.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.formula.api as smf

IN_CSV = ("/user/geog/falejandraperez/sea-ice-phase/data/merged/"
          "analysis_table_daily_anomaly_periodclim.csv")
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
SPLIT_YEAR = 2016

SECTORS = ["Amundsen-Bellingshausen", "Weddell", "King Haakon VII",
           "East Antarctica", "Ross-Amundsen"]
SHORT = {"Amundsen-Bellingshausen": "ABS", "Weddell": "WED",
         "King Haakon VII": "KHV", "East Antarctica": "EA",
         "Ross-Amundsen": "RA"}

# match fig_sector_map_poster.py
SECTOR_COLORS = {
    "Amundsen-Bellingshausen": "#2196F3",
    "Weddell":                 "#F44336",
    "King Haakon VII":         "#FFC107",
    "East Antarctica":         "#FF9800",
    "Ross-Amundsen":           "#4CAF50",
}

OUT = "fig_variance_decomposition.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def lighten(hex_color, factor=0.45):
    """Return a lighter version of a hex color for the post-2016 bars."""
    rgb = mcolors.to_rgb(hex_color)
    return tuple(c + (1 - c) * factor for c in rgb)


def main():
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df = df[~df.date.dt.year.isin(EXCLUDE_YEARS)]
    df["post"] = (df.date.dt.year >= SPLIT_YEAR).astype(int)

    rows = []
    for s in SECTORS:
        for p, lab in [(0, "pre"), (1, "post")]:
            sub = df[(df.sector == s) & (df.post == p)].dropna(
                subset=["delta_SIA_anomaly", "wind_stress_anomaly"])
            m = smf.ols("delta_SIA_anomaly ~ wind_stress_anomaly", data=sub).fit()
            var_y = sub.delta_SIA_anomaly.var()
            var_r = m.resid.var()
            var_w = sub.wind_stress_anomaly.var()
            wind_share = 1 - var_r / var_y
            rows.append({
                "sector": s, "short": SHORT[s], "period": lab,
                "var_dSIA": var_y, "var_wind": var_w,
                "wind_share": wind_share * 100,
                "resid_share": (var_r / var_y) * 100,
            })

    res = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    bar_h = 0.35

    for panel, (ax, col, xlabel, title) in enumerate(zip(axes,
        ["var_dSIA", "wind_share", "var_wind"],
        ["var(ΔSIA)  (×10⁹ km⁴)", "wind's share of var(ΔSIA)  (%)",
         "var(wind stress anom)  (×10⁻⁶)"],
        ["Total daily variability", "Wind's share",
         "Wind variance"])):

        for i, s in enumerate(SECTORS):
            pre = res[(res.sector == s) & (res.period == "pre")].iloc[0]
            post = res[(res.sector == s) & (res.period == "post")].iloc[0]
            color = SECTOR_COLORS[s]
            light = lighten(color)

            if col == "var_dSIA":
                v_pre, v_post = pre[col] / 1e9, post[col] / 1e9
            elif col == "var_wind":
                v_pre, v_post = pre[col] * 1e6, post[col] * 1e6
            else:
                v_pre, v_post = pre[col], post[col]

            ax.barh(i + bar_h/2, v_pre, height=bar_h, color=color,
                    edgecolor="none",
                    label="pre-2016" if i == 0 and panel == 0 else "")
            ax.barh(i - bar_h/2, v_post, height=bar_h, color=light,
                    edgecolor=color, linewidth=1.2,
                    label="post-2016" if i == 0 and panel == 0 else "")

            if col == "var_dSIA":
                pct = (post[col] - pre[col]) / pre[col] * 100
                ax.text(max(v_pre, v_post) + 0.05, i,
                        f"{pct:+.0f}%", va="center", fontsize=11,
                        fontweight="bold")
            elif col == "wind_share":
                diff = post[col] - pre[col]
                ax.text(max(v_pre, v_post) + 0.2, i,
                        f"{diff:+.1f}pp", va="center", fontsize=11)

        ax.set_yticks(range(len(SECTORS)))
        ax.set_yticklabels([SHORT[s] for s in SECTORS], fontsize=13)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

    axes[0].legend(fontsize=11, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")

    print(res[["short", "period", "var_dSIA", "wind_share",
               "resid_share"]].to_string(index=False))


if __name__ == "__main__":
    main()