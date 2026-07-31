"""
fig_variance_decomposition.py

Poster figure: wind's share of daily area-change variance, pre vs post-2016,
by sector. The key finding: wind's share ROSE in every sector — not because
wind got stronger, but because the residual (non-wind) variability collapsed.

Shows three things per sector:
  - var(ΔSIA) pre vs post (total variability collapsed)
  - wind's share pre vs post (rose everywhere)
  - var(wind) pre vs post (unchanged — confirms the shift isn't forcing-driven)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUT = "fig_variance_decomposition.png"


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

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: var(ΔSIA) collapsed
    ax = axes[0]
    for i, s in enumerate(SECTORS):
        pre = res[(res.sector == s) & (res.period == "pre")].iloc[0]
        post = res[(res.sector == s) & (res.period == "post")].iloc[0]
        pct = (post.var_dSIA - pre.var_dSIA) / pre.var_dSIA * 100
        ax.barh(i - 0.15, pre.var_dSIA / 1e9, height=0.3, color="steelblue",
                label="pre-2016" if i == 0 else "")
        ax.barh(i + 0.15, post.var_dSIA / 1e9, height=0.3, color="coral",
                label="post-2016" if i == 0 else "")
        ax.text(max(pre.var_dSIA, post.var_dSIA) / 1e9 + 0.05,
                i, f"{pct:+.0f}%", va="center", fontsize=9)
    ax.set_yticks(range(len(SECTORS)))
    ax.set_yticklabels([SHORT[s] for s in SECTORS])
    ax.set_xlabel("var(ΔSIA)  (×10⁹ km⁴)")
    ax.set_title("Total daily variability\ncollapsed 30–70%")
    ax.legend(fontsize=9)

    # Panel 2: wind's share rose
    ax = axes[1]
    for i, s in enumerate(SECTORS):
        pre = res[(res.sector == s) & (res.period == "pre")].iloc[0]
        post = res[(res.sector == s) & (res.period == "post")].iloc[0]
        ax.barh(i - 0.15, pre.wind_share, height=0.3, color="steelblue")
        ax.barh(i + 0.15, post.wind_share, height=0.3, color="coral")
        ax.text(max(pre.wind_share, post.wind_share) + 0.2,
                i, f"{post.wind_share - pre.wind_share:+.1f}pp", va="center",
                fontsize=9)
    ax.set_yticks(range(len(SECTORS)))
    ax.set_yticklabels([SHORT[s] for s in SECTORS])
    ax.set_xlabel("wind's share of var(ΔSIA)  (%)")
    ax.set_title("Wind's share rose\n(not because wind got stronger)")

    # Panel 3: var(wind) unchanged
    ax = axes[2]
    for i, s in enumerate(SECTORS):
        pre = res[(res.sector == s) & (res.period == "pre")].iloc[0]
        post = res[(res.sector == s) & (res.period == "post")].iloc[0]
        ax.barh(i - 0.15, pre.var_wind * 1e6, height=0.3, color="steelblue")
        ax.barh(i + 0.15, post.var_wind * 1e6, height=0.3, color="coral")
    ax.set_yticks(range(len(SECTORS)))
    ax.set_yticklabels([SHORT[s] for s in SECTORS])
    ax.set_xlabel("var(wind stress anom)  (×10⁻⁶)")
    ax.set_title("Wind variance\nunchanged")

    fig.suptitle("The atmosphere pushed steadily — the system got quieter",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    # print the table too
    print(res[["short", "period", "var_dSIA", "wind_share",
               "resid_share"]].to_string(index=False))


if __name__ == "__main__":
    main()