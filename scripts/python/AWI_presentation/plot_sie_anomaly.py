"""
Circumpolar sea ice extent anomaly, daily, 5-day running mean, filled above
and below zero (Labe-style). Colourblind-safe (Okabe-Ito). Transparent
background, no title, so it drops straight into a slide.

Usage:
    python plot_sie_anomaly.py                      # uses DEFAULT_CSV and SIE_circumpolar
    python plot_sie_anomaly.py --col SIE_Weddell    # any other column in the file
    python plot_sie_anomaly.py --baseline 1981-01-01,2010-12-31
    python plot_sie_anomaly.py --out anomaly --width 12 --height 4

Input: the daily SIE CSV with a `time` column and one column per region.
"""

import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Okabe-Ito
BLUE = "#0072B2"        # positive anomaly
VERMILLION = "#D55E00"  # negative anomaly
INK = "#222222"         # axes and text; set to "#EEEEEE" for a dark slide

FONT = None             # e.g. "Arial"; None keeps the matplotlib default
BASELINE_DEFAULT = ("1979-01-01", "2015-12-31")
STEP_YEAR = None        # e.g. 2016 for a dashed marker; None to omit
LABEL_LAST = True       # label the last year at the end of the line


DEFAULT_CSV = ("/Users/fridaperez/Research/repos/sea-ice-phase/data/raw/"
               "SIE_daily_sector_and_circumpolar_million_km2.csv")
DATE_COL = "time"
VALUE_COL = "SIE_circumpolar"


def load(path, col=VALUE_COL):
    df = pd.read_csv(path, parse_dates=[DATE_COL]).sort_values(DATE_COL).set_index(DATE_COL)
    s = df[col].astype(float)
    # SMMR era is every other day: fill gaps of up to 2 days
    return s.asfreq("D").interpolate(limit=2)


def anomaly(s, baseline):
    """Daily anomaly from a 31-day-smoothed day-of-year climatology over `baseline`."""
    base = s.loc[baseline[0]:baseline[1]]
    clim = base.groupby(base.index.dayofyear).mean()
    clim = clim.reindex(range(1, 367)).interpolate(limit_direction="both")
    pad = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]])
    clim_s = pad.rolling(31, center=True).mean().iloc[15:-15]
    clim_s.index = range(1, 367)
    return s - clim_s.reindex(s.index.dayofyear).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=DEFAULT_CSV)
    ap.add_argument("--col", default=VALUE_COL, help="column to plot, e.g. a sector")
    ap.add_argument("--baseline", default="early",
                    help="'early' = 1979-2015 (default), 'full', or 'YYYY-MM-DD,YYYY-MM-DD'")
    ap.add_argument("--window", type=int, default=5, help="running-mean window, days")
    ap.add_argument("--out", default="anomaly")
    ap.add_argument("--width", type=float, default=12.0)
    ap.add_argument("--height", type=float, default=4.0)
    args = ap.parse_args()

    s = load(args.csv, args.col)
    if args.baseline == "early":
        baseline = BASELINE_DEFAULT
    elif args.baseline == "full":
        baseline = (str(s.index.min().date()), str(s.index.max().date()))
    else:
        baseline = tuple(args.baseline.split(","))

    a = anomaly(s, baseline).rolling(args.window, center=True, min_periods=1).mean()

    if FONT:
        matplotlib.rcParams["font.family"] = FONT
    matplotlib.rcParams.update({
        "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.labelcolor": INK, "text.color": INK, "font.size": 14,
    })

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    x, y = a.index, a.values
    ax.fill_between(x, 0, y, where=y >= 0, color=BLUE, alpha=0.85, lw=0, interpolate=True)
    ax.fill_between(x, 0, y, where=y < 0, color=VERMILLION, alpha=0.85, lw=0, interpolate=True)
    ax.plot(x, y, color=INK, lw=0.4, alpha=0.6)
    ax.axhline(0, color=INK, lw=1.0)

    if STEP_YEAR is not None:
        ax.axvline(pd.Timestamp(f"{STEP_YEAR}-01-01"), color=INK, lw=1.2, ls=(0, (4, 3)), alpha=0.8)

    if LABEL_LAST:
        last = a.dropna().index[-1]
        ax.annotate(str(last.year), (last, a.loc[last]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=12,
                    color=VERMILLION if a.loc[last] < 0 else BLUE, fontweight="bold")

    ax.set_ylabel("Extent anomaly (10$^6$ km$^2$)")
    ax.set_xlim(pd.Timestamp("1979-01-01"), s.index.max() + pd.Timedelta(days=200))
    ymax = max(abs(a.min()), abs(a.max())) * 1.05
    ax.set_ylim(-ymax, ymax)
    ticks = pd.date_range("1980-01-01", s.index.max(), freq="5YS")
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t.year) for t in ticks])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=4)
    fig.tight_layout()

    fig.savefig(f"{args.out}.svg", transparent=True)
    fig.savefig(f"{args.out}.png", dpi=300, transparent=True)
    print(f"wrote {args.out}.svg/.png  baseline {baseline[0]}..{baseline[1]}  window {args.window} d")


if __name__ == "__main__":
    main()
