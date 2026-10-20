#!/usr/bin/env python
"""
windfactor_analysis.py -- daily wind factor, turning angle and residual drift,
pre (1992-2015) vs post (2016-2020), on fixed cells, with year-block bootstrap.

Model, per sector-season, over daily sector-mean vectors (east/north frame):
    U_ice = A * U_wind + C          A, C complex;  alpha = |A|, theta = arg(A)
Internal-stress release raises alpha (toward free drift ~0.02) and moves theta;
an ocean-current change leaves A alone and changes C.

Input : results/ch4/tables/uv_daily_sector_fixedcells_1992-2024.csv
         (from compute_uv_daily_sector_fixedcells.py)
Output: results/ch4/tables/windfactor_prepost.csv
        results/ch4/figures/{manuscript,slides}/fig07_windfactor.*

    python windfactor_analysis.py            # table + figure, both modes
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_figstyle as fs  # noqa: E402

ROOT = fs.ROOT
CSV = f"{ROOT}/results/ch4/tables/uv_daily_sector_fixedcells_1992-2024.csv"
OUT_TAB = f"{ROOT}/results/ch4/tables/windfactor_prepost.csv"
BASE0, BASE1, POST0, POST1 = 1992, 2015, 2016, 2020
SEASONS = ["MAM", "JJA", "SON"]
N_BOOT, SEED = 2000, 42
FREE_DRIFT_ALPHA = 0.02           # Thorndike & Colony (1982) reference


def fit(ui, uw):
    """Complex least squares U_ice = A*U_wind + C. Returns A, C, R^2."""
    X = np.column_stack([uw, np.ones_like(uw)])
    coef, *_ = np.linalg.lstsq(X, ui, rcond=None)
    A, C = coef
    resid = ui - X @ coef
    r2 = 1 - np.sum(np.abs(resid) ** 2) / np.sum(np.abs(ui - ui.mean()) ** 2)
    return A, C, float(r2)


def summarise(A, C):
    return dict(alpha=np.abs(A), theta=np.degrees(np.angle(A)), resid=np.abs(C), c=C)


def table():
    df = pd.read_csv(CSV, parse_dates=["date"]).dropna()
    df["sector"] = df["sector"].map(fs.SECTOR_ID)
    df["year"] = df.date.dt.year + (df.date.dt.month == 12).astype(int)
    df["ui"] = df.ui_e + 1j * df.ui_n
    df["uw"] = df.uw_e + 1j * df.uw_n
    rng = np.random.default_rng(SEED)
    rows, dc_boots = [], {}
    for (sec, sea), g in df[df.season.isin(SEASONS)].groupby(["sector", "season"]):
        per = {"pre": g[(g.year >= BASE0) & (g.year <= BASE1)], "post": g[(g.year >= POST0) & (g.year <= POST1)]}
        if len(per["pre"].year.unique()) < 10 or len(per["post"].year.unique()) < 3:
            continue
        est, boots = {}, {}
        for p, gp in per.items():
            A, C, r2 = fit(gp.ui.values, gp.uw.values)
            est[p] = summarise(A, C) | {"r2": r2, "n_days": len(gp), "n_years": gp.year.nunique()}
            yrs = gp.year.unique()
            byyear = {y: gp[gp.year == y] for y in yrs}
            b = {k: np.empty(N_BOOT) for k in ("alpha", "theta", "resid")}
            b["c"] = np.empty(N_BOOT, dtype=complex)
            for i in range(N_BOOT):
                pick = rng.choice(yrs, len(yrs), replace=True)
                gb = pd.concat([byyear[y] for y in pick])
                A, C, _ = fit(gb.ui.values, gb.uw.values)
                s = summarise(A, C)
                for k in b:
                    b[k][i] = s[k]
            boots[p] = b
        row = dict(sector=sec, season=sea)
        for p in ("pre", "post"):
            for k in ("alpha", "theta", "resid"):
                lo, hi = np.percentile(boots[p][k], [2.5, 97.5])
                row[f"{k}_{p}"], row[f"{k}_{p}_lo"], row[f"{k}_{p}_hi"] = est[p][k], lo, hi
            row[f"r2_{p}"], row[f"ndays_{p}"] = est[p]["r2"], est[p]["n_days"]
        for k in ("alpha", "theta", "resid"):
            diff = boots["post"][k] - boots["pre"][k]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            row[f"{k}_diff"], row[f"{k}_diff_lo"], row[f"{k}_diff_hi"] = est["post"][k] - est["pre"][k], lo, hi
            row[f"{k}_sig"] = (lo > 0) or (hi < 0)
        # residual as a VECTOR: components, change vector, and its heading (deg, 0 = east, 90 = north)
        dc = est["post"]["c"] - est["pre"]["c"]
        dcb = boots["post"]["c"] - boots["pre"]["c"]
        row.update(c_e_pre=est["pre"]["c"].real, c_n_pre=est["pre"]["c"].imag,
                   c_e_post=est["post"]["c"].real, c_n_post=est["post"]["c"].imag,
                   dc_e=dc.real, dc_n=dc.imag, dc_mag=abs(dc), dc_dir=np.degrees(np.angle(dc)),
                   dc_mag_lo=np.percentile(np.abs(dcb), 2.5), dc_mag_hi=np.percentile(np.abs(dcb), 97.5),
                   # fraction of bootstrap change vectors within 45 deg of the point estimate's heading
                   dc_dir_consistency=float(np.mean(np.abs(np.angle(dcb * np.conj(dc))) < np.pi / 4)))
        dc_boots[f"{sec}_{sea}"] = dcb
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_TAB, index=False)
    np.savez(OUT_TAB.replace(".csv", "_dc_boots.npz"), **dc_boots)
    print(f"wrote {OUT_TAB}")
    return t


def fig07(t):
    panels = [("alpha", "Wind factor α\n|U$_{ice}$| / |U$_{wind}$|", FREE_DRIFT_ALPHA, "free drift"),
              ("theta", "Turning angle θ (°)\n+ = ice to left of wind", None, None),
              ("resid", "Residual drift |C|\n(m s$^{-1}$)", 0.0, None)]
    if fs.mode() == "manuscript":
        fig, axes = plt.subplots(1, 3, figsize=fs.size(1.0, 0.75), sharey=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=fs.size(1.0, 0.5), sharey=True)
    ypos, y = {}, 0.0
    for s in fs.SECTOR_ORDER:
        for se in SEASONS:
            ypos[(s, se)] = -y
            y += 1
        y += 0.7
    for ax, (k, label, ref, reflab) in zip(axes, panels):
        if ref is not None:
            ax.axvline(ref, color=fs.INK, lw=0.8, ls="--" if ref else "-", zorder=1)
            if reflab:
                ax.text(ref, 1.0, reflab, ha="center", va="bottom", color=fs.INK,
                        transform=ax.get_xaxis_transform(), fontsize=plt.rcParams["legend.fontsize"])
        for (s, se), yy in ypos.items():
            r = t[(t.sector == s) & (t.season == se)]
            if r.empty:
                continue
            r = r.iloc[0]
            c = fs.SECTOR_COLORS[s]
            a = 1.0 if r[f"{k}_sig"] else 0.45
            ax.plot([r[f"{k}_pre"], r[f"{k}_post"]], [yy, yy], color=c, lw=fs.lw(0.5), alpha=0.5 * a, zorder=2)
            for p, dy, hollow in (("pre", +0.18, True), ("post", -0.18, False)):
                ax.plot([r[f"{k}_{p}_lo"], r[f"{k}_{p}_hi"]], [yy + dy, yy + dy], color=c, lw=fs.lw(0.7),
                        alpha=a, solid_capstyle="round", zorder=2)
                ax.plot(r[f"{k}_{p}"], yy + dy, marker=fs.SEASON_MARKERS[se], ls="none",
                        markerfacecolor="none" if hollow else c, markeredgecolor=c,
                        markeredgewidth=fs.lw(0.9), ms=fs.ms(0.7), alpha=a, zorder=3)
            if ax is axes[0]:
                ax.text(-0.02, yy, se, ha="right", va="center", color=fs.INK, transform=ax.get_yaxis_transform(),
                        fontsize=plt.rcParams["ytick.labelsize"])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(label)
    for s in fs.SECTOR_ORDER:
        ys = [ypos[(s, se)] for se in SEASONS]
        axes[0].text(-0.12, np.mean(ys), fs.SECTOR_NAME[s], transform=axes[0].get_yaxis_transform(),
                     ha="right", va="center", color=fs.SECTOR_COLORS[s],
                     fontproperties=fs.st.bold_font_properties(plt.rcParams["axes.labelsize"]))
    h = [Line2D([0], [0], marker="o", ls="none", markerfacecolor="none", markeredgecolor="#666",
                markeredgewidth=fs.lw(0.9), ms=fs.ms(0.7), label=f"{BASE0}–{BASE1}"),
         Line2D([0], [0], marker="o", ls="none", color="#666", ms=fs.ms(0.7), label=f"{POST0}–{POST1}"),
         Line2D([0], [0], marker="o", ls="none", color="#666", ms=fs.ms(0.7), alpha=0.45,
                label="faded: post − pre CI spans 0")]
    axes[-1].legend(handles=h, loc="lower right")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=plt.rcParams["xtick.labelsize"] * 0.9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    fig.subplots_adjust(wspace=0.18)
    fs.save(fig, "fig07_windfactor")


if __name__ == "__main__":
    t = table() if not os.path.exists(OUT_TAB) else pd.read_csv(OUT_TAB)
    pd.set_option("display.width", 200)
    print(t[["sector", "season", "alpha_pre", "alpha_post", "alpha_diff_lo", "alpha_diff_hi", "alpha_sig",
             "theta_pre", "theta_post", "theta_sig", "resid_pre", "resid_post", "resid_diff_lo", "resid_diff_hi",
             "resid_sig", "r2_pre", "r2_post"]].round(3).to_string(index=False))
    print("\nResidual change as a vector (dc_dir: 0 = east, 90 = north, -90 = south, +-180 = west):")
    print(t[["sector", "season", "c_e_pre", "c_n_pre", "c_e_post", "c_n_post", "dc_mag", "dc_mag_lo", "dc_mag_hi",
             "dc_dir", "dc_dir_consistency"]].round(3).to_string(index=False))
    for m in ("manuscript", "slide"):
        fs.set_mode(m)
        fig07(t)
