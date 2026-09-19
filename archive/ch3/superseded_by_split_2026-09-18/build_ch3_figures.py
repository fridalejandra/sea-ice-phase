#!/usr/bin/env python3
"""
build_ch3_figures.py — Figures 6 and 7, built directly from the canonical
pipeline's own output tables (not hand-transcribed numbers).

Run this from the same directory as ch3_config.py (e.g.
scripts/python/plotting/Ch3/figures/), AFTER the canonical pipeline has been
run in this order:
    01_fit_apac.R  ->  compute_atmospheric_correlations.py  ->  ch3_stats.py

Reads:
    results/ch3/tables/t33_phase_amp_splits.csv   (Fig 6)
    data/ch3/annual_params.csv                    (Fig 7)

Writes:
    results/ch3/figures/fig6_coupling_pooled.png
    results/ch3/figures/fig7_abs_growth_season.png

If a column name below doesn't match your actual table (schemas can drift),
this will raise a KeyError that prints the real column list — paste that
back and the fix is a one-line rename, not a rewrite.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import (
    ANNUAL_CSV, TABLES_DIR, OUTPUT_DIR, SECTORS, NUMBERS_CSV,
    SECTOR_ORDER_BY_LONGITUDE, SECTOR_LABELS,
)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find a column for '{label}'. Tried: {candidates}\n"
        f"Actual columns in this table: {list(df.columns)}"
    )


def load_table(name):
    path = os.path.join(TABLES_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run the canonical pipeline first.")
    return pd.read_csv(path)


# ============================================================ Fig 6: coupling
splits = load_table("t33_phase_amp_splits.csv")
row_p = splits[(splits["split"] == 2016) & (splits["method"] == "pearson")].iloc[0]
row_s = splits[(splits["split"] == 2016) & (splits["method"] == "spearman")].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# --- left panel: pooled forest plot ---
ax = axes[0]
forest_rows = [
    ("Pooled, pre-2016\n(1979-2015, Pearson)", row_p["r_pre"], row_p["lo_pre"], row_p["hi_pre"]),
    ("Pooled, 2016+\n(2016-2023, Pearson)", row_p["r_post"], row_p["lo_post"], row_p["hi_post"]),
]
y = np.arange(len(forest_rows))[::-1]
for i, (lab, r, lo, hi) in enumerate(forest_rows):
    ax.errorbar(r, y[i], xerr=[[r - lo], [hi - r]], fmt="o", ms=9,
                color="#C0392B" if i == 1 else "#7f8c8d", capsize=4, lw=2, zorder=3)
ax.axvline(0, color="k", lw=0.8, ls="--", zorder=1)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in forest_rows])
ax.set_xlim(-0.3, 0.8)
ax.set_xlabel("Pooled r (day of max, amplitude)")
ax.set_title(
    f"(a) Pooled coupling, split at 2016\n"
    f"p={row_p['p_post']:.3f} (post); shift p={row_p['p_shift']:.3f} (Pearson), "
    f"{row_s['p_shift']:.3f} (Spearman)", fontsize=9.5)

# --- right panel: per-sector 2016-2023 dot plot ---
ax = axes[1]
sector_cols = [c for c in splits.columns if c.startswith("post_")]
sector_names = [c.replace("post_", "") for c in sector_cols]
r2016 = [row_p[c] for c in sector_cols]
order = np.argsort(r2016)[::-1]
sector_names = [sector_names[i] for i in order]
r2016 = [r2016[i] for i in order]
colors = ["#2E7D32" if r > 0 else "#C0392B" for r in r2016]
y = np.arange(len(sector_names))[::-1]
ax.barh(y, r2016, color=colors, height=0.6)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(sector_names, fontsize=9)
pad = 0.15
ax.set_xlim(min(r2016) - pad, max(r2016) + pad)
ax.set_xlabel("r (day of max, amplitude), 2016-2023, n=8")
ax.set_title("(b) Per-sector coupling, 2016-2023\n(check n=8 significance before calling any one sector significant)", fontsize=9.5)
for i, r in enumerate(r2016):
    ax.text(r + (0.02 if r >= 0 else -0.02), y[i], f"{r:+.2f}",
            va="center", ha="left" if r >= 0 else "right", fontsize=8.5)

fig.suptitle("Timing-amplitude coupling before and after 2016", fontweight="bold", y=1.02)
fig.tight_layout()
out6 = os.path.join(OUTPUT_DIR, "fig6_coupling_pooled.png")
fig.savefig(out6, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out6}")
print(f"  pooled pre-2016 r={row_p['r_pre']:.3f} (p={row_p['p_pre']:.3f}) | "
      f"2016+ r={row_p['r_post']:.3f} (p={row_p['p_post']:.4f}) | shift p={row_p['p_shift']:.3f}")

# ============================================================ Fig 7: ABS growth season
ann = pd.read_csv(ANNUAL_CSV)
abs_sector = [s for s in SECTORS if "Amundsen" in s][0]
year_col = pick_col(ann, ["Year", "year"], "year")
sector_col = pick_col(ann, ["sector", "Sector"], "sector")

ann_abs = ann[(ann[sector_col] == abs_sector) & (ann[year_col].between(2016, 2023))].sort_values(year_col)
if len(ann_abs) == 0:
    raise ValueError(
        f"No rows matched sector=={abs_sector!r} in {ANNUAL_CSV}. "
        f"Actual sector values present: {sorted(ann[sector_col].unique())}"
    )

amp_col = pick_col(ann_abs, ["amplitude_raw_yr", "amplitude_raw", "amplitude", "amplitude_raw_anom"], "ABS amplitude")
max_col = pick_col(ann_abs, ["max_doy_raw", "max_doy", "max_doy_raw_anom"], "ABS day of max")
min_col = pick_col(ann_abs, ["min_doy_raw", "min_doy", "min_doy_raw_anom"], "ABS day of min")
using_anom = amp_col.endswith("_anom")

years = ann_abs[year_col].tolist()
growth_len = (ann_abs[max_col] - ann_abs[min_col]).tolist()
amp = ann_abs[amp_col].tolist()
r, p = pearsonr(growth_len, amp)

fig, ax = plt.subplots(figsize=(5.6, 4.6))
sc = ax.scatter(growth_len, amp, c=years, cmap="viridis", s=90, edgecolor="k", linewidth=0.6, zorder=3)
for x, yv, yr in zip(growth_len, amp, years):
    ax.annotate(str(int(yr)), (x, yv), textcoords="offset points", xytext=(6, 4), fontsize=8)
m, b = np.polyfit(growth_len, amp, 1)
xx = np.linspace(min(growth_len) - 5, max(growth_len) + 5, 50)
ax.plot(xx, m * xx + b, color="#C0392B", lw=1.6, ls="--", zorder=2,
        label=f"r = {r:+.2f} (2016-2023, this run, p={p:.3f})")
ax.set_xlabel("Growth-season length, day of max - day of min (days)"
              + (" [anomaly]" if using_anom else ""))
ax.set_ylabel("Amplitude" + (" anomaly" if using_anom else "") + " (10⁶ km²)")
ax.set_title("Amundsen-Bellingshausen: amplitude vs. growth-season length\n"
              "(2016-2023; compare to the pre-2016 r=+0.09 reported in the text)", fontsize=9.5)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
fig.colorbar(sc, ax=ax, label="Year")
fig.tight_layout()
out7 = os.path.join(OUTPUT_DIR, "fig7_abs_growth_season.png")
fig.savefig(out7, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out7}")
print(f"  r={r:.3f}, p={p:.4f}, using {'ANOMALY (raw columns not found)' if using_anom else 'raw'} columns")

# ============================================================ Fig 9: Ross-ASL non-stationarity
detail = load_table("t36_ross_asl_detail.csv")
seasons_order = ["annual", "DJF", "MAM", "JJA", "SON", "ADV", "RET"]
season_rows = detail[detail["test"] == "season"].set_index("key").reindex(seasons_order)

sweep = detail[detail["test"] == "split_sweep"].copy()
sweep["key"] = sweep["key"].astype(int)
sweep = sweep.sort_values("key")

loo_row = detail[detail["test"] == "loo_shift"].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

ax = axes[0]
x = np.arange(len(seasons_order)); w = 0.36
ax.bar(x - w / 2, season_rows["r_pre"], w, label="1979-2000", color="#2c7fb8")
ax.bar(x + w / 2, season_rows["r_post"], w, label="2001-2023", color="#e8703a")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(seasons_order)
ax.set_ylabel("r (ASL relative central pressure, Ross amplitude)")
ax.set_title(
    f"(a) By season (annual shift p={season_rows.loc['annual', 'p_shift']:.1g})", fontsize=9.5)
ax.legend(fontsize=8.5, frameon=False)

ax = axes[1]
ax.plot(sweep["key"], sweep["r_pre"], "o-", color="#2c7fb8", label="r before boundary", lw=2)
ax.plot(sweep["key"], sweep["r_post"], "o-", color="#e8703a", label="r after boundary", lw=2)
for xv, yv, pv in zip(sweep["key"], sweep["r_post"], sweep["p_shift"]):
    ax.annotate(f"p={pv:.2g}", (xv, yv), textcoords="offset points", xytext=(0, -14),
                fontsize=7.5, ha="center", color="#555")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Candidate breakpoint year")
ax.set_ylabel("r (ASL annual, Ross amplitude)")
sig_years = sweep.loc[sweep["p_shift"] < 0.05, "key"].tolist()
ax.set_title(f"(b) Split-year sweep—shift p<0.05 for boundaries {sig_years}", fontsize=9.5)
ax.legend(fontsize=8.5, frameon=False)

fig.suptitle("Amundsen Sea Low – Ross Sea amplitude correlation, by season and split-year boundary",
              fontweight="bold", y=1.03)
fig.tight_layout()
out9 = os.path.join(OUTPUT_DIR, "fig9_ross_asl_nonstationarity.png")
fig.savefig(out9, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out9}")
print(f"  annual: r_pre={season_rows.loc['annual','r_pre']:.3f}, "
      f"r_post={season_rows.loc['annual','r_post']:.3f}, "
      f"shift p={season_rows.loc['annual','p_shift']:.2g}")
print(f"  split-year sweep, shift p by boundary: "
      + ", ".join(f"{int(k)}: {p:.2g}" for k, p in zip(sweep['key'], sweep['p_shift'])))
print(f"  LOO worst shift p: {loo_row['p_shift']:.2g}")

# ============================================================ Fig 10: seven pre-specified pairs
pairs = load_table("t35_primary_pairs.csv")

TARGET_LABEL = {"amplitude_raw_anom": "amplitude", "max_doy_raw_anom": "max-day"}


def family_of(index_col):
    for prefix, fam in [("Nino34", "ENSO"), ("SAM", "SAM"), ("ASL", "ASL"), ("ZW3", "ZW3")]:
        if index_col.startswith(prefix):
            return fam
    return "?"


pairs = pairs.copy()
pairs["label"] = pairs["sector"] + " · " + pairs["target"].map(TARGET_LABEL) + " ~ " + pairs["index"]
pairs["family"] = pairs["index"].map(family_of)
pairs = pairs.reindex(pairs["r"].abs().sort_values().index)

fig, ax = plt.subplots(figsize=(8, 4.8))
fam_colors = {"ENSO": "#1f77b4", "SAM": "#d62728", "ASL": "#2ca02c", "ZW3": "#9467bd"}
y = np.arange(len(pairs))[::-1]
for i, (_, row) in enumerate(pairs.iterrows()):
    ax.barh(y[i], row["r"], color=fam_colors.get(row["family"], "#888"), height=0.6)
    ax.text(row["r"] + (0.015 if row["r"] >= 0 else -0.015), y[i],
            f"r={row['r']:+.2f}, p_Bonf={row['p_bonf7']:.3f}",
            va="center", ha="left" if row["r"] >= 0 else "right", fontsize=8)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(pairs["label"], fontsize=9)
lim = max(0.75, pairs["r"].abs().max() + 0.2)
ax.set_xlim(-lim, lim)
ax.set_xlabel("Full-record correlation (1979-2023, detrended)")
ax.set_title("Seven pre-specified atmosphere-component relationships\n"
              "(Bonferroni over 7; all significant)", fontsize=9.5)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_colors.values()]
ax.legend(handles, fam_colors.keys(), loc="lower right", frameon=False, fontsize=8.5, ncol=4)
fig.tight_layout()
out10 = os.path.join(OUTPUT_DIR, "fig10_atmosphere_sevenpairs.png")
fig.savefig(out10, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out10}")
for _, row in pairs.iterrows():
    print(f"  {row['label']}: r={row['r']:+.3f}, p_bonf7={row['p_bonf7']:.4f}, LOO worst p={row['loo_worst_p']:.4f}")

# ============================================================ Fig 8: day-to-day volatility (3.4c)
numbers = pd.read_csv(NUMBERS_CSV)
vol = numbers[numbers["section"] == "3.4c"].copy()
vol["value"] = pd.to_numeric(vol["value"], errors="coerce")


def parse_ci(extra):
    m = re.search(r"\[([-\d.]+),\s*([-\d.]+)\]", str(extra))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


vol[["ci_lo", "ci_hi"]] = vol["extra"].apply(lambda e: pd.Series(parse_ci(e)))
vol["kind"] = np.where(vol["quantity"].str.contains(r"\(dSIE\)"), "raw dSIE",
                        np.where(vol["quantity"].str.contains("residual_apac"), "residual (APAC)", "?"))

sector_order = [SECTOR_LABELS[s] for s in SECTOR_ORDER_BY_LONGITUDE if s in SECTOR_LABELS] + ["Circumpolar"]
sector_order = [s for s in sector_order if s in vol["sector"].unique()]

fig, ax = plt.subplots(figsize=(7.2, 4.8))
y_raw = np.arange(len(sector_order))[::-1] * 2 + 0.32
y_res = np.arange(len(sector_order))[::-1] * 2 - 0.32

for sector, yr, yres in zip(sector_order, y_raw, y_res):
    r_raw = vol[(vol["sector"] == sector) & (vol["kind"] == "raw dSIE")].iloc[0]
    r_res = vol[(vol["sector"] == sector) & (vol["kind"] == "residual (APAC)")].iloc[0]
    ax.errorbar(r_raw["value"], yr, xerr=[[r_raw["value"] - r_raw["ci_lo"]], [r_raw["ci_hi"] - r_raw["value"]]],
                fmt="o", ms=8, color="#C0392B", capsize=4, lw=1.8, zorder=3)
    ax.errorbar(r_res["value"], yres, xerr=[[r_res["value"] - r_res["ci_lo"]], [r_res["ci_hi"] - r_res["value"]]],
                fmt="o", ms=8, color="#7f8c8d", capsize=4, lw=1.8, zorder=3)

ax.axvline(1, color="k", lw=1.0, ls="--", zorder=1, label="no change (ratio = 1)")
yticks = np.arange(len(sector_order))[::-1] * 2
ax.set_yticks(yticks)
ax.set_yticklabels(sector_order)
ax.set_xlabel("Day-to-day volatility ratio, 2016+ / pre-2016 (bootstrap 95% CI)")
ax.set_title(
    "Day-to-day SIE volatility, 2016+ vs. pre-2016\n"
    "(red = raw dSIE, gray = APAC residual — season+sensor fixed, n_boot=100)",
    fontsize=9.5)
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#C0392B", markersize=8, label="raw dSIE"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f8c8d", markersize=8, label="APAC residual"),
    plt.Line2D([0], [0], color="k", lw=1.0, ls="--", label="no change (ratio = 1)"),
]
ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="lower right")
fig.tight_layout()
out8 = os.path.join(OUTPUT_DIR, "fig8_volatility_raw_vs_residual.png")
fig.savefig(out8, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out8}")
for sector in sector_order:
    r_raw = vol[(vol["sector"] == sector) & (vol["kind"] == "raw dSIE")].iloc[0]
    r_res = vol[(vol["sector"] == sector) & (vol["kind"] == "residual (APAC)")].iloc[0]
    print(f"  {sector:16s} raw dSIE {r_raw['value']:.2f} [{r_raw['ci_lo']:.2f},{r_raw['ci_hi']:.2f}]  "
          f"residual {r_res['value']:.2f} [{r_res['ci_lo']:.2f},{r_res['ci_hi']:.2f}]")

# ============================================================ Fig 11: index x component heatmap
comp_path = os.path.join(TABLES_DIR, "t37_component_comparison.csv")
if not os.path.exists(comp_path):
    print(f"\nSKIPPING Fig 11 — {comp_path} not found. "
          f"Run compute_component_comparison.py first, then re-run this script.")
else:
    comp = pd.read_csv(comp_path)
    cols = ["r_raw_sie", "r_amplitude", "r_phase", "r_residual"]
    pcols = ["p_raw_sie", "p_amplitude", "p_phase", "p_residual"]
    col_labels = ["Raw seasonal\nSIE (detrended)", "Amplitude", "Phase\n(day of max/min)", "Residual\nvolatility (SD)"]

    comp["row_label"] = comp["sector"] + " · " + comp["index"]
    mat = comp.set_index("row_label")[cols].astype(float)
    pmat = comp.set_index("row_label")[pcols].astype(float)
    pmat.columns = cols

    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(mat) + 1.6))
    vmax = np.nanmax(np.abs(mat.values))
    vmax = max(vmax, 0.3)
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            r = mat.values[i, j]
            p = pmat.values[i, j]
            if np.isnan(r):
                txt = "n/a"
            else:
                star = "*" if (not np.isnan(p) and p < 0.05) else ""
                txt = f"{r:+.2f}{star}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if (not np.isnan(r) and abs(r) > vmax * 0.55) else "black", fontsize=9)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=9)
    ax.set_title(
        "Correlation with the same index and season, by measurement type\n"
        "(* p<.05, uncorrected, single comparison)",
        fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    out11 = os.path.join(OUTPUT_DIR, "fig11_component_comparison_heatmap.png")
    fig.savefig(out11, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out11}")
    print(comp[["sector", "index", "season", "r_raw_sie", "r_amplitude", "r_phase", "r_residual"]].to_string(index=False))