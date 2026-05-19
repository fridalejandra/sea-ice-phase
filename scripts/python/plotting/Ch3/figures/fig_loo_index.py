"""
fig_loo_index.py
================
Leave-one-index-out figure — two panels per sector/variable combination,
styled after Fig 4 / Fig 6 from the reference paper.

Layout:
    Rows = sector × variable combinations (all sectors, phase + amplitude)
    Each row has two subplots:
        (left)  Skill metrics — RMSE and R² bars for each scenario
        (right) Residual timeseries — absolute residuals 1979–2023,
                one line per scenario

Scenarios shown:
    ALL, noSAM, noZW3, noASL, noNiño3.4
    (single-predictor models are in the CSV but not plotted here —
     they make the timeseries panel too busy; use them for supplementary)

Input:
    processed/loo_index_skill.csv
    processed/loo_index_residuals.csv

Output:
    figures/fig_loo_index.pdf  +  .png  (300 dpi)

Usage:
    python fig_loo_index.py
    python fig_loo_index.py --skill path/to/skill.csv --resid path/to/resid.csv --outdir figures/
"""

import argparse
import pathlib
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SKILL = pathlib.Path("processed/loo_index_skill.csv")
DEFAULT_RESID = pathlib.Path("processed/loo_index_residuals.csv")
DEFAULT_OUT   = pathlib.Path("figures")

# ── Display order ─────────────────────────────────────────────────────────────
SECTOR_ORDER   = ["EA", "Ross", "ABS", "Weddell", "King Haakon"]
VARIABLE_ORDER = ["amplitude", "phase"]

SECTOR_LABELS  = {
    "EA"        : "East Antarctica",
    "Ross"      : "Ross",
    "ABS"       : "ABS",
    "Weddell"   : "Weddell",
    "King Haakon": "King Haakon",
}
VAR_LABELS = {"amplitude": "Amplitude anomaly (σ)",
              "phase"    : "Phase anomaly (days)"}

# Scenarios to show in the figure (LOO set — not single-predictor)
LOO_SCENARIOS = ["ALL", "noSAM", "noZW3", "noASL", "noNiño3.4"]
# Fallback if Niño3.4 stored differently
NINO_ALIASES  = ["noNiño3.4", "noNino34", "noNiño34"]

SCENARIO_COLORS = {
    "ALL"       : "#222222",
    "noSAM"     : "#e6194b",   # red
    "noZW3"     : "#f58231",   # orange
    "noASL"     : "#4363d8",   # blue
    "noNiño3.4" : "#3cb44b",   # green
    "noNino34"  : "#3cb44b",
    "noNiño34"  : "#3cb44b",
}
SCENARIO_DISPLAY = {
    "ALL"       : "ALL",
    "noSAM"     : "no SAM",
    "noZW3"     : "no ZW3",
    "noASL"     : "no ASL",
    "noNiño3.4" : "no Niño3.4",
    "noNino34"  : "no Niño3.4",
    "noNiño34"  : "no Niño3.4",
}

FIG_WIDTH   = 15.0
ROW_HEIGHT  = 2.6   # inches per sector×variable row
HSPACE      = 0.55
WSPACE      = 0.30
LEFT_FRAC   = 0.06
RIGHT_FRAC  = 0.97


# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_scenarios(df_scenarios):
    """Return the LOO scenario names actually present in the data."""
    present = set(df_scenarios)
    out = ["ALL"]
    for s in ["noSAM", "noZW3", "noASL"]:
        if s in present:
            out.append(s)
    # Niño3.4 may be stored under different spellings
    for alias in NINO_ALIASES:
        if alias in present:
            out.append(alias)
            break
    return out


def color_for(scenario):
    return SCENARIO_COLORS.get(scenario, "#888888")


def label_for(scenario):
    return SCENARIO_DISPLAY.get(scenario, scenario)


# ── Panel drawing ─────────────────────────────────────────────────────────────
def draw_skill_panel(ax, skill_sub, scenarios, row_label):
    """
    Grouped bar chart: RMSE (left y-axis, black) and R² (right y-axis, red).
    One group per scenario.
    """
    n = len(scenarios)
    x = np.arange(n)
    w = 0.35

    rmse_vals = [skill_sub.loc[skill_sub["scenario"] == s, "rmse"].values
                 for s in scenarios]
    r2_vals   = [skill_sub.loc[skill_sub["scenario"] == s, "r2"].values
                 for s in scenarios]

    rmse_vals = [v[0] if len(v) else np.nan for v in rmse_vals]
    r2_vals   = [v[0] if len(v) else np.nan for v in r2_vals]

    # RMSE bars — left axis (note: inverted so higher skill = top)
    ax.bar(x - w / 2, rmse_vals, width=w,
           color=[color_for(s) for s in scenarios],
           alpha=0.85, label="RMSE", zorder=3)

    ax.set_ylabel("RMSE", fontsize=7.5, color="black")
    ax.tick_params(axis="y", labelsize=7, colors="black")
    ax.invert_yaxis()   # higher skill toward top, matching reference fig

    # R² bars — right axis
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, r2_vals, width=w,
            color=[color_for(s) for s in scenarios],
            alpha=0.45, hatch="///", label="R²", zorder=3)
    ax2.set_ylabel("R²", fontsize=7.5, color="#cc0000")
    ax2.tick_params(axis="y", labelsize=7, colors="#cc0000")
    ax2.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels([label_for(s) for s in scenarios],
                       fontsize=7, rotation=25, ha="right")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_title(f"{row_label}\nskill metrics", fontsize=8,
                 fontweight="bold", pad=3)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def draw_resid_panel(ax, resid_sub, scenarios, row_label):
    """
    Timeseries of absolute residuals, one line per scenario.
    """
    for scenario in scenarios:
        sub = resid_sub[resid_sub["scenario"] == scenario].sort_values("year")
        if sub.empty:
            continue
        ax.plot(sub["year"], sub["abs_resid"],
                color=color_for(scenario),
                linewidth=1.4 if scenario == "ALL" else 0.9,
                alpha=1.0    if scenario == "ALL" else 0.75,
                label=label_for(scenario),
                zorder=3     if scenario == "ALL" else 2)

    ax.set_ylabel("|residual|", fontsize=7.5)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlim(1979, 2023)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"{row_label}\nresiduals 1979–2023",
                 fontsize=8, fontweight="bold", pad=3)


# ── Figure assembly ───────────────────────────────────────────────────────────
def make_figure(skill_path, resid_path, outdir):
    skill_df = pd.read_csv(skill_path)
    resid_df = pd.read_csv(resid_path)

    # Normalise sector/variable names
    for df in (skill_df, resid_df):
        df["sector"]   = df["sector"].str.strip()
        df["variable"] = df["variable"].str.strip().str.lower()

    # Build ordered row list
    rows = [(sec, var)
            for var in VARIABLE_ORDER
            for sec in SECTOR_ORDER
            if not skill_df[(skill_df["sector"] == sec) &
                            (skill_df["variable"] == var)].empty]

    if not rows:
        raise ValueError("No matching sector/variable combinations found in skill CSV.")

    n_rows    = len(rows)
    scenarios = resolve_scenarios(skill_df["scenario"].unique())

    fig_height = n_rows * ROW_HEIGHT + 1.2
    fig, axes  = plt.subplots(
        n_rows, 2,
        figsize=(FIG_WIDTH, fig_height),
        gridspec_kw={"wspace": WSPACE, "hspace": HSPACE,
                     "left": LEFT_FRAC, "right": RIGHT_FRAC}
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_i, (sec, var) in enumerate(rows):
        row_label = f"{SECTOR_LABELS.get(sec, sec)} — {VAR_LABELS.get(var, var)}"

        skill_sub = skill_df[(skill_df["sector"]   == sec) &
                             (skill_df["variable"] == var)]
        resid_sub = resid_df[(resid_df["sector"]   == sec) &
                             (resid_df["variable"] == var)]

        draw_skill_panel(axes[row_i, 0], skill_sub, scenarios, row_label)
        draw_resid_panel(axes[row_i, 1], resid_sub, scenarios, row_label)

    # ── Shared legend ─────────────────────────────────────────────────────────
    handles = [
        mpl.lines.Line2D([0], [0], color=color_for(s), linewidth=2,
                         label=label_for(s))
        for s in scenarios
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(scenarios), fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "Leave-one-index-out regression skill — phase and amplitude anomalies",
        fontsize=11, fontweight="bold", y=1.002
    )

    # ── Panel labels (a), (b), (c)… ──────────────────────────────────────────
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    for i, axrow in enumerate(axes):
        for j, ax in enumerate(axrow):
            label = f"({panel_letters[i * 2 + j]})"
            ax.text(-0.01, 1.05, label, transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="bottom", ha="right")

    # ── Save ──────────────────────────────────────────────────────────────────
    outdir.mkdir(parents=True, exist_ok=True)
    stem = "fig_loo_index"
    for ext in ("pdf", "png"):
        fpath = outdir / f"{stem}.{ext}"
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        print(f"Saved → {fpath}")
    plt.close(fig)
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Leave-one-index-out skill and residual figure"
    )
    parser.add_argument("--skill",  type=pathlib.Path, default=DEFAULT_SKILL)
    parser.add_argument("--resid",  type=pathlib.Path, default=DEFAULT_RESID)
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    make_figure(args.skill, args.resid, args.outdir)