"""
fig_loo_index.py
================
Leave-one-index-out residual timeseries figures.

For each variable (amplitude / phase), produces a figure with one panel
per sector (5 panels total), showing absolute residuals 1979–2023 for
each single-predictor scenario plus ALL as reference.

Scenarios shown:
    ALL           — all four indices as predictors (reference, black)
    SAM_only      — SAM alone
    ZW3_only      — ZW3 alone
    ASL_only      — ASL alone
    Niño3.4_only  — Niño3.4 alone

Two output figures:
    fig_loo_amplitude.pdf/.png
    fig_loo_phase.pdf/.png

Input:
    loo_index_residuals.csv   (from compute_loo_index.py)

Usage:
    python fig_loo_index.py
    python fig_loo_index.py --resid /full/path/to/loo_index_residuals.csv --outdir figures/
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RESID = pathlib.Path("processed/loo_index_residuals.csv")
DEFAULT_OUT   = pathlib.Path("figures")

# ── Display order ─────────────────────────────────────────────────────────────
SECTOR_ORDER = ["EA", "Ross", "ABS", "Weddell", "King Haakon"]
SECTOR_LABELS = {
    "EA"         : "East Antarctica",
    "Ross"       : "Ross",
    "ABS"        : "ABS",
    "Weddell"    : "Weddell",
    "King Haakon": "King Haakon",
}
VAR_YLABELS = {
    "amplitude": "|residual| (σ)",
    "phase"    : "|residual| (days)",
}
VAR_TITLES = {
    "amplitude": "Leave-one-index-out — Amplitude anomaly",
    "phase"    : "Leave-one-index-out — Phase anomaly",
}

# Single-predictor scenarios + ALL reference (aliases handled below)
SCENARIO_PREFS = ["ALL", "SAM_only", "ZW3_only", "ASL_only", "Niño3.4_only", "Nino34_only"]

SCENARIO_STYLE = {
    "ALL"          : dict(color="#222222", lw=1.8, alpha=1.0,  zorder=4, ls="-"),
    "SAM_only"     : dict(color="#e6194b", lw=1.1, alpha=0.85, zorder=3, ls="-"),
    "ZW3_only"     : dict(color="#f58231", lw=1.1, alpha=0.85, zorder=3, ls="-"),
    "ASL_only"     : dict(color="#4363d8", lw=1.1, alpha=0.85, zorder=3, ls="-"),
    "Niño3.4_only" : dict(color="#3cb44b", lw=1.1, alpha=0.85, zorder=3, ls="-"),
    "Nino34_only"  : dict(color="#3cb44b", lw=1.1, alpha=0.85, zorder=3, ls="-"),
}
SCENARIO_LABELS = {
    "ALL"          : "ALL",
    "SAM_only"     : "SAM",
    "ZW3_only"     : "ZW3",
    "ASL_only"     : "ASL",
    "Niño3.4_only" : "Niño3.4",
    "Nino34_only"  : "Niño3.4",
}

FIG_WIDTH  = 11.0
ROW_HEIGHT = 2.2
LEFT_FRAC  = 0.08
RIGHT_FRAC = 0.97


# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_scenarios(present):
    """Return scenario names to plot, deduplicated by display label."""
    seen, out = set(), []
    for s in SCENARIO_PREFS:
        if s in present:
            lbl = SCENARIO_LABELS.get(s, s)
            if lbl not in seen:
                out.append(s)
                seen.add(lbl)
    return out


# ── Figure builder ────────────────────────────────────────────────────────────
def make_figure(resid_df, variable, scenarios, outdir):
    sectors = [s for s in SECTOR_ORDER
               if not resid_df[(resid_df["sector"] == s) &
                               (resid_df["variable"] == variable)].empty]
    if not sectors:
        print(f"  No data for variable={variable}, skipping.")
        return

    n     = len(sectors)
    fig_h = n * ROW_HEIGHT + 1.0

    fig, axes = plt.subplots(
        n, 1,
        figsize=(FIG_WIDTH, fig_h),
        gridspec_kw={
            "hspace" : 0.55,
            "left"   : LEFT_FRAC,
            "right"  : RIGHT_FRAC,
            "top"    : 1 - 0.4 / fig_h,
            "bottom" : 0.7 / fig_h,
        }
    )
    if n == 1:
        axes = [axes]

    letters = "abcdefghijklmnopqrstuvwxyz"

    for i, sec in enumerate(sectors):
        ax  = axes[i]
        sub = resid_df[(resid_df["sector"]   == sec) &
                       (resid_df["variable"] == variable)]

        for scenario in scenarios:
            s_sub = sub[sub["scenario"] == scenario].sort_values("year")
            if s_sub.empty:
                continue
            style = SCENARIO_STYLE.get(
                scenario, dict(color="#aaaaaa", lw=0.9, alpha=0.7, zorder=2, ls="-")
            )
            ax.plot(s_sub["year"], s_sub["abs_resid"],
                    label=SCENARIO_LABELS.get(scenario, scenario),
                    **style)

        ax.set_xlim(1979, 2023)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax.tick_params(axis="both", labelsize=8)
        ax.set_ylabel(VAR_YLABELS.get(variable, "|residual|"), fontsize=8)
        ax.grid(linestyle="--", linewidth=0.4, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        ax.text(-0.005, 1.03, f"({letters[i]})",
                transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="right")
        ax.set_title(SECTOR_LABELS.get(sec, sec),
                     fontsize=9, fontweight="bold", pad=3, loc="left")

        if i < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Year", fontsize=8)

    # Shared legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(scenarios),
               fontsize=8.5, frameon=True,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(VAR_TITLES[variable], fontsize=11,
                 fontweight="bold", y=1.002)

    stem = f"fig_loo_{variable}"
    for ext in ("pdf", "png"):
        fpath = outdir / f"{stem}.{ext}"
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        print(f"  Saved → {fpath}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(resid_path, outdir):
    print("Loading residuals...")
    resid_df = pd.read_csv(resid_path)
    resid_df["sector"]   = resid_df["sector"].str.strip()
    resid_df["variable"] = resid_df["variable"].str.strip().str.lower()
    resid_df["scenario"] = resid_df["scenario"].str.strip()

    scenarios = resolve_scenarios(resid_df["scenario"].unique())
    print(f"  Scenarios: {scenarios}")

    outdir.mkdir(parents=True, exist_ok=True)

    for variable in ["amplitude", "phase"]:
        print(f"\nPlotting {variable}...")
        make_figure(resid_df, variable, scenarios, outdir)

    print("\nDone.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LOO index residual timeseries figures (amplitude + phase)"
    )
    parser.add_argument("--resid",  type=pathlib.Path, default=DEFAULT_RESID)
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.resid, args.outdir)