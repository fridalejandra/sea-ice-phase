#!/usr/bin/env python3
"""
run_all.py -- regenerate the Chapter 3 tables and figures in manuscript order.

    python run_all.py              everything, in dependency order
    python run_all.py fig_ross     one step (substring of the script name)
    python run_all.py --list       show the steps and what each produces
    python run_all.py --figures    figures only (fig_, draft_, supp_; skip the R fits and tables)

Python steps run in-process (runpy) so a failure shows a normal traceback;
R steps run via Rscript. A missing script is reported and skipped.

The R fits take hours and are not rerun unless named explicitly
("python run_all.py 01_fit" or "07_vol").

File prefixes (2026-10-04 reorganization):
    fig_    figure placed in ch3_main.tex
    draft_  figure candidate not placed yet
    supp_   supplement figure
    table_  builds a table
Descriptions below say what each step makes and where it's used, not a
figure number -- LaTeX numbers the figures, and those numbers change.
"""
import os
import sys
import runpy
import subprocess
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import ROOT

R_DIR = os.path.join(ROOT, "scripts", "R", "Ch3")
PROC_DIR = os.path.normpath(os.path.join(HERE, "..", "processing"))

# (script, runner, what it makes)
STEPS = [
    # -- fits (slow; run only when named) --------------------------------------
    (os.path.join(R_DIR, "01_fit_apac.R"),        "R",  "APAC fit: daily_fitted.csv, annual_params.csv"),
    (os.path.join(R_DIR, "07_volatility_only.R"), "R",  "GAMLSS volatility tables t34c (Sect. 3.4, supplement)"),
    # -- indices and tables -----------------------------------------------------
    (os.path.join(PROC_DIR, "compute_atmospheric_correlations.py"), "py", "master_index_detrended.csv, correlations_output.csv"),
    (os.path.join(HERE, "ch3_stats.py"),                    "py", "ch3_numbers ledger; t35 scan; seven pairs; t36 Ross-ASL"),
    (os.path.join(HERE, "table_relationships.py"),          "py", "Table 3 rows from the ledger (pass --tex PATH to check ch3_main.tex)"),
    (os.path.join(HERE, "table_rmse.py"),                   "py", "RMSE table (Sect. 3.1)"),
    (os.path.join(HERE, "table_share_trends.py"),           "py", "Trend-share growth tests (Sect. 3.1)"),
    (os.path.join(HERE, "table_variance_ratios.py"),        "py", "post/pre-2016 variance ratios (Sect. 3.4)"),
    (os.path.join(HERE, "draft_component_comparison_data.py"), "py", "t37, t37b, t37c; same-day wind coupling"),
    # -- figures in the manuscript, in text order --------------------------------
    (os.path.join(HERE, "fig_cycle_mechanisms.py"),   "py", "Intro: three changes, one anomaly"),
    (os.path.join(HERE, "fig_phase_adjustment.py"),   "py", "Methods: phase adjustment diagram"),
    (os.path.join(HERE, "fig_relationships_heatmap.py"), "py", "Methods/3.2: index vs component heatmap (referenced, not placed yet)"),
    (os.path.join(HERE, "fig_level_leak.py"),         "py", "Sect. 3.1: level leak, Weddell / Nino3.4"),
    (os.path.join(HERE, "fig_ross_asl_break.py"),     "py", "Sect. 3.3: Ross-ASL break by season and break year"),
    (os.path.join(HERE, "fig_ross_asl_composite.py"), "py", "Sect. 3.3: Ross composite cycles, high vs low ASL"),
    (os.path.join(HERE, "fig_level_vs_amplitude.py"), "py", "Sect. 3.4: level vs amplitude, sector levels"),
    (os.path.join(HERE, "checks", "check_raw_anomaly_persistence.py"), "py", "Sect. 3.4: raw-anomaly persistence"),
    # -- candidates, not placed yet ------------------------------------------------
    (os.path.join(HERE, "draft_sector_map.py"),         "py", "sector map (commented out in Data)"),
    (os.path.join(HERE, "draft_components_by_cycle.py"), "py", "attribution by component, per cycle"),
    (os.path.join(HERE, "draft_rolling_phase_amp.py"),  "py", "Sect. 3.4 candidate: timing-amplitude correlation"),
    (os.path.join(HERE, "draft_abs_growth_season.py"),  "py", "Sect. 3.4.2 candidate: ABS growth season"),
    # -- supplement -------------------------------------------------------------
    (os.path.join(HERE, "supp_fitted_vs_observed.py"),   "py", "supplement: fitted vs observed"),
    (os.path.join(HERE, "supp_annual_min_max_trend.py"), "py", "supplement: annual min/max trend"),
    (os.path.join(HERE, "archived/fig_s03_volatility_seasonal_curves.py"),    "py", "supplement: volatility seasonal curves (script lives in archived/)"),
    (os.path.join(HERE, "archived/fig_s04_annual_volatility_persistence.py"), "py", "supplement: year-by-year variance (script lives in archived/)"),
]
# Removed 2026-10-04: draft_component_comparison.py (duplicate of fig_relationships_heatmap.py,
# archived), fig_05_component_comparison_heatmap.py (never existed under that name),
# archived/_dropped_atmosphere_sevenpairs.py (dropped from the manuscript), and
# archived/retired_raw_anomaly_persistence.py (raw-anomaly standalone analysis was cut;
# add it back here if its Table 4 is still used).
SLOW = ("01_fit_apac", "07_volatility_only")
FIGURE_PREFIXES = ("fig_", "draft_", "supp_")


def run(script, runner):
    print(f"\n=== {os.path.basename(script)}")
    if not os.path.exists(script):
        print(f"    no script at {script}; skipped")
        return False
    try:
        if runner == "R":
            subprocess.run(["Rscript", script], check=True, cwd=os.path.dirname(script))
        else:
            cwd = os.getcwd()
            os.chdir(os.path.dirname(script))
            try:
                runpy.run_path(script, run_name="__main__")
            finally:
                os.chdir(cwd)
        return True
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"    exited with {e.code}")
            return False
        return True
    except Exception:
        traceback.print_exc()
        return False


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if "--list" in args:
        for s, r, what in STEPS:
            print(f"  {os.path.basename(s):46s} {what}")
        sys.exit(0)
    figures_only = "--figures" in args
    names = [a for a in args if not a.startswith("--")]
    ok, bad = [], []
    for s, r, what in STEPS:
        base = os.path.basename(s)
        if names and not any(n in base for n in names):
            continue
        if not names and any(k in base for k in SLOW):
            print(f"\n=== {base}: slow R fit, run it by name to rerun")
            continue
        if figures_only and not base.startswith(FIGURE_PREFIXES):
            continue
        (ok if run(s, r) else bad).append(base)
    print(f"\ndone: {len(ok)} ran" + (f", {len(bad)} failed/skipped: {bad}" if bad else ""))
