#!/usr/bin/env python3
"""
run_all.py -- regenerate the Chapter 3 tables and figures in manuscript order.

    python run_all.py            everything, in dependency order
    python run_all.py fig_07     one step (substring of the script name)
    python run_all.py --list     show the steps and what each produces
    python run_all.py --figures  figures only (skip the R fits and stats)

Python steps run in-process (runpy) so a failure shows a normal traceback;
R steps run via Rscript. A missing script is reported and skipped.

The R fits take hours and are not rerun unless named explicitly
("python run_all.py 01_fit" or "07_vol").
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
    (os.path.join(R_DIR, "01_fit_apac.R"),        "R",      "APAC fit: daily_fitted.csv, annual_params.csv"),
    (os.path.join(R_DIR, "07_volatility_only.R"), "R",      "GAMLSS volatility tables t34c (Sect. 3.4, Fig. S3)"),
    # -- indices and tables -----------------------------------------------------
    (os.path.join(PROC_DIR, "compute_atmospheric_correlations.py"), "py", "master_index_detrended.csv, correlations_output.csv"),
    (os.path.join(HERE, "ch3_stats.py"),                   "py", "ch3_numbers.csv ledger; t35 scan; seven pairs; Ross-ASL"),
    (os.path.join(HERE, "table_02_03_rmse.py"),            "py", "Tables 2-3 (RMSE)"),
    (os.path.join(HERE, "compute_t32_share_trends.py"),    "py", "Sect. 3.2 share-trend tests"),
    (os.path.join(HERE, "compute_t33_variance_ratios.py"), "py", "Sect. 3.3 variance ratios"),
    (os.path.join(HERE, "compute_fig07_component_comparison.py"), "py", "t37, t37b, t37c; daily wind lags (Table 8)"),
    # -- main figures -----------------------------------------------------------
    (os.path.join(HERE, "fig_01_conflation.py"),                  "py", "Fig. 1 concept"),
    (os.path.join(HERE, "fig_02_sector_map.py"),                  "py", "Fig. 2 sector map"),
    (os.path.join(HERE, "fig_03-04_attribution_annual.py"),       "py", "Figs. 3-4 attribution, era shares"),
    (os.path.join(HERE, "fig_05_rolling_phase_amp.py"),           "py", "Fig. 5 timing-amplitude correlation"),
    (os.path.join(HERE, "fig_06_raw_anomaly_persistence.py"),     "py", "Fig. 6 raw-anomaly autocorrelation; Table 4"),
    (os.path.join(HERE, "fig_07_component_comparison_heatmap.py"), "py", "Fig. 7 index/wind vs components"),
    (os.path.join(HERE, "fig_08_atmosphere_sevenpairs.py"),       "py", "Fig. 8 seven relationships"),
    (os.path.join(HERE, "fig_09_ross_asl_nonstationarity.py"),    "py", "Fig. 9 Ross-ASL"),
    (os.path.join(HERE, "fig_10_abs_growth_season.py"),           "py", "Fig. 10 ABS growth season"),
    # -- supplement -------------------------------------------------------------
    (os.path.join(HERE, "fig_s01_fitted_vs_observed.py"),         "py", "Fig. S1"),
    (os.path.join(HERE, "fig_s02_annual_min_max_trend.py"),       "py", "Fig. S2"),
    (os.path.join(HERE, "fig_s03_volatility_seasonal_curves.py"), "py", "Fig. S3 (s03b raw anomaly; s03a dSIE unused)"),
    (os.path.join(HERE, "fig_s04_annual_volatility_persistence.py"), "py", "Fig. S4 year-by-year variance"),
]
SLOW = ("01_fit_apac", "07_volatility_only")


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
        if figures_only and not base.startswith("fig_"):
            continue
        (ok if run(s, r) else bad).append(base)
    print(f"\ndone: {len(ok)} ran" + (f", {len(bad)} failed/skipped: {bad}" if bad else ""))