"""
run_all.py — regenerate every Ch3 figure in dependency order.
=============================================================
    python run_all.py            # everything available
    python run_all.py fig04      # one figure
    python run_all.py --list     # show what exists and what is blocked
"""

import os, sys, runpy, traceback
from ch3_config import DAILY_CSV, ANNUAL_CSV, CORR_CSV, LAG_CSV, CONT_CSV, MONTHLY_XC_CSV

# (script, human name, list of required input files)
FIGURES = [
    ("fig03_rolling_phase_amp_corr.py", "fig03 rolling r(phase,amp)", [ANNUAL_CSV]),
    ("fig04_phase_timeseries.py",       "fig04 timing timeseries",    [ANNUAL_CSV]),
    ("fig04_dot_timeline.py",           "fig04b dominance timeline",  [DAILY_CSV]),
    ("fig05_amplitude_timeseries.py",   "fig05 amplitude timeseries", [ANNUAL_CSV]),
    ("fig06_rolling_sd.py",             "fig06 rolling SD",           [ANNUAL_CSV]),
    ("fig07_era_sd_bars.py",            "fig07 pre/post-2016 SD",     [ANNUAL_CSV]),
    ("figS01_fitted_vs_raw.py",         "S01 fitted vs observed",     [ANNUAL_CSV]),
    ("figS05_decomp_2016_2023.py",      "S05 decomposition",          [DAILY_CSV, ANNUAL_CSV]),
    ("figS06_case_study_zscores.py",    "S06 case-study z-scores",    [ANNUAL_CSV]),
    # --- blocked until the correlation pipeline is rerun on annual_params_B ---
    ("fig08_correlation_heatmap.py",    "fig08 index heatmap",        [CORR_CSV]),
    ("fig09_monthly_lag_corr.py",       "fig09 monthly lag corr",     [MONTHLY_XC_CSV]),
    ("fig10_rolling_index_corr.py",     "fig10 rolling index corr",   [CORR_CSV]),
]


def status(reqs, script):
    if not os.path.exists(script):
        return "no script"
    missing = [os.path.basename(r) for r in reqs if not os.path.exists(r)]
    return "missing: " + ", ".join(missing) if missing else "ready"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--list" in sys.argv:
        print(f"{'figure':<32} {'script':<36} status")
        print("-" * 96)
        for script, name, reqs in FIGURES:
            print(f"{name:<32} {script:<36} {status(reqs, script)}")
        return

    todo = [f for f in FIGURES if not args or any(a in f[0] for a in args)]
    ok, skipped, failed = [], [], []

    for script, name, reqs in todo:
        st = status(reqs, script)
        if st != "ready":
            print(f"SKIP  {name:<32} ({st})")
            skipped.append(name)
            continue
        print(f"\n=== {name} ===")
        try:
            runpy.run_path(script, run_name="__main__")
            ok.append(name)
        except SystemExit as e:
            print(f"STOPPED: {e}")
            failed.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)

    print("\n" + "=" * 60)
    print(f"built {len(ok)}   skipped {len(skipped)}   failed {len(failed)}")
    if failed:
        print("failed: " + ", ".join(failed))
    if skipped:
        print("skipped: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
