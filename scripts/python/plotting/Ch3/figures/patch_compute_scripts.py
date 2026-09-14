#!/usr/bin/env python3
"""
patch_compute_scripts.py
========================
Rewires the compute_*.py pipeline onto ch3_config.py and switches it from the
FITTED metrics to the OBSERVED ones.

WHY
---
Four of the six compute scripts correlate against `max_doy_anom` and
`amplitude_anom`. `max_doy_anom` is the anomaly of `max_doy_fitted` — the
argmax of the fitted APAC curve, which is dominated by the fixed s(DOY) term.
Its SD is 2-4x smaller than the observed maximum date, it correlates with the
observed maximum at only r = 0.02-0.46, and in Ross it correlates with the
observed MINIMUM at r = -0.74. It is not a timing metric.

Results currently resting on it include the ABS phase-ASL DJF finding and the
Ross phase result in the outlier diagnostic.

Separately, every script hardcodes its own paths and they have drifted:
compute_outlier_diagnostic.py reads master_index_detrended.csv from
Ch3/figures/ while the rest read it from Ch3/data/.

WHAT THIS DOES
--------------
Per script:
  * repoint ANNUAL_CSV / DAILY_CSV at the Pipeline B outputs
  * fix the INDEX_CSV path drift
  * replace the local APAC_VARS with the observed set
  * leave everything else untouched

Run with --dry-run first to see the diff. Originals are backed up to
*.py.bak before anything is written.

    python patch_compute_scripts.py --dry-run
    python patch_compute_scripts.py
    python patch_compute_scripts.py --dir=../processing

NOTE: this edits the scripts IN PLACE (backing up to *.py.bak). It does not
create new files. If you would rather not run it, pre-patched copies of all
six scripts are provided alongside this one.
"""

import os
import re
import sys
import shutil
import difflib

# Target directory. Defaults to ./processing relative to this file, then to
# this file's own directory. Override with --dir=/path/to/processing
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULTS = [os.path.join(_HERE, "processing"),
             os.path.join(os.path.dirname(_HERE), "processing"),
             _HERE]
SCRIPT_DIR = next((d for d in _DEFAULTS
                   if os.path.exists(os.path.join(
                       d, "compute_atmospheric_correlations.py"))), _HERE)
for _a in sys.argv[1:]:
    if _a.startswith("--dir="):
        SCRIPT_DIR = os.path.expanduser(_a.split("=", 1)[1])

# ── Replacements, per file ────────────────────────────────────────────────────
# Each entry: (pattern, replacement, description). Applied in order.

PATHS = [
    ('^(DATA_DIR\\s*=\\s*)"/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"',
     '# --- Repo root: resolves local first, cluster as fallback -----------------\n_ROOTS = [os.environ.get("SEAICE_ROOT"),\n          os.path.expanduser("~/Research/repos/sea-ice-phase"),\n          "/Users/fridaperez/Research/repos/sea-ice-phase",\n          "/user/geog/falejandraperez/sea-ice-phase"]\nROOT = next((r for r in _ROOTS if r and os.path.isdir(os.path.join(r, "scripts"))), None)\nif ROOT is None:\n    raise SystemExit("Cannot locate sea-ice-phase repo. Set SEAICE_ROOT.")\n\\1os.path.join(ROOT, "scripts", "R", "Ch3", "data")',
     'DATA_DIR -> resolved root'),
    ('^(INDEX_DIR\\s*=\\s*)"/user/geog/falejandraperez/sea-ice-phase/data/indices"',
     '\\1os.path.join(ROOT, "data", "indices")',
     'INDEX_DIR -> resolved root'),
    ('ANNUAL_CSV = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data/annual_params_B\\.csv"',
     '# --- Repo root: resolves local first, cluster as fallback -----------------\n_ROOTS = [os.environ.get("SEAICE_ROOT"),\n          os.path.expanduser("~/Research/repos/sea-ice-phase"),\n          "/Users/fridaperez/Research/repos/sea-ice-phase",\n          "/user/geog/falejandraperez/sea-ice-phase"]\nROOT = next((r for r in _ROOTS if r and os.path.isdir(os.path.join(r, "scripts"))), None)\nif ROOT is None:\n    raise SystemExit("Cannot locate sea-ice-phase repo. Set SEAICE_ROOT.")\nANNUAL_CSV = os.path.join(ROOT, "scripts", "R", "Ch3", "data", "annual_params_B.csv")',
     'ANNUAL_CSV -> resolved root'),
    ('INDEX_CSV  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data/master_index_detrended\\.csv"',
     'INDEX_CSV  = os.path.join(ROOT, "scripts", "R", "Ch3", "data", "master_index_detrended.csv")',
     'INDEX_CSV -> resolved root'),
    ('OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/figures/"',
     'OUTPUT_DIR = os.path.join(ROOT, "scripts", "python", "plotting", "Ch3", "figures")',
     'OUTPUT_DIR -> resolved root'),
]

COMMON = PATHS + [
    (r'ANNUAL_CSV\s*=\s*os\.path\.join\(DATA_DIR,\s*"annual_params\.csv"\)',
     'ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params_B.csv")',
     "annual_params -> annual_params_B"),
    (r'DAILY_CSV\s*=\s*os\.path\.join\(DATA_DIR,\s*"daily_fitted\.csv"\)',
     'DAILY_CSV   = os.path.join(DATA_DIR, "daily_fitted_B.csv")',
     "daily_fitted -> daily_fitted_B"),
]

OBSERVED_VARS_BLOCK = '''APAC_VARS = {
    # OBSERVED metrics — see ch3_config.ANALYSIS_VARS for why the fitted
    # quantities (max_doy_anom / amplitude_anom) are not used here.
    "max_doy_raw_anom"  : "phase_max",
    "min_doy_raw_anom"  : "phase_min",
    "amplitude_raw_anom": "amplitude",
}'''

PATCHES = {
    "compute_atmospheric_correlations.py": COMMON + [
        # already runs all four variants; just add the minimum date
        (r'APAC_VARS = \{\n(\s+"amplitude_anom".*\n)(\s+"max_doy_anom".*\n)'
         r'(\s+"amplitude_raw_anom".*\n)(\s+"max_doy_raw_anom".*\n)\}',
         'APAC_VARS = {\n'
         '    "amplitude_anom"    : "amplitude_apac",\n'
         '    "max_doy_anom"      : "phase_apac",\n'
         '    "amplitude_raw_anom": "amplitude_raw",\n'
         '    "max_doy_raw_anom"  : "phase_max_raw",\n'
         '    "min_doy_raw_anom"  : "phase_min_raw",\n'
         '}',
         "add min_doy_raw_anom; rename phase_raw -> phase_max_raw"),
    ],

    "compute_monthly_lagged_correlations.py": COMMON + [
        (r'APAC_VARS = \{\n\s+"max_doy_anom"\s*:\s*"phase",\n'
         r'\s+"amplitude_anom":\s*"amplitude",\n\}',
         OBSERVED_VARS_BLOCK,
         "FITTED -> OBSERVED metrics"),
        (r'for col in \["max_doy_anom", "amplitude_anom"\]:',
         'for col in list(APAC_VARS.keys()):',
         "detrend loop follows APAC_VARS"),
        # KEY_PAIRS reference the fitted column names
        (r'"amplitude_anom"', '"amplitude_raw_anom"',
         "KEY_PAIRS amplitude column"),
        (r'"max_doy_anom"', '"max_doy_raw_anom"',
         "KEY_PAIRS phase column"),
    ],

    "compute_monthly_corr_both.py": COMMON + [
        (r'for ice_var, var_label in \[\("amplitude_anom","amplitude"\),\s*'
         r'\("max_doy_anom","phase"\)\]:',
         'for ice_var, var_label in [("amplitude_raw_anom", "amplitude"),\n'
         '                                   ("max_doy_raw_anom", "phase_max"),\n'
         '                                   ("min_doy_raw_anom", "phase_min")]:',
         "FITTED -> OBSERVED metrics"),
    ],

    "compute_loo_index.py": COMMON + [
        (r'APAC_VARS = \{\n\s+"amplitude_anom":\s*"amplitude",\n'
         r'\s+"max_doy_anom"\s*:\s*"phase",\n\}',
         OBSERVED_VARS_BLOCK,
         "FITTED -> OBSERVED metrics"),
    ],

    "compute_outlier_diagnostic.py": [
        (r'ANNUAL_CSV = ".*?annual_params\.csv"\nINDEX_CSV  = ".*?master_index_detrended\.csv"\nOUTPUT_DIR = ".*?"',
         '# --- Repo root: resolves local first, cluster as fallback -----------------\n_ROOTS = [os.environ.get("SEAICE_ROOT"),\n          os.path.expanduser("~/Research/repos/sea-ice-phase"),\n          "/Users/fridaperez/Research/repos/sea-ice-phase",\n          "/user/geog/falejandraperez/sea-ice-phase"]\nROOT = next((r for r in _ROOTS if r and os.path.isdir(os.path.join(r, "scripts"))), None)\nif ROOT is None:\n    raise SystemExit("Cannot locate sea-ice-phase repo. Set SEAICE_ROOT.")\nDATA_DIR   = os.path.join(ROOT, "scripts", "R", "Ch3", "data")\n\nANNUAL_CSV = os.path.join(DATA_DIR, "annual_params_B.csv")\n# was Ch3/figures/master_index_detrended.csv — stale copy\nINDEX_CSV  = os.path.join(DATA_DIR, "master_index_detrended.csv")\nOUTPUT_DIR = os.path.join(ROOT, "scripts", "python", "plotting", "Ch3", "figures")',
         "paths -> resolved root; PATH DRIFT figures/ -> data/; annual_params_B"),
        (r'\["Year","max_doy_anom"\]', '["Year","max_doy_raw_anom"]',
         "Ross phase -> observed"),
        (r'"Year", "max_doy_anom"', '"Year", "max_doy_raw_anom"',
         "Ross phase -> observed"),
        (r'detrend\(ross, "Year", "max_doy_anom"\)',
         'detrend(ross, "Year", "max_doy_raw_anom")',
         "detrend call"),
        (r'\["Year","amplitude_anom"\]', '["Year","amplitude_raw_anom"]',
         "EA amplitude -> observed"),
        (r'detrend\(ea, "Year", "amplitude_anom"\)',
         'detrend(ea, "Year", "amplitude_raw_anom")',
         "detrend call"),
        (r'"max_doy_anom"', '"max_doy_raw_anom"', "remaining refs"),
        (r'"amplitude_anom"', '"amplitude_raw_anom"', "remaining refs"),
    ],

    "compute_phase_amplitude_monthly.py": COMMON + [
        # the trend-free reference replaces fitted_invariant for anomalies
        (r'inv_by_doy = sec_daily\.groupby\("DOY"\)\["fitted_invariant"\]\.mean\(\)',
         'inv_by_doy = sec_daily.groupby("DOY")["iac_notrend"].mean()',
         "use trend-free climatology"),
        (r'invar   = mo_data\["fitted_invariant"\]\.values',
         'invar   = mo_data["iac_notrend"].values',
         "use trend-free climatology"),
    ],
}


def apply_patches(path, rules, dry_run):
    if not os.path.exists(path):
        print(f"  SKIP  {os.path.basename(path)} — not found")
        return False

    original = open(path).read()
    text = original
    applied, skipped = [], []

    for pattern, repl, desc in rules:
        new, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
        if n:
            text = new
            applied.append(f"{desc} ({n}x)")
        else:
            skipped.append(desc)

    name = os.path.basename(path)
    if text == original:
        print(f"  ---   {name}: no changes "
              f"(already patched, or patterns did not match)")
        for s in skipped:
            print(f"          unmatched: {s}")
        return False

    print(f"  PATCH {name}")
    for a in applied:
        print(f"          {a}")
    for s in skipped:
        print(f"          unmatched: {s}")

    if dry_run:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile=name, tofile=name + " (patched)", n=1)
        print("".join("          " + d for d in diff))
    else:
        shutil.copy2(path, path + ".bak")
        open(path, "w").write(text)
        print(f"          written (backup: {name}.bak)")
    return True


def main():
    dry = "--dry-run" in sys.argv
    print("patch_compute_scripts.py" + ("  [DRY RUN]" if dry else ""))
    print(f"working in: {SCRIPT_DIR}")
    if not os.path.exists(os.path.join(SCRIPT_DIR,
                                       "compute_atmospheric_correlations.py")):
        print("\nNo compute_*.py found there. Pass --dir=/path/to/processing")
        return
    print()

    changed = 0
    for fname, rules in PATCHES.items():
        changed += apply_patches(os.path.join(SCRIPT_DIR, fname), rules, dry)

    print(f"\n{changed} file(s) {'would be ' if dry else ''}changed.")
    if not dry and changed:
        print("\nNow rerun the pipeline in this order:")
        print("  1. compute_atmospheric_correlations.py   (writes master_index_detrended.csv)")
        print("  2. compute_phase_amplitude_monthly.py    (writes monthly_params.csv)")
        print("  3. compute_monthly_lagged_correlations.py")
        print("  4. compute_monthly_corr_both.py")
        print("  5. compute_loo_index.py")
        print("  6. compute_outlier_diagnostic.py")
        print("\nThen: python run_all.py")
    print("\nUnmatched patterns are expected where a script has already been")
    print("edited by hand — check those individually before rerunning.")


if __name__ == "__main__":
    main()
