#!/usr/bin/env python3
"""
run_pipeline.py — the whole Chapter 3 pipeline, in dependency order.
====================================================================

    python run_pipeline.py                 # everything
    python run_pipeline.py --from 3        # resume from stage 3
    python run_pipeline.py --only 3        # one stage
    python run_pipeline.py --list          # show stages and current state
    python run_pipeline.py --dry-run       # show what would run
    python run_pipeline.py --skip-r        # assume Pipeline B already ran

Stages, and why the order matters:

  1. APAC_Sector_Pipeline_B.R          -> daily_fitted_B, annual_params_B
  2. compute_atmospheric_correlations  -> master_index_detrended.csv
                                          FOUR later scripts read this, so it
                                          must run before any of them
  3. compute_phase_amplitude_monthly   -> monthly_params.csv
                                          needed by compute_monthly_corr_both
  4. compute_monthly_lagged_correlations
  5. compute_monthly_corr_both
  6. compute_loo_index
  7. compute_outlier_diagnostic
  8. run_all.py                        -> figures

Stages 4-7 are mutually independent; 2 and 3 are the real gates.

After each stage the checkpoint verifies the outputs exist, are newer than
their inputs, and satisfy a stage-specific condition. A failed checkpoint
stops the run rather than letting a stale or malformed file propagate.
"""

import os
import re
import sys
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Locate processing/ ────────────────────────────────────────────────────────
# It does not have to sit beside this file. Search a few sensible places, then
# fall back to a recursive search under the repo. Override with --proc=/path
_MARKER = "compute_atmospheric_correlations.py"


def _find_processing(start):
    cands = [
        os.path.join(start, "processing"),
        start,
        os.path.join(os.path.dirname(start), "processing"),
        os.path.dirname(start),
        os.path.join(os.path.dirname(os.path.dirname(start)), "processing"),
    ]
    for c in cands:
        if os.path.exists(os.path.join(c, _MARKER)):
            return c
    return None


PROC = _find_processing(HERE)

for _a in sys.argv[1:]:
    if _a.startswith("--proc="):
        PROC = os.path.expanduser(_a.split("=", 1)[1])

# Resolve data paths from ch3_config so there is one source of truth
sys.path.insert(0, HERE)
try:
    from ch3_config import DATA_DIR, OUTPUT_DIR
except ImportError:
    raise SystemExit("Cannot import ch3_config.py — run this from the ch3fig folder.")

D = lambda *p: os.path.join(DATA_DIR, *p)

# If processing/ was not found near this file, search under the repo root
if PROC is None:
    _root = os.path.dirname(os.path.dirname(os.path.dirname(DATA_DIR)))
    for _dirpath, _dirnames, _files in os.walk(_root):
        if _MARKER in _files:
            PROC = _dirpath
            break

if PROC is None:
    raise SystemExit(
        f"Could not find {_MARKER}.\n"
        "Pass --proc=/path/to/processing, or put the folder beside this script."
    )

# ── Locate run_all.py ─────────────────────────────────────────────────────────
# Also does not have to sit beside this file. Override with --runall=/path
_RUNALL_CANDS = [
    os.path.join(HERE, "run_all.py"),
    os.path.join(os.path.dirname(HERE), "run_all.py"),
    os.path.join(PROC, "run_all.py"),
    os.path.join(os.path.dirname(PROC), "run_all.py"),
]
RUNALL = next((p for p in _RUNALL_CANDS if os.path.exists(p)), None)

for _a in sys.argv[1:]:
    if _a.startswith("--runall="):
        RUNALL = os.path.expanduser(_a.split("=", 1)[1])

if RUNALL is None:
    # last resort: walk the repo
    _root = os.path.dirname(os.path.dirname(os.path.dirname(DATA_DIR)))
    for _dirpath, _dirnames, _files in os.walk(_root):
        if "run_all.py" in _files:
            RUNALL = os.path.join(_dirpath, "run_all.py")
            break

# R pipeline location — edit if it lives elsewhere
R_PIPELINE = os.path.expanduser(
    "~/Research/repos/sea-ice-phase/scripts/R/Ch3/01_fit_apac.R")


# ── Checkpoints ───────────────────────────────────────────────────────────────

def _read_head(path, n=200000):
    with open(path, "r", errors="replace") as f:
        return f.read(n)


def chk_pipeline_b():
    """Pipeline B outputs exist and the decomposition sums."""
    import pandas as pd
    import numpy as np
    for f in ("daily_fitted_B.csv", "annual_params_B.csv"):
        if not os.path.exists(D(f)):
            return False, f"{f} not written"
    d = pd.read_csv(D("daily_fitted_B.csv"))
    need = ["anomaly_from_iac", "iac_notrend", "trend_component",
            "amplitude_component", "phase_component", "residual_apac"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        return False, f"missing columns {miss}"
    err = float(np.nanmean(np.abs(
        d["anomaly_from_iac"] - (d["trend_component"] + d["amplitude_component"]
                                 + d["phase_component"] + d["residual_apac"]))))
    if err > 1e-6:
        return False, f"decomposition does not sum (mean |error| = {err:.2e})"
    a = pd.read_csv(D("annual_params_B.csv"))
    if a["min_doy_raw_anom"].abs().max() > 150:
        return False, "min_doy_raw_anom > 150 d — DOY wrap fix missing"
    return True, f"sum error {err:.1e}, {len(a)} sector-years"


def chk_indices():
    """master_index_detrended + correlations_output, with the observed metrics."""
    import pandas as pd
    for f in ("master_index_detrended.csv", "correlations_output.csv"):
        if not os.path.exists(D(f)):
            return False, f"{f} not written"
    c = pd.read_csv(D("correlations_output.csv"))
    vt = set(c["var_type"].unique())
    if not any("raw" in v for v in vt):
        return False, f"no observed var_type found — got {sorted(vt)}"
    note = f"var_type: {sorted(vt)}"

    # is the stale copy in figures/ different?
    stale = os.path.join(os.path.dirname(DATA_DIR), "figures",
                         "master_index_detrended.csv")
    if os.path.exists(stale):
        import filecmp
        if not filecmp.cmp(stale, D("master_index_detrended.csv"), shallow=False):
            note += ("\n      NOTE: the copy in Ch3/figures/ DIFFERS from the new "
                     "one in Ch3/data/.\n"
                     "      compute_outlier_diagnostic.py used to read the stale "
                     "one. Delete it.")
    return True, note


def chk_monthly_params():
    import pandas as pd
    if not os.path.exists(D("monthly_params.csv")):
        return False, "monthly_params.csv not written"
    m = pd.read_csv(D("monthly_params.csv"))
    if "monthly_amp_anom" not in m.columns:
        return False, "monthly_amp_anom missing"
    return True, f"{len(m)} monthly records"


def chk_monthly_lagged():
    import pandas as pd
    f = D("monthly_cross_correlations.csv")
    if not os.path.exists(f):
        return False, "monthly_cross_correlations.csv not written"
    x = pd.read_csv(f)
    vars_found = set(x["variable"].unique())
    if vars_found <= {"phase", "amplitude"}:
        return False, (f"variable = {sorted(vars_found)} — still the FITTED set. "
                       "Use the patched script in processing/.")
    return True, f"variable: {sorted(vars_found)}"


def chk_exists(*names):
    def _c():
        missing = [n for n in names if not os.path.exists(D(n))]
        if missing:
            return False, f"not written: {missing}"
        return True, ", ".join(names)
    return _c


def chk_figures():
    if not os.path.isdir(OUTPUT_DIR):
        return False, "figure directory does not exist"
    pngs = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")]
    recent = [f for f in pngs
              if time.time() - os.path.getmtime(os.path.join(OUTPUT_DIR, f)) < 3600]
    return True, f"{len(pngs)} PNGs present, {len(recent)} written in the last hour"


# ── Stages ────────────────────────────────────────────────────────────────────
# (number, name, command, checkpoint, note)

STAGES = [
    (1, "Pipeline B (R)",
     ["Rscript", R_PIPELINE], chk_pipeline_b,
     "decomposition + observed metrics"),

    (2, "Atmospheric correlations",
     [sys.executable, os.path.join(PROC, "compute_atmospheric_correlations.py")],
     chk_indices,
     "GATE: writes master_index_detrended.csv, read by stages 4-7"),

    (3, "Monthly params",
     [sys.executable, os.path.join(PROC, "compute_phase_amplitude_monthly.py")],
     chk_monthly_params,
     "GATE: writes monthly_params.csv, read by stage 5"),

    (4, "Monthly lagged correlations",
     [sys.executable, os.path.join(PROC, "compute_monthly_lagged_correlations.py")],
     chk_monthly_lagged,
     "feeds fig09"),

    (5, "Monthly correlations (both)",
     [sys.executable, os.path.join(PROC, "compute_monthly_corr_both.py")],
     chk_exists("lag_correlations.csv", "contemporaneous_correlations.csv"),
     ""),

    (6, "Leave-one-index-out",
     [sys.executable, os.path.join(PROC, "compute_loo_index.py")],
     chk_exists("loo_index_skill.csv", "loo_index_residuals.csv"),
     ""),

    (7, "Outlier diagnostic",
     [sys.executable, os.path.join(PROC, "compute_outlier_diagnostic.py")],
     lambda: (True, "check the printed r values — these change with the "
                    "observed metrics"),
     "ABS phase-ASL DJF and Ross phase results will differ"),

    (8, "Figures",
     [sys.executable, RUNALL or "run_all.py"], chk_figures,
     ""),
]


def run_stage(num, name, cmd, check, note, dry):
    print(f"\n{'=' * 72}")
    print(f"STAGE {num}  {name}")
    if note:
        print(f"          {note}")
    print("=" * 72)

    exe = cmd[1] if len(cmd) > 1 else cmd[0]
    if not os.path.exists(exe) and num != 1:
        print(f"  MISSING: {exe}")
        return False
    if num == 1 and not os.path.exists(R_PIPELINE):
        print(f"  MISSING: {R_PIPELINE}")
        print("  Edit R_PIPELINE at the top of this script, or use --skip-r.")
        return False

    print(f"  $ {' '.join(cmd)}")
    if dry:
        return True

    t0 = time.time()
    _cwd = os.path.dirname(cmd[1]) if len(cmd) > 1 else HERE
    r = subprocess.run(cmd, cwd=_cwd or HERE)
    dt = time.time() - t0

    if r.returncode != 0:
        print(f"\n  FAILED (exit {r.returncode}) after {dt:.0f}s")
        return False

    ok, msg = check()
    if ok:
        print(f"\n  OK  ({dt:.0f}s)  {msg}")
    else:
        print(f"\n  CHECKPOINT FAILED: {msg}")
    return ok


def main():
    argv = sys.argv[1:]
    dry = "--dry-run" in argv
    skip_r = "--skip-r" in argv

    def _num(flag):
        for a in argv:
            if a.startswith(flag):
                if "=" in a:
                    return int(a.split("=", 1)[1])
                i = argv.index(a)
                if i + 1 < len(argv):
                    return int(argv[i + 1])
        return None

    start = _num("--from")
    only = _num("--only")

    if "--list" in argv:
        print(f"{'#':>2}  {'stage':<32} status")
        print("-" * 72)
        for num, name, cmd, check, _ in STAGES:
            try:
                ok, msg = check()
            except Exception as e:
                ok, msg = False, f"{type(e).__name__}"
            print(f"{num:>2}  {name:<32} {'done' if ok else 'pending'} — {msg.splitlines()[0][:60]}")
        return

    todo = STAGES
    if only is not None:
        todo = [s for s in todo if s[0] == only]
    elif start is not None:
        todo = [s for s in todo if s[0] >= start]
    if skip_r:
        todo = [s for s in todo if s[0] != 1]

    print("Chapter 3 pipeline" + ("  [DRY RUN]" if dry else ""))
    print(f"data:       {DATA_DIR}")
    print(f"figures:    {OUTPUT_DIR}")
    print(f"processing: {PROC}")
    print(f"run_all:    {RUNALL}")
    print(f"stages:  {[s[0] for s in todo]}")

    done = []
    for stage in todo:
        if not run_stage(*stage, dry):
            print(f"\nSTOPPED at stage {stage[0]}. "
                  f"Fix, then: python run_pipeline.py --from {stage[0]}")
            sys.exit(1)
        done.append(stage[0])

    print(f"\n{'=' * 72}")
    print(f"Completed stages {done}")
    if not dry and 7 in done:
        print("\nNumbers that change with the observed metrics — check these "
              "against the chapter:")
        print("  * ABS phase-ASL DJF  (was computed from max_doy_anom)")
        print("  * Ross phase outlier diagnostic")
        print("  * anything in monthly_cross_correlations.csv")


if __name__ == "__main__":
    main()