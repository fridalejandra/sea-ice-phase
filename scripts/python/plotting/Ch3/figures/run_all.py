"""
run_all.py -- regenerate the Ch3 pipeline and figures in manuscript order.
============================================================================
Replaces the previous run_all.py, which listed only the old rolling-window
/ monthly-lag-correlation figure scripts and never called the current
pipeline (build_ch3_figures.py, fig04_attribution_annual.py,
compute_component_comparison.py, t34_coupling_shift_test.py) at all -- see
the archived copy for what it used to do.

    python run_all.py            # everything available, in dependency order
    python run_all.py fig03      # one step (matches on script filename --
                                  #   e.g. fig03_attribution_annual.py; see
                                  #   the note at that STEPS entry below
                                  #   about its still-unresolved figure number)
    python run_all.py --list     # show every step, its manuscript figure,
                                  #   and whether its inputs exist yet

Honesty about what's confirmed vs inferred: every script/table relationship
below either came up directly in this chapter's own working session (build
was read in full, plot_fig7_sectors.R was read and run, compute_fig11_
component_comparison.py, check_timing_variance_shift.py, check_coupling_
shift.py, and check_component_share_trend.py were all read in full and
confirmed live) or is a reasonable inference from ch3_config.py's own paths
and comments. One thing is not independently confirmed and is marked below
rather than asserted: the full table-by-table output list of ch3_stats.py,
which is treated here as "run it, then check --list again" rather than
enumerated.

RESOLVED 2026-09-18: the previous version of this docstring flagged
05_volatility_gamlss.R's relationship to ch3_numbers.csv section 3.4c as
"unverified." It's now verified, and worse than unverified -- it was
silently wrong. 01_fit_apac.R's own tail end ("SECTION 7", its own comment:
"ported from 05_volatility_gamlss.R (Handcock's suggestion, email Sep
2026)") is a newer, corrected port of the exact same GAMLSS volatility
model, and it already runs as part of the 01_fit_apac.R step above -- no
separate step needed. Both SECTION 7 and the standalone
05_volatility_gamlss.R write to the SAME two files
(t34c_volatility_gamlss_post2016.csv, t34c_volatility_seasonal_curves.csv),
but 05_volatility_gamlss.R hardcodes `Year <= 2023` (its own line 48) while
SECTION 7 uses `Date <= PERIODS[["FULL"]]`, the same dynamic full-record end
date used everywhere else in this pipeline (currently 2025-12-31). Since
05_volatility_gamlss.R used to run as a later STEPS entry, its stale,
2-year-shorter output was silently overwriting SECTION 7's correct output on
every full pipeline run. Confirmed from Frida's real run: SECTION 7
(83202 daily rows, 1988-2025) gave SIE_Ross dSIE post2016 x1.632 (63% MORE
volatile after 2016); the standalone script (78816 rows, 1988-2023) gave
x0.919 (8% LESS volatile) for the identical quantity -- a qualitatively
opposite reading, not a rounding difference. The row-count gap (4386) is
almost exactly 2 extra years x 6 sectors x ~365.5 days. Fix: the
05_volatility_gamlss.R STEPS entry below has been removed (SECTION 7
already covers it, correctly, every time 01_fit_apac.R runs); the standalone
file itself should be archived as superseded -- see
archive_volatility_gamlss.sh. Any Fig 8 / section 3.4c prose already
written from the old on-disk numbers needs to be re-checked against a fresh
pipeline run.

Renamed 2026-09-18 (paths below updated to match, logic in every file
unchanged): t33_variance_prepost.py -> check_timing_variance_shift.py;
t34_coupling_shift_test.py -> check_coupling_shift.py;
test_component_dominance.py -> table2_component_dominance.py (it's the
script behind Table 2, so this ties the name to that); test_trend_share_
over_time.py -> check_component_share_trend.py; compute_component_
comparison.py -> compute_fig11_component_comparison.py (ties it to the
figure it feeds). Confirmed table2_component_dominance.py's mention of the
other two check_*.py scripts is prose only (a "same convention as..."
comment), not a real import -- so the rename order across these four
doesn't matter functionally, only for the comment text staying accurate.
Output table/figure filenames (t33_variance_prepost.csv,
t34_coupling_shift_test.csv, etc.) were deliberately NOT renamed -- those
are tracked in _provenance_audit.csv, a separate, higher-stakes decision
than tidying up script names.

04_chapter_analyses.R (in R/Ch3) IS included below, as of 2026-09-18 -- it
was investigated, not assumed. It used to read daily_fitted_E.csv /
annual_params_E.csv, an old "Pipeline-E" data variant that turned out not
to exist on disk anymore (confirmed via `ls data/ch3/*.csv`) -- the script
would have failed outright with file-not-found if run as found. The
provenance audit (results/ch3/tables/_provenance_audit.csv) confirmed it is
still the sole, current writer of s335_residual_variability.csv and
s34_internal_correlations.csv -- nothing else in the pipeline produces
those §3.3.5/§3.4 numbers, so this was a real, broken gap, not dead code.
Fixed to read the canonical daily_fitted.csv / annual_params.csv (filtered
to period == "FULL", the same overlap bug already fixed elsewhere in this
chapter) and to fail loudly if either file is missing an expected column,
since the daily-side columns (residual_apac, volatility) weren't
independently re-verified this pass. Its own external-index-correlation
block stays disabled (if (FALSE && ...)), per its own comment that this was
already migrated to ch3_stats.py.

R steps run via `Rscript`; Python steps run in-process via runpy (so a
failure prints a normal Python traceback, not a subprocess dump).
"""

import os
import sys
import runpy
import subprocess
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import (
    ROOT, DAILY_CSV, ANNUAL_CSV, INDEX_CSV, NUMBERS_CSV, TABLES_DIR,
)

R_DIR = os.path.join(ROOT, "scripts", "R", "Ch3")  # confirmed via `pwd` in-session

# processing/'s exact nesting under scripts/python wasn't independently
# confirmed the way R_DIR and HERE (this file's own directory) were -- HERE
# turned out to be scripts/python/plotting/Ch3/figures, not scripts/python/
# figures as first guessed (see reorg_ch3_scripts.sh's fix), so rather than
# repeat that mistake, this searches for it instead of asserting a path.
def _find_processing_dir():
    search_root = os.path.join(ROOT, "scripts", "python")
    hits = []
    for dirpath, dirnames, _ in os.walk(search_root):
        # don't descend more than ~3 levels below scripts/python -- this repo
        # doesn't nest deeper than that anywhere we've seen
        depth = dirpath[len(search_root):].count(os.sep)
        if depth >= 3:
            dirnames[:] = []
            continue
        if os.path.basename(dirpath).lower() == "processing":
            hits.append(dirpath)
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        print(f"WARNING: no 'processing' directory found under {search_root} -- "
              f"steps that need it will report 'no script'.")
    else:
        print(f"WARNING: multiple 'processing' directories found under {search_root}: "
              f"{hits} -- picking the first; check this is right.")
    return hits[0] if hits else os.path.join(search_root, "processing")


PROC_DIR = _find_processing_dir()

t = lambda name: os.path.join(TABLES_DIR, name)

# (script path, runner, human name / manuscript figure, required input files)
STEPS = [
    # ---- precursor pipeline: fit, then atmospheric indices, then the stats ledger
    (os.path.join(R_DIR, "01_fit_apac.R"), "R",
     "fit APAC (writes daily_fitted.csv, annual_params.csv)", []),
    (os.path.join(PROC_DIR, "compute_atmospheric_correlations.py"), "python",
     "atmospheric indices -> master_index_detrended.csv", [DAILY_CSV, ANNUAL_CSV]),
    (os.path.join(HERE, "ch3_stats.py"), "python",
     "chapter statistics ledger (ch3_numbers.csv + most t3x_*.csv tables)",
     [DAILY_CSV, ANNUAL_CSV, INDEX_CSV]),
    # 05_volatility_gamlss.R REMOVED 2026-09-18: it duplicated 01_fit_apac.R's
    # own "SECTION 7" (a newer, corrected port of the same GAMLSS model, using
    # the dynamic PERIODS[["FULL"]] end date instead of this script's
    # hardcoded Year <= 2023), and both wrote to the same two output files --
    # this stale step ran later and was silently overwriting SECTION 7's
    # correct, current output. See the docstring above and
    # archive_volatility_gamlss.sh. Fig 8's day-to-day volatility numbers now
    # come entirely from the 01_fit_apac.R step above.
    (os.path.join(HERE, "compute_fig11_component_comparison.py"), "python",
     "raw/amplitude/phase/residual comparison -> t37_component_comparison.csv (Fig 11)",
     [t("t35_primary_pairs.csv")]),
    (os.path.join(R_DIR, "04_chapter_analyses.R"), "R",
     "Sec 3.3.5 residual variability + Sec 3.4 internal seasonal-SIE "
     "correlations -> s335_residual_variability.csv, s34_internal_correlations.csv "
     "(fixed 2026-09-18: was reading nonexistent _E data files)",
     [DAILY_CSV, ANNUAL_CSV]),

    # ---- figures, in manuscript order ----
    # NAMES CORRECTED 2026-09-18: this file previously guessed
    # fig_01_conflation.py / fig_02_sector_map.py for these two steps
    # (flagged "(?)" -- never independently confirmed against your repo).
    # You sent the real files: they're actually named
    # fig01_concept_manuscript.py and fig_sector_map.py (no "02_" prefix on
    # the second one). Both would have silently reported "no script" and
    # been skipped on every run until now. Paths below fixed to match; the
    # figure numbering itself (Fig 1 = concept, Fig 2 = sector map) was
    # already baked into each script's own output filename
    # (fig01_concept_manuscript.png / fig02_sector_map.png), so that part
    # wasn't actually in question -- just double-check you're still happy
    # with that as the Fig 1/Fig 2 order.
    (os.path.join(HERE, "fig01_concept_manuscript.py"), "python",
     "Fig 1: phase-vs-amplitude conflation concept (schematic)", []),
    (os.path.join(HERE, "fig_sector_map.py"), "python",
     "Fig 2: sector map (polar + rectangular panels; needs cartopy)", []),
    # NAME CORRECTED 2026-09-18: this entry pointed at fig_04_attribution_annual.py
    # for a while, on the assumption a prior rename had gone through. It
    # hadn't -- the real file Frida sent is still named
    # fig03_attribution_annual.py (its own docstring, and its two PNG outputs
    # fig03_attribution_by_cycle.png / fig03_attribution_era_shares.png, all
    # still say "03"). Path fixed below so this step actually runs again.
    # RESOLVED 2026-09-18 (Frida's call): Fig 3 = fig03_attribution_by_cycle.png,
    # Fig 4 = fig03_attribution_era_shares.png -- KEPT in the main text
    # (reverses ch3_section32_draft.md's earlier "drop" note and its old
    # "Figure 8" label; see the RESOLVED block in that file). The era-shares
    # panel earns its keep because the same NET-vs-GROSS contrast it shows is
    # what backs the trend-share finding: phase nets out to ~0 under NET by
    # construction (a timing shift cancels within a cycle), so GROSS is the
    # fairer read on whether phase's role is shrinking.
    (os.path.join(HERE, "fig03_attribution_annual.py"), "python",
     "Figs 3 + 4: year-by-year attribution bars (fig03_attribution_by_cycle.png) "
     "and NET-vs-GROSS era shares (fig03_attribution_era_shares.png) -- writes "
     "t32_attribution_by_cycle*.csv, t32_component_dominance_era.csv", [DAILY_CSV]),
    # NEW 2026-09-18 (Frida's call): rolling phase-amplitude correlation figure
    # -- was floated earlier as "should this be Fig 3?" but Fig 3 is now the
    # attribution-by-cycle figure above, so this is Fig 5, placed in Sect. 3.3
    # (the section after the Fig 3/4 attribution results) rather than Sect. 3.1.
    (os.path.join(HERE, "fig03_rolling_phase_amp_corr.py"), "python",
     "Fig 5: trailing 10-yr rolling Spearman correlation, observed timing vs. "
     "amplitude anomalies, by sector -- writes t33_phase_amp_era_split.csv "
     "(NOTE: filename still says fig03_ -- that's the old working name from "
     "before Fig 3/4/5 were sorted out; left alone since nothing else "
     "references it by name and a rename risks breaking that assumption)",
     [ANNUAL_CSV]),
    # NOTE 2026-09-18: this script's circumpolar HR20-reproduction panel
    # (fig07a_circumpolar_2016.png) is no longer cited as Fig S1 -- Frida
    # replaced it with fig_s01_fitted_vs_observed.py below (broader check,
    # all 6 sectors, fitted vs observed amp/timing rather than a digitized
    # comparison to HR20's own published figure). Still run this step: its
    # six-panel sector grid (fig07_sectors_2016.png) was floated as "candidate
    # Fig 3" earlier, but Fig 3 is now definitively fig03_attribution_by_cycle.png
    # (Frida's call, 2026-09-18) -- so this grid is STILL UNASSIGNED a
    # manuscript number and needs a fresh decision (drop it, fold into an
    # existing figure, or give it its own number after Fig 11). Separately,
    # what happens to the old S1 panel itself (drop it, or give it a new
    # supplementary number) is also still undecided -- see the tenth-pass
    # note in ch3_paper_draft_v2.md.
    (os.path.join(R_DIR, "plot_fig7_sectors.R"), "R",
     "UNASSIGNED (was 'candidate Fig 3', now needs a new number -- see note "
     "above): 6-panel sector-anatomy grid; also still writes the old "
     "fig07a_circumpolar_2016.png panel, no longer cited as Fig S1",
     [DAILY_CSV]),
    (os.path.join(HERE, "fig_s01_fitted_vs_observed.py"), "python",
     "Fig S1 (replaces the old HR20-reproduction panel): fitted vs. "
     "observed amplitude + timing, all 6 sectors -- backs the r>=0.99 / "
     "variance-ratio claims in Sec 3.2 (VERIFIED 2026-09-18 against Frida's "
     "real annual_params.csv: fitted-side columns are amplitude_anom/"
     "max_doy_anom/min_doy_anom, not the absolute-scale *_fitted columns "
     "originally guessed -- confirmed day-of-min r>=0.99 as claimed)",
     [ANNUAL_CSV]),
    # build_ch3_figures.py (one file, six figures) was split 2026-09-18 into
    # the six fig_##_name.py scripts below, one per manuscript figure --
    # the archived original is kept for reference, not run from here.
    (os.path.join(HERE, "fig_06_coupling_pooled.py"), "python",
     "Fig 6: timing-amplitude coupling, pooled + per-sector, split at 2016",
     [t("t33_phase_amp_splits.csv")]),
    (os.path.join(HERE, "fig_07_abs_growth_season.py"), "python",
     "Fig 7: ABS amplitude vs. growth-season length, 2016-2023", [ANNUAL_CSV]),
    (os.path.join(HERE, "fig_08_volatility_raw_vs_residual.py"), "python",
     "Fig 8: day-to-day volatility ratio, raw + APAC residual (needs ch3_numbers.csv "
     "section 3.4c current -- now written by 01_fit_apac.R's SECTION 7 only, "
     "see that script's own docstring)", [NUMBERS_CSV]),
    (os.path.join(HERE, "fig_09_ross_asl_nonstationarity.py"), "python",
     "Fig 9: Ross-ASL non-stationarity, by season + split-year sweep",
     [t("t36_ross_asl_detail.csv")]),
    (os.path.join(HERE, "fig_10_atmosphere_sevenpairs.py"), "python",
     "Fig 10: seven pre-specified atmosphere-component pairs", [t("t35_primary_pairs.csv")]),
    (os.path.join(HERE, "fig_11_component_comparison_heatmap.py"), "python",
     "Fig 11: raw/amplitude/phase/residual comparison heatmap",
     [t("t37_component_comparison.csv")]),

    # ---- optional: Table 2 + supplementary statistical checks, not
    # figures, meant to be run once with console output pasted into the
    # draft rather than regenerated silently -- run last ----
    (os.path.join(HERE, "table2_component_dominance.py"), "python",
     "Table 2: magnitude-dominance + variance-share decomposition (optional)",
     [t("t32_attribution_by_cycle.csv"), t("t32_attribution_by_cycle_gross.csv")]),
    (os.path.join(HERE, "check_timing_variance_shift.py"), "python",
     "Sec 3.3 supplementary: pre/post-2016 variance cross-check, observed vs. "
     "fitted phase (paste console output into the text)",
     [t("t32_attribution_by_cycle.csv"), t("t32_attribution_by_cycle_gross.csv")]),
    (os.path.join(HERE, "check_coupling_shift.py"), "python",
     "Sec 3.3 supplementary: cross-sector timing-amplitude coupling shift test "
     "(Fisher z + Stouffer + sign test; paste console output into the text)",
     [ANNUAL_CSV]),
    (os.path.join(HERE, "check_component_share_trend.py"), "python",
     "Supplementary: boundary-free component-share trend test, NET+GROSS, "
     "pooled across sectors (paste console output into the text)",
     [t("t32_attribution_by_cycle.csv"), t("t32_attribution_by_cycle_gross.csv")]),
    (os.path.join(HERE, "audit_ch3_correlations.py"), "python",
     "cross-check every quoted correlation against its source table (optional QA)", []),
]


def status(reqs, script):
    if not os.path.exists(script):
        return "no script"
    missing = [os.path.basename(r) for r in reqs if not os.path.exists(r)]
    return "missing: " + ", ".join(missing) if missing else "ready"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if "--list" in sys.argv:
        print(f"{'step':<70} {'runner':<7} status")
        print("-" * 110)
        for script, runner, name, reqs in STEPS:
            print(f"{name:<70} {runner:<7} {status(reqs, script)}")
        return

    todo = [s for s in STEPS if not args or any(a in os.path.basename(s[0]) for a in args)]
    ok, skipped, failed = [], [], []

    for script, runner, name, reqs in todo:
        st = status(reqs, script)
        if st != "ready":
            print(f"SKIP  {name:<70} ({st})")
            skipped.append(name)
            continue
        print(f"\n=== {name} ===")
        try:
            if runner == "R":
                subprocess.run(["Rscript", script], check=True, cwd=os.path.dirname(script))
            else:
                runpy.run_path(script, run_name="__main__")
            ok.append(name)
        except SystemExit as e:
            print(f"STOPPED: {e}")
            failed.append(name)
        except subprocess.CalledProcessError as e:
            print(f"R SCRIPT FAILED (exit {e.returncode}): {script}")
            failed.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)

    print("\n" + "=" * 70)
    print(f"built {len(ok)}   skipped {len(skipped)}   failed {len(failed)}")
    if failed:
        print("failed: " + ", ".join(failed))
    if skipped:
        print("skipped: " + ", ".join(skipped))


if __name__ == "__main__":
    main()