# Ch3 figure package

Fresh rewrite. Every figure reads Pipeline B output through a shared data
layer that validates before plotting, so a broken or stale input fails loudly
instead of producing a plausible-looking wrong figure.

## Layout

    ch3_config.py   paths, sectors, colours, constants — change things HERE
    ch3_data.py     loading, validation, derived quantities
    ch3_plot.py     shared layout helpers (sector grids, break marker, save)
    ch3_style.py    YOUR existing style module — unchanged, not in this package
    run_all.py      build everything, or one figure, or show status

## Use

    python run_all.py --list     # what exists, what is blocked and why
    python run_all.py            # build everything available
    python run_all.py fig04      # build one

## Guards

`load_daily()` refuses Pipeline A output: it checks the decomposition columns
exist and that they sum to `anomaly_from_iac` within 1e-6. Under Pipeline B
the error is ~1e-19; under Pipeline A the trend was double-counted and
`phase_component` was built from the wrong curve.

`load_annual()` refuses input where `min_doy_raw_anom` exceeds 150 days, which
indicates the DOY wrap fix is missing (Pipeline A gave Weddell 1980 and 2005
an anomaly of +305).

`load_correlations()` warns when a correlation CSV is older than
`annual_params_B.csv`, i.e. when it still encodes the old anomalies.

## Metric policy

The chapter uses OBSERVED quantities throughout: `min_doy_raw_anom`,
`max_doy_raw_anom`, `amplitude_raw_anom`. Fitted timing (`max_doy_fitted`)
appears only in S01, as the evidence for why it is not used.

## Status

Ready — need only `daily_fitted_B.csv` / `annual_params_B.csv`:

    fig03  rolling r(timing, amplitude)
    fig04  observed timing timeseries
    fig04b component dominance timeline   (rebuilt; original script was lost)
    fig05  observed amplitude timeseries
    fig06  rolling 10-yr SD
    fig07  pre/post-2016 SD bars
    S01    fitted vs observed timing
    S05    decomposition, 2016 and 2023
    S06    case-study z-scores

Blocked — need the correlation pipeline rerun on `annual_params_B.csv` first,
because the existing CSVs were built from the old anomalies:

    fig08  index correlation heatmap
    fig09  monthly lag correlations
    fig10  rolling index correlations

Not yet written: fig01 (conflation schematic), fig02 (sector map),
S02, S03, S04, S07, S08.

## Compute pipeline

`ch3_config.py` now covers the `compute_*.py` scripts too, not just figures.
`patch_compute_scripts.py` rewires them:

    python patch_compute_scripts.py --dry-run   # inspect
    python patch_compute_scripts.py             # apply (writes *.py.bak)

What it changes:

* **Metrics.** Four scripts correlated against `max_doy_anom` /
  `amplitude_anom` — the FITTED quantities. `max_doy_fitted` is the argmax of
  the fitted curve, dominated by the fixed `s(DOY)` term: SD 2–4× smaller than
  observed, r with the observed maximum only 0.02–0.46, and in Ross r with the
  observed MINIMUM is −0.74. Results resting on it include the ABS phase–ASL
  DJF finding and the Ross phase outlier diagnostic. Now switched to
  `max_doy_raw_anom`, `min_doy_raw_anom`, `amplitude_raw_anom`.
* **The minimum date.** No compute script had ever used it. Added, since the
  chapter reports min-date and max-date as a pair.
* **Path drift.** `compute_outlier_diagnostic.py` read
  `master_index_detrended.csv` from `Ch3/figures/` while every other script
  read it from `Ch3/data/` — it could silently use a stale copy. Fixed.
* **Inputs.** Repointed at `annual_params_B.csv` / `daily_fitted_B.csv`.
* **Monthly params.** `compute_phase_amplitude_monthly.py` now uses
  `iac_notrend` rather than `fitted_invariant` as its climatological
  reference, matching the Pipeline B decomposition.

Rerun order after patching:

    1. compute_atmospheric_correlations.py    -> master_index_detrended.csv
    2. compute_phase_amplitude_monthly.py     -> monthly_params.csv
    3. compute_monthly_lagged_correlations.py
    4. compute_monthly_corr_both.py
    5. compute_loo_index.py
    6. compute_outlier_diagnostic.py
    7. python run_all.py

`fig09` will refuse to plot until step 3 has been rerun — it inspects the
`variable` column and stops if it sees the fitted set.

## Running the whole thing

    python run_pipeline.py --list      # what is done, what is pending
    python run_pipeline.py             # everything, in dependency order
    python run_pipeline.py --from 3    # resume after a failure
    python run_pipeline.py --skip-r    # Pipeline B already ran
    python run_pipeline.py --dry-run   # show commands without running

Stage 1 needs `R_PIPELINE` at the top of `run_pipeline.py` to point at
`APAC_Sector_Pipeline_B.R`; everything else resolves from `ch3_config.py`.

Order is not arbitrary. Stage 2 writes `master_index_detrended.csv`, which
stages 4–7 read; stage 3 writes `monthly_params.csv`, which stage 5 reads.
Stages 4–7 are independent of each other.

Each stage has a checkpoint that runs after the command. A failed checkpoint
stops the run so a stale or malformed file cannot propagate. The checks:

  1. decomposition sums to ~1e-19; `min_doy_raw_anom` within ±150 d
  2. `correlations_output.csv` contains observed `var_type` values, and
     warns if the stale `Ch3/figures/master_index_detrended.csv` differs
     from the newly written one in `Ch3/data/`
  3. `monthly_params.csv` has `monthly_amp_anom`
  4. `variable` is NOT the fitted set — refuses `{phase, amplitude}`
  5–6. expected CSVs written
  8. counts PNGs written in the last hour

After stage 7, these numbers will have changed and need checking against the
chapter text: ABS phase–ASL DJF, the Ross phase outlier diagnostic, and
anything drawn from `monthly_cross_correlations.csv`.
