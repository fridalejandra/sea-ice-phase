# Chapter 3 Figures

This directory contains all figure scripts for Chapter 3.
Figure numbers refer to the current thesis draft.

Supplementary figures live in `figures/supplementary/`.
Shared style constants and helpers are in `utils/ch3_style.py`.

---

## Main Figures

| Doc # | Script                                | Description                                                                                                | Status |
|-------|---------------------------------------|------------------------------------------------------------------------------------------------------------|--------|
| 3.1   | `fig_concept_apac_framework.py`       | Conceptual schematic of the APAC phase/amplitude decomposition framework                                   | Done   |
| 3.2   | `fig_conflation_argument.py`          | SIE anomaly vs APAC method — shows how traditional anomaly conflates phase and amplitude signals           | IP     |
| 3.3   | `fig_rmse_validation.py`              | Sequential RMSE improvement by sector — Invariant → Amplitude → Phase → Amp+Phase                          | Done   |
| 3.4   | `fig_phase_amplitude_independence.py` | Phase and amplitude independence demonstrated in model output and observations                             | NEED   |
| 3.5   | `fig_phase_amplitude_timeseries.py`   | Phase and amplitude anomaly timeseries, 10-yr rolling variance, pre/post-2016 variability bars             | Done   |
| 3.6   | `fig_case_study_2016_2023.py`         | 2016 vs 2023 case study — z-scored phase and amplitude anomalies by sector                                 | Done   |
| 3.7   | `fig_correlation_heatmap.py`          | Pearson and Spearman correlation heatmap — phase/amplitude vs atmospheric indices (SAM, ZW3, ASL, Niño3.4) | Done   |
| 3.8   | `fig_phase_amplitude_scatter.py`      | Phase vs amplitude scatter — atmospheric driver relationships by sector                                    | Done   |
| 3.9   | `fig_rolling_window_correlations.py`  | 10-yr rolling window correlations — non-stationarity of atmospheric drivers post-2016                      | Done   |
| 3.10  | `fig_volatility_gamlss.py`            | Volatility and variance regime change — GAMLSS location and scale model fitted to phase and amplitude      | NEED   |

---

## Supplementary Figures

| Thesis # | Script                                  | Description                                                                                    | Status |
|----------|-----------------------------------------|------------------------------------------------------------------------------------------------|--------|
| S3.1     | `figS_rolling_window_comparison_raw.py` | Rolling window comparison — EA amplitude ~ SAM vs raw EA SIE ~ SAM (hypothesis B test)         | Done   |
| S3.2     | `figS_decomposition_2016_2023.py`       | Sector-level phase and amplitude decomposition for 2016 and 2023 individually                  | IP     |
| S3.3     | `figS_spearman_pearson_comparison.py`   | Spearman vs Pearson correlation comparison across sectors and indices                          | Done   |
| S3.4     | `figS_outlier_diagnostics.py`           | Outlier diagnostics — 2023 extremes, Weddell/Ross dipole, leverage and influence               | Done   |
| S3.5     | `figS_season_length.py`                 | Growth and retreat season length anomaly by sector (tabled from main figures)                  | Done   |
| S3.6     | `figS_block_bootstrap_CIs.py`           | Block bootstrap confidence intervals for key correlations — addresses temporal autocorrelation | NEED   |
| S3.7     | `figS_lagged_monthly_correlations.py`   | Shoulder season and monthly lagged correlations — peak atmospheric influence windows           | NEED   |

---

## Key Results Reference

For cross-referencing during writing. Update as analysis develops.

| Result                       | Value                            | Figure    |
|------------------------------|----------------------------------|-----------|
| EA amplitude ~ SAM annual    | r = +0.47                        | 3.7       |
| EA amplitude ~ SAM pre-2016  | r = +0.617                       | 3.9       |
| EA amplitude ~ SAM post-2016 | r = +0.254                       | 3.9       |
| Ross amplitude ~ ZW3 annual  | r = −0.41 (stable)               | 3.7       |
| ABS phase ~ ASL DJF          | r = −0.36 (non-stationary)       | 3.7, 3.9  |
| Weddell amplitude 2023       | +1.75σ                           | 3.6, S3.4 |
| Ross amplitude 2023          | −2.61σ (most extreme in dataset) | 3.6, S3.4 |

---

## Processing Scripts

Figure scripts load precomputed results from `Ch3/processing/`.
No figure script should compute results from raw data directly (messy)!!

| Script                                | Outputs                                   | Used by        |
|---------------------------------------|-------------------------------------------|----------------|
| `compute_atmospheric_correlations.py` | correlation matrices, r values, p values  | 3.7, 3.8, S3.3 |
| `compute_rolling_correlations.py`     | rolling r timeseries by sector/index pair | 3.9, S3.1      |
| `compute_outlier_diagnostics.py`      | z-scores, leverage, Cook's D              | S3.4           |
| `compute_phase_amplitude.py`          | annual_params.csv, daily_fitted.csv       | 3.2–3.6, 3.10  |
| `block_bootstrap_CIs.py`              | bootstrap CI table (.csv)                 | S3.6, 3.7      |

---

*Last updated: 2026-05*