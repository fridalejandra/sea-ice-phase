# Chapter 2: Phase Sensitivity Analysis – Task Plan (Due Tuesday)

## Data & Code Prep
- [ ] Ensure `.gitignore` excludes cluster outputs (`data/`, `results/`, `*.log`, `*.out`).
- [ ] Verify cluster repo is on `feat/fsms-wrapper` branch and synced with GitHub.
- [ ] Confirm `advance_retreat_smmr.py` runs to completion on cluster with nohup/logging.

## Sensitivity Runs
- [ ] Threshold sensitivity (FS/MS at k=5):
  - [ ] Run thresholds = 10%, 15%, 30%.
  - [ ] Save yearly NetCDF outputs to `results/phase/SMMR/FS_thrXX_k5/` and `MS_thrXX_k5/`.
- [ ] Window sensitivity (FS/MS at thr=15%):
  - [ ] Run windows = 3, 5, 7 days.
  - [ ] Save yearly NetCDF outputs to `results/phase/SMMR/FS_thr15_kK/` and `MS_thr15_kK/`.

## Post-processing
- [ ] Compute Δ using circular day-of-year difference (shortest signed difference).
- [ ] Summarize distributions:
  - [ ] Median, IQR, %(|Δ|>5).
  - [ ] Year-wise mean, std, %(|Δ|>5).
- [ ] Generate maps:
  - [ ] Median Δ across years.
  - [ ] Fraction of years with |Δ|>5.
- [ ] Generate distributions:
  - [ ] Violin plots (Δ for FS/MS, 3–5 vs 7–5).
  - [ ] Histograms of Δ.

## Writing
- [ ] Finalize Methods subsection (static definition + sensitivity design).
- [ ] Draft Results subsections with placeholders for numbers/figures:
  - [ ] Threshold sensitivity narrative.
  - [ ] Window sensitivity narrative.
- [ ] Cross-reference figures and YAML config for reproducibility.

## Deliverables (by Tuesday)
- [ ] All NetCDF outputs saved in `results/phase/SMMR/`.
- [ ] Figures: violin plots + maps of sensitivity.
- [ ] Methods text integrated in Chapter 2 draft.
- [ ] Results text drafted with placeholders.
- [ ] `config/phase_static.yml` committed in repo.
