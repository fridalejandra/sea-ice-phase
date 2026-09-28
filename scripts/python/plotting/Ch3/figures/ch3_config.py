"""
ch3_config.py — single source of truth for paths, sectors, and constants.
=========================================================================
Imported by BOTH layers:
  compute_*.py  — the correlation / monthly pipeline
  fig*.py       — the figure package

Change a path once, not in nine files. Previously each compute script
hardcoded its own; they had drifted — compute_outlier_diagnostic.py read
master_index_detrended.csv from Ch3/figures/ while every other script read it
from Ch3/data/, so it could silently use a stale copy.
"""

import os

# ── Repo root ─────────────────────────────────────────────────────────────────
# The APAC work is local; the cluster paths are kept as a fallback. First
# existing candidate wins. Override with the SEAICE_ROOT environment variable:
#     export SEAICE_ROOT=/some/other/sea-ice-phase
_CANDIDATE_ROOTS = [
    os.environ.get("SEAICE_ROOT"),
    os.path.expanduser("~/Research/repos/sea-ice-phase"),
    "/Users/fridaperez/Research/repos/sea-ice-phase",
    "/user/geog/falejandraperez/sea-ice-phase",
]

ROOT = next((r for r in _CANDIDATE_ROOTS
             if r and os.path.isdir(os.path.join(r, "scripts"))), None)

if ROOT is None:
    raise SystemExit(
        "Could not locate the sea-ice-phase repo.\n"
        "Tried:\n  " + "\n  ".join(str(r) for r in _CANDIDATE_ROOTS if r) +
        "\nSet SEAICE_ROOT, or edit _CANDIDATE_ROOTS in ch3_config.py."
    )

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(ROOT, "scripts", "R", "Ch3", "data")
INDEX_DIR  = os.path.join(ROOT, "data", "indices")
OUTPUT_DIR = os.path.join(ROOT, "scripts", "python", "plotting", "Ch3", "figures")
GDRIVE     = "gdrive:sea-ice-phase/results/Ch3_Figures/"

# Raw index files (all read from INDEX_DIR)
INDEX_FILES = {
    "SAM":    "marshall_sam_monthly.txt",
    "Nino34": "nina34.data",
    "ASL":    "asli_era5_v3-latest.csv",
    "ZW3R":   "ZW3_raphael_monthly.csv",
    "ZW3G":   "ZW3_goyal_monthly.csv",
}

# Pipeline B outputs. Rename these two constants (drop "_B") once you have
# renamed the files on disk — nothing else in the package needs touching.
DAILY_CSV  = os.path.join(DATA_DIR, "daily_fitted_D.csv")
ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params_D.csv")
RMSE_CSV   = os.path.join(DATA_DIR, "rmse_summary_D.csv")

# Correlation-pipeline outputs (regenerate these from annual_params_B.csv
# before using any figure that depends on them — the anomalies changed).
CORR_CSV       = os.path.join(DATA_DIR, "correlations_output.csv")
LAG_CSV        = os.path.join(DATA_DIR, "lag_correlations.csv")
CONT_CSV       = os.path.join(DATA_DIR, "contemporaneous_correlations.csv")
MONTHLY_XC_CSV = os.path.join(DATA_DIR, "monthly_cross_correlations.csv")
INDEX_CSV      = os.path.join(DATA_DIR, "master_index_detrended.csv")

# ── Sectors ───────────────────────────────────────────────────────────────────
# Order used top-to-bottom / left-to-right in every multi-panel figure.
SECTORS = [
    "SIE_Weddell",
    "SIE_Amundsen_Bellingshausen",
    "SIE_Ross",
    "SIE_East_Antarctica",
    "SIE_King_Haakon",
    "SIE_circumpolar",
]

SECTOR_LABELS = {
    "SIE_Weddell":                 "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross":                    "Ross",
    "SIE_East_Antarctica":         "East Antarctica",
    "SIE_King_Haakon":             "King Haakon",
    "SIE_circumpolar":             "Circumpolar",
}

SECTOR_COLORS = {
    "SIE_Weddell":                 "#2196F3",
    "SIE_Amundsen_Bellingshausen": "#F44336",
    "SIE_Ross":                    "#4CAF50",
    "SIE_East_Antarctica":         "#FF9800",
    "SIE_King_Haakon":             "#9C27B0",
    "SIE_circumpolar":             "#2C2C2A",
}

# Sectors excluding circumpolar, for panels where the aggregate would mislead
SECTORS_ONLY = [s for s in SECTORS if s != "SIE_circumpolar"]

# ── Decomposition components ──────────────────────────────────────────────────
COMPONENTS = ["Trend", "Amplitude", "Phase", "Residual"]

COMPONENT_COLS = {
    "Trend":     "trend_component",
    "Amplitude": "amplitude_component",
    "Phase":     "phase_component",
    "Residual":  "residual_apac",
}

COMPONENT_COLORS = {
    "Trend":     "#C77B26",
    "Amplitude": "#378ADD",
    "Phase":     "#D4537E",
    "Residual":  "#BDBDBD",
}

# ── Analysis constants ────────────────────────────────────────────────────────
YEAR_START = 1980          # first full cycle year after the pipeline drop
YEAR_END   = 2023
BREAK_YEAR = 2016          # regime shift

ROLL_SHORT = 10            # rolling window, years — SD and r(phase, amp)
ROLL_LONG  = 15            # rolling window, years — index correlations

# Decade bins used in several figures
DECADE_BINS   = [1979, 1989, 1999, 2009, 2015, 2023]
DECADE_LABELS = ["1980s", "1990s", "2000s", "2010–15", "2016–23"]

# ── Metric naming ─────────────────────────────────────────────────────────────
# The chapter uses OBSERVED metrics throughout. Fitted quantities exist only
# to be shown in S01 as a demonstration of why they are not used.
METRICS_OBSERVED = {
    "min_doy_raw_anom":   "Minimum-date anomaly (days)",
    "max_doy_raw_anom":   "Maximum-date anomaly (days)",
    "amplitude_raw_anom": "Amplitude anomaly (million km²)",
}

METRICS_FITTED_DO_NOT_USE = {
    "min_doy_anom":   "Fitted minimum-date anomaly (ARTIFACT)",
    "max_doy_anom":   "Fitted maximum-date anomaly (ARTIFACT)",
    "amplitude_anom": "Fitted amplitude anomaly",
}

# ── THE metric set for the compute pipeline ──────────────────────────────────
# Every compute_*.py should use ANALYSIS_VARS. Previously each script defined
# its own APAC_VARS, and four of them used max_doy_anom / amplitude_anom —
# the FITTED quantities. max_doy_fitted is the argmax of the fitted curve,
# dominated by the fixed s(DOY) term: SD 2-4x smaller than observed, r with
# the observed maximum only 0.02-0.46, and in Ross r with the observed
# MINIMUM is -0.74. Correlations against it are correlations against a
# compressed, partly spurious series.
#
# Note also that no compute script has ever used the minimum date. It is
# included here because the chapter reports min-date and max-date as a pair.
ANALYSIS_VARS = {
    "max_doy_raw_anom":   "phase_max",
    "min_doy_raw_anom":   "phase_min",
    "amplitude_raw_anom": "amplitude",
}

# Fitted equivalents, for the supplementary comparison figure ONLY.
FITTED_VARS = {
    "max_doy_anom":   "phase_max_fitted",
    "amplitude_anom": "amplitude_fitted",
}

# ── Compute-pipeline derived outputs ─────────────────────────────────────────
MONTHLY_PARAMS_CSV = os.path.join(DATA_DIR, "monthly_params.csv")
AUTOCORR_CSV       = os.path.join(DATA_DIR, "atmospheric_autocorrelations.csv")
PARTIAL_CSV        = os.path.join(DATA_DIR, "partial_correlations.csv")
LOO_SKILL_CSV      = os.path.join(DATA_DIR, "loo_index_skill.csv")
LOO_RESID_CSV      = os.path.join(DATA_DIR, "loo_index_residuals.csv")

# ── Analysis window ──────────────────────────────────────────────────────────
YEAR_MIN = 1979    # index records start here
YEAR_MAX = 2023

# Sector mapping used by the compute scripts (short labels, no circumpolar —
# the correlation pipeline covers the five sectors only)
SECTORS_COMPUTE = {
    "SIE_Weddell":                 "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross":                    "Ross",
    "SIE_East_Antarctica":         "East Antarctica",
    "SIE_King_Haakon":             "King Haakon",
}

# Significance threshold for n = 44, two-tailed p = 0.05
R_SIG_44 = 0.297