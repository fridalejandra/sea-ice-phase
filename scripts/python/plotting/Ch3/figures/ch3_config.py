"""
ch3_config.py — single source of truth for paths, sectors, and constants.
=========================================================================
Imported by BOTH layers:
  processing/compute_*.py — the correlation / monthly pipeline
  figures/fig*.py         — the figure package
  figures/ch3_stats.py    — every statistic quoted in the chapter

Change a path once, not in nine files.

Repo layout (since the Sep-2026 reorganisation, tag ch3-pipeline-v1):
  data/raw/         untouched inputs (NSIDC SIE, ERA5 winds, ...)
  data/indices/     raw atmospheric index files
  data/ch3/         Pipeline-E outputs + master_index_detrended.csv
  R/ch3/            01_fit_apac.R, 03_make_indices.R, 04_chapter_analyses.R
  results/ch3/      tables/ (every derived CSV) and figures/ (every PNG)
  archive/ch3/      superseded pipelines and outputs — never read from here
"""

import os

# ── Repo root ─────────────────────────────────────────────────────────────────
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
DATA_DIR    = os.path.join(ROOT, "data", "ch3")               # Pipeline-E outputs
INDEX_DIR   = os.path.join(ROOT, "data", "indices")           # raw index files
TABLES_DIR  = os.path.join(ROOT, "results", "ch3", "tables")  # every derived CSV
OUTPUT_DIR  = os.path.join(ROOT, "results", "ch3", "figures") # every PNG
GDRIVE      = "gdrive:sea-ice-phase/results/Ch3_Figures/"

for _d in (TABLES_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# Raw index files (all read from INDEX_DIR)
INDEX_FILES = {
    "SAM":    "marshall_sam_monthly.txt",
    "Nino34": "nina34.data",
    "ASL":    "asli_era5_v3-latest.csv",   # ASL = RelCenPres (relative central pressure)
    "ZW3R":   "ZW3_raphael_monthly.csv",
    "ZW3G":   "ZW3_goyal_monthly.csv",
}

# Pipeline-E outputs (written by R/ch3/01_fit_apac.R; fit starts 1979-01-01)
DAILY_CSV  = os.path.join(DATA_DIR, "daily_fitted_E.csv")
ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params_E.csv")
RMSE_CSV   = os.path.join(DATA_DIR, "rmse_summary_E.csv")

# Detrended seasonal index table (written by processing/compute_atmospheric_correlations.py)
INDEX_CSV  = os.path.join(DATA_DIR, "master_index_detrended.csv")

# Correlation-pipeline outputs (results, not data)
CORR_CSV       = os.path.join(TABLES_DIR, "correlations_output.csv")
LAG_CSV        = os.path.join(TABLES_DIR, "lag_correlations.csv")
CONT_CSV       = os.path.join(TABLES_DIR, "contemporaneous_correlations.csv")
MONTHLY_XC_CSV = os.path.join(TABLES_DIR, "monthly_cross_correlations.csv")

# Chapter statistics ledger (written by figures/ch3_stats.py). Every number in
# the chapter text has a row here naming the section and the script.
NUMBERS_CSV = os.path.join(TABLES_DIR, "ch3_numbers.csv")

# ── Sectors ───────────────────────────────────────────────────────────────────
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

SECTORS_ONLY = [s for s in SECTORS if s != "SIE_circumpolar"]

# ── Decomposition components (Pipeline E bookkeeping) ────────────────────────
#   trend_component     = s(tdate)
#   amplitude_component = fitted_amp  - iac_notrend - trend_component
#   phase_component     = fitted_apac - fitted_amp
#   residual_apac       = Extent - fitted_apac   (== raw_anomaly)
#   anomaly_from_iac    = Extent - iac_notrend   == sum of the four above
# Fitted components are used ONLY for the per-year attribution (§3.2 / Fig 7).
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
YEAR_MIN   = 1979          # first full year in the fit; all annual statistics start here
YEAR_MAX   = 2023
YEAR_START = 1980          # daily-based figures start here (1979 residual sd is ~1.5x later years: edge effect)
YEAR_END   = 2023
BREAK_YEAR = 2016          # regime shift
SPLIT_YEAR = 2001          # half-record split for atmosphere stationarity tests

ROLL_SHORT = 10            # rolling window, years — SD and r(phase, amp)
ROLL_LONG  = 15            # rolling window, years — index correlations

DECADE_BINS   = [1979, 1989, 1999, 2009, 2015, 2023]
DECADE_LABELS = ["1980s", "1990s", "2000s", "2010–15", "2016–23"]

# ── Seasons in master_index_detrended.csv (compute_atmospheric_correlations.py)
#   annual, DJF, MAM, JJA, SON  (DJF assigned to the year of Jan/Feb)
#   ADV = Mar–Aug  (advance)     RET = Oct–Jan (retreat; Jan assigned to preceding year)
SEASONS = ["annual", "DJF", "MAM", "JJA", "SON", "ADV", "RET"]

# ── Metric policy ─────────────────────────────────────────────────────────────
# The chapter uses OBSERVED metrics for every statistical test. Fitted timing is
# a construct of the beta backfit (r with observed day-of-max 0.11 Ross, 0.15
# circumpolar, 0.29-0.55 elsewhere); fitted amplitude equals observed (r = 1.00).
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

ANALYSIS_VARS = {
    "max_doy_raw_anom":   "phase_max",
    "min_doy_raw_anom":   "phase_min",
    "amplitude_raw_anom": "amplitude",
}

FITTED_VARS = {
    "max_doy_anom":   "phase_max_fitted",
    "amplitude_anom": "amplitude_fitted",
}

# ── Pre-specified atmosphere pairs (§3.5 / §3.6) ─────────────────────────────
# (sector, observed variable, index column in master_index_detrended.csv, literature basis)
# Sector names corrected 2026-09-14: the Weddell and Amundsen-Bellingshausen
# columns of the sector CSV were exchanged until that date (see
# scripts/python/checks/check_sectors.py). The ENSO and ZW3 amplitude pairs are
# in the Amundsen-Bellingshausen sector and the SAM (JJA) amplitude pair in the
# Weddell; the relationships themselves are unchanged.
PRIMARY_PAIRS = [
    ("SIE_Amundsen_Bellingshausen", "amplitude_raw_anom", "Nino34_SON",  "ENSO, Pacific pole of the Antarctic Dipole via the ASL (Yuan 2004; Stammerjohn et al. 2008)"),
    ("SIE_King_Haakon",             "amplitude_raw_anom", "Nino34_annual", "ENSO, Atlantic/Indian pole (Yuan 2004)"),
    ("SIE_Ross",                    "amplitude_raw_anom", "ASL_annual",  "ASL–Ross (Raphael et al. 2016; Hosking 2013)"),
    ("SIE_Weddell",                 "amplitude_raw_anom", "SAM_JJA",     "SAM–Weddell (Lefebvre et al. 2004)"),
    ("SIE_East_Antarctica",         "max_doy_raw_anom",   "SAM_RET",     "SAM–East Antarctic retreat (Stammerjohn 2008)"),
    ("SIE_King_Haakon",             "max_doy_raw_anom",   "ZW3R_SON",    "ZW3 (Raphael 2004)"),
    ("SIE_Amundsen_Bellingshausen", "amplitude_raw_anom", "ZW3R_annual", "ZW3 (Raphael 2004)"),
]

# Significance threshold for n = 45 (1979-2023), two-tailed p = 0.05
R_SIG_45 = 0.294
R_SIG_44 = 0.297   # kept for older scripts

SECTORS_COMPUTE = {
    "SIE_Weddell":                 "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross":                    "Ross",
    "SIE_East_Antarctica":         "East Antarctica",
    "SIE_King_Haakon":             "King Haakon",
}

# ── Compute-pipeline derived outputs (results, not data) ─────────────────────
MONTHLY_PARAMS_CSV = os.path.join(TABLES_DIR, "monthly_params.csv")
AUTOCORR_CSV       = os.path.join(TABLES_DIR, "atmospheric_autocorrelations.csv")
PARTIAL_CSV        = os.path.join(TABLES_DIR, "partial_correlations.csv")
LOO_SKILL_CSV      = os.path.join(TABLES_DIR, "loo_index_skill.csv")
LOO_RESID_CSV      = os.path.join(TABLES_DIR, "loo_index_residuals.csv")