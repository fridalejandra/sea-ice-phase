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
  data/ch3/         canonical fit outputs + master_index_detrended.csv
  R/ch3/            01_fit_apac.R (APAC fit + RMSE + volatility, all periods);
                    04_chapter_analyses.R; atmospheric-indices consolidation TBD
  results/ch3/      tables/ (every derived CSV) and figures/ (every PNG)
  archive/ch3/      superseded pipelines and outputs — never read from here
"""

import os

# ── Repo root ─────────────────────────────────────────────────────────────────
_CANDIDATE_ROOTS = [
    os.environ.get("SEAICE_ROOT"),
    "/home/claude",
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
DATA_DIR    = os.path.join(ROOT, "data", "ch3")               # canonical fit outputs
RAW_DATA_DIR = os.path.join(ROOT, "data", "raw")               # ADDED 2026-09-18: raw, pre-fit input files
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

# Canonical fit outputs (written by R/ch3/01_fit_apac.R; fit starts 1979-01-01).
# Renamed 2026-09 to drop the "_E" pipeline-version suffix now that there is
# only one canonical fitting script — was daily_fitted_E.csv / annual_params_E.csv
# / rmse_summary_E.csv. Both now carry a `period` column (see ch3_data.py).
DAILY_CSV  = os.path.join(DATA_DIR, "daily_fitted.csv")
ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")
RMSE_CSV   = os.path.join(DATA_DIR, "rmse_summary.csv")

# ADDED 2026-09-18: raw (pre-fit) daily SIE by sector, million km^2 -- needed
# for fig_s02_annual_min_max_trend.py, which reads the actual extent value at
# the annual min/max rather than the APAC fit's amplitude/timing parameters
# (annual_params.csv has no column for the raw extremum's value). CONFIRMED
# location (Frida's real repo, data/raw/, per screenshot 2026-09-18) -- this
# lives alongside the other raw inputs (ERA5 winds, Bootstrap/NSIDC extent,
# AMSR-E), not under data/ch3/ with the canonical fit outputs.
DAILY_RAW_CSV = os.path.join(RAW_DATA_DIR, "SIE_daily_sector_and_circumpolar_million_km2.csv")

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

# ── Sector boundaries (Raphael & Hobbs 2014) ─────────────────────────────────
# Confirmed 2026-09-14 against canonical_sectors.nc's own attrs
# (sector_bounds_degE, sector_N_name) during the Weddell/ABS label-swap fix --
# see scripts/python/checks/check_sectors.py and check_mask.py. Degrees East,
# continuous (not wrapped to -180..180) so the sequence is monotonic all the
# way around; take `% 360` (or subtract 360 to land in -180..180) if a caller
# needs a conventional longitude. Ross is the only sector that straddles the
# antimeridian in a -180..180 frame (162E -> -180/180 -> -110W); leaving the
# whole table in this continuous frame means no sector needs special-casing.
#   Weddell         290 -> 346  (70W -> 14W)
#   King Haakon VII 346 -> 431  (14W -> 71E, i.e. wraps 0 at 360)
#   East Antarctica  71 -> 162  (71E -> 162E)
#   Ross            162 -> 250  (162E -> 110W)
#   ABS             250 -> 290  (110W -> 70W)
SECTOR_BOUNDS_DEG = {
    "SIE_Weddell":                 (290, 346),
    "SIE_King_Haakon":             (346, 431),
    "SIE_East_Antarctica":         (71, 162),
    "SIE_Ross":                    (162, 250),
    "SIE_Amundsen_Bellingshausen": (250, 290),
}
# Draw order (Weddell first, eastward around the pole) — also the order used
# for any figure that lays sectors out by longitude rather than by SECTORS.
SECTOR_ORDER_BY_LONGITUDE = [
    "SIE_Weddell", "SIE_King_Haakon", "SIE_East_Antarctica",
    "SIE_Ross", "SIE_Amundsen_Bellingshausen",
]

# ── Decomposition components (canonical-pipeline bookkeeping) ───────────────
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
# YEAR_MAX / YEAR_END FIXED 2026-09-18: both were hardcoded to 2023, but the
# record now runs through 2025 (confirmed: daily SIE data ends 2025-12-31;
# annual_params.csv period=="FULL" has n=47, i.e. 1979-2025). ch3_doctor.py
# already warns about exactly this failure mode ("scripts filtering on
# YEAR_MAX will silently drop the tail") -- it was right. Known scripts that
# read one of these two constants and were silently truncating to 2023:
# compute_asl_ross_sweep.py (filters ann to YEAR_MIN..YEAR_MAX -- this is the
# Ross/ASL nonstationarity result cited in the abstract/discussion, so it's
# worth re-running to confirm 2024-2025 don't change it) and
# compute_rolling_diagnostics.py (its rolling-window loop stopped generating
# new windows after 2023, so the "Diagnostic 3" rolling phase-SIE/amp-SIE
# figure was likely missing its last ~2 years even though the underlying
# data already had them). Both should be re-run now that this is fixed.
YEAR_MAX   = 2025
YEAR_START = 1980          # daily-based figures start here (1979 residual sd is ~1.5x later years: edge effect)
YEAR_END   = 2025
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

# ── Figure sizing ─────────────────────────────────────────────────────────────
# Graduate Division requires >=300 dpi (already ch3_plot.save()'s default) but
# says nothing about physical size, so this defaults to a standard 1in-margin
# 8.5x11in page (6.5in usable width) until you confirm your program's actual
# margin requirement — if it differs, change TEXT_WIDTH_IN here and every
# figure using FIGSIZE/figsize() picks it up.
#
# Height is capped by panel layout rather than one fixed aspect ratio, so a
# stacked 2-panel figure (e.g. Fig 2's map + Southern Ocean strip) doesn't
# balloon to a full page the way a single timeseries panel would if forced
# into the same ratio. ch3_plot.figsize(shape) looks these up; ch3_plot.save()
# warns if a figure's actual width exceeds TEXT_WIDTH_IN.
TEXT_WIDTH_IN = 6.5

FIGSIZE = {
    "single":  (TEXT_WIDTH_IN, 3.0),   # one axes: a timeseries, a bar panel
    "row2":    (TEXT_WIDTH_IN, 3.0),   # 1x2 side-by-side (width splits, height stays "single"-scale)
    "row3":    (TEXT_WIDTH_IN, 3.0),   # 1x3 side-by-side
    "stack2":  (TEXT_WIDTH_IN, 5.0),   # 2 panels stacked vertically
    "grid2x2": (13.0, 11.0),           # 2x2 grid — wide for better readability, user approved wider format
    "grid2x3": (TEXT_WIDTH_IN, 6.0),   # 2x3 sector grid (sector_grid()'s default layout)
}

# ── Compute-pipeline derived outputs (results, not data) ─────────────────────
MONTHLY_PARAMS_CSV = os.path.join(TABLES_DIR, "monthly_params.csv")
AUTOCORR_CSV       = os.path.join(TABLES_DIR, "atmospheric_autocorrelations.csv")
PARTIAL_CSV        = os.path.join(TABLES_DIR, "partial_correlations.csv")
LOO_SKILL_CSV      = os.path.join(TABLES_DIR, "loo_index_skill.csv")
LOO_RESID_CSV      = os.path.join(TABLES_DIR, "loo_index_residuals.csv")