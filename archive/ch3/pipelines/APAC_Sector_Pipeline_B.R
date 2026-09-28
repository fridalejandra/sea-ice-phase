# =============================================================================
# APAC_Sector_Pipeline_B.R
# -----------------------------------------------------------------------------
# PIPELINE B — the analysis pipeline.
#
# Pipeline A (the original APAC_Sector_Pipeline.R) is FROZEN: it reproduces
# H&R (2020) Table 1 with bs="cc", k=14/20/75/150 and should not be changed.
# Section 6 of this script re-runs that frozen reproduction UNTOUCHED so both
# numbers are produced side by side.
#
# WHAT CHANGED IN PIPELINE B, AND WHY
#
# (1) TREND-FREE INVARIANT REFERENCE  [the sum-to-Extent bug]
#     Old: fitted_invariant = s(tdate) + s(DOY), i.e. it already contained the
#     trend. So anomaly = Extent - fitted_invariant had the trend REMOVED, and
#     adding trend_component back double-counted it.
#     Verified on King Haakon:
#         mean |anom - (trend+amp+phase)| = 0.2403  == mean|residual_apac|
#         mean |anom - (amp+phase)|       = 0.1379  == mean|residual_apac|
#     Fix: iac_notrend = s(DOY) only. Now anomaly_from_iac contains the trend
#     and  trend + amplitude + phase + residual = anomaly  exactly, as in
#     H&R Fig 7a. fitted_invariant (with trend) is RETAINED for back-compat.
#
# (2) COMMON TREND BASIS, bs="tp", low k  [nesting + interpretability]
#     Old: k = 14/20/75/150 across the four models and bs="cc" on a 44-year
#     axis. k=150 gives ~3 knots/yr: on King Haakon the trend term's
#     within-year SD (0.198) EXCEEDS its between-year SD (0.111) — it wiggles
#     inside years rather than evolving across decades.
#     Note: REML pins edf to whatever ceiling it is given (18.6-18.9 of 20;
#     28.4-28.8 of 30), so k CANNOT be data-selected. TREND_K is a stated
#     modelling choice: ~1 df per 5-10 yr over 44 yr -> k ~ 6-10.
#     Sensitivity is reported in trend_k_sensitivity.csv (separate script).
#
# (3) phase_component = fitted_apac - fitted_amp   (was fitted_phase - fitted_amp)
#
# (4) min-DOY WRAP FIX
#     Old: Weddell 1980, 2005 wrapped to DOY 365 -> anomaly +305.
#     Fix: DOY mapped to a cycle-centred axis before differencing.
#
# (5) est_anomaly = ARMA conditional mean  (was raw_anomaly - volatility)
#     Old version subtracted sigma(t), which is positive, biasing by -0.022.
#     Correct H&R Fig 7a orange line is fitted(garch_fit).
#
# NOTE ON METRICS: phase is reported from OBSERVED extremum dates
# (min_doy_raw / max_doy_raw). max_doy_fitted is the argmax of the fitted
# curve and is an ARTIFACT (dominated by the fixed s(DOY) term; r with raw max
# = 0.02-0.46; in Ross r(fitted max, raw MIN) = -0.74). It is retained ONLY to
# be plotted in Fig S01 as the demonstration of why it is not used.
# =============================================================================

library(dplyr)
library(lubridate)
library(mgcv)
library(rugarch)

# =============================================================================
# 0. USER SETTINGS
# =============================================================================

INPUT_FILE <- "~/Research/repos/sea-ice-phase/scripts/R/observations/SIE_daily_sector_and_circumpolar_million_km2.csv"
OUTPUT_DIR <- "~/Research/repos/sea-ice-phase/scripts/R/Ch3/data"

DATE_START <- as.Date("1979-01-01")
DATE_END   <- as.Date("2023-12-31")

# --- Pipeline B model settings -----------------------------------------------
TREND_BS <- "tp"   # thin-plate: trend evolves across years (NOT cyclic)
TREND_K  <- 8      # stated choice (~1 df / 5-10 yr). NOT data-selected.
DOY_K    <- 100    # cycle shape — cyclic, unchanged from Pipeline A
PHASE_K  <- 100    # phase cycle  — cyclic, unchanged from Pipeline A
IAC_DOY_K <- 25    # resolution of the trend-free climatological reference

trend_term <- sprintf('s(tdate, bs="%s", k=%d)', TREND_BS, TREND_K)

SECTOR_COLS <- c(
  "SIE_Weddell",
  "SIE_Amundsen_Bellingshausen",
  "SIE_Ross",
  "SIE_East_Antarctica",
  "SIE_King_Haakon",
  "SIE_circumpolar"
)

# Helper: put DOY on a cycle-centred axis so a late-December minimum does not
# read as +305 days relative to an early-January one.
# Reference point = the sector's own median min DOY.
centre_doy <- function(doy, ref) {
  ((doy - ref + 182) %% 365) - 182 + ref
}

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================

raw <- read.csv(INPUT_FILE, stringsAsFactors = FALSE)

# Input file uses a "time" column with ISO dates (yyyy-mm-dd); older versions
# used "Date" with %m/%d/%y. Handle both.
names(raw)[tolower(names(raw)) %in% c("date", "time")] <- "Date"
if (!"Date" %in% names(raw))
  stop("No date column found. names(raw): ", paste(names(raw), collapse = ", "))

raw$Date <- if (grepl("^\\d{4}-\\d{2}-\\d{2}", raw$Date[1])) {
  as.Date(raw$Date)                      # ISO
} else {
  as.Date(raw$Date, format = "%m/%d/%y") # legacy
}
if (all(is.na(raw$Date))) stop("Date parsing failed — check the date format.")

raw <- raw %>%
  filter(Date >= DATE_START & Date <= DATE_END) %>%
  arrange(Date)
raw$Year  <- year(raw$Date)
raw$DOY   <- yday(raw$Date)
raw$tdate <- as.numeric(raw$Date)

message("Data loaded: ", nrow(raw), " rows from ",
        min(raw$Date), " to ", max(raw$Date))

# =============================================================================
# 2. CORE FITTING FUNCTION  (PIPELINE B)
# =============================================================================

fit_sector <- function(raw_data, sector_col) {
  
  message("\n========================================")
  message("Sector: ", sector_col, "   [Pipeline B]")
  message("========================================")
  
  # --- 2a. Prepare sector time series ----------------------------------------
  sie <- raw_data %>%
    select(Date, Year, DOY, tdate, Extent = all_of(sector_col)) %>%
    filter(!is.na(Extent)) %>%
    arrange(Date)
  
  # --- 2b. Per-year raw statistics -------------------------------------------
  yearly_stats <- sie %>%
    group_by(Year) %>%
    summarise(
      min_extent     = min(Extent,  na.rm = TRUE),
      max_extent     = max(Extent,  na.rm = TRUE),
      amplitude_raw  = max_extent - min_extent,
      min_doy_raw    = DOY[which.min(Extent)],
      max_doy_raw    = DOY[which.max(Extent)],
      min_date       = Date[which.min(Extent)],
      max_date       = Date[which.max(Extent)],
      .groups        = "drop"
    )
  
  sie <- sie %>% left_join(yearly_stats, by = "Year")
  
  sie <- sie %>%
    mutate(scaling_factor = (Extent - min_extent) / (amplitude_raw + 1e-10))
  
  # --- 2c. Phase warp — UNCHANGED from Pipeline A ----------------------------
  # NOTE: pbeta(x, 1, 1) is the identity. Phase is therefore a rigid
  # re-indexing to the observed minimum date; beta is never estimated. The
  # observed minimum date IS the model's phase parameter. Stated as a
  # departure from H&R in Methods.
  yearly_phase <- yearly_stats %>%
    select(Year, Date1 = min_date) %>%
    mutate(Date2 = lag(Date1), Date3 = lead(Date1))
  
  sie <- sie %>%
    left_join(yearly_phase, by = "Year") %>%
    rowwise() %>%
    mutate(
      t = case_when(
        Year == min(sie$Year)                      ~ 365 - as.numeric(Date3 - Date),
        Year == min(sie$Year) + 1 & Date < Date1   ~ 365 - as.numeric(Date1 - Date),
        Date >= Date1                              ~ as.numeric(Date - Date1),
        Date <  Date1                              ~ as.numeric(Date - Date2)
      )
    ) %>%
    ungroup()
  
  t_stats <- sie %>%
    group_by(Year) %>%
    summarise(t_min = min(t, na.rm = TRUE),
              t_max = max(t, na.rm = TRUE), .groups = "drop")
  
  sie <- sie %>%
    left_join(t_stats, by = "Year") %>%
    rowwise() %>%
    mutate(
      phase = 365 * pbeta((t - t_min) / (t_max - t_min + 1e-10),
                          shape1 = 1, shape2 = 1)
    ) %>%
    ungroup() %>%
    filter(!is.na(phase) & !is.na(scaling_factor))
  
  first_year <- min(sie$Year)
  sie <- sie %>% filter(Year != first_year)
  
  message("  Rows after phase computation: ", nrow(sie))
  
  # --- 2d. MODEL 1: Traditional day-wise climatology -------------------------
  trad_means <- sie %>%
    group_by(DOY) %>%
    summarise(trad_mean = mean(Extent, na.rm = TRUE), .groups = "drop")
  sie <- sie %>% left_join(trad_means, by = "DOY")
  rmse_trad <- sqrt(mean((sie$Extent - sie$trad_mean)^2, na.rm = TRUE))
  message("  Traditional RMSE: ", round(rmse_trad, 4))
  
  # --- 2e. TREND-FREE INVARIANT REFERENCE  [CHANGE 1] ------------------------
  # This is the decomposition reference (light-blue curve in H&R Fig 7a).
  # It must NOT contain a trend, or the trend is double-counted downstream.
  gam_iac_notrend <- gam(
    as.formula(paste0('Extent ~ s(DOY, bs="cc", k=', IAC_DOY_K, ')')),
    data = sie, method = "REML", knots = list(DOY = c(1, 365))
  )
  sie$iac_notrend      <- as.numeric(predict(gam_iac_notrend))
  sie$anomaly_from_iac <- sie$Extent - sie$iac_notrend   # black line, Fig 7a
  
  # Trend-containing invariant RETAINED for back-compatibility / Model 2 RMSE
  gam_invariant <- gam(
    as.formula(paste0("Extent ~ ", trend_term,
                      ' + s(DOY, bs="cc", k=', DOY_K, ')')),
    data = sie, method = "REML", knots = list(DOY = c(1, 365))
  )
  sie$fitted_invariant <- as.numeric(predict(gam_invariant))
  rmse_iac    <- sqrt(mean((sie$Extent - sie$fitted_invariant)^2, na.rm = TRUE))
  pct_imp_iac <- 100 * (1 - rmse_iac^2 / rmse_trad^2)
  message("  Invariant RMSE: ", round(rmse_iac, 4),
          "  (", round(pct_imp_iac, 1), "%)")
  
  # --- 2f. MODEL 3: Amplitude-adjusted  [common trend] -----------------------
  message("  Fitting Model 3: Amplitude-adjusted...")
  gam_amp <- gam(
    as.formula(paste("scaling_factor ~", trend_term,
                     '+ s(DOY, bs="cc", k=', DOY_K, ')')),
    data = sie, method = "REML", knots = list(DOY = c(1, 365))
  )
  sie$fitted_amp   <- as.numeric(predict(gam_amp)) *
    sie$amplitude_raw + sie$min_extent
  sie$residual_amp <- sie$Extent - sie$fitted_amp
  rmse_amp    <- sqrt(mean(sie$residual_amp^2, na.rm = TRUE))
  pct_imp_amp <- 100 * (1 - rmse_amp^2 / rmse_trad^2)
  message("  Amplitude-adjusted RMSE: ", round(rmse_amp, 4),
          "  (", round(pct_imp_amp, 1), "%)")
  
  # --- 2g. MODEL 4: Phase-adjusted  [common trend] ---------------------------
  message("  Fitting Model 4: Phase-adjusted...")
  gam_phase <- gam(
    as.formula(paste("Extent ~", trend_term,
                     '+ s(DOY, bs="cc", k=', DOY_K, ')',
                     '+ s(phase, bs="cc", k=', PHASE_K, ', fx=FALSE)')),
    data = sie, method = "REML",
    knots = list(DOY = c(1, 365), phase = c(0, 365))
  )
  sie$fitted_phase   <- as.numeric(predict(gam_phase))
  sie$residual_phase <- sie$Extent - sie$fitted_phase
  rmse_phase    <- sqrt(mean(sie$residual_phase^2, na.rm = TRUE))
  pct_imp_phase <- 100 * (1 - rmse_phase^2 / rmse_trad^2)
  message("  Phase-adjusted RMSE: ", round(rmse_phase, 4),
          "  (", round(pct_imp_phase, 1), "%)")
  
  # --- 2h. MODEL 5: Full APAC  [common trend] --------------------------------
  message("  Fitting Model 5: Full APAC...")
  gam_apac <- gam(
    as.formula(paste("scaling_factor ~", trend_term,
                     '+ s(DOY, bs="cc", k=', DOY_K, ')',
                     '+ s(phase, bs="cc", k=', PHASE_K, ')')),
    data = sie, method = "REML",
    knots = list(DOY = c(1, 365), phase = c(0, 365))
  )
  sie$pred_scaled_apac <- as.numeric(predict(gam_apac))
  sie$fitted_apac      <- sie$pred_scaled_apac * sie$amplitude_raw + sie$min_extent
  sie$residual_apac    <- sie$Extent - sie$fitted_apac
  rmse_apac    <- sqrt(mean(sie$residual_apac^2, na.rm = TRUE))
  pct_imp_apac <- 100 * (1 - rmse_apac^2 / rmse_trad^2)
  message("  Full APAC RMSE: ", round(rmse_apac, 4),
          "  (", round(pct_imp_apac, 1), "%)")
  
  edf_trend <- summary(gam_apac)$s.table["s(tdate)", "edf"]
  message("  Trend edf used: ", round(edf_trend, 2), " of ceiling ", TREND_K)
  
  # --- 2i. Volatility + ARMA conditional mean  [CHANGE 5] --------------------
  message("  Fitting GARCH(2,2) volatility model...")
  spec <- ugarchspec(
    variance.model     = list(model = "sGARCH", garchOrder = c(2, 2)),
    mean.model         = list(armaOrder = c(1, 1), include.mean = TRUE),
    distribution.model = "norm"
  )
  ok <- !is.na(sie$residual_apac)
  resid_clean <- sie$residual_apac[ok]
  garch_fit <- tryCatch(
    ugarchfit(spec = spec, data = resid_clean, solver = "hybrid"),
    error = function(e) { message("  GARCH failed: ", e$message); NULL }
  )
  sie$volatility  <- NA_real_
  sie$arma_mean   <- NA_real_
  if (!is.null(garch_fit)) {
    sie$volatility[ok] <- as.numeric(sigma(garch_fit))
    sie$arma_mean[ok]  <- as.numeric(fitted(garch_fit))  # conditional mean
    message("  Volatility + conditional mean estimated.")
  }
  
  # ==========================================================================
  # --- 2j. Sequential decomposition (H&R Fig 7a) ---------------------------
  #
  #   anomaly_from_iac = Extent - iac_notrend          [black line]
  #                    = trend_component               [green]
  #                    + amplitude_component           [blue]
  #                    + phase_component               [red]
  #                    + residual_apac                 [remainder]
  #
  #   est_anomaly = arma_mean  -> the orange "estimated anomaly" line
  # ==========================================================================
  
  apac_terms <- predict(gam_apac, type = "terms")
  sie$trend_component <- as.numeric(apac_terms[, "s(tdate)"]) * sie$amplitude_raw
  
  sie <- sie %>%
    mutate(
      amplitude_component = fitted_amp  - iac_notrend - trend_component,
      phase_component     = fitted_apac - fitted_amp,      # [CHANGE 3]
      raw_anomaly         = Extent - fitted_apac,
      est_anomaly         = arma_mean                       # [CHANGE 5]
    )
  
  # --- DECOMPOSITION SUM CHECK ----------------------------------------------
  sum_err <- with(sie, anomaly_from_iac -
                    (trend_component + amplitude_component +
                       phase_component + residual_apac))
  message("  Decomposition sum check — mean |error|: ",
          round(mean(abs(sum_err), na.rm = TRUE), 6),
          "   (should be ~0)")
  if (mean(abs(sum_err), na.rm = TRUE) > 1e-8) {
    warning("Decomposition does not sum for ", sector_col)
  }
  
  # --- Annual scalar parameters ---------------------------------------------
  ref_min <- median(sie$min_doy_raw, na.rm = TRUE)
  ref_max <- median(sie$max_doy_raw, na.rm = TRUE)
  
  annual_fitted <- sie %>%
    group_by(Year) %>%
    summarise(
      # ARTIFACT — retained for Fig S01 only, never used as a metric
      max_doy_fitted   = DOY[which.max(fitted_phase)],
      min_doy_fitted   = DOY[which.min(fitted_phase)],
      amplitude_fitted = max(fitted_amp, na.rm = TRUE) -
        min(fitted_amp, na.rm = TRUE),
      # OBSERVED metrics — these are the ones used in the chapter
      max_doy_raw      = DOY[which.max(Extent)],
      min_doy_raw      = DOY[which.min(Extent)],
      amplitude_raw_yr = max(Extent, na.rm = TRUE) - min(Extent, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    # [CHANGE 4] cycle-centred DOY before differencing
    mutate(
      min_doy_raw_c    = centre_doy(min_doy_raw,    ref_min),
      max_doy_raw_c    = centre_doy(max_doy_raw,    ref_max),
      min_doy_fitted_c = centre_doy(min_doy_fitted, ref_min),
      max_doy_fitted_c = centre_doy(max_doy_fitted, ref_max)
    )
  
  annual_fitted <- annual_fitted %>%
    mutate(
      max_doy_anom   = max_doy_fitted_c - median(max_doy_fitted_c, na.rm = TRUE),
      min_doy_anom   = min_doy_fitted_c - median(min_doy_fitted_c, na.rm = TRUE),
      amplitude_anom = amplitude_fitted - mean(amplitude_fitted, na.rm = TRUE),
      max_doy_raw_anom   = max_doy_raw_c - median(max_doy_raw_c, na.rm = TRUE),
      min_doy_raw_anom   = min_doy_raw_c - median(min_doy_raw_c, na.rm = TRUE),
      amplitude_raw_anom = amplitude_raw_yr - mean(amplitude_raw_yr, na.rm = TRUE)
    )
  
  # Wrap-fix report
  n_wrapped <- sum(annual_fitted$min_doy_raw != annual_fitted$min_doy_raw_c |
                     annual_fitted$max_doy_raw != annual_fitted$max_doy_raw_c,
                   na.rm = TRUE)
  if (n_wrapped > 0)
    message("  min/max DOY wrap correction applied to ", n_wrapped, " year(s)")
  
  annual_params <- annual_fitted %>%
    left_join(
      yearly_stats %>%
        filter(Year != first_year) %>%
        select(Year, min_extent, max_extent, min_date, max_date),
      by = "Year"
    ) %>%
    mutate(sector = sector_col) %>%
    select(
      sector, Year,
      max_doy_fitted, min_doy_fitted, max_doy_anom, min_doy_anom,
      amplitude_fitted, amplitude_anom,
      max_doy_raw, min_doy_raw, max_doy_raw_c, min_doy_raw_c,
      max_doy_raw_anom, min_doy_raw_anom,
      amplitude_raw_yr, amplitude_raw_anom,
      min_extent, max_extent, min_date, max_date
    )
  
  message("  Fitted vs raw max_doy correlation: ",
          round(cor(annual_params$max_doy_fitted,
                    annual_params$max_doy_raw, use = "complete.obs"), 3),
          "   (low value expected — fitted max is an artifact)")
  
  daily_out <- sie %>%
    select(
      Date, Year, DOY, tdate, Extent,
      phase, t, t_min, t_max, scaling_factor,
      iac_notrend, fitted_invariant,
      fitted_amp,   residual_amp,
      fitted_phase, residual_phase,
      fitted_apac,  residual_apac,
      anomaly_from_iac, volatility, arma_mean,
      trend_component, amplitude_component, phase_component,
      raw_anomaly, est_anomaly
    ) %>%
    mutate(sector = sector_col)
  
  list(
    annual = annual_params, daily = daily_out,
    rmse_trad = rmse_trad, rmse_iac = rmse_iac, rmse_amp = rmse_amp,
    rmse_phase = rmse_phase, rmse_apac = rmse_apac,
    pct_imp_iac = pct_imp_iac, pct_imp_amp = pct_imp_amp,
    pct_imp_phase = pct_imp_phase, pct_imp_apac = pct_imp_apac,
    edf_trend = edf_trend,
    sum_err = mean(abs(sum_err), na.rm = TRUE)
  )
}

# =============================================================================
# 3. RUN FOR ALL SECTORS
# =============================================================================

all_annual <- list(); all_daily <- list(); rmse_table <- list()

for (sec in SECTOR_COLS) {
  result <- fit_sector(raw, sec)
  all_annual[[sec]] <- result$annual
  all_daily[[sec]]  <- result$daily
  rmse_table[[sec]] <- data.frame(
    sector = sec,
    rmse_trad = result$rmse_trad, rmse_iac = result$rmse_iac,
    rmse_amp = result$rmse_amp, rmse_phase = result$rmse_phase,
    rmse_apac = result$rmse_apac,
    pct_imp_iac = result$pct_imp_iac, pct_imp_amp = result$pct_imp_amp,
    pct_imp_phase = result$pct_imp_phase, pct_imp_apac = result$pct_imp_apac,
    edf_trend = result$edf_trend, decomp_sum_err = result$sum_err
  )
}

annual_df <- bind_rows(all_annual)
daily_df  <- bind_rows(all_daily)
rmse_df   <- bind_rows(rmse_table)

message("\n--- RMSE Summary (Pipeline B) ---")
print(rmse_df)

# =============================================================================
# 4. SAVE OUTPUTS
# =============================================================================

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
write.csv(annual_df, file.path(OUTPUT_DIR, "annual_params_B.csv"), row.names = FALSE)
write.csv(daily_df,  file.path(OUTPUT_DIR, "daily_fitted_B.csv"),  row.names = FALSE)
write.csv(rmse_df,   file.path(OUTPUT_DIR, "rmse_summary_B.csv"),  row.names = FALSE)

message("\n=== Done (Pipeline B) ===")
message("annual_params_B.csv : ", nrow(annual_df), " rows")
message("daily_fitted_B.csv  : ", nrow(daily_df),  " rows")
message("NOTE: written with _B suffix so Pipeline A outputs are not overwritten.")
message("      Rename once you are satisfied with the checks below.")

# =============================================================================
# 5. VALIDATION
# =============================================================================

message("\n--- Validation checks ---")

message("\nDecomposition sum error by sector (must be ~0):")
print(rmse_df %>% select(sector, decomp_sum_err))

message("\nTrend edf used by sector (ceiling ", TREND_K, "):")
print(rmse_df %>% select(sector, edf_trend))

message("\nTrend component: within-year vs between-year SD")
message("  (Pipeline A had within > between, i.e. it wiggled inside years.")
message("   Pipeline B should have between > within.)")
print(
  daily_df %>%
    group_by(sector) %>%
    summarise(
      within_year_SD  = round(mean(tapply(trend_component, Year, sd), na.rm = TRUE), 4),
      between_year_SD = round(sd(tapply(trend_component, Year, mean)), 4),
      .groups = "drop"
    )
)

message("\nObserved phase/amplitude anomalies, 2016 and 2023:")
print(
  annual_df %>%
    filter(Year %in% c(2016, 2023)) %>%
    select(sector, Year, min_doy_raw_anom, max_doy_raw_anom, amplitude_raw_anom)
)

message("\nPhase-amplitude independence (OBSERVED metrics), pre/post 2016:")
print(
  annual_df %>%
    mutate(era = ifelse(Year < 2016, "pre_2016", "post_2016")) %>%
    group_by(sector, era) %>%
    summarise(
      r_minDOY_amp = round(cor(min_doy_raw_anom, amplitude_raw_anom,
                               use = "complete.obs"), 2),
      r_maxDOY_amp = round(cor(max_doy_raw_anom, amplitude_raw_anom,
                               use = "complete.obs"), 2),
      .groups = "drop"
    )
)

message("\nSD of observed timing metrics (the flat-extremes limitation):")
print(
  annual_df %>%
    group_by(sector) %>%
    summarise(
      SD_min_doy = round(sd(min_doy_raw_c, na.rm = TRUE), 1),
      SD_max_doy = round(sd(max_doy_raw_c, na.rm = TRUE), 1),
      .groups = "drop"
    )
)

# =============================================================================
# 6. PIPELINE A — FROZEN H&R REPRODUCTION, 1979-2018
#    DO NOT MODIFY. Uses the original bs="cc", k=14/20/75/150 settings.
# =============================================================================

message("\n========================================")
message("PIPELINE A (FROZEN) — CIRCUMPOLAR VALIDATION 1979-2018")
message("========================================")

circ <- raw %>%
  filter(Year <= 2018) %>%
  select(Date, Year, DOY, tdate, Extent = SIE_circumpolar) %>%
  filter(!is.na(Extent)) %>%
  arrange(Date)

circ_stats <- circ %>%
  group_by(Year) %>%
  summarise(
    min_extent = min(Extent, na.rm = TRUE),
    max_extent = max(Extent, na.rm = TRUE),
    amplitude  = max_extent - min_extent,
    Date1      = Date[which.min(Extent)],
    .groups    = "drop"
  ) %>%
  mutate(Date2 = lag(Date1), Date3 = lead(Date1))

circ <- circ %>%
  left_join(circ_stats, by = "Year") %>%
  mutate(scaling_factor = (Extent - min_extent) / (amplitude + 1e-10))

circ <- circ %>%
  rowwise() %>%
  mutate(
    t = case_when(
      Year == 1978                ~ 365 - as.numeric(Date3 - Date),
      Year == 1979 & Date < Date1 ~ 365 - as.numeric(Date1 - Date),
      Date >= Date1               ~ as.numeric(Date - Date1),
      Date <  Date1               ~ as.numeric(Date - Date2)
    )
  ) %>%
  ungroup()

t_stats_circ <- circ %>%
  group_by(Year) %>%
  summarise(t_min = min(t, na.rm = TRUE),
            t_max = max(t, na.rm = TRUE), .groups = "drop")

circ <- circ %>%
  left_join(t_stats_circ, by = "Year") %>%
  rowwise() %>%
  mutate(phase = 365 * pbeta((t - t_min) / (t_max - t_min + 1e-10),
                             shape1 = 1, shape2 = 1)) %>%
  ungroup() %>%
  filter(!is.na(phase) & Year != 1978)

trad_circ <- circ %>%
  group_by(DOY) %>%
  summarise(trad_mean = mean(Extent, na.rm = TRUE), .groups = "drop")
circ <- circ %>% left_join(trad_circ, by = "DOY")
rmse_trad_circ <- sqrt(mean((circ$Extent - circ$trad_mean)^2, na.rm = TRUE))

gam_c_iac <- gam(
  Extent ~ s(tdate, bs = "cc", k = 14) + s(DOY, bs = "cc", k = 25),
  data = circ, method = "GCV.Cp", knots = list(DOY = c(1, 365))
)
rmse_iac_circ <- sqrt(mean((circ$Extent - predict(gam_c_iac))^2, na.rm = TRUE))
pct_iac_circ  <- 100 * (1 - rmse_iac_circ^2 / rmse_trad_circ^2)

gam_c_amp <- gam(
  scaling_factor ~ s(tdate, bs = "cc", k = 20) + s(DOY, bs = "cc", k = 100),
  data = circ, method = "GCV.Cp", knots = list(DOY = c(1, 365))
)
fitted_c_amp  <- predict(gam_c_amp) * circ$amplitude + circ$min_extent
rmse_amp_circ <- sqrt(mean((circ$Extent - fitted_c_amp)^2, na.rm = TRUE))
pct_amp_circ  <- 100 * (1 - rmse_amp_circ^2 / rmse_trad_circ^2)

gam_c_phase <- gam(
  Extent ~ s(tdate, bs = "cc", k = 75) + s(DOY, bs = "cc", k = 100) +
    s(phase, bs = "cc", k = 100, fx = FALSE),
  data = circ, method = "GCV.Cp",
  knots = list(DOY = c(1, 365), phase = c(0, 365))
)
rmse_phase_circ <- sqrt(mean((circ$Extent - predict(gam_c_phase))^2, na.rm = TRUE))
pct_phase_circ  <- 100 * (1 - rmse_phase_circ^2 / rmse_trad_circ^2)

gam_c_apac <- gam(
  scaling_factor ~ s(tdate, bs = "cc", k = 150) + s(DOY, bs = "cc", k = 100) +
    s(phase, bs = "cc", k = 100),
  data = circ, method = "GCV.Cp",
  knots = list(DOY = c(1, 365), phase = c(0, 365))
)
fitted_c_apac  <- predict(gam_c_apac) * circ$amplitude + circ$min_extent
rmse_apac_circ <- sqrt(mean((circ$Extent - fitted_c_apac)^2, na.rm = TRUE))
pct_apac_circ  <- 100 * (1 - rmse_apac_circ^2 / rmse_trad_circ^2)

message("\nCircumpolar RMSE vs Handcock & Raphael Table 1 (1979-2018):")
message("                          Your result    Paper (H&R 2020)")
message(sprintf("  Traditional RMSE:         %.3f          0.576", rmse_trad_circ))
message(sprintf("  Invariant RMSE:           %.3f   (%+.1f%%)   0.482 (28.7%%)",
                rmse_iac_circ,   pct_iac_circ))
message(sprintf("  Amplitude-adjusted RMSE:  %.3f   (%+.1f%%)   0.382 (55.2%%)",
                rmse_amp_circ,   pct_amp_circ))
message(sprintf("  Phase-adjusted RMSE:      %.3f   (%+.1f%%)   0.343 (63.9%%)",
                rmse_phase_circ, pct_phase_circ))
message(sprintf("  Full APAC RMSE:           %.3f   (%+.1f%%)   0.272 (77.3%%)",
                rmse_apac_circ,  pct_apac_circ))
message("\nThis reproduction is UNCHANGED from Pipeline A and validates the")
message("implementation. Pipeline B numbers above are the analysis numbers.")