# =============================================================================
# APAC_Sector_Pipeline.R
# Implements the Handcock & Raphael (2020) framework for 5 Antarctic
# sea ice sectors following the user's own validated implementation.
#
# Models fitted per sector:
#   1. Traditional annual cycle
#   2. Invariant annual cycle         — Extent ~ s(tdate) + s(DOY)
#   3. Amplitude-adjusted             — scaling_factor ~ s(tdate) + s(DOY)
#   4. Phase-adjusted                 — Extent ~ s(tdate) + s(DOY) + s(phase)
#   5. Full APAC                      — scaling_factor ~ s(tdate) + s(DOY) + s(phase)
#
# Sequential decomposition (Eq. 13 / Fig. 7 of H&R 2020):
#   amplitude_component = fitted_amp   - fitted_invariant
#   phase_component     = fitted_phase - fitted_amp
#
# Parameter extraction — ASYMMETRIC by design:
#   max_doy_fitted   — DOY of peak of FULL fitted_phase curve
#                      (phase GAM contains no amplitude scaling, so it is
#                       already amplitude-free; no subtraction needed)
#   amplitude_fitted — max - min of amplitude_component
#                      (amplitude GAM uses scaling_factor, so subtracting
#                       the IAC baseline isolates the pure magnitude signal)
#
# References:
#   Handcock & Raphael (2020), The Cryosphere, 14, 2159-2172.
#   Raphael et al. (2020), JGR Oceans, 125, e2020JC016459.
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

SECTOR_COLS <- c(
  "SIE_Weddell",
  "SIE_Amundsen_Bellingshausen",
  "SIE_Ross",
  "SIE_East_Antarctica",
  "SIE_King_Haakon",
  "SIE_circumpolar"
)

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================

raw <- read.csv(INPUT_FILE, stringsAsFactors = FALSE)
raw$Date  <- as.Date(raw$Date, format = "%m/%d/%y")
raw <- raw %>%
  filter(Date >= DATE_START & Date <= DATE_END) %>%
  arrange(Date)
raw$Year  <- year(raw$Date)
raw$DOY   <- yday(raw$Date)
raw$tdate <- as.numeric(raw$Date)

message("Data loaded: ", nrow(raw), " rows from ",
        min(raw$Date), " to ", max(raw$Date))

# =============================================================================
# 2. CORE FITTING FUNCTION
# =============================================================================

fit_sector <- function(raw_data, sector_col) {
  
  message("\n========================================")
  message("Sector: ", sector_col)
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
  
  # Scaling factor
  sie <- sie %>%
    mutate(scaling_factor = (Extent - min_extent) / (amplitude_raw + 1e-10))
  
  # --- 2c. Phase warp — user's validated t-value logic -----------------------
  yearly_phase <- yearly_stats %>%
    select(Year, Date1 = min_date) %>%
    mutate(
      Date2 = lag(Date1),
      Date3 = lead(Date1)
    )
  
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
    summarise(
      t_min = min(t, na.rm = TRUE),
      t_max = max(t, na.rm = TRUE),
      .groups = "drop"
    )
  
  sie <- sie %>%
    left_join(t_stats, by = "Year") %>%
    rowwise() %>%
    mutate(
      phase = 365 * pbeta(
        (t - t_min) / (t_max - t_min + 1e-10),
        shape1 = 1, shape2 = 1
      )
    ) %>%
    ungroup() %>%
    filter(!is.na(phase) & !is.na(scaling_factor))
  
  # Remove first year (incomplete cycle)
  first_year <- min(sie$Year)
  sie <- sie %>% filter(Year != first_year)
  
  message("  Rows after phase computation: ", nrow(sie))
  
  # --- 2d. MODEL 1: Traditional ----------------------------------------------
  trad_means <- sie %>%
    group_by(DOY) %>%
    summarise(trad_mean = mean(Extent, na.rm = TRUE), .groups = "drop")
  sie <- sie %>% left_join(trad_means, by = "DOY")
  rmse_trad <- sqrt(mean((sie$Extent - sie$trad_mean)^2, na.rm = TRUE))
  message("  Traditional RMSE: ", round(rmse_trad, 4))
  
  # --- 2e. MODEL 2: Invariant ------------------------------------------------
  gam_invariant <- gam(
    Extent ~ s(tdate, bs = "cc", k = 14) + s(DOY, bs = "cc", k = 25),
    data = sie, method = "GCV.Cp",
    knots = list(DOY = c(1, 365))
  )
  sie$fitted_invariant <- as.numeric(predict(gam_invariant))
  sie$anomaly_from_iac <- sie$Extent - sie$fitted_invariant
  rmse_iac    <- sqrt(mean(sie$anomaly_from_iac^2, na.rm = TRUE))
  pct_imp_iac <- 100 * (1 - rmse_iac^2 / rmse_trad^2)
  message("  Invariant RMSE: ", round(rmse_iac, 4),
          "  (", round(pct_imp_iac, 1), "%)")
  
  # Trend term — extracted from gam_apac s(tdate) after models are fitted.
  # Placeholder column set here; filled in section 2j after gam_apac exists.
  sie$trend_component <- NA_real_
  
  # --- 2f. MODEL 3: Amplitude-adjusted ---------------------------------------
  message("  Fitting Model 3: Amplitude-adjusted...")
  gam_amp <- gam(
    scaling_factor ~ s(tdate, bs = "cc", k = 20) +
      s(DOY,   bs = "cc", k = 100),
    data = sie, method = "GCV.Cp",
    knots = list(DOY = c(1, 365))
  )
  # Back-transform to SIE units — same scale as fitted_invariant and fitted_phase
  sie$fitted_amp   <- as.numeric(predict(gam_amp)) *
    sie$amplitude_raw + sie$min_extent
  sie$residual_amp <- sie$Extent - sie$fitted_amp
  rmse_amp    <- sqrt(mean(sie$residual_amp^2, na.rm = TRUE))
  pct_imp_amp <- 100 * (1 - rmse_amp^2 / rmse_trad^2)
  message("  Amplitude-adjusted RMSE: ", round(rmse_amp, 4),
          "  (", round(pct_imp_amp, 1), "%)")
  
  # --- 2g. MODEL 4: Phase-adjusted -------------------------------------------
  message("  Fitting Model 4: Phase-adjusted...")
  gam_phase <- gam(
    Extent ~ s(tdate, bs = "cc", k = 75) +
      s(DOY,   bs = "cc", k = 100) +
      s(phase, bs = "cc", k = 100, fx = FALSE),
    data = sie, method = "GCV.Cp",
    knots = list(DOY = c(1, 365), phase = c(0, 365))
  )
  sie$fitted_phase   <- as.numeric(predict(gam_phase))
  sie$residual_phase <- sie$Extent - sie$fitted_phase
  rmse_phase    <- sqrt(mean(sie$residual_phase^2, na.rm = TRUE))
  pct_imp_phase <- 100 * (1 - rmse_phase^2 / rmse_trad^2)
  message("  Phase-adjusted RMSE: ", round(rmse_phase, 4),
          "  (", round(pct_imp_phase, 1), "%)")
  
  # --- 2h. MODEL 5: Full APAC ------------------------------------------------
  message("  Fitting Model 5: Full APAC...")
  gam_apac <- gam(
    scaling_factor ~ s(tdate, bs = "cc", k = 150) +
      s(DOY,   bs = "cc", k = 100)  +
      s(phase, bs = "cc", k = 100),
    data = sie, method = "GCV.Cp",
    knots = list(DOY = c(1, 365), phase = c(0, 365))
  )
  sie$pred_scaled_apac <- as.numeric(predict(gam_apac))
  sie$fitted_apac      <- sie$pred_scaled_apac * sie$amplitude_raw + sie$min_extent
  sie$residual_apac    <- sie$Extent - sie$fitted_apac
  rmse_apac    <- sqrt(mean(sie$residual_apac^2, na.rm = TRUE))
  pct_imp_apac <- 100 * (1 - rmse_apac^2 / rmse_trad^2)
  message("  Full APAC RMSE: ", round(rmse_apac, 4),
          "  (", round(pct_imp_apac, 1), "%)")
  
  # --- 2i. Volatility on APAC residuals --------------------------------------
  message("  Fitting GARCH(2,2) volatility model...")
  spec <- ugarchspec(
    variance.model     = list(model = "sGARCH", garchOrder = c(2, 2)),
    mean.model         = list(armaOrder = c(1, 1), include.mean = TRUE),
    distribution.model = "norm"
  )
  resid_clean <- sie$residual_apac[!is.na(sie$residual_apac)]
  garch_fit <- tryCatch(
    ugarchfit(spec = spec, data = resid_clean, solver = "hybrid"),
    error = function(e) { message("  GARCH failed: ", e$message); NULL }
  )
  sie$volatility <- NA_real_
  if (!is.null(garch_fit)) {
    sie$volatility[!is.na(sie$residual_apac)] <- as.numeric(sigma(garch_fit))
    message("  Volatility estimated successfully.")
  }
  
  # ==========================================================================
  # --- 2j. Sequential decomposition (Eq. 13 / Fig. 7, H&R 2020) -----------
  #
  # All five components are strictly additive — they sum exactly to Extent:
  #
  #   Extent(t) = fitted_invariant          [IAC baseline]
  #             + trend_component           [s(tdate) from gam_apac]
  #             + amplitude_component       [fitted_amp   - fitted_invariant]
  #             + phase_component           [fitted_phase - fitted_amp]
  #             + raw_anomaly               [Extent - fitted_apac]
  #
  # raw_anomaly = est_anomaly + volatility  [GARCH decomposition]
  #
  # trend_component is extracted from gam_apac's s(tdate) partial term so it
  # sits on the same model chain as amplitude and phase. This matches H&R Fig 7
  # where the trend is the APAC multi-decadal component (green curve).
  #
  # For plotting (Fig. 7 style), each component is shown as a deviation from
  # its own long-run mean so all series oscillate around zero.
  #
  # Annual scalar parameters — ASYMMETRIC extraction by design:
  #   max_doy_fitted   = DOY[ argmax( fitted_phase ) ]     — pure timing
  #   amplitude_fitted = max(amplitude_component) - min   — pure magnitude
  # ==========================================================================
  
  # --- Extract trend from gam_apac s(tdate) partial term --------------------
  # predict(..., type="terms") gives the contribution of each smooth centred
  # at zero. We want the trend in SIE units on the same scale as the other
  # components, so we scale by (amplitude_raw) and add min_extent to match
  # the back-transform used for fitted_apac, then take only the tdate partial.
  apac_terms <- predict(gam_apac, type = "terms")
  # s(tdate) partial is in scaled (scaling_factor) space — convert to SIE units
  sie$trend_component <- as.numeric(apac_terms[, "s(tdate)"]) *
    sie$amplitude_raw
  
  sie <- sie %>%
    mutate(
      # Amplitude component: year-specific magnitude shift above the IAC
      amplitude_component = fitted_amp - fitted_invariant,
      
      # Phase component: additional shift from timing variation, on top of
      # amplitude adjustment (sequential — trend already inside fitted_amp
      # via the shared s(tdate) in gam_amp, but phase component captures
      # what phase-adjustment adds beyond that)
      phase_component     = fitted_phase - fitted_amp,
      
      # Raw anomaly: everything not explained by the APAC
      raw_anomaly         = Extent - fitted_apac,
      
      # Estimated (smoothed) anomaly: raw anomaly minus volatility noise
      est_anomaly         = raw_anomaly - volatility
    )
  
  annual_fitted <- sie %>%
    group_by(Year) %>%
    summarise(
      # --- PHASE: DOY of peak of the FULL phase-adjusted fitted curve --------
      # gam_phase is fitted on raw Extent (not scaling_factor), so it carries
      # no amplitude-varying rescaling. Its peak is a pure timing signal —
      # no subtraction of the IAC is needed here.
      max_doy_fitted   = DOY[which.max(fitted_phase)],
      min_doy_fitted   = DOY[which.min(fitted_phase)],
      
      # --- AMPLITUDE: range of the ISOLATED amplitude component -------------
      # gam_amp IS fitted on scaling_factor, so back-transformed fitted_amp
      # absorbs both shape and magnitude. Subtracting fitted_invariant removes
      # the common IAC baseline, leaving the pure year-specific magnitude signal.
      amplitude_fitted = max(amplitude_component, na.rm = TRUE) -
        min(amplitude_component, na.rm = TRUE),
      
      # --- RAW values for comparison ----------------------------------------
      max_doy_raw      = DOY[which.max(Extent)],
      min_doy_raw      = DOY[which.min(Extent)],
      amplitude_raw_yr = max(Extent, na.rm = TRUE) - min(Extent, na.rm = TRUE),
      
      .groups = "drop"
    )
  
  # Compute anomalies relative to long-run median/mean
  annual_fitted <- annual_fitted %>%
    mutate(
      # Fitted anomalies — primary output
      max_doy_anom   = max_doy_fitted  - median(max_doy_fitted,  na.rm = TRUE),
      min_doy_anom   = min_doy_fitted  - median(min_doy_fitted,  na.rm = TRUE),
      amplitude_anom = amplitude_fitted - mean(amplitude_fitted, na.rm = TRUE),
      
      # Raw anomalies — kept for comparison
      max_doy_raw_anom   = max_doy_raw   - median(max_doy_raw,     na.rm = TRUE),
      min_doy_raw_anom   = min_doy_raw   - median(min_doy_raw,     na.rm = TRUE),
      amplitude_raw_anom = amplitude_raw_yr - mean(amplitude_raw_yr, na.rm = TRUE)
    )
  
  # Merge with original yearly stats for dates and extents
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
      # Fitted parameters — primary
      max_doy_fitted, min_doy_fitted,
      max_doy_anom,   min_doy_anom,
      amplitude_fitted, amplitude_anom,
      # Raw parameters — for reference
      max_doy_raw, min_doy_raw,
      max_doy_raw_anom, min_doy_raw_anom,
      amplitude_raw_yr, amplitude_raw_anom,
      # Extents and dates
      min_extent, max_extent, min_date, max_date
    )
  
  message("  Fitted vs raw max_doy correlation: ",
          round(cor(annual_params$max_doy_fitted,
                    annual_params$max_doy_raw, use = "complete.obs"), 3))
  message("  Fitted vs raw amplitude correlation: ",
          round(cor(annual_params$amplitude_fitted,
                    annual_params$amplitude_raw_yr, use = "complete.obs"), 3))
  
  # --- 2k. Daily output table ------------------------------------------------
  daily_out <- sie %>%
    select(
      Date, Year, DOY, tdate, Extent,
      phase, t, t_min, t_max,
      scaling_factor,
      fitted_invariant,
      fitted_amp,   residual_amp,
      fitted_phase, residual_phase,
      fitted_apac,  residual_apac,
      anomaly_from_iac, volatility,
      # Fig. 7 sequential decomposition components
      trend_component,
      amplitude_component,
      phase_component,
      raw_anomaly,
      est_anomaly
    ) %>%
    mutate(sector = sector_col)
  
  list(
    annual        = annual_params,
    daily         = daily_out,
    rmse_trad     = rmse_trad,
    rmse_iac      = rmse_iac,
    rmse_amp      = rmse_amp,
    rmse_phase    = rmse_phase,
    rmse_apac     = rmse_apac,
    pct_imp_iac   = pct_imp_iac,
    pct_imp_amp   = pct_imp_amp,
    pct_imp_phase = pct_imp_phase,
    pct_imp_apac  = pct_imp_apac,
    gam_invariant = gam_invariant,
    gam_amp       = gam_amp,
    gam_phase     = gam_phase,
    gam_apac      = gam_apac
  )
}

# =============================================================================
# 3. RUN FOR ALL SECTORS
# =============================================================================

all_annual <- list()
all_daily  <- list()
rmse_table <- list()

for (sec in SECTOR_COLS) {
  result <- fit_sector(raw, sec)
  all_annual[[sec]] <- result$annual
  all_daily[[sec]]  <- result$daily
  rmse_table[[sec]] <- data.frame(
    sector        = sec,
    rmse_trad     = result$rmse_trad,
    rmse_iac      = result$rmse_iac,
    rmse_amp      = result$rmse_amp,
    rmse_phase    = result$rmse_phase,
    rmse_apac     = result$rmse_apac,
    pct_imp_iac   = result$pct_imp_iac,
    pct_imp_amp   = result$pct_imp_amp,
    pct_imp_phase = result$pct_imp_phase,
    pct_imp_apac  = result$pct_imp_apac
  )
}

annual_df <- bind_rows(all_annual)
daily_df  <- bind_rows(all_daily)
rmse_df   <- bind_rows(rmse_table)

message("\n--- RMSE Summary ---")
print(rmse_df)

# =============================================================================
# 4. SAVE OUTPUTS
# =============================================================================

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
write.csv(annual_df, file.path(OUTPUT_DIR, "annual_params.csv"),  row.names = FALSE)
write.csv(daily_df,  file.path(OUTPUT_DIR, "daily_fitted.csv"),   row.names = FALSE)
write.csv(rmse_df,   file.path(OUTPUT_DIR, "rmse_summary.csv"),   row.names = FALSE)

message("\n=== Done ===")
message("annual_params.csv : ", nrow(annual_df), " rows")
message("daily_fitted.csv  : ", nrow(daily_df),  " rows")
message("rmse_summary.csv  : ", nrow(rmse_df),   " rows")

# =============================================================================
# 5. VALIDATION
# =============================================================================

message("\n--- Validation checks ---")

message("Years per sector:")
print(table(annual_df$sector))

message("\nMedian timing and mean amplitude per sector:")
print(
  annual_df %>%
    group_by(sector) %>%
    summarise(
      median_max_doy_fitted = median(max_doy_fitted),
      median_max_doy_raw    = median(max_doy_raw),
      mean_amp_fitted       = round(mean(amplitude_fitted), 3),
      mean_amp_raw          = round(mean(amplitude_raw_yr), 3)
    )
)

message("\nFitted vs raw correlation check:")
print(
  annual_df %>%
    group_by(sector) %>%
    summarise(
      r_max_doy   = round(cor(max_doy_fitted, max_doy_raw,
                              use = "complete.obs"), 3),
      r_amplitude = round(cor(amplitude_fitted, amplitude_raw_yr,
                              use = "complete.obs"), 3)
    )
)

message("\n2016 anomalies (fitted):")
print(
  annual_df %>%
    filter(Year == 2016) %>%
    select(sector, Year, max_doy_anom, amplitude_anom,
           max_doy_raw_anom, amplitude_raw_anom)
)

message("\n2023 anomalies (fitted):")
print(
  annual_df %>%
    filter(Year == 2023) %>%
    select(sector, Year, max_doy_anom, amplitude_anom,
           max_doy_raw_anom, amplitude_raw_anom)
)

message("\nIndependence check — correlation between fitted phase and amplitude anomalies:")
print(
  annual_df %>%
    group_by(sector) %>%
    summarise(
      r_fitted = round(cor(max_doy_anom, amplitude_anom,
                           use = "complete.obs"), 3),
      r_raw    = round(cor(max_doy_raw_anom, amplitude_raw_anom,
                           use = "complete.obs"), 3)
    )
)

message("\nFull RMSE comparison table:")
print(rmse_df)

# =============================================================================
# 6. CIRCUMPOLAR VALIDATION — compare to Handcock & Raphael Table 1
# =============================================================================

message("\n========================================")
message("CIRCUMPOLAR VALIDATION — 1979-2018")
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
  summarise(
    t_min = min(t, na.rm = TRUE),
    t_max = max(t, na.rm = TRUE),
    .groups = "drop"
  )

circ <- circ %>%
  left_join(t_stats_circ, by = "Year") %>%
  rowwise() %>%
  mutate(
    phase = 365 * pbeta(
      (t - t_min) / (t_max - t_min + 1e-10),
      shape1 = 1, shape2 = 1
    )
  ) %>%
  ungroup() %>%
  filter(!is.na(phase) & Year != 1978)

trad_circ <- circ %>%
  group_by(DOY) %>%
  summarise(trad_mean = mean(Extent, na.rm = TRUE), .groups = "drop")
circ <- circ %>% left_join(trad_circ, by = "DOY")
rmse_trad_circ <- sqrt(mean((circ$Extent - circ$trad_mean)^2, na.rm = TRUE))

gam_c_iac <- gam(
  Extent ~ s(tdate, bs = "cc", k = 14) + s(DOY, bs = "cc", k = 25),
  data = circ, method = "GCV.Cp",
  knots = list(DOY = c(1, 365))
)
rmse_iac_circ <- sqrt(mean((circ$Extent - predict(gam_c_iac))^2, na.rm = TRUE))
pct_iac_circ  <- 100 * (1 - rmse_iac_circ^2 / rmse_trad_circ^2)

gam_c_amp <- gam(
  scaling_factor ~ s(tdate, bs = "cc", k = 20) +
    s(DOY,   bs = "cc", k = 100),
  data = circ, method = "GCV.Cp",
  knots = list(DOY = c(1, 365))
)
fitted_c_amp  <- predict(gam_c_amp) * circ$amplitude + circ$min_extent
rmse_amp_circ <- sqrt(mean((circ$Extent - fitted_c_amp)^2, na.rm = TRUE))
pct_amp_circ  <- 100 * (1 - rmse_amp_circ^2 / rmse_trad_circ^2)

gam_c_phase <- gam(
  Extent ~ s(tdate, bs = "cc", k = 75) +
    s(DOY,   bs = "cc", k = 100) +
    s(phase, bs = "cc", k = 100, fx = FALSE),
  data = circ, method = "GCV.Cp",
  knots = list(DOY = c(1, 365), phase = c(0, 365))
)
rmse_phase_circ <- sqrt(mean((circ$Extent - predict(gam_c_phase))^2,
                             na.rm = TRUE))
pct_phase_circ  <- 100 * (1 - rmse_phase_circ^2 / rmse_trad_circ^2)

gam_c_apac <- gam(
  scaling_factor ~ s(tdate, bs = "cc", k = 150) +
    s(DOY,   bs = "cc", k = 100)  +
    s(phase, bs = "cc", k = 100),
  data = circ, method = "GCV.Cp",
  knots = list(DOY = c(1, 365), phase = c(0, 365))
)
fitted_c_apac  <- predict(gam_c_apac) * circ$amplitude + circ$min_extent
rmse_apac_circ <- sqrt(mean((circ$Extent - fitted_c_apac)^2, na.rm = TRUE))
pct_apac_circ  <- 100 * (1 - rmse_apac_circ^2 / rmse_trad_circ^2)

message("\nCircumpolar RMSE vs Handcock & Raphael Table 1 (1979-2018):")
message("                          Your result    Paper (H&R 2020)")
message(sprintf("  Traditional RMSE:         %.3f          0.576",
                rmse_trad_circ))
message(sprintf("  Invariant RMSE:           %.3f   (%+.1f%%)   0.482 (28.7%%)",
                rmse_iac_circ,   pct_iac_circ))
message(sprintf("  Amplitude-adjusted RMSE:  %.3f   (%+.1f%%)   0.382 (55.2%%)",
                rmse_amp_circ,   pct_amp_circ))
message(sprintf("  Phase-adjusted RMSE:      %.3f   (%+.1f%%)   0.343 (63.9%%)",
                rmse_phase_circ, pct_phase_circ))
message(sprintf("  Full APAC RMSE:           %.3f   (%+.1f%%)   0.272 (77.3%%)",
                rmse_apac_circ,  pct_apac_circ))


# =============================================================================
# 7. ANNUAL PARAMETER TIME SERIES — per sector
# =============================================================================
# One figure per sector: two-panel plot showing
#   (a) max_doy_anom   — phase anomaly (days) with raw overlay
#   (b) amplitude_anom — amplitude anomaly (million km²) with raw overlay
# 2016 and 2023 highlighted. Fitted = solid, raw = dashed grey.
# =============================================================================

library(ggplot2)
library(tidyr)
library(patchwork)

SECTOR_LABELS <- c(
  "SIE_Weddell"                 = "Weddell",
  "SIE_Amundsen_Bellingshausen" = "Amundsen–Bellingshausen",
  "SIE_Ross"                    = "Ross",
  "SIE_East_Antarctica"         = "East Antarctica",
  "SIE_King_Haakon"             = "King Haakon"
)

FIG_DIR <- "~/Research/repos/sea-ice-phase/scripts/R/chapter3/figures"
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

HIGHLIGHT_YEARS <- c(2016, 2023)
HIGHLIGHT_COLS  <- c("2016" = "#D85A30", "2023" = "#BA7517")

plot_annual_params <- function(df_sec, sector_col) {
  
  slabel <- SECTOR_LABELS[sector_col]
  
  # --- shared theme ----------------------------------------------------------
  base_theme <- theme_bw(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "grey92", linewidth = 0.3),
      axis.title.x     = element_blank(),
      legend.position  = "none"
    )
  
  # --- vertical highlight bands for key years --------------------------------
  hi_bands <- list(
    geom_vline(xintercept = 2016, colour = HIGHLIGHT_COLS["2016"],
               linewidth = 0.5, linetype = "dashed", alpha = 0.7),
    geom_vline(xintercept = 2023, colour = HIGHLIGHT_COLS["2023"],
               linewidth = 0.5, linetype = "dashed", alpha = 0.7)
  )
  
  # --- Panel (a): phase anomaly ----------------------------------------------
  pa <- ggplot(df_sec, aes(x = Year)) +
    hi_bands +
    geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.4) +
    # raw — dashed grey behind
    geom_line(aes(y = max_doy_raw_anom),
              colour = "grey60", linewidth = 0.6, linetype = "dashed",
              na.rm = TRUE) +
    # fitted — solid colour on top
    geom_line(aes(y = max_doy_anom),
              colour = "#185FA5", linewidth = 1.0, na.rm = TRUE) +
    geom_point(
      data    = df_sec %>% filter(Year %in% HIGHLIGHT_YEARS),
      aes(y   = max_doy_anom, fill = factor(Year)),
      shape   = 21, size = 3, colour = "white", stroke = 0.5,
      show.legend = FALSE
    ) +
    scale_fill_manual(values = HIGHLIGHT_COLS) +
    scale_x_continuous(breaks = seq(1980, 2023, by = 5)) +
    labs(
      y        = "Phase anomaly (days)",
      subtitle = "Timing of maximum — fitted (solid) vs raw (dashed)"
    ) +
    base_theme
  
  # --- Panel (b): amplitude anomaly ------------------------------------------
  pb <- ggplot(df_sec, aes(x = Year)) +
    hi_bands +
    geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.4) +
    geom_line(aes(y = amplitude_raw_anom),
              colour = "grey60", linewidth = 0.6, linetype = "dashed",
              na.rm = TRUE) +
    geom_line(aes(y = amplitude_anom),
              colour = "#1D9E75", linewidth = 1.0, na.rm = TRUE) +
    geom_point(
      data    = df_sec %>% filter(Year %in% HIGHLIGHT_YEARS),
      aes(y   = amplitude_anom, fill = factor(Year)),
      shape   = 21, size = 3, colour = "white", stroke = 0.5,
      show.legend = FALSE
    ) +
    scale_fill_manual(values = HIGHLIGHT_COLS) +
    scale_x_continuous(breaks = seq(1980, 2023, by = 5)) +
    labs(
      y        = "Amplitude anomaly (million km\u00b2)",
      subtitle = "Annual amplitude — fitted (solid) vs raw (dashed)",
      x        = "Year"
    ) +
    base_theme + theme(axis.title.x = element_text())
  
  # --- Combine with patchwork ------------------------------------------------
  p <- pa / pb +
    plot_annotation(
      title   = paste0("Annual parameters — ", slabel),
      caption = paste0(
        "Dashed vertical lines: ",
        paste(HIGHLIGHT_YEARS, collapse = ", "),
        "   \u25CF coloured dots mark those years on fitted series"
      ),
      theme = theme(
        plot.title   = element_text(size = 12, face = "bold"),
        plot.caption = element_text(size = 8,  colour = "grey45")
      )
    )
  
  fname <- file.path(FIG_DIR,
                     sprintf("annual_params_%s.png", gsub("SIE_", "", sector_col)))
  ggsave(fname, p, width = 8, height = 6, dpi = 150)
  message("Saved: ", fname)
  invisible(p)
}

for (sec in SECTOR_COLS) {
  plot_annual_params(annual_df %>% filter(sector == sec), sec)
}

# =============================================================================
# 8. SECTOR COMPARISON PANEL — phase and amplitude side by side
# =============================================================================
# Single figure: 5 rows (sectors) x 2 columns (phase | amplitude)
# Fitted anomaly only. Sectors ordered geographically. free_y scales.
# =============================================================================

plot_sector_comparison <- function(annual_df) {
  
  sector_order <- SECTOR_LABELS[SECTOR_COLS]
  
  # Build long data frame with both parameters
  df_all <- annual_df %>%
    mutate(sector_label = factor(SECTOR_LABELS[sector], levels = sector_order)) %>%
    select(sector_label, Year, max_doy_anom, amplitude_anom) %>%
    pivot_longer(
      cols      = c(max_doy_anom, amplitude_anom),
      names_to  = "param",
      values_to = "value"
    ) %>%
    mutate(
      param = factor(param,
                     levels = c("max_doy_anom", "amplitude_anom"),
                     labels = c("Phase anomaly (days)",
                                "Amplitude anomaly (million km\u00b2)")
      ),
      colour = ifelse(param == "Phase anomaly (days)", "#185FA5", "#1D9E75")
    )
  
  df_hi <- df_all %>% filter(Year %in% HIGHLIGHT_YEARS)
  
  p <- ggplot(df_all, aes(x = Year, y = value)) +
    geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.35) +
    geom_vline(xintercept = 2016, colour = HIGHLIGHT_COLS["2016"],
               linewidth = 0.4, linetype = "dashed", alpha = 0.65) +
    geom_vline(xintercept = 2023, colour = HIGHLIGHT_COLS["2023"],
               linewidth = 0.4, linetype = "dashed", alpha = 0.65) +
    geom_line(aes(colour = param), linewidth = 0.85, na.rm = TRUE) +
    geom_point(
      data  = df_hi,
      aes(fill = factor(Year)),
      shape = 21, size = 2.5, colour = "white", stroke = 0.5,
      show.legend = TRUE
    ) +
    scale_colour_manual(
      values = c("Phase anomaly (days)"              = "#185FA5",
                 "Amplitude anomaly (million km\u00b2)" = "#1D9E75"),
      name   = NULL,
      guide  = "none"
    ) +
    scale_fill_manual(
      values = HIGHLIGHT_COLS,
      name   = NULL,
      labels = as.character(HIGHLIGHT_YEARS)
    ) +
    scale_x_continuous(breaks = seq(1980, 2020, by = 10)) +
    facet_grid(sector_label ~ param, scales = "free_y", switch = "y") +
    labs(
      title   = "Phase and amplitude anomalies by sector (1980\u20132023)",
      x       = "Year",
      y       = NULL,
      caption = "Fitted anomaly from long-run mean   |   dashed lines: 2016 (orange), 2023 (gold)"
    ) +
    theme_bw(base_size = 11) +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major   = element_line(colour = "grey92", linewidth = 0.3),
      strip.background   = element_rect(fill = "grey96", colour = "grey80"),
      strip.text.x       = element_text(size = 10, face = "bold"),
      strip.text.y.left  = element_text(angle = 0, size = 9, hjust = 1),
      strip.placement    = "outside",
      legend.position    = "bottom",
      plot.title         = element_text(size = 12, face = "bold"),
      plot.caption       = element_text(size = 8,  colour = "grey45"),
      panel.spacing.x    = unit(0.8, "cm"),
      panel.spacing.y    = unit(0.4, "cm")
    ) +
    guides(fill = guide_legend(override.aes = list(size = 3)))
  
  fname <- file.path(FIG_DIR, "sector_comparison_phase_amplitude.png")
  ggsave(fname, p, width = 10, height = 11, dpi = 150)
  message("Saved: ", fname)
  invisible(p)
}

plot_sector_comparison(annual_df)

message("\n=== Figures complete ===")
message("Per-sector time series : figures/annual_params_*.png")
message("Sector comparison      : figures/sector_comparison_phase_amplitude.png")

# =============================================================================
# 9. FIGURE 1 REPRODUCTION — tabled for later
# =============================================================================
# All component columns needed are present in daily_fitted.csv:
#   trend_component, amplitude_component, phase_component,
#   raw_anomaly, est_anomaly, fitted_invariant
# x-axis: t (days since annual minimum, 0-365)
# Day 0 reference: median min DOY per sector
# =============================================================================

message("\nFig 1 plotting tabled — all component columns saved in daily_fitted.csv")
message("\nFig 1 plotting tabled — all component columns saved in daily_fitted.csv")