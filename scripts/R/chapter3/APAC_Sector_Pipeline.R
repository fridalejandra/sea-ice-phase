# =============================================================================
# APAC Sector Pipeline
# Implements the full Handcock & Raphael (2020) framework for 5 Antarctic
# sea ice sectors, faithfully following the mathematical specification
# in Sections 3.1 and 3.2 of the paper.
#
# Outputs:
#   1. annual_params.csv  — per-year, per-sector phase & amplitude parameters
#   2. daily_fitted.csv   — daily fitted APAC, invariant cycle, residuals, trend
#   3. volatility.csv     — per-sector volatility (daily SD from GARCH model)
#
# Reference: Handcock & Raphael (2020), The Cryosphere, 14, 2159-2172.
#            Raphael et al. (2020), JGR Oceans, 125, e2020JC016459.
# =============================================================================

library(dplyr)
library(lubridate)
library(mgcv)      # for gam() with cyclic cubic splines and thin-plate splines
library(rugarch)   # for GARCH volatility model

# =============================================================================
# 0. USER SETTINGS — adjust paths here only
# =============================================================================

INPUT_FILE  <- "~/Research/repos/sea-ice-phase/scripts/R/Sea_Ice_Sheets/SIE_daily_sector_and_circumpolar_million_km2.csv"
OUTPUT_DIR  <- "~/Research/repos/sea-ice-phase/scripts/R/chapter3/data"

# Date range to analyse (paper uses 1979-2018 for observations)
DATE_START <- as.Date("1979-01-01")
DATE_END   <- as.Date("2023-12-31")
# =============================================================================
# APAC Sector Pipeline — Path A (Linear Phase Warp)
#
# Implements the amplitude-adjusted annual cycle with linear phase warp
# (Beta = 1,1) for 5 Antarctic sea ice sectors, following the regional
# application in Raphael et al. (2020, JGR Oceans) and the mathematical
# framework of Handcock & Raphael (2020, The Cryosphere).
#
# With Beta=(1,1), phase is a linear rescaling of time between the observed
# annual minimum and maximum days. This captures WHEN transitions happen
# (the key quantity for Ch3 atmospheric regressions) without requiring
# the iterative Beta optimisation of the full APAC.
#
# Outputs:
#   annual_params.csv — per-year, per-sector phase & amplitude parameters
#   daily_fitted.csv  — daily fitted cycles, residuals, trend, volatility
#   rmse_summary.csv  — RMSE comparison table (Table 1 analogue)
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

INPUT_FILE <- "~/Research/repos/sea-ice-phase/scripts/R/Sea_Ice_Sheets/SIE_daily_sector_and_circumpolar_million_km2.csv"
OUTPUT_DIR <- "~/Research/repos/sea-ice-phase/scripts/R/chapter3/data"

DATE_START <- as.Date("1979-01-01")
DATE_END   <- as.Date("2023-12-31")

SECTOR_COLS <- c(
  "SIE_Weddell",
  "SIE_Amundsen_Bellingshausen",
  "SIE_Ross",
  "SIE_East_Antarctica",
  "SIE_King_Haakon"
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
raw$tdate <- raw$Year + (raw$DOY - 1) / 365.25

message("Data loaded: ", nrow(raw), " rows from ",
        min(raw$Date), " to ", max(raw$Date))

# =============================================================================
# 2. CORE FITTING FUNCTION
# =============================================================================

fit_sector <- function(raw_data, sector_col) {
  
  message("\n========================================")
  message("Sector: ", sector_col)
  message("========================================")
  
  # --- 2a. Prepare sector time series ---------------------------------------
  sie <- raw_data %>%
    select(Date, Year, DOY, tdate, Extent = all_of(sector_col)) %>%
    filter(!is.na(Extent)) %>%
    arrange(Date)
  
  # --- 2b. Per-year amplitude statistics (Eq. 4-6) --------------------------
  yearly_stats <- sie %>%
    group_by(Year) %>%
    summarise(
      min_extent = min(Extent,  na.rm = TRUE),
      max_extent = max(Extent,  na.rm = TRUE),
      amplitude  = max_extent - min_extent,
      min_doy    = DOY[which.min(Extent)],
      max_doy    = DOY[which.max(Extent)],
      min_date   = Date[which.min(Extent)],
      max_date   = Date[which.max(Extent)],
      .groups    = "drop"
    )
  
  sie <- sie %>% left_join(yearly_stats, by = "Year")
  
  # Standardised extent: u_A[s] = (Extent - min) / (max - min)
  sie <- sie %>%
    mutate(scaled_extent = (Extent - min_extent) / (amplitude + 1e-10))
  
  # --- 2c. Linear phase warp Beta(1,1) (Eq. 8-9) ----------------------------
  # Phase maps each day linearly onto [0,365] between the annual min date
  # and the next year's min date. Beta(1,1) means no shape warp — just a
  # linear stretch/compress of the calendar between those anchors.
  # This is equivalent to pbeta(t_norm, 1, 1) = t_norm.
  
  min_dates <- yearly_stats %>% select(Year, cycle_min_date = min_date)
  prev_min  <- yearly_stats %>%
    mutate(Year = Year + 1L) %>%
    select(Year, prev_min_date = min_date)
  next_min  <- yearly_stats %>%
    mutate(Year = Year - 1L) %>%
    select(Year, next_min_date = min_date)
  
  sie <- sie %>%
    left_join(min_dates, by = "Year") %>%
    left_join(prev_min,  by = "Year") %>%
    left_join(next_min,  by = "Year") %>%
    mutate(
      t_from_min = if_else(
        Date >= cycle_min_date,
        as.numeric(Date - cycle_min_date),
        as.numeric(Date - prev_min_date)
      ),
      cycle_length = as.numeric(next_min_date - cycle_min_date),
      t_norm = pmax(0, pmin(1, t_from_min / (cycle_length + 1e-10))),
      phase  = pmax(0.001, pmin(364.999, 365 * t_norm))
    ) %>%
    filter(!is.na(phase) & !is.na(scaled_extent))
  
  message("  Rows after phase computation: ", nrow(sie))
  
  # --- 2d. Invariant annual cycle + trend (Eq. 3, 15) -----------------------
  message("  Fitting invariant annual cycle + trend...")
  
  gam_invariant <- gam(
    Extent ~ s(DOY, bs = "cc", k = 100) + s(tdate, bs = "tp"),
    data   = sie,
    method = "GCV.Cp",
    knots  = list(DOY = c(1, 365))
  )
  
  sie$fitted_invariant <- as.numeric(predict(gam_invariant))
  sie$fitted_trend     <- as.numeric(
    predict(gam_invariant, type = "terms")[, "s(tdate)"] +
      coef(gam_invariant)[1]
  )
  sie$anomaly_from_iac <- sie$Extent - sie$fitted_invariant
  rmse_iac <- sqrt(mean(sie$anomaly_from_iac^2, na.rm = TRUE))
  message("  IAC RMSE: ", round(rmse_iac, 4), " million km2")
  
  # --- 2e. Amplitude-adjusted annual cycle (Eq. 4-6) ------------------------
  message("  Fitting amplitude-adjusted annual cycle...")
  
  gam_amp <- gam(
    scaled_extent ~ s(phase, bs = "cc", k = 100),
    data   = sie,
    method = "GCV.Cp",
    knots  = list(phase = c(0, 365))
  )
  
  sie$pred_scaled_amp <- as.numeric(predict(gam_amp))
  sie$fitted_amp      <- sie$pred_scaled_amp * sie$amplitude + sie$min_extent
  sie$residual_amp    <- sie$Extent - sie$fitted_amp
  
  rmse_amp    <- sqrt(mean(sie$residual_amp^2, na.rm = TRUE))
  pct_imp_amp <- 100 * (1 - rmse_amp^2 / rmse_iac^2)
  message("  Amplitude-adjusted RMSE: ", round(rmse_amp, 4),
          " million km2  (", round(pct_imp_amp, 1), "% improvement)")
  
  # --- 2f. Volatility: GARCH(2,2) on residuals (Section 3.2.1) -------------
  message("  Fitting GARCH(2,2) volatility model...")
  
  spec <- ugarchspec(
    variance.model     = list(model = "sGARCH", garchOrder = c(2, 2)),
    mean.model         = list(armaOrder = c(1, 1), include.mean = TRUE),
    distribution.model = "norm"
  )
  
  resid_clean <- sie$residual_amp[!is.na(sie$residual_amp)]
  
  garch_fit <- tryCatch(
    ugarchfit(spec = spec, data = resid_clean, solver = "hybrid"),
    error = function(e) { message("  GARCH failed: ", e$message); NULL }
  )
  
  sie$volatility <- NA_real_
  if (!is.null(garch_fit)) {
    sie$volatility[!is.na(sie$residual_amp)] <- as.numeric(sigma(garch_fit))
    message("  Volatility estimated successfully.")
  }
  
  # --- 2g. Annual parameter table (key Ch3 output) --------------------------
  annual_params <- yearly_stats %>%
    mutate(
      sector         = sector_col,
      min_doy_anom   = min_doy - median(min_doy, na.rm = TRUE),
      max_doy_anom   = max_doy - median(max_doy, na.rm = TRUE),
      amplitude_anom = amplitude - mean(amplitude, na.rm = TRUE)
    ) %>%
    select(
      sector, Year,
      min_extent, max_extent, amplitude, amplitude_anom,
      min_doy, max_doy, min_date, max_date,
      min_doy_anom, max_doy_anom
    )
  
  # --- 2h. Daily output table -----------------------------------------------
  daily_out <- sie %>%
    select(
      Date, Year, DOY, tdate, Extent,
      phase, t_norm, t_from_min, cycle_length,
      scaled_extent,
      fitted_invariant, fitted_trend,
      fitted_amp, residual_amp,
      anomaly_from_iac, volatility
    ) %>%
    mutate(sector = sector_col)
  
  list(
    annual        = annual_params,
    daily         = daily_out,
    rmse_iac      = rmse_iac,
    rmse_amp      = rmse_amp,
    pct_imp_amp   = pct_imp_amp,
    gam_invariant = gam_invariant,
    gam_amp       = gam_amp
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
    sector      = sec,
    rmse_iac    = result$rmse_iac,
    rmse_amp    = result$rmse_amp,
    pct_imp_amp = result$pct_imp_amp
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
write.csv(annual_df, file.path(OUTPUT_DIR, "annual_params.csv"), row.names = FALSE)
write.csv(daily_df,  file.path(OUTPUT_DIR, "daily_fitted.csv"),  row.names = FALSE)
write.csv(rmse_df,   file.path(OUTPUT_DIR, "rmse_summary.csv"),  row.names = FALSE)

message("\n=== Done ===")
message("annual_params.csv : ", nrow(annual_df), " rows")
message("daily_fitted.csv  : ", nrow(daily_df),  " rows")

# =============================================================================
# 5. VALIDATION PLOTS
# =============================================================================

if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  library(tidyr)
  
  # Fig 5 style: amplitude-adjusted cycle shape by sector
  plot_shape <- daily_df %>%
    mutate(phase_bin = round(phase)) %>%
    group_by(sector, phase_bin) %>%
    summarise(mean_scaled = mean(scaled_extent, na.rm = TRUE), .groups = "drop")
  
  p1 <- ggplot(plot_shape, aes(x = phase_bin, y = mean_scaled, colour = sector)) +
    geom_line(linewidth = 0.8) +
    labs(title    = "Amplitude-adjusted annual cycle by sector",
         subtitle = "Reproduces Fig. 5 style — Raphael et al. (2020)",
         x = "Day of cycle (0 = annual minimum)",
         y = "Standardised SIE", colour = "Sector") +
    theme_minimal(base_size = 12)
  
  ggsave(file.path(OUTPUT_DIR, "cycle_shape_by_sector.png"),
         p1, width = 10, height = 5, dpi = 150)
  
  # Phase anomaly time series — primary Ch3 input
  p2 <- ggplot(annual_df, aes(x = Year, y = min_doy_anom, colour = sector)) +
    geom_line(linewidth = 0.7) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
    labs(title    = "Annual minimum timing anomaly by sector",
         subtitle = "Positive = later than median; Negative = earlier",
         x = "Year", y = "Min DOY anomaly (days)", colour = "Sector") +
    theme_minimal(base_size = 12)
  
  ggsave(file.path(OUTPUT_DIR, "phase_anomaly_by_sector.png"),
         p2, width = 10, height = 5, dpi = 150)
  
  message("Plots saved.")
  
  
  
  ##### Check
  # Load the output
  annual <- read.csv(file.path(OUTPUT_DIR, "annual_params.csv"))
  
  # 1. How many years per sector?
  table(annual$sector)
  
  # 2. Median min and max DOY per sector
  # (should roughly match Table 1 in Raphael et al. 2020)
  annual %>%
    group_by(sector) %>%
    summarise(
      median_min_doy = median(min_doy),
      median_max_doy = median(max_doy),
      mean_amplitude = round(mean(amplitude), 3)
    )
  
  # 3. Check the 2016 anomaly — should be a large negative min_doy_anom
  # (ice retreated much earlier than normal)
  annual %>%
    filter(Year == 2016) %>%
    select(sector, Year, min_doy_anom, max_doy_anom, amplitude_anom)
  
  # 4. Quick check for any NA in key columns
  colSums(is.na(annual))
}

# 1. How many years per sector?
table(annual$sector)

# 2. Median min and max DOY per sector
annual %>%
  group_by(sector) %>%
  summarise(
    median_min_doy = median(min_doy),
    median_max_doy = median(max_doy),
    mean_amplitude = round(mean(amplitude), 3)
  )

# 3. Check the 2016 anomaly
annual %>%
  filter(Year == 2016) %>%
  select(sector, Year, min_doy_anom, max_doy_anom, amplitude_anom)

annual %>%
  group_by(sector) %>%
  summarise(
    median_min_doy = median(min_doy),
    median_max_doy = median(max_doy),
    mean_amplitude = round(mean(amplitude), 3)
  )