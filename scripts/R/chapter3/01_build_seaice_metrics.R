# ============================================================
# Chapter 3: Build sea ice metrics
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(lubridate)
  library(ggplot2)
  library(mgcv)
  library(readr)
})

# -----------------------------
# 0. PATHS
# -----------------------------
BASE_DIR <- "~/Research/repos/sea-ice-phase/scripts/R/"
INFILE <- file.path(BASE_DIR, "Sea_Ice_Sheets/SIE_daily_sector_and_circumpolar_million_km2.csv")
OUTDIR <- file.path(BASE_DIR, "chapter3/data")

dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

message("Reading input: ", INFILE)

# -----------------------------
# 1. LOAD + RESHAPE
# -----------------------------
df_wide <- read_csv(INFILE, show_col_types = FALSE)

if (!"Date" %in% names(df_wide)) {
  stop("Input file must contain a Date column.")
}

sie_cols <- names(df_wide)[grepl("^SIE_", names(df_wide))]
if (length(sie_cols) == 0) {
  stop("No columns starting with 'SIE_' found in input file.")
}

df_long <- df_wide %>%
  mutate(Date = mdy(Date)) %>%
  filter(!is.na(Date)) %>%
  pivot_longer(
    cols = all_of(sie_cols),
    names_to = "sector",
    values_to = "Extent"
  ) %>%
  mutate(
    sector = str_remove(sector, "^SIE_"),
    Extent = as.numeric(Extent),
    Year = year(Date),
    Month = month(Date),
    Day = day(Date)
  ) %>%
  filter(!is.na(Extent)) %>%
  filter(!(Month == 2 & Day == 29)) %>%
  mutate(DOY = yday(Date)) %>%
  arrange(sector, Date)

sector_levels <- c(
  "Amundsen_Bellingshausen",
  "Ross",
  "Weddell",
  "East_Antarctica",
  "King_Haakon",
  "circumpolar"
)

present_levels <- sector_levels[sector_levels %in% unique(df_long$sector)]
other_levels <- setdiff(unique(df_long$sector), present_levels)

df_long <- df_long %>%
  mutate(sector = factor(sector, levels = c(present_levels, other_levels)))

# -----------------------------
# 2. HELPERS
# -----------------------------
shift_vec <- function(x, lag_days) {
  n <- length(x)
  lag_days <- lag_days %% n
  if (lag_days == 0) return(x)
  c(tail(x, n - lag_days), head(x, lag_days))
}

compute_runs <- function(sign_vec, doy_vec, target = c("positive", "negative")) {
  target <- match.arg(target)
  keep <- if (target == "positive") sign_vec > 0 else sign_vec < 0
  r <- rle(keep)
  
  lengths <- r$lengths
  values <- r$values
  ends <- cumsum(lengths)
  starts <- ends - lengths + 1
  
  out <- tibble(
    keep = values,
    start_idx = starts,
    end_idx = ends,
    run_length = lengths
  ) %>%
    filter(keep)
  
  if (nrow(out) == 0) {
    return(tibble(
      run_length = NA_integer_,
      start_doy = NA_integer_,
      end_doy = NA_integer_
    ))
  }
  
  longest <- out %>%
    slice_max(run_length, n = 1, with_ties = FALSE)
  
  tibble(
    run_length = longest$run_length,
    start_doy = doy_vec[longest$start_idx],
    end_doy = doy_vec[longest$end_idx]
  )
}

# -----------------------------
# 3. FIT IAC PER SECTOR
# -----------------------------
message("Fitting invariant annual cycle per sector...")

df_long_iac <- df_long %>%
  group_by(sector) %>%
  group_modify(~{
    mod <- gam(
      Extent ~ s(DOY, bs = "cc", k = 25),
      data = .x,
      method = "REML"
    )
    .x$IAC_pred <- predict(mod, newdata = .x)
    .x
  }) %>%
  ungroup()

# -----------------------------
# 4. BUILD DAILY IAC + ENVELOPE
# -----------------------------
iac_daily <- df_long_iac %>%
  group_by(sector, DOY) %>%
  summarise(
    IAC = mean(IAC_pred, na.rm = TRUE),
    Traditional = mean(Extent, na.rm = TRUE),
    p10 = quantile(Extent, 0.10, na.rm = TRUE),
    p25 = quantile(Extent, 0.25, na.rm = TRUE),
    p75 = quantile(Extent, 0.75, na.rm = TRUE),
    p90 = quantile(Extent, 0.90, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(sector) %>%
  arrange(DOY, .by_group = TRUE) %>%
  mutate(
    IAC_norm = (IAC - min(IAC, na.rm = TRUE)) /
      (max(IAC, na.rm = TRUE) - min(IAC, na.rm = TRUE)),
    dIAC = c(NA_real_, diff(IAC)),
    spread_10_90 = p90 - p10,
    spread_25_75 = p75 - p25
  ) %>%
  ungroup()

write_csv(iac_daily, file.path(OUTDIR, "sector_IAC_daily.csv"))
write_csv(iac_daily, file.path(OUTDIR, "sector_seasonal_envelope.csv"))

# -----------------------------
# 5. ANNUAL PHASE + AMPLITUDE
# -----------------------------
message("Computing annual phase scalar and amplitude...")

lags_to_test <- -40:40

phase_amp_sector <- df_long %>%
  group_by(sector, Year) %>%
  group_modify(~{
    sec_name <- as.character(.y$sector[[1]])
    
    dat <- .x %>%
      group_by(DOY) %>%
      summarise(Extent = mean(Extent, na.rm = TRUE), .groups = "drop") %>%
      arrange(DOY)
    
    if (nrow(dat) < 300) {
      return(tibble(
        phase_scalar = NA_real_,
        sse_min = NA_real_,
        amplitude_range = NA_real_,
        min_doy = NA_integer_,
        max_doy = NA_integer_,
        min_extent = NA_real_,
        max_extent = NA_real_
      ))
    }
    
    ref <- iac_daily %>%
      filter(as.character(sector) == sec_name) %>%
      select(DOY, IAC, IAC_norm)
    
    dat2 <- dat %>%
      left_join(ref, by = "DOY")
    
    obs_min <- min(dat2$Extent, na.rm = TRUE)
    obs_max <- max(dat2$Extent, na.rm = TRUE)
    obs_rng <- obs_max - obs_min
    
    if (is.na(obs_rng) || obs_rng == 0) {
      return(tibble(
        phase_scalar = NA_real_,
        sse_min = NA_real_,
        amplitude_range = NA_real_,
        min_doy = NA_integer_,
        max_doy = NA_integer_,
        min_extent = NA_real_,
        max_extent = NA_real_
      ))
    }
    
    obs_norm <- (dat2$Extent - obs_min) / obs_rng
    ref_norm <- dat2$IAC_norm
    
    sse_vals <- sapply(lags_to_test, function(L) {
      shifted_ref <- shift_vec(ref_norm, L)
      sum((obs_norm - shifted_ref)^2, na.rm = TRUE)
    })
    
    best_lag <- lags_to_test[which.min(sse_vals)]
    
    tibble(
      phase_scalar = best_lag,
      sse_min = min(sse_vals, na.rm = TRUE),
      amplitude_range = obs_rng,
      min_doy = dat2$DOY[which.min(dat2$Extent)],
      max_doy = dat2$DOY[which.max(dat2$Extent)],
      min_extent = obs_min,
      max_extent = obs_max
    )
  }) %>%
  ungroup() %>%
  group_by(sector) %>%
  mutate(
    phase_anom = phase_scalar - mean(phase_scalar, na.rm = TRUE),
    amp_anom = amplitude_range - mean(amplitude_range, na.rm = TRUE)
  ) %>%
  ungroup()

write_csv(phase_amp_sector, file.path(OUTDIR, "sector_annual_phase_amplitude.csv"))

# -----------------------------
# 6. GROWTH / RETREAT DURATION
# -----------------------------
duration_tbl <- iac_daily %>%
  group_by(sector) %>%
  group_modify(~{
    dat <- .x %>%
      arrange(DOY) %>%
      filter(!is.na(dIAC))
    
    growth <- compute_runs(dat$dIAC, dat$DOY, target = "positive")
    retreat <- compute_runs(dat$dIAC, dat$DOY, target = "negative")
    
    tibble(
      growth_days = growth$run_length,
      growth_start_doy = growth$start_doy,
      growth_end_doy = growth$end_doy,
      retreat_days = retreat$run_length,
      retreat_start_doy = retreat$start_doy,
      retreat_end_doy = retreat$end_doy,
      max_growth_doy = dat$DOY[which.max(dat$dIAC)],
      max_retreat_doy = dat$DOY[which.min(dat$dIAC)],
      max_growth_rate = max(dat$dIAC, na.rm = TRUE),
      max_retreat_rate = min(dat$dIAC, na.rm = TRUE)
    )
  }) %>%
  ungroup()

write_csv(duration_tbl, file.path(OUTDIR, "sector_growth_retreat_duration.csv"))

# -----------------------------
# 7. SUMMARY TABLE
# -----------------------------
summary_tbl <- phase_amp_sector %>%
  group_by(sector) %>%
  summarise(
    mean_phase = mean(phase_scalar, na.rm = TRUE),
    sd_phase = sd(phase_scalar, na.rm = TRUE),
    mean_amp = mean(amplitude_range, na.rm = TRUE),
    sd_amp = sd(amplitude_range, na.rm = TRUE),
    mean_sse = mean(sse_min, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(duration_tbl, by = "sector")

write_csv(summary_tbl, file.path(OUTDIR, "sector_summary_metrics.csv"))

message("Done.")
print(list.files(OUTDIR, full.names = FALSE))