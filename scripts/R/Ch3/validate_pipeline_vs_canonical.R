# validate_pipeline_vs_canonical.R
# =================================
# Runs all four canonical APAC models on the v4 circumpolar data
# and compares RMSE values against the pipeline output.
#
# Also compares annual amplitude and phase scalars year-by-year
# to verify the pipeline extracts the same values as the canonical approach.
#
# Run on your Mac from the repo root.

library(dplyr)
library(lubridate)
library(mgcv)

# ── Paths ─────────────────────────────────────────────────────────────────────
SIE_FILE     <- "~/Research/repos/sea-ice-phase/scripts/R/observations/SIE_daily_sector_and_circumpolar_million_km2.csv"
PIPELINE_CSV <- "~/Research/repos/sea-ice-phase/scripts/R/Ch3/data/annual_params.csv"
DAILY_CSV    <- "~/Research/repos/sea-ice-phase/scripts/R/Ch3/data/daily_fitted.csv"

YEAR_MIN <- 1979
YEAR_MAX <- 2018   # Match H&R validation period

cat("Loading data...\n")
raw <- read.csv(SIE_FILE)
# Handle both 'Date' and 'time' column names across file versions
if ("time" %in% names(raw)) {
  raw$Date <- as.Date(raw$time)
} else {
  raw$Date <- as.Date(raw$Date, format="%m/%d/%y")
}
raw$Year  <- year(raw$Date)
raw$Month <- month(raw$Date)
raw$DOY   <- yday(raw$Date)
raw$tdate <- as.numeric(raw$Date)

# Use circumpolar for validation
df <- raw %>%
  filter(Year >= YEAR_MIN, Year <= YEAR_MAX) %>%
  select(Date, Year, Month, DOY, tdate, Extent = SIE_circumpolar) %>%
  filter(!is.na(Extent))

cat(sprintf("  %d rows, %d years\n", nrow(df), n_distinct(df$Year)))

# ── 1. Traditional ────────────────────────────────────────────────────────────
cat("\n1. Traditional annual cycle...\n")
trad_clim <- df %>% group_by(DOY) %>% summarise(mean_extent = mean(Extent, na.rm=TRUE))
df <- df %>% left_join(trad_clim, by="DOY")
rmse_trad <- sqrt(mean((df$mean_extent - df$Extent)^2, na.rm=TRUE))
cat(sprintf("   RMSE traditional: %.4f\n", rmse_trad))

# ── 2. Invariant ──────────────────────────────────────────────────────────────
cat("\n2. Invariant annual cycle...\n")
gam_inv <- gam(Extent ~ s(tdate, bs="cc", k=14) + s(DOY, bs="cc", k=25),
               data=df, method="GCV.Cp",
               knots=list(DOY=c(1,365)))
df$fitted_invariant <- predict(gam_inv)
rmse_inv <- sqrt(mean((df$fitted_invariant - df$Extent)^2, na.rm=TRUE))
pct_inv  <- 100 * (1 - rmse_inv^2 / rmse_trad^2)
cat(sprintf("   RMSE invariant:   %.4f  (%.1f%%)\n", rmse_inv, pct_inv))

# ── 3. Amplitude-adjusted ─────────────────────────────────────────────────────
cat("\n3. Amplitude-adjusted annual cycle...\n")
yr_stats <- df %>%
  group_by(Year) %>%
  summarise(max_ext = max(Extent, na.rm=TRUE),
            min_ext = min(Extent, na.rm=TRUE), .groups="drop")
df <- df %>%
  left_join(yr_stats, by="Year") %>%
  mutate(amplitude      = max_ext - min_ext,
         scaling_factor = (Extent - min_ext) / amplitude)

gam_amp <- gam(scaling_factor ~ s(tdate, bs="cc", k=20) + s(DOY, bs="cc", k=100),
               data=df, method="GCV.Cp",
               knots=list(DOY=c(1,365)))
df$pred_scaling  <- predict(gam_amp)
df$fitted_amp    <- df$pred_scaling * df$amplitude + df$min_ext
rmse_amp  <- sqrt(mean((df$fitted_amp - df$Extent)^2, na.rm=TRUE))
pct_amp   <- 100 * (1 - rmse_amp^2 / rmse_trad^2)
cat(sprintf("   RMSE amplitude:   %.4f  (%.1f%%)\n", rmse_amp, pct_amp))

# ── 4. Phase warp ─────────────────────────────────────────────────────────────
cat("\n4. Computing phase warp...\n")
yr_min <- df %>%
  group_by(Year) %>%
  summarise(Date1 = Date[which.min(Extent)], .groups="drop") %>%
  mutate(Date1 = as.Date(Date1),
         Date2 = as.Date(lag(Date1)),
         Date3 = as.Date(lead(Date1)))

df <- df %>% left_join(yr_min, by="Year")
df$Date  <- as.Date(df$Date)
df$Date1 <- as.Date(df$Date1)
df$Date2 <- as.Date(df$Date2)
df$Date3 <- as.Date(df$Date3)

df <- df %>%
  rowwise() %>%
  mutate(t = case_when(
    Year == YEAR_MIN & Date < Date1 ~ 365 - as.numeric(Date1 - Date),
    Date >= Date1                   ~ as.numeric(Date - Date1),
    Date <  Date1                   ~ as.numeric(Date - Date2)
  )) %>%
  ungroup()

t_stats <- df %>%
  group_by(Year) %>%
  summarise(t_min = min(t, na.rm=TRUE),
            t_max = max(t, na.rm=TRUE), .groups="drop")
df <- df %>% left_join(t_stats, by="Year")
df <- df %>%
  rowwise() %>%
  mutate(phase = 365 * pbeta((t - t_min) / (t_max - t_min + 1e-10),
                             shape1=1, shape2=1)) %>%
  ungroup()

# ── 5. Full APAC ──────────────────────────────────────────────────────────────
cat("\n5. Full APAC (amplitude + phase)...\n")
gam_apac <- gam(scaling_factor ~ s(tdate, bs="cc", k=150) +
                  s(DOY, bs="cc", k=100) +
                  s(phase, bs="cc", k=100),
                data=df, method="GCV.Cp",
                knots=list(DOY=c(1,365), phase=c(0,365)))
df$pred_apac   <- predict(gam_apac)
df$fitted_apac <- df$pred_apac * df$amplitude + df$min_ext
rmse_apac <- sqrt(mean((df$fitted_apac - df$Extent)^2, na.rm=TRUE))
pct_apac  <- 100 * (1 - rmse_apac^2 / rmse_trad^2)
cat(sprintf("   RMSE full APAC:   %.4f  (%.1f%%)\n", rmse_apac, pct_apac))

# ── 6. Summary vs H&R and pipeline ────────────────────────────────────────────
cat("\n")
cat(strrep("=", 60), "\n")
cat("CANONICAL vs PIPELINE vs H&R 2020\n")
cat(strrep("=", 60), "\n")
cat(sprintf("%-25s %10s %10s %10s\n", "Model", "Canonical", "Pipeline", "H&R 2020"))
cat(strrep("-", 55), "\n")

# Load pipeline RMSE
pipe_rmse <- read.csv("~/Research/repos/sea-ice-phase/scripts/R/Ch3/data/rmse_scenarios/v4_1979_2018/rmse_summary.csv")
pipe <- pipe_rmse %>% filter(sector == "SIE_circumpolar")

cat(sprintf("%-25s %10.4f %10.4f %10s\n", "Traditional",  rmse_trad, pipe$rmse_trad,  "0.5760"))
cat(sprintf("%-25s %10.4f %10.4f %10s\n", "Invariant",    rmse_inv,  pipe$rmse_iac,   "0.4820"))
cat(sprintf("%-25s %10.4f %10.4f %10s\n", "Amplitude",    rmse_amp,  pipe$rmse_amp,   "0.3820"))
cat(sprintf("%-25s %10.4f %10.4f %10s\n", "Full APAC",    rmse_apac, pipe$rmse_apac,  "0.2720"))

# ── 7. Annual amplitude comparison ────────────────────────────────────────────
cat("\n")
cat(strrep("=", 60), "\n")
cat("ANNUAL AMPLITUDE: Canonical vs Pipeline (circumpolar)\n")
cat(strrep("=", 60), "\n")

# Canonical annual amplitude from fitted_amp
canon_amp <- df %>%
  group_by(Year) %>%
  summarise(
    canon_amp_fitted = max(fitted_amp, na.rm=TRUE) - min(fitted_amp, na.rm=TRUE),
    canon_amp_raw    = max(Extent,     na.rm=TRUE) - min(Extent,     na.rm=TRUE),
    .groups="drop"
  )

# Pipeline annual amplitude
pipe_daily <- read.csv(DAILY_CSV)
pipe_amp <- pipe_daily %>%
  filter(sector == "SIE_circumpolar",
         Year >= YEAR_MIN, Year <= YEAR_MAX) %>%
  group_by(Year) %>%
  summarise(pipe_amp_fitted = max(fitted_amp, na.rm=TRUE) - min(fitted_amp, na.rm=TRUE),
            .groups="drop")

amp_comp <- canon_amp %>% inner_join(pipe_amp, by="Year")
r_amp    <- cor(amp_comp$canon_amp_fitted, amp_comp$pipe_amp_fitted)
cat(sprintf("  Correlation canonical vs pipeline amplitude: r = %.4f\n", r_amp))
cat(sprintf("  Mean canonical: %.4f  |  Mean pipeline: %.4f\n",
            mean(amp_comp$canon_amp_fitted), mean(amp_comp$pipe_amp_fitted)))

# ── 8. Annual phase comparison ────────────────────────────────────────────────
cat("\n")
cat(strrep("=", 60), "\n")
cat("ANNUAL PHASE: Canonical vs Pipeline (circumpolar)\n")
cat(strrep("=", 60), "\n")

# Canonical max DOY from fitted_apac
canon_phase <- df %>%
  group_by(Year) %>%
  summarise(canon_max_doy = DOY[which.max(fitted_apac)], .groups="drop")

pipe_annual <- read.csv(PIPELINE_CSV)
pipe_phase <- pipe_annual %>%
  filter(sector == "SIE_circumpolar",
         Year >= YEAR_MIN, Year <= YEAR_MAX) %>%
  select(Year, pipe_max_doy = max_doy_fitted)

phase_comp <- canon_phase %>% inner_join(pipe_phase, by="Year")
r_phase    <- cor(phase_comp$canon_max_doy, phase_comp$pipe_max_doy)
cat(sprintf("  Correlation canonical vs pipeline phase: r = %.4f\n", r_phase))
cat(sprintf("  Mean canonical: %.1f  |  Mean pipeline: %.1f\n",
            mean(phase_comp$canon_max_doy), mean(phase_comp$pipe_max_doy)))

cat("\nValidation complete.\n")
