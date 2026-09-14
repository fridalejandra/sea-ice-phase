# =============================================================================
# HR_reproduce_v2.R
#
# Reproduce H&R (2020) Fig 7a using Frida's CANONICAL model specs (which
# produced her Table 2), with a separate bs="tp" trend fit for Fig 7a's
# green line.
#
# This uses the practical implementation that WORKS for RMSE, not the
# stripped-down Eqs 10/12 which require per-year β estimation to fit well.
#
# Models (from canonical_files 1-4, all use bs="cc" on tdate as originals):
#   * Invariant:    Extent  ~ s(DOY, bs="cc", k=25)              [canonical_files1]
#   * Amp-only:     scaling ~ s(tdate, bs="cc", k=20)  + s(DOY, bs="cc", k=100)
#                                                                [canonical_files2]
#   * Phase-only:   Extent  ~ s(tdate, bs="cc", k=75)  + s(DOY, bs="cc", k=100)
#                                                                + s(phase, bs="cc", k=100)  [canonical_files3]
#   * APAC:         scaling ~ s(tdate, bs="cc", k=150) + s(DOY, bs="cc", k=100)
#                                                                + s(phase, bs="cc", k=100)  [canonical_files4]
#
# Separately for the Fig 7a trend line only:
#   * TrendExtract: Extent ~ s(tdate, bs="tp") + s(DOY, bs="cc", k=50)
#                                                                [Handcock's spec]
#
# Fig 7a decomposition:
#   invariant  = iac_notrend                           [no trend, DOY only]
#   trend      = s(tdate,tp) partial from TrendExtract [proper hook shape]
#   amplitude  = fitted_amp   - fitted_iac_notrend - trend
#   phase      = fitted_APAC  - fitted_amp
#   raw_anom   = Extent       - fitted_APAC
# =============================================================================

library(dplyr)
library(lubridate)
library(mgcv)

# ── SETTINGS ────────────────────────────────────────────────────────────────
DATE_START <- as.Date("1979-01-01")
DATE_END   <- as.Date("2018-12-31")
SECTOR     <- "SIE_circumpolar"

.roots <- c(Sys.getenv("SEAICE_ROOT", unset = NA),
            path.expand("~/Research/repos/sea-ice-phase"),
            "/Users/fridaperez/Research/repos/sea-ice-phase")
ROOT <- NA
for (r in .roots) if (!is.na(r) && dir.exists(file.path(r, "scripts"))) { ROOT <- r; break }
INPUT_FILE <- file.path(ROOT, "scripts", "R", "observations",
                        "SIE_daily_sector_and_circumpolar_million_km2.csv")
OUTPUT_DIR <- file.path(ROOT, "scripts", "R", "Ch3", "data")

# ── 1. LOAD ─────────────────────────────────────────────────────────────────
raw <- read.csv(INPUT_FILE, stringsAsFactors = FALSE)
names(raw)[tolower(names(raw)) %in% c("date", "time")] <- "Date"
if (grepl("^\\d{4}-\\d{2}-\\d{2}", raw$Date[1])) {
  raw$Date <- as.Date(raw$Date)
} else {
  raw$Date <- as.Date(raw$Date, format = "%m/%d/%y")
}
raw <- raw %>% filter(Date >= DATE_START, Date <= DATE_END) %>% arrange(Date)
raw$Year   <- year(raw$Date)
raw$DOY    <- yday(raw$Date)
raw$tdate  <- as.numeric(raw$Date)
raw$Extent <- as.numeric(raw[[SECTOR]])

sie <- raw %>% select(Date, Year, DOY, tdate, Extent) %>% filter(!is.na(Extent))
message("Loaded ", nrow(sie), " rows, ", min(sie$Date), " to ", max(sie$Date))

# ── 2. YEARLY MIN/MAX + SCALING + PHASE (canonical) ─────────────────────────
yr <- sie %>% group_by(Year) %>%
  summarise(min_extent = min(Extent),
            max_extent = max(Extent),
            amplitude  = max_extent - min_extent,
            Date1      = Date[which.min(Extent)],
            .groups    = "drop") %>%
  mutate(Date2 = lag(Date1), Date3 = lead(Date1))

sie <- sie %>% left_join(yr, by = "Year") %>%
  mutate(scaling = (Extent - min_extent) / (amplitude + 1e-10))

sie <- sie %>% rowwise() %>%
  mutate(t = case_when(
    Year == 1979 & Date < Date1 ~ 365 - as.numeric(Date1 - Date),
    Date >= Date1               ~ as.numeric(Date - Date1),
    Date <  Date1               ~ as.numeric(Date - Date2)
  )) %>% ungroup() %>% filter(!is.na(t))

t_stats <- sie %>% group_by(Year) %>%
  summarise(t_min = min(t), t_max = max(t), .groups = "drop")
sie <- sie %>% left_join(t_stats, by = "Year") %>%
  mutate(phase = 365 * pbeta((t - t_min) / (t_max - t_min + 1e-10), 1, 1)) %>%
  filter(!is.na(phase))

# ── 3. FIT MODELS EXACTLY AS FRIDA'S CANONICAL SCRIPTS ──────────────────────

# Model 1: Traditional
trad_mean <- sie %>% group_by(DOY) %>%
  summarise(fitted_trad = mean(Extent), .groups = "drop")
sie <- sie %>% left_join(trad_mean, by = "DOY")
rmse_trad <- sqrt(mean((sie$Extent - sie$fitted_trad)^2))

# Model 2: Invariant (from canonical_files1) — DOY only, no trend
message("Fitting invariant (canonical)...")
gam_iac_notrend <- gam(Extent ~ s(DOY, bs = "cc", k = 25), data = sie)
sie$iac_notrend <- as.numeric(predict(gam_iac_notrend))
rmse_iac_notrend <- sqrt(mean((sie$Extent - sie$iac_notrend)^2))

# Model 3: Amplitude-adjusted (from canonical_files2)
message("Fitting amplitude (canonical)...")
gam_amp <- gam(scaling ~ s(tdate, bs = "cc", k = 20) +
                 s(DOY,   bs = "cc", k = 100),
               data = sie)
sie$fitted_amp <- as.numeric(predict(gam_amp)) * sie$amplitude + sie$min_extent
rmse_amp <- sqrt(mean((sie$Extent - sie$fitted_amp)^2))

# Model 4: Phase-adjusted (from canonical_files3)
message("Fitting phase (canonical)...")
gam_phase <- gam(Extent ~ s(tdate, bs = "cc", k = 75) +
                   s(DOY,   bs = "cc", k = 100) +
                   s(phase, bs = "cc", k = 100, fx = FALSE),
                 data = sie)
sie$fitted_phase <- as.numeric(predict(gam_phase))
rmse_phase <- sqrt(mean((sie$Extent - sie$fitted_phase)^2))

# Model 5: APAC (from canonical_files4)
message("Fitting APAC (canonical)...")
gam_apac <- gam(scaling ~ s(tdate, bs = "cc", k = 150) +
                  s(DOY,   bs = "cc", k = 100) +
                  s(phase, bs = "cc", k = 100),
                data = sie)
sie$fitted_apac   <- as.numeric(predict(gam_apac)) * sie$amplitude + sie$min_extent
sie$residual_apac <- sie$Extent - sie$fitted_apac
rmse_apac <- sqrt(mean(sie$residual_apac^2))

# ── SEPARATE TREND FIT FOR FIG 7a GREEN LINE ────────────────────────────────
# canonical bs="cc" on tdate suppresses trend. We fit ONE extra model with
# bs="tp" purely to extract a proper trend for Fig 7a.
message("Fitting trend-extraction model (bs=tp, Handcock spec)...")
gam_trend_extract <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 50, fx = FALSE),
                         data = sie, method = "REML",
                         knots = list(DOY = c(0, 365)))
trend_terms <- predict(gam_trend_extract, type = "terms")
sie$trend_component <- as.numeric(trend_terms[, "s(tdate)"])

# ── 4. FIG 7a COMPONENTS ────────────────────────────────────────────────────
sie <- sie %>% mutate(
  invariant_component = iac_notrend,
  amplitude_component = fitted_amp  - iac_notrend - trend_component,
  phase_component     = fitted_apac - fitted_amp,
  raw_anomaly         = Extent - fitted_apac
)

# ── 5. RESULTS ──────────────────────────────────────────────────────────────
pct <- function(r) 100 * (1 - r^2 / rmse_trad^2)

cat("\n============= H&R Table 1 REPRODUCTION =============\n")
cat(sprintf("                          Yours          H&R Paper\n"))
cat(sprintf("  Traditional:            %.3f          0.576\n", rmse_trad))
cat(sprintf("  Invariant (DOY-only):   %.3f (%.1f%%)  0.482 (28.7%%)\n",
            rmse_iac_notrend, pct(rmse_iac_notrend)))
cat(sprintf("  Amplitude-adjusted:     %.3f (%.1f%%)  0.382 (55.2%%)\n",
            rmse_amp, pct(rmse_amp)))
cat(sprintf("  Phase-adjusted:         %.3f (%.1f%%)  0.343 (63.9%%)\n",
            rmse_phase, pct(rmse_phase)))
cat(sprintf("  APAC:                   %.3f (%.1f%%)  0.272 (77.3%%)\n",
            rmse_apac, pct(rmse_apac)))

c2016 <- sie %>% filter(Year == 2016)
cat("\n============= 2016 COMPONENT MAGNITUDES =============\n")
cat("(H&R Fig 7a targets: trend ~3, amp ~3, phase ~11)\n\n")
for (col in c("invariant_component", "trend_component", "amplitude_component",
              "phase_component", "raw_anomaly")) {
  v <- c2016[[col]]
  cat(sprintf("  %-22s  min %7.3f   max %7.3f   range %7.3f\n",
              col, min(v), max(v), max(v) - min(v)))
}

# ── 6. SAVE ─────────────────────────────────────────────────────────────────
daily_out <- sie %>%
  select(Date, Year, DOY, tdate, Extent, phase, t, t_min, t_max, scaling,
         min_extent, max_extent, amplitude,
         iac_notrend, fitted_amp, fitted_phase, fitted_apac,
         residual_apac,
         invariant_component, trend_component, amplitude_component,
         phase_component, raw_anomaly) %>%
  mutate(sector = SECTOR)

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
outfile <- file.path(OUTPUT_DIR, "daily_HR_v2.csv")
write.csv(daily_out, outfile, row.names = FALSE)
cat("\nWrote ", outfile, "\n")