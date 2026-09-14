# =============================================================================
# quick_HR_check.R  --  3-minute sanity check against H&R Table 1
#
# Circumpolar only, 1979-2018 (H&R's window), Beta = (1,1), no GARCH.
# Not the final pipeline — just verifies the Handcock fixes reproduce the paper.
#
# H&R Table 1 targets: IAC 28.7 | Amp 55.2 | Phase 63.9 | APAC 77.3
# =============================================================================

library(dplyr); library(lubridate); library(mgcv)

# --- locate input file (same logic as Pipeline D) ---------------------------
.roots <- c(Sys.getenv("SEAICE_ROOT", unset = NA),
            path.expand("~/Research/repos/sea-ice-phase"),
            "/Users/fridaperez/Research/repos/sea-ice-phase")
ROOT <- NA
for (r in .roots) if (!is.na(r) && dir.exists(file.path(r, "scripts"))) { ROOT <- r; break }
if (is.na(ROOT)) stop("Cannot locate sea-ice-phase repo. Set SEAICE_ROOT env var.")
INPUT_FILE <- file.path(ROOT, "scripts", "R", "observations",
                        "SIE_daily_sector_and_circumpolar_million_km2.csv")
if (!file.exists(INPUT_FILE)) stop("Input CSV not found at: ", INPUT_FILE)

# --- load ------------------------------------------------------------------
raw <- read.csv(INPUT_FILE, stringsAsFactors = FALSE)
cat("Columns found:\n"); print(names(raw))
cat("First row:\n"); print(head(raw, 1))

# Normalize Date column name (Pipeline D convention)
names(raw)[tolower(names(raw)) %in% c("date", "time")] <- "Date"
if (!"Date" %in% names(raw)) stop("No Date/date/time column found. Columns: ",
                                  paste(names(raw), collapse = ", "))

# Also normalize circumpolar column name in case it varies
if (!"SIE_circumpolar" %in% names(raw)) {
  cand <- grep("circumpolar|CIRCUMPOLAR", names(raw), value = TRUE)
  if (length(cand) == 1) {
    names(raw)[names(raw) == cand] <- "SIE_circumpolar"
    cat("Renamed", cand, "-> SIE_circumpolar\n")
  } else {
    stop("Cannot find circumpolar column. Columns: ",
         paste(names(raw), collapse = ", "))
  }
}

# --- parse date -------------------------------------------------------------
if (length(raw$Date) == 0) stop("Date column is empty after normalization.")

first_date <- raw$Date[1]
cat("First date raw value: '", first_date, "'\n", sep = "")

if (grepl("^\\d{4}-\\d{2}-\\d{2}", first_date)) {
  raw$Date <- as.Date(raw$Date)
} else {
  raw$Date <- as.Date(raw$Date, format = "%m/%d/%y")
}
if (all(is.na(raw$Date))) stop("Date parsing failed. First raw value: ", first_date)

# --- H&R window, numeric cast (Handcock's flagged bug) ----------------------
raw <- raw %>% filter(Date >= as.Date("1979-01-01"),
                      Date <= as.Date("2018-12-31")) %>% arrange(Date)
raw$Year  <- year(raw$Date)
raw$DOY   <- yday(raw$Date)
raw$tdate <- as.numeric(raw$Date)
raw$SIE_circumpolar <- as.numeric(raw$SIE_circumpolar)

cat("Loaded", nrow(raw), "rows from", format(min(raw$Date)), "to",
    format(max(raw$Date)), "\n")

sie <- raw %>% select(Date, Year, DOY, tdate, Extent = SIE_circumpolar) %>%
  filter(!is.na(Extent))
cat("After NA filter:", nrow(sie), "rows\n")

# --- per-year stats, min-anchored phase, scaling factor ---------------------
yr <- sie %>% group_by(Year) %>%
  summarise(min_extent = min(Extent), max_extent = max(Extent),
            amplitude = max_extent - min_extent,
            Date1 = Date[which.min(Extent)], .groups = "drop") %>%
  mutate(Date2 = lag(Date1), Date3 = lead(Date1))

sie <- sie %>% left_join(yr, by = "Year") %>%
  mutate(scaling_factor = (Extent - min_extent) / (amplitude + 1e-10)) %>%
  rowwise() %>%
  mutate(t = case_when(
    Year == 1979 & Date < Date1 ~ 365 - as.numeric(Date1 - Date),
    Date >= Date1               ~ as.numeric(Date - Date1),
    Date <  Date1               ~ as.numeric(Date - Date2))) %>%
  ungroup() %>% filter(!is.na(t))

t_stats <- sie %>% group_by(Year) %>%
  summarise(t_min = min(t), t_max = max(t), .groups = "drop")
sie <- sie %>% left_join(t_stats, by = "Year") %>%
  mutate(phase = 365 * pbeta((t - t_min) / (t_max - t_min + 1e-10), 1, 1)) %>%
  filter(!is.na(phase))

cat("Ready to fit GAMs on", nrow(sie), "rows\n\n")

# --- Model 1: traditional ---------------------------------------------------
cat("Fitting Model 1: traditional...\n")
trad <- sie %>% group_by(DOY) %>%
  summarise(trad_mean = mean(Extent), .groups = "drop")
sie <- sie %>% left_join(trad, by = "DOY")
rmse_trad <- sqrt(mean((sie$Extent - sie$trad_mean)^2))

# --- Model 2: invariant (Handcock's exact spec) -----------------------------
cat("Fitting Model 2: invariant...\n")
g_iac <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 50, fx = FALSE),
             data = sie, method = "REML", knots = list(DOY = c(0, 365)))
rmse_iac <- sqrt(mean((sie$Extent - predict(g_iac))^2))

# --- Model 3: amplitude-adjusted --------------------------------------------
cat("Fitting Model 3: amplitude...\n")
g_amp <- gam(scaling_factor ~ s(tdate) + s(DOY, bs = "cc", k = 50, fx = FALSE),
             data = sie, method = "REML", knots = list(DOY = c(0, 365)))
fitted_amp <- predict(g_amp) * sie$amplitude + sie$min_extent
rmse_amp <- sqrt(mean((sie$Extent - fitted_amp)^2))

# --- Model 4: phase-adjusted ------------------------------------------------
cat("Fitting Model 4: phase...\n")
g_phase <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 50, fx = FALSE) +
                 s(phase, bs = "cc", k = 50, fx = FALSE),
               data = sie, method = "REML",
               knots = list(DOY = c(0, 365), phase = c(0, 365)))
rmse_phase <- sqrt(mean((sie$Extent - predict(g_phase))^2))

# --- Model 5: APAC ----------------------------------------------------------
cat("Fitting Model 5: APAC...\n")
g_apac <- gam(scaling_factor ~ s(tdate) + s(DOY, bs = "cc", k = 50, fx = FALSE) +
                s(phase, bs = "cc", k = 50, fx = FALSE),
              data = sie, method = "REML",
              knots = list(DOY = c(0, 365), phase = c(0, 365)))
fitted_apac <- predict(g_apac) * sie$amplitude + sie$min_extent
rmse_apac <- sqrt(mean((sie$Extent - fitted_apac)^2))

# --- Report -----------------------------------------------------------------
pct <- function(r) 100 * (1 - r^2 / rmse_trad^2)
cat("\n===================== H&R TABLE 1 CHECK =====================\n")
cat(sprintf("Circumpolar 1979-2018      YOUR RESULT       H&R Table 1\n"))
cat(sprintf("  Traditional RMSE:        %.3f              0.576\n", rmse_trad))
cat(sprintf("  Invariant RMSE:          %.3f (%5.1f%%)     0.482 (28.7%%)\n",
            rmse_iac,   pct(rmse_iac)))
cat(sprintf("  Amplitude-adjusted:      %.3f (%5.1f%%)     0.382 (55.2%%)\n",
            rmse_amp,   pct(rmse_amp)))
cat(sprintf("  Phase-adjusted:          %.3f (%5.1f%%)     0.343 (63.9%%)\n",
            rmse_phase, pct(rmse_phase)))
cat(sprintf("  Full APAC:               %.3f (%5.1f%%)     0.272 (77.3%%)\n",
            rmse_apac,  pct(rmse_apac)))
cat("\n")
cat("If your % improvements are within a few points of H&R's,\n")
cat("the Handcock fixes reproduce the paper. Green light for the full pipeline.\n")