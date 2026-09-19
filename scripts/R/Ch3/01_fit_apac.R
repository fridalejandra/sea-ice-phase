# =============================================================================
# APAC_Sector_Pipeline_FINAL.R                          [CANONICAL FILE 1 of 2]
# -----------------------------------------------------------------------------
# Production APAC pipeline, rebuilt in native mgcv around the validated
# reproduction of Handcock & Raphael (2020) — including the pieces H&R never
# published, recovered by scoring against their digitized Figure 7a:
#
#   VALIDATED RECIPE (reproduces their Fig 7a: trend 0.03 / amp 0.49 /
#   phase 1.10 RMSE against the digitized curves; Table 1 within 1-2 pp)
#   -------------------------------------------------------------------
#   * Record starts 26 Oct 1978 (their exact record; the 1978 tail matters)
#   * Phase is CONTINUOUS across New Year: t anchored on the most recent
#     minimum, normalized by that cycle's own min-to-min span
#   * Model 3 amplitude is PURE  scaling ~ s(DOY)        (Eq 6 — no tdate)
#   * Models 4/5 are PURE phase  ~ s(phase)              (Eq 10/12) with
#     PER-YEAR Beta(b1,b2) warp estimated by backfitting, then shrunk
#     0.85 toward (1,1) (approximates their joint PSE minimization)
#   * Trend is the s(tdate) partial of  Extent ~ s(tdate) + s(DOY) (Eq 15);
#     both the GCV-chosen and a heavily-smoothed k=6 variant are exported
#     (the k=6 variant matched the published trend curve almost exactly)
#   * Sequential components:  amp = A - IAC - trend ;  phase = APAC - A ;
#     raw = SIE - APAC.  u-space predictions are exported so figures can
#     re-normalize however they need (incl. the retreat-limb blend).
#   * FIGURE CONVENTION (H&R's, unlabeled in their paper): invariant in
#     raw Mkm^2 centered; all components as PERCENT OF THE YEAR'S
#     AMPLITUDE. Percent columns are exported alongside Mkm^2 ones.
#
#   2026 CONSOLIDATION — what changed and why
#   -------------------------------------------------------------------
#   * DUAL-PERIOD FIT: everything in Sections 1-6 is now fit TWICE per
#     sector, once per entry in PERIODS — a full independent refit for
#     each DATE_END, not a single fit with a restricted-window evaluation.
#     This matches how the earlier v3/v4 Table 2 comparison was actually
#     produced, so the two period rows stay apples-to-apples with that
#     prior work. Runtime is ~2x a single pass (see RUNTIME note below).
#   * VOLATILITY (Section 7) is the ported, as-validated content of the
#     former 05_volatility_gamlss.R (Handcock's suggestion, email Sep
#     2026), NOT a from-scratch model — it now runs at the end of this
#     script directly on the assembled daily table in memory, instead of
#     as a separate script re-reading daily_fitted.csv from disk. It is
#     run ONCE, on period == "FULL" only: the post-2016 regime-shift test
#     needs a real post-2016 sample (8 yrs under FULL vs. 3 yrs under
#     HR2018), so duplicating it per period would be close to meaningless
#     for HR2018 and wastes the bootstrap's runtime. The old GARCH-based
#     per-day `volatility` column is retired (see Section 3h) in favor of
#     this — it was a different, less-validated quantity.
#   * BREAKING CSV SCHEMA CHANGE: annual_params.csv and daily_fitted.csv
#     now carry a `period` column (values: names(PERIODS), e.g. "HR2018",
#     "FULL") and contain ONE FULL COPY OF THE OUTPUT PER PERIOD — years
#     1979-2018 appear under BOTH periods (as independent refits, so their
#     fitted values can differ slightly between the two), while years after
#     2018 appear only under "FULL". EVERY downstream script that reads
#     these two files must now filter to a single period before use —
#     normal figures/analysis should filter to period == "FULL"; only the
#     new H&R-comparison validation table should use period == "HR2018".
#     Cleanest fix: add that filter once, as the default, inside the shared
#     loader (ch3_data.py's load_annual()/load_daily(), if that's the one
#     shared entry point) rather than touching every consumer script.
#
# OUTPUTS:
#   data/ch3/daily_fitted.csv       every day x sector x period: fits,
#                                     u-space, components (Mkm^2 AND % of
#                                     amplitude). `volatility` column is now
#                                     a retired/NA placeholder — see Section 7.
#   data/ch3/annual_params.csv      per year x sector x period: timing/
#                                     amplitude params, anomalies, beta1/beta2
#   data/ch3/rmse_summary.csv       Table-1-style comparison, all sectors x period
#   results/ch3/tables/t34c_volatility_gamlss_post2016.csv    (Section 7)
#   results/ch3/tables/t34c_volatility_seasonal_curves.csv    (Section 7)
#
# RUNTIME: the Beta backfit refits a gam 3x per response per sector, PER
#   PERIOD (~5-10 min/sector/period). Expect roughly double the old
#   single-pass runtime (~80-120 min for both periods), plus Section 7's
#   gamlss/bootstrap pass at the end (NBOOT=100 default; set NBOOT=0 via
#   env var to skip the bootstrap for a quick pass). For a quicker pass
#   set BETA_ITERS <- 2, SHARE_BETA <- TRUE, or PERIODS <- PERIODS["FULL"].
# =============================================================================

library(dplyr)
library(lubridate)
library(mgcv)
library(gamlss)

# ── 0. SETTINGS ─────────────────────────────────────────────────────────────

DATE_START <- as.Date("1979-01-01")   # first full year; partial 1978 cycle excluded (methods 2.1.1)

# Each period is fit completely independently (full refit, not a shared fit
# with restricted-window evaluation). "HR2018" mirrors Handcock & Raphael
# (2020) Table 1 exactly; "FULL" is the whole record. Add/remove entries
# here (e.g. an eventual 2025 extension) without touching anything below.
PERIODS <- list(
  HR2018 = as.Date("2018-12-31"),
  FULL   = as.Date("2025-12-31")
)

BETA_ITERS  <- 3      # backfitting passes for the Beta warp
BETA_SHRINK <- 0.85   # shrink estimated betas toward (1,1); validated value
SHARE_BETA  <- FALSE  # TRUE: Model 4 reuses Model 5's betas (halves runtime)
GCV         <- "GCV.Cp"   # H&R's stated smoothing selection

# --- Section 7 (volatility) settings, ported from 05_volatility_gamlss.R ---
VOL_START         <- as.Date("1988-01-01")  # SSM/I onward; SMMR (pre-1988) is
# every-other-day sampled and its
# GARCH-era volatility reads 5-10x
# high for that reason alone
VOL_BREAK         <- 2016                   # post2016 = Year >= VOL_BREAK
VOL_SENSOR_SWITCH <- 2008                   # SSM/I 1988-2007, SSMIS 2008-
VOL_NBOOT         <- as.integer(Sys.getenv("NBOOT", "100"))  # 0 skips the bootstrap

# ── HARDCODED PATHS (2026-09-16) ───────────────────────────────────────────
# Direct specification to avoid ROOT-finding ambiguity
ROOT <- "/Users/fridaperez/Research/repos/sea-ice-phase"
INPUT_FILE <- file.path(ROOT, "data", "raw",
                        "SIE_daily_sector_and_circumpolar_million_km2.csv")
OUTPUT_DIR <- file.path(ROOT, "data", "ch3")
TABLES_DIR <- file.path(ROOT, "results", "ch3", "tables")

# Verify input file exists
if (!file.exists(INPUT_FILE)) {
  stop(sprintf("Input file not found: %s", INPUT_FILE))
}

# Create output directories if they don't exist
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)

message(sprintf("ROOT = %s", ROOT))
message(sprintf("INPUT_FILE = %s", INPUT_FILE))
message(sprintf("OUTPUT_DIR = %s", OUTPUT_DIR))
message(sprintf("TABLES_DIR = %s", TABLES_DIR))

SECTOR_COLS <- c("SIE_circumpolar", "SIE_Weddell", "SIE_Amundsen_Bellingshausen",
                 "SIE_Ross", "SIE_East_Antarctica", "SIE_King_Haakon")

# ── 1. LOAD (once, full record; each period filters this below) ────────────

raw_all <- read.csv(INPUT_FILE, stringsAsFactors = FALSE)
names(raw_all)[tolower(names(raw_all)) %in% c("date", "time")] <- "Date"
date_str <- raw_all$Date
raw_all$Date <- as.Date(date_str)
if (all(is.na(raw_all$Date))) raw_all$Date <- as.Date(date_str, format = "%m/%d/%y")
raw_all <- raw_all %>% filter(Date >= DATE_START) %>% arrange(Date)
raw_all$Year  <- year(raw_all$Date)
raw_all$DOY   <- yday(raw_all$Date)
raw_all$tdate <- as.numeric(raw_all$Date)
for (col in SECTOR_COLS) if (col %in% names(raw_all))
  raw_all[[col]] <- as.numeric(raw_all[[col]])
message("Loaded ", nrow(raw_all), " rows: ", min(raw_all$Date), " to ", max(raw_all$Date))

# ── 2. HELPERS ──────────────────────────────────────────────────────────────

# Per-year Beta-warp backfit (the piece H&R estimated but never published).
# response: "scaling" (Eq 12 / APAC) or "Extent" (Eq 10 / phase-only).
fit_beta_model <- function(sie, response, iters = BETA_ITERS,
                           shrink = BETA_SHRINK, label = "") {
  yrs <- sort(unique(sie$anchor_year))
  B <- matrix(1, length(yrs), 2, dimnames = list(as.character(yrs), NULL))
  phase_of <- function(B) {
    365 * pbeta(sie$frac, B[as.character(sie$anchor_year), 1],
                B[as.character(sie$anchor_year), 2])
  }
  fml <- as.formula(paste(response, "~ s(phase, bs='cc', k=100)"))
  for (it in seq_len(iters)) {
    sie$phase <- phase_of(B)
    g <- gam(fml, data = sie, method = GCV, knots = list(phase = c(0, 365)))
    message(sprintf("    %s beta iter %d/%d  fit-RMSE %.4f", label, it,
                    iters, sqrt(mean((sie[[response]] - fitted(g))^2))))
    for (yy in yrs) {
      m <- sie$anchor_year == yy
      if (sum(m) < 60) next
      fy <- sie$frac[m]; uv <- sie[[response]][m]
      sse <- function(p) {
        if (any(p <= 0.4) || any(p >= 2.5)) return(1e9)
        phy <- 365 * pbeta(fy, p[1], p[2])
        sum((uv - predict(g, newdata = data.frame(phase = phy)))^2)
      }
      gr <- expand.grid(a = c(0.7, 0.85, 1, 1.2, 1.5),
                        b = c(0.7, 0.85, 1, 1.2, 1.5))
      v  <- apply(gr, 1, sse)
      st <- as.numeric(gr[which.min(v), ])
      op <- optim(st, sse, method = "Nelder-Mead", control = list(maxit = 60))
      B[as.character(yy), ] <- if (op$value < min(v)) op$par else st
    }
  }
  B <- 1 + shrink * (B - 1)
  sie$phase <- phase_of(B)
  g <- gam(fml, data = sie, method = GCV, knots = list(phase = c(0, 365)))
  list(fit = g, B = B, phase = sie$phase,
       pred = as.numeric(fitted(g)))
}

# ── 3. FIT ONE SECTOR, ONE PERIOD ────────────────────────────────────────────

fit_sector <- function(raw_data, sector_col, period_label, date_end) {
  message("\n==== ", sector_col, "  [period: ", period_label,
          ", through ", date_end, "] ====")
  sie <- raw_data %>%
    select(Date, Year, DOY, tdate, Extent = all_of(sector_col)) %>%
    filter(!is.na(Extent)) %>% arrange(Date)
  
  # --- 3a. calendar-year stats -------------------------------------------
  yearly <- sie %>% group_by(Year) %>%
    summarise(min_extent = min(Extent), max_extent = max(Extent),
              amplitude  = max_extent - min_extent,
              min_doy_raw = DOY[which.min(Extent)],
              max_doy_raw = DOY[which.max(Extent)],
              Date1 = Date[which.min(Extent)], .groups = "drop") %>%
    arrange(Year) %>%
    mutate(Date2 = lag(Date1), Date3 = lead(Date1))
  sie <- sie %>% left_join(yearly, by = "Year") %>%
    mutate(scaling = (Extent - min_extent) / (amplitude + 1e-10))
  
  # --- 3b. continuous cycle clock (validated v4 form) ----------------------
  after    <- sie$Date >= sie$Date1
  anchor   <- as.Date(ifelse(after, sie$Date1, sie$Date2), origin = "1970-01-01")
  next_min <- as.Date(ifelse(after, sie$Date3, sie$Date1), origin = "1970-01-01")
  anchor[is.na(anchor)]     <- next_min[is.na(anchor)] - 365
  next_min[is.na(next_min)] <- anchor[is.na(next_min)] + 365
  sie$t      <- as.numeric(sie$Date - anchor)
  sie$t_span <- as.numeric(next_min - anchor)
  sie <- sie %>% filter(t_span > 0, t >= 0, t < t_span)
  sie$frac <- pmin(pmax(sie$t / sie$t_span, 1e-9), 1 - 1e-9)
  sie$anchor_year <- ifelse(sie$Date >= sie$Date1, sie$Year, sie$Year - 1L)
  sie$phase0 <- 365 * sie$frac   # UNWARPED phase (beta = (1,1)) — kept
  # pristine for the canonical benchmarks
  message("  rows after cycle clock: ", nrow(sie))
  
  # --- 3c. Model 1: traditional --------------------------------------------
  trad <- sie %>% group_by(DOY) %>%
    summarise(trad_mean = mean(Extent), .groups = "drop")
  sie <- sie %>% left_join(trad, by = "DOY")
  rmse_trad <- sqrt(mean((sie$Extent - sie$trad_mean)^2))
  
  # --- 3d. Model 2: invariant (Eq 3) + trend (Eq 15) ------------------------
  g_iac <- gam(Extent ~ s(DOY, bs = "cc", k = 25), data = sie,
               method = GCV, knots = list(DOY = c(0, 365)))
  sie$iac_notrend <- as.numeric(fitted(g_iac))
  rmse_iac <- sqrt(mean((sie$Extent - sie$iac_notrend)^2))
  
  g_tr  <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 25), data = sie,
               method = GCV, knots = list(DOY = c(0, 365)))
  g_tr6 <- gam(Extent ~ s(tdate, k = 6) + s(DOY, bs = "cc", k = 25),
               data = sie, method = GCV, knots = list(DOY = c(0, 365)))
  sie$trend_component    <- as.numeric(predict(g_tr,  type = "terms")[, "s(tdate)"])
  sie$trend_component_k6 <- as.numeric(predict(g_tr6, type = "terms")[, "s(tdate)"])
  rmse_iac_trend <- sqrt(mean((sie$Extent - fitted(g_tr))^2))
  message(sprintf("  trend EDF: GCV %.1f | k6 %.1f",
                  summary(g_tr)$s.table["s(tdate)", "edf"],
                  summary(g_tr6)$s.table["s(tdate)", "edf"]))
  
  # --- 3e. Model 3: amplitude (Eq 6, PURE s(DOY)) ---------------------------
  g_amp <- gam(scaling ~ s(DOY, bs = "cc", k = 100), data = sie,
               method = GCV, knots = list(DOY = c(0, 365)))
  sie$u_amp      <- as.numeric(fitted(g_amp))
  sie$fitted_amp <- sie$u_amp * sie$amplitude + sie$min_extent
  rmse_amp <- sqrt(mean((sie$Extent - sie$fitted_amp)^2))
  
  # --- 3f. Model 5: APAC (Eq 12, PURE s(phase) + per-year beta) -------------
  message("  Model 5 (APAC) beta backfit...")
  apac <- fit_beta_model(sie, "scaling", label = "APAC")
  sie$phase       <- apac$phase
  sie$u_apac      <- apac$pred
  sie$fitted_apac <- sie$u_apac * sie$amplitude + sie$min_extent
  sie$residual_apac <- sie$Extent - sie$fitted_apac
  rmse_apac <- sqrt(mean(sie$residual_apac^2))
  
  # --- 3g. Model 4: phase-only (Eq 10, Extent-response) ---------------------
  if (SHARE_BETA) {
    B4 <- apac$B
    ph4 <- 365 * pbeta(sie$frac, B4[as.character(sie$anchor_year), 1],
                       B4[as.character(sie$anchor_year), 2])
    g_ph <- gam(Extent ~ s(phase, bs = "cc", k = 100),
                data = transform(sie, phase = ph4),
                method = GCV, knots = list(phase = c(0, 365)))
    sie$fitted_phase <- as.numeric(fitted(g_ph))
  } else {
    message("  Model 4 (phase-only) beta backfit...")
    ph  <- fit_beta_model(sie, "Extent", label = "PH ")
    sie$fitted_phase <- ph$pred
  }
  rmse_phase <- sqrt(mean((sie$Extent - sie$fitted_phase)^2))
  
  # --- 3g2. CANONICAL benchmark models — Frida's exact Table-2 forms --------
  # (high-k cyclic s(tdate) interannual absorber + UNWARPED phase0; these
  #  are the forms that match published Table 1 / chapter Table 2)
  message("  canonical benchmark models (RMSE table)...")
  g_amp_can <- gam(scaling ~ s(tdate, bs = "cc", k = 20) +
                     s(DOY,   bs = "cc", k = 100),
                   data = sie)
  fitted_amp_can <- as.numeric(fitted(g_amp_can)) * sie$amplitude +
    sie$min_extent
  rmse_amp_can <- sqrt(mean((sie$Extent - fitted_amp_can)^2))
  
  g_ph_can <- gam(Extent ~ s(tdate,  bs = "cc", k = 75) +
                    s(DOY,    bs = "cc", k = 100) +
                    s(phase0, bs = "cc", k = 100),
                  data = sie)
  rmse_phase_can <- sqrt(mean((sie$Extent - fitted(g_ph_can))^2))
  
  g_ap_can <- gam(scaling ~ s(tdate,  bs = "cc", k = 150) +
                    s(DOY,    bs = "cc", k = 100) +
                    s(phase0, bs = "cc", k = 100),
                  data = sie)
  fitted_apac_can <- as.numeric(fitted(g_ap_can)) * sie$amplitude +
    sie$min_extent
  rmse_apac_can <- sqrt(mean((sie$Extent - fitted_apac_can)^2))
  
  pct <- function(r) 100 * (1 - r^2 / rmse_trad^2)
  message(sprintf(
    "  Trad %.3f | IAC %.3f (%.1f%%) | IAC+tr %.3f (%.1f%%) | Amp %.3f (%.1f%%) | Phase %.3f (%.1f%%) | APAC %.3f (%.1f%%)",
    rmse_trad, rmse_iac, pct(rmse_iac), rmse_iac_trend, pct(rmse_iac_trend),
    rmse_amp, pct(rmse_amp), rmse_phase, pct(rmse_phase),
    rmse_apac, pct(rmse_apac)))
  message(sprintf(
    "  CANONICAL: Amp %.3f (%.1f%%) | Phase %.3f (%.1f%%) | APAC %.3f (%.1f%%)   <- Table-2 benchmark forms",
    rmse_amp_can, pct(rmse_amp_can),
    rmse_phase_can, pct(rmse_phase_can), rmse_apac_can, pct(rmse_apac_can)))
  if (identical(period_label, "HR2018"))
    message("  H&R circumpolar 1979-2018: IAC 28.7 | Amp 55.2 | Phase 63.9 | APAC 77.3")
  
  # --- 3h. daily `volatility` column: retired ------------------------------
  # Formerly the per-day sigma from a GARCH(2,2)/ARMA(1,1) fit to
  # residual_apac. Superseded by the seasonal-gamlss volatility analysis in
  # Section 7 (ported from 05_volatility_gamlss.R), which works on dSIE /
  # residual_apac across all sectors at once and produces its own tables
  # (t34c_volatility_gamlss_post2016.csv, t34c_volatility_seasonal_curves.csv)
  # rather than a per-day column here. Left as NA so the column still exists
  # for anything that expects it, but nothing populates it anymore — check
  # whether any current downstream script reads daily_fitted.csv$volatility
  # before relying on it being NA.
  sie$volatility <- NA_real_
  
  # --- 3i. sequential components (validated decomposition) ------------------
  sie <- sie %>% mutate(
    amplitude_component = fitted_amp  - iac_notrend - trend_component,
    phase_component     = fitted_apac - fitted_amp,
    raw_anomaly         = Extent - fitted_apac,
    anomaly_from_iac    = Extent - iac_notrend,   # full anomaly = trend + amplitude + phase + residual
    # est_anomaly: an 11-day smoothed version of raw_anomaly, used only as a
    # plotting stand-in for Fig 7.
    est_anomaly = as.numeric(stats::filter(raw_anomaly, rep(1/11, 11), sides = 2)),
    # percent-of-amplitude versions (H&R's Fig 7 plotting convention)
    trend_pct = trend_component     / amplitude * 100,
    amp_pct   = amplitude_component / amplitude * 100,
    phase_pct = phase_component     / amplitude * 100,
    raw_pct   = raw_anomaly         / amplitude * 100)
  
  # --- 3j. annual parameters (incl. the estimated betas) --------------------
  Bdf <- data.frame(anchor_year = as.integer(rownames(apac$B)),
                    beta1 = apac$B[, 1], beta2 = apac$B[, 2])
  annual <- sie %>% group_by(Year) %>%
    summarise(
      sie_annual = mean(Extent),
      sie_DJF = mean(Extent[month(Date) %in% c(12, 1, 2)]),
      sie_MAM = mean(Extent[month(Date) %in% 3:5]),
      sie_JJA = mean(Extent[month(Date) %in% 6:8]),
      sie_SON = mean(Extent[month(Date) %in% 9:11]),
      max_doy_fitted = DOY[which.max(fitted_apac)],
      min_doy_fitted = DOY[which.min(fitted_apac)],
      amplitude_fitted = max(fitted_amp) - min(fitted_amp),
      max_doy_raw = first(max_doy_raw), min_doy_raw = first(min_doy_raw),
      amplitude_raw_yr = first(amplitude),
      .groups = "drop") %>%
    mutate(
      max_doy_anom   = max_doy_fitted - median(max_doy_fitted),
      min_doy_anom   = min_doy_fitted - median(min_doy_fitted),
      amplitude_anom = amplitude_fitted - mean(amplitude_fitted),
      max_doy_raw_anom   = max_doy_raw - median(max_doy_raw),
      min_doy_raw_anom   = min_doy_raw - median(min_doy_raw),
      amplitude_raw_anom = amplitude_raw_yr - mean(amplitude_raw_yr)) %>%
    left_join(Bdf, by = c(Year = "anchor_year")) %>%
    mutate(beta_asym = beta1 - beta2, sector = sector_col, period = period_label)
  
  daily <- sie %>%
    select(Date, Year, DOY, tdate, Extent, anchor_year, t, t_span, frac,
           phase, scaling, min_extent, max_extent, amplitude,
           iac_notrend, trend_component, trend_component_k6,
           u_amp, fitted_amp, u_apac, fitted_apac, fitted_phase,
           residual_apac, volatility,
           amplitude_component, phase_component, raw_anomaly, anomaly_from_iac, est_anomaly,
           trend_pct, amp_pct, phase_pct, raw_pct) %>%
    mutate(sector = sector_col, period = period_label)
  
  list(annual = annual, daily = daily,
       rmse = data.frame(sector = sector_col, period = period_label,
                         date_end = date_end,
                         rmse_trad = rmse_trad,
                         rmse_iac = rmse_iac, rmse_iac_trend = rmse_iac_trend,
                         rmse_amp = rmse_amp, rmse_phase = rmse_phase,
                         rmse_apac = rmse_apac,
                         rmse_amp_can = rmse_amp_can,
                         rmse_phase_can = rmse_phase_can,
                         rmse_apac_can = rmse_apac_can,
                         pct_iac = pct(rmse_iac),
                         pct_iac_trend = pct(rmse_iac_trend),
                         pct_amp = pct(rmse_amp),
                         pct_phase = pct(rmse_phase), pct_apac = pct(rmse_apac),
                         pct_amp_can = pct(rmse_amp_can),
                         pct_phase_can = pct(rmse_phase_can),
                         pct_apac_can = pct(rmse_apac_can)))
}

# ── 4. RUN — every sector, every period ─────────────────────────────────────

A <- list(); D <- list(); R <- list()
for (period_label in names(PERIODS)) {
  date_end <- PERIODS[[period_label]]
  raw_p <- raw_all %>% filter(Date <= date_end)
  message(sprintf("\n########## PERIOD %s  (%s through %s) ##########",
                  period_label, min(raw_p$Date), date_end))
  for (sec in SECTOR_COLS) {
    key <- paste(period_label, sec, sep = "__")
    res <- fit_sector(raw_p, sec, period_label = period_label, date_end = date_end)
    A[[key]] <- res$annual; D[[key]] <- res$daily; R[[key]] <- res$rmse
  }
}
annual_df <- bind_rows(A); daily_df <- bind_rows(D); rmse_df <- bind_rows(R)

# ── 5. SAVE (Sections 1-6 outputs) ──────────────────────────────────────────

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
write.csv(annual_df, file.path(OUTPUT_DIR, "annual_params.csv"), row.names = FALSE)
write.csv(daily_df,  file.path(OUTPUT_DIR, "daily_fitted.csv"),  row.names = FALSE)
write.csv(rmse_df,   file.path(OUTPUT_DIR, "rmse_summary.csv"),  row.names = FALSE)

# ── 6. VALIDATION PRINTS ────────────────────────────────────────────────────

for (period_label in names(PERIODS)) {
  message(sprintf("\n===== RMSE TABLE — CANONICAL benchmark forms — period %s =====", period_label))
  print(rmse_df %>% filter(period == period_label) %>%
          select(sector, pct_iac_trend, pct_amp_can, pct_phase_can, pct_apac_can) %>%
          rename(pct_invariant = pct_iac_trend) %>%
          mutate(across(where(is.numeric), ~round(.x, 1))))
  message(sprintf("===== PURE Eq-10/12 forms with approximate beta — period %s =====", period_label))
  print(rmse_df %>% filter(period == period_label) %>%
          select(sector, pct_phase, pct_apac) %>%
          mutate(across(where(is.numeric), ~round(.x, 1))))
}
if ("HR2018" %in% names(PERIODS))
  message("\nH&R circumpolar 1979-2018: IAC 28.7 | Amp 55.2 | Phase 63.9 | APAC 77.3")

# ── Table 1: H&R (2020) validation check — circumpolar only, HR2018 period ──
# Published Handcock & Raphael (2020) circumpolar, 1979-2018 performance (% improvement),
# checked against this replication's CANONICAL benchmark forms (the
# high-k cyclic-s(tdate) interannual absorber + unwarped phase0 models —
# see fit_sector()'s canonical-benchmark section — not the "pure Eq-10/12"
# per-year-Beta-warped forms, which read further from H&R's own numbers).
if ("HR2018" %in% names(PERIODS)) {
  hr_row <- rmse_df %>% filter(sector == "SIE_circumpolar", period == "HR2018")
  table1 <- data.frame(
    Model      = c("Invariant (IAC+trend)", "Amplitude", "Phase", "APAC"),
    HR_2020    = c(28.7, 55.2, 63.9, 77.3),
    This_Study = round(c(hr_row$pct_iac_trend, hr_row$pct_amp_can,
                         hr_row$pct_phase_can, hr_row$pct_apac_can), 1)
  )
  table1$Diff <- round(table1$This_Study - table1$HR_2020, 1)
  dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
  write.csv(table1, file.path(TABLES_DIR, "table1_hr_validation.csv"), row.names = FALSE)
  message("\n===== TABLE 1: H&R (2020) validation, circumpolar, HR2018 (% improvement vs traditional) =====")
  print(table1, row.names = FALSE)
}

# ── Table 2: sector extension — FULL period (1979-2023), canonical forms ───
# The chapter's headline result: H&R's circumpolar-only analysis extended to
# all sectors and to the full record. Percentage improvement vs traditional.
table2 <- rmse_df %>%
  filter(period == "FULL") %>%
  transmute(Sector = sector,
            Invariant = round(pct_iac_trend, 1), Amplitude = round(pct_amp_can, 1),
            Phase = round(pct_phase_can, 1), APAC = round(pct_apac_can, 1))
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
write.csv(table2, file.path(TABLES_DIR, "table2_sector_rmse_full.csv"), row.names = FALSE)
message("\n===== TABLE 2: sector extension, FULL period (1979-2023), canonical forms (% improvement vs traditional) =====")
print(table2, row.names = FALSE)

message("\nbeta(2016) by sector x period (phase-warp parameters, previously unpublished):")
print(annual_df %>% filter(Year == 2016) %>%
        select(sector, period, beta1, beta2, beta_asym) %>%
        mutate(across(where(is.numeric), ~round(.x, 2))))

# =============================================================================
# ── 7. VOLATILITY — seasonal gamlss, pre/post-2016, ported from
#      05_volatility_gamlss.R (Handcock's suggestion, email Sep 2026).
# -----------------------------------------------------------------------------
# Runs ONCE, on period == "FULL" only (see note in the file header on why).
# Models the SCALE of the daily anomaly with a cyclic spline in day-of-year
# (an "annual cycle in volatility"), with an indicator for post-2016 and a
# sensor term so retrieval effects are held constant within an instrument.
#
# Data: this script's own daily_df, restricted to VOL_START (1988-01-01,
# SSM/I onward — the SMMR period has every-other-day sampling and its GARCH
# volatility read 5-10x higher for that reason alone).
#
# Two response variables. PRIMARY: the daily tendency dSIE = Extent(t) -
# Extent(t-1) (consecutive days only), which is what the 2020 paper's GARCH
# innovations measure — day-to-day change — and which removes level, trend
# and nearly all amplitude/phase variation by construction. CHECK: the APAC
# residual (residual_apac). (The unadjusted anomaly was tried first: its
# scale term absorbs year-to-year level differences even with a period
# intercept in the location, so it is not a day-to-day measure. Kept out.)
#
# Model (per sector): y ~ pbc(DOY) + post2016 + sensor        [location]
#                    sigma ~ pbc(DOY) + sensor + post2016     [log-scale]
#   post2016 = one multiplicative change in volatility after 2016, all seasons.
# Shape comparison: the same model fitted SEPARATELY to pre- and post-2016
#   (Handcock: let the seasonal volatility differ by period); the ratio of
#   the two fitted sigma curves by day of year, summarised by season.
# Uncertainty: year-block bootstrap (resample calendar years with
#   replacement, refit), VOL_NBOOT reps, because daily residuals are
#   autocorrelated and the model's own SEs assume independence.
#
# Outputs (results/ch3/tables/):
#   t34c_volatility_gamlss_post2016.csv   sector, response, exp(coef), boot CI, by-season ratios
#   t34c_volatility_seasonal_curves.csv   sector, response, DOY, sigma_pre, sigma_post
# =============================================================================

message("\n\n===== SECTION 7: seasonal volatility (gamlss, period == 'FULL' only) =====")

vold <- daily_df %>%
  filter(period == "FULL", Date >= VOL_START, Date <= PERIODS[["FULL"]]) %>%
  arrange(sector, Date) %>%
  group_by(sector) %>%
  mutate(dSIE = ifelse(as.numeric(Date - lag(Date)) == 1, Extent - lag(Extent), NA_real_)) %>%
  ungroup() %>%
  mutate(post2016 = factor(Year >= VOL_BREAK, levels = c(FALSE, TRUE)),
         sensor   = factor(ifelse(Year >= VOL_SENSOR_SWITCH, "SSMIS", "SSMI"),
                           levels = c("SSMI", "SSMIS")))
message("  volatility rows: ", nrow(vold), "  ", min(vold$Date), " to ", max(vold$Date))

# ADDED 2026-09-19 (Frida's call, following up on Handcock's email): a
# continuous-trend companion to the post2016 step-change model below, since
# "has volatility changed" doesn't have to mean "did it jump at 2016" --
# cYear centers Year at the record's own midpoint so the fitted sigma
# intercept stays interpretable (the value at the middle of the record,
# not at Year=0).
VOL_YEAR_CENTER <- mean(range(vold$Year))

# FIXED 2026-09-19: raised from gamlss's default n.cyc=20 and used everywhere
# a volatility model is fit (point estimate AND every bootstrap replicate, in
# both this post2016 step model and Section 7b's continuous-trend model)
# after the first real run of 7b showed EVERY dSIE sector's point estimate
# falling outside its own bootstrap CI -- a sign some replicates ran to the
# default iteration cap without converging and were silently kept (tryCatch
# only catches thrown errors, not "didn't converge but returned a number").
# This run's Section 7 bootstrap happened not to show the problem, but it
# uses the exact same fitting machinery, so the same guard is applied here
# too rather than assuming it's fine just because it looked fine once.
VOL_GAMLSS_CONTROL <- gamlss.control(n.cyc = 100, trace = FALSE)

vol_frame_of <- function(dd, resp)   # plain data.frame, modelled columns only, NA rows dropped
  na.omit(data.frame(y = as.numeric(dd[[resp]]), DOY = as.numeric(dd$DOY), post2016 = dd$post2016,
                     sensor = dd$sensor, Year = dd$Year, cYear = dd$Year - VOL_YEAR_CENTER))

vol_fit_one <- function(dd, resp) {
  dd <- vol_frame_of(dd, resp)
  # location carries the period level and sensor too, otherwise the post-2016
  # mean offset of the unadjusted anomaly is booked as scale
  m <- gamlss(y ~ pbc(DOY) + post2016 + sensor, sigma.formula = ~ pbc(DOY) + sensor + post2016,
              family = NO, data = dd, trace = FALSE, control = VOL_GAMLSS_CONTROL)
  co <- coef(m, what = "sigma")
  list(model = m, post = unname(co["post2016TRUE"]), sens = unname(co["sensorSSMIS"]), data = dd,
       converged = isTRUE(m$converged))
}
vol_fit_period <- function(dd, resp) {  # one period only: no post2016 term; sensor only if both present
  dd <- vol_frame_of(dd, resp)
  if (length(unique(dd$sensor)) > 1)
    m <- gamlss(y ~ pbc(DOY) + sensor, sigma.formula = ~ pbc(DOY) + sensor, family = NO, data = dd,
                trace = FALSE, control = VOL_GAMLSS_CONTROL)
  else
    m <- gamlss(y ~ pbc(DOY), sigma.formula = ~ pbc(DOY), family = NO, data = dd,
                trace = FALSE, control = VOL_GAMLSS_CONTROL)
  if (!isTRUE(m$converged))
    message("    WARNING: vol_fit_period() did not converge for this subset -- seasonal sigma_pre/sigma_post curve may be unreliable")
  list(model = m, data = dd)
}
VOL_SEASON_OF <- function(doy) { m <- as.integer(format(as.Date(doy - 1, origin = "2001-01-01"), "%m"))
c("DJF","DJF","MAM","MAM","MAM","JJA","JJA","JJA","SON","SON","SON","DJF")[m] }

vol_rows <- list(); vol_curves <- list()
for (sec in unique(vold$sector)) {
  for (resp in c("dSIE", "residual_apac")) {
    dd <- vold[vold$sector == sec, ]
    f  <- vol_fit_one(dd, resp)
    if (!f$converged)
      message(sprintf("  WARNING: %s %s POINT-ESTIMATE fit did not converge (n.cyc=%d) -- treat with caution",
                      sec, resp, VOL_GAMLSS_CONTROL$n.cyc))
    message(sprintf("  %-28s %-17s post2016 x%.3f  sensor x%.3f", sec, resp, exp(f$post), exp(f$sens)))
    
    # seasonal shape by period: separate fits, sigma curves at the SSMIS level
    fp <- vol_fit_period(dd[dd$Year <  VOL_BREAK, ], resp)
    fq <- vol_fit_period(dd[dd$Year >= VOL_BREAK, ], resp)
    nd_p <- data.frame(DOY = 1:365, sensor = factor("SSMIS", levels = c("SSMI", "SSMIS")))
    sig_pre  <- predict(fp$model, what = "sigma", newdata = nd_p, type = "response", data = fp$data)
    sig_post <- predict(fq$model, what = "sigma", newdata = data.frame(DOY = 1:365), type = "response", data = fq$data)
    cv <- data.frame(sector = sec, response = resp, DOY = 1:365, sigma_pre = sig_pre, sigma_post = sig_post)
    cv$season <- VOL_SEASON_OF(cv$DOY)
    vol_curves[[paste(sec, resp)]] <- cv
    by_season <- cv %>% group_by(season) %>% summarise(ratio = mean(sigma_post) / mean(sigma_pre), .groups = "drop")
    message(sprintf("     separate-period fits: post/pre sigma ratio  DJF %.2f  MAM %.2f  JJA %.2f  SON %.2f   (full-year %.2f)",
                    by_season$ratio[by_season$season=="DJF"], by_season$ratio[by_season$season=="MAM"],
                    by_season$ratio[by_season$season=="JJA"], by_season$ratio[by_season$season=="SON"],
                    mean(sig_post) / mean(sig_pre)))
    
    # year-block bootstrap of the post2016 coefficient
    boot <- NA_real_; n_nonconverged <- 0L
    if (VOL_NBOOT > 0) {
      yrs <- unique(dd$Year); set.seed(1)
      boot <- replicate(VOL_NBOOT, {
        ys <- sample(yrs, length(yrs), replace = TRUE)
        db <- do.call(rbind, lapply(seq_along(ys), function(i) { x <- dd[dd$Year == ys[i], ]; x$Year <- 10000 + i; x }))
        db$post2016 <- factor(ys[match(db$Year - 10000, seq_along(ys))] >= VOL_BREAK, levels = c(FALSE, TRUE))
        db$sensor <- factor(ifelse(ys[match(db$Year - 10000, seq_along(ys))] >= VOL_SENSOR_SWITCH, "SSMIS", "SSMI"),
                            levels = c("SSMI", "SSMIS"))
        if (length(unique(db$post2016)) < 2 || length(unique(db$sensor)) < 2) return(NA_real_)
        fit <- tryCatch(vol_fit_one(db, resp), error = function(e) NULL)
        if (is.null(fit) || !fit$converged) return(NA_real_)  # drop non-converged, not just errored
        fit$post
      })
      n_nonconverged <- sum(is.na(boot))
      boot <- boot[is.finite(boot)]
      message(sprintf("    (%d/%d bootstrap replicates converged and were used for the CI)",
                      length(boot), VOL_NBOOT))
    }
    vol_rows[[paste(sec, resp)]] <- data.frame(
      sector = sec, response = resp, n_days = nrow(dd),
      vol_ratio_post2016 = exp(f$post),
      boot_lo = if (length(boot) > 10) exp(quantile(boot, 0.025)) else NA,
      boot_hi = if (length(boot) > 10) exp(quantile(boot, 0.975)) else NA,
      n_boot = length(boot), n_boot_nonconverged = n_nonconverged,
      point_converged = f$converged,
      vol_ratio_ssmis_vs_ssmi = exp(f$sens),
      ratio_fullyear_separate_fits = mean(cv$sigma_post) / mean(cv$sigma_pre),
      ratio_DJF = by_season$ratio[by_season$season=="DJF"], ratio_MAM = by_season$ratio[by_season$season=="MAM"],
      ratio_JJA = by_season$ratio[by_season$season=="JJA"], ratio_SON = by_season$ratio[by_season$season=="SON"])
  }
}
vol_res <- do.call(rbind, vol_rows); rownames(vol_res) <- NULL
vol_cur <- do.call(rbind, vol_curves); rownames(vol_cur) <- NULL

dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
write.csv(vol_res, file.path(TABLES_DIR, "t34c_volatility_gamlss_post2016.csv"), row.names = FALSE)
write.csv(vol_cur, file.path(TABLES_DIR, "t34c_volatility_seasonal_curves.csv"), row.names = FALSE)
cat("\n== post-2016 multiplicative change in day-to-day volatility (season and sensor held fixed) ==\n")
print(vol_res %>% mutate(across(where(is.numeric), ~ round(.x, 3))), row.names = FALSE)
message("wrote t34c_volatility_gamlss_post2016.csv, t34c_volatility_seasonal_curves.csv")

# =============================================================================
# 7b. VOLATILITY TREND — continuous-Year companion to the post2016 step model
#     above (Frida's call 2026-09-19, following Handcock's email). Same two
#     responses (dSIE primary, residual_apac as the robustness check -- per
#     Handcock: "these adjustments should affect the volatility only in
#     minor ways... to be on the safe side, redo everything for the APAC and
#     show that the effects are small"), same seasonal cyclic spline and
#     sensor control, but cYear (continuous, centered) replaces post2016 in
#     both the location and sigma formulas -- asking whether log-volatility
#     has been drifting over the whole record, not whether it stepped at any
#     one year.
#
# NOT YET DONE, a natural follow-up if this is worth pursuing further:
# whether the SEASONAL SHAPE of volatility (not just its overall level) is
# itself drifting -- a pbc(DOY):cYear interaction in the sigma formula --
# analogous to the pre/post separate-fit shape comparison above, generalized
# to a continuous trend. Left out here to keep this first pass simple and
# fast to check; worth adding if the plain trend term looks interesting.
#
# Outputs (results/ch3/tables/):
#   t34c_volatility_gamlss_trend.csv   sector, response, %/decade, boot CI
# =============================================================================

message("\n\n===== SECTION 7b: seasonal volatility, CONTINUOUS TREND (gamlss) =====")


# FIXED 2026-09-19: the first real run of this section (full record, cYear
# continuous) showed "Algorithm RS has not yet converged" warnings and, in
# the output table, EVERY dSIE sector's pct_change_per_decade fell entirely
# outside its own reported boot_lo_pct/boot_hi_pct (2 of 6 residual_apac
# sectors too) -- e.g. circumpolar dSIE +67.19%/decade against a reported CI
# of [-26.44, +30.22]. A well-formed bootstrap CI should straddle the point
# estimate; one that doesn't, systematically, across every sector for one
# specific response, means some bootstrap replicates are silently including
# non-converged fits -- tryCatch() only catches thrown errors, not a fit that
# ran to n.cyc without converging and returned a coefficient anyway. Section
# 7's analogous post2016-factor model (same data, same two responses) did NOT
# show this problem, which points at the continuous cYear term specifically:
# a resampled year-block can land cYear/DOY combinations near the edge of the
# design that make the sigma link struggle to converge within gamlss's
# default n.cyc=20. Fix: raise n.cyc, and explicitly check convergence rather
# than trusting a returned coefficient -- a non-converged replicate is now
# dropped from the bootstrap the same way an errored one already was.
VOL_GAMLSS_CONTROL <- gamlss.control(n.cyc = 100, trace = FALSE)

vol_fit_trend_one <- function(dd, resp) {
  dd <- vol_frame_of(dd, resp)
  m <- gamlss(y ~ pbc(DOY) + cYear + sensor, sigma.formula = ~ pbc(DOY) + sensor + cYear,
              family = NO, data = dd, trace = FALSE, control = VOL_GAMLSS_CONTROL)
  co <- coef(m, what = "sigma")
  list(model = m, trend = unname(co["cYear"]), sens = unname(co["sensorSSMIS"]), data = dd,
       converged = isTRUE(m$converged))
}

vol_trend_rows <- list()
for (sec in unique(vold$sector)) {
  for (resp in c("dSIE", "residual_apac")) {
    dd <- vold[vold$sector == sec, ]
    f  <- vol_fit_trend_one(dd, resp)
    if (!f$converged)
      message(sprintf("  WARNING: %s %s POINT-ESTIMATE fit did not converge (n.cyc=%d) -- treat with caution",
                      sec, resp, VOL_GAMLSS_CONTROL$n.cyc))
    pct_decade <- (exp(f$trend * 10) - 1) * 100
    message(sprintf("  %-28s %-17s %+.2f%% per decade  (sigma at record midpoint x%.3f rel. to SSMI)",
                    sec, resp, pct_decade, exp(f$sens)))
    
    boot <- NA_real_; n_nonconverged <- 0L
    if (VOL_NBOOT > 0) {
      yrs <- unique(dd$Year); set.seed(1)
      boot <- replicate(VOL_NBOOT, {
        ys <- sample(yrs, length(yrs), replace = TRUE)
        db <- do.call(rbind, lapply(seq_along(ys), function(i) { x <- dd[dd$Year == ys[i], ]; x$Year <- 10000 + i; x }))
        yr_lookup <- ys[match(db$Year - 10000, seq_along(ys))]
        db$cYear  <- yr_lookup - VOL_YEAR_CENTER
        db$sensor <- factor(ifelse(yr_lookup >= VOL_SENSOR_SWITCH, "SSMIS", "SSMI"),
                            levels = c("SSMI", "SSMIS"))
        if (length(unique(db$sensor)) < 2) return(NA_real_)
        fit <- tryCatch(vol_fit_trend_one(db, resp), error = function(e) NULL)
        if (is.null(fit) || !fit$converged) return(NA_real_)  # drop non-converged, not just errored
        fit$trend
      })
      n_nonconverged <- sum(is.na(boot))
      boot <- boot[is.finite(boot)]
      message(sprintf("    (%d/%d bootstrap replicates converged and were used for the CI)",
                      length(boot), VOL_NBOOT))
    }
    pct_lo <- if (length(boot) > 10) (exp(quantile(boot, 0.025) * 10) - 1) * 100 else NA
    pct_hi <- if (length(boot) > 10) (exp(quantile(boot, 0.975) * 10) - 1) * 100 else NA
    
    vol_trend_rows[[paste(sec, resp)]] <- data.frame(
      sector = sec, response = resp, n_days = nrow(vol_frame_of(dd, resp)),
      pct_change_per_decade = pct_decade, boot_lo_pct = pct_lo, boot_hi_pct = pct_hi,
      n_boot = length(boot), n_boot_nonconverged = n_nonconverged,
      point_converged = f$converged, vol_ratio_ssmis_vs_ssmi = exp(f$sens))
  }
}
vol_trend_res <- do.call(rbind, vol_trend_rows); rownames(vol_trend_res) <- NULL
write.csv(vol_trend_res, file.path(TABLES_DIR, "t34c_volatility_gamlss_trend.csv"), row.names = FALSE)
cat("\n== continuous trend in day-to-day volatility, %/decade (season and sensor held fixed) ==\n")
print(vol_trend_res %>% mutate(across(where(is.numeric), ~ round(.x, 3))), row.names = FALSE)
message("wrote t34c_volatility_gamlss_trend.csv")

message("\nWrote *.csv outputs (annual_params.csv / daily_fitted.csv / rmse_summary.csv, ")
message("each with a `period` column) plus results/ch3/tables/t34c_volatility_*.csv. ")
message("Fig 7 plotting: use *_pct columns, filtered to period == 'FULL'; the invariant is ")
message("plotted in raw Mkm^2 centered (H&R's mixed convention). The H&R-comparison ")
message("validation table should filter to period == 'HR2018'. daily_fitted.csv$volatility ")
message("is now an unused NA placeholder -- volatility lives in t34c_volatility_*.csv.")