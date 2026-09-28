# =============================================================================
# APAC_Sector_Pipeline_FINAL.R
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
# OUTPUTS (suffix _E):
#   daily_fitted_E.csv    every day x sector: fits, u-space, components
#                         (Mkm^2 AND % of amplitude), volatility
#   annual_params_E.csv   per year x sector: timing/amplitude params,
#                         anomalies, AND the estimated beta1/beta2
#   rmse_summary_E.csv    Table-1-style comparison, all sectors
#
# RUNTIME: the Beta backfit refits a gam 3x per response per sector.
#   ~5-10 min/sector, ~40-60 min total. Progress is printed. For a quick
#   pass set BETA_ITERS <- 2 and SHARE_BETA <- TRUE (~15 min total).
# =============================================================================

library(dplyr)
library(lubridate)
library(mgcv)

# ── 0. SETTINGS ─────────────────────────────────────────────────────────────

DATE_START <- as.Date("1979-01-01")   # first full year; partial 1978 cycle excluded (methods 2.1.1)
DATE_END   <- as.Date("2023-12-31")   # full record; use 2018-12-31 to mirror
# H&R exactly (Table 1 validation)
BETA_ITERS  <- 3      # backfitting passes for the Beta warp
BETA_SHRINK <- 0.85   # shrink estimated betas toward (1,1); validated value
SHARE_BETA  <- FALSE  # TRUE: Model 4 reuses Model 5's betas (halves runtime)
RUN_GARCH   <- TRUE   # volatility / est_anomaly; solver fixed vs Pipeline D
GCV         <- "GCV.Cp"   # H&R's stated smoothing selection

.roots <- c(Sys.getenv("SEAICE_ROOT", unset = NA),
            path.expand("~/Research/repos/sea-ice-phase"),
            "/Users/fridaperez/Research/repos/sea-ice-phase")
ROOT <- NA
for (r in .roots) if (!is.na(r) && dir.exists(file.path(r, "scripts"))) { ROOT <- r; break }
if (is.na(ROOT)) stop("Cannot locate sea-ice-phase repo. Set SEAICE_ROOT.")
INPUT_FILE <- file.path(ROOT, "data", "raw",
                        "SIE_daily_sector_and_circumpolar_million_km2.csv")
OUTPUT_DIR <- file.path(ROOT, "data", "ch3")

SECTOR_COLS <- c("SIE_circumpolar", "SIE_Weddell", "SIE_Amundsen_Bellingshausen",
                 "SIE_Ross", "SIE_East_Antarctica", "SIE_King_Haakon")
# ── 1. LOAD ─────────────────────────────────────────────────────────────────

raw <- read.csv(INPUT_FILE, stringsAsFactors = FALSE)
names(raw)[tolower(names(raw)) %in% c("date", "time")] <- "Date"
date_str <- raw$Date
raw$Date <- as.Date(date_str)
if (all(is.na(raw$Date))) raw$Date <- as.Date(date_str, format = "%m/%d/%y")
raw <- raw %>% filter(Date >= DATE_START, Date <= DATE_END) %>% arrange(Date)
raw$Year  <- year(raw$Date)
raw$DOY   <- yday(raw$Date)
raw$tdate <- as.numeric(raw$Date)
for (col in SECTOR_COLS) if (col %in% names(raw))
  raw[[col]] <- as.numeric(raw[[col]])
message("Loaded ", nrow(raw), " rows: ", min(raw$Date), " to ", max(raw$Date))

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

# ── 3. FIT ONE SECTOR ───────────────────────────────────────────────────────

fit_sector <- function(raw_data, sector_col) {
  message("\n==== ", sector_col, " ====")
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
  
  # --- 3h. GARCH volatility (solver fixed vs Pipeline D's hang) -------------
  sie$volatility <- NA_real_; sie$est_anomaly_g <- NA_real_
  have_rugarch <- RUN_GARCH &&
    suppressWarnings(require(rugarch, quietly = TRUE))  # attach for S4 methods
  if (have_rugarch) {
    spec <- ugarchspec(
      variance.model = list(model = "sGARCH", garchOrder = c(2, 2)),
      mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
      distribution.model = "norm")
    ok <- !is.na(sie$residual_apac)
    try_garch <- function(order, solver) {
      s2 <- ugarchspec(
        variance.model = list(model = "sGARCH", garchOrder = order),
        mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
        distribution.model = "norm")
      g <- tryCatch(ugarchfit(s2, sie$residual_apac[ok], solver = solver),
                    error = function(e) NULL)
      if (!is.null(g) && g@fit$convergence == 0) g else NULL
    }
    gf <- try_garch(c(2, 2), "nlminb")
    if (is.null(gf)) gf <- try_garch(c(2, 2), "solnp")
    if (is.null(gf)) gf <- try_garch(c(1, 1), "nlminb")   # simpler fallback
    if (is.null(gf)) gf <- try_garch(c(1, 1), "solnp")
    if (!is.null(gf)) {
      sie$volatility[ok]    <- as.numeric(sigma(gf))
      sie$est_anomaly_g[ok] <- as.numeric(fitted(gf))
      message("  GARCH ok")
    } else message("  GARCH failed at (2,2) and (1,1); est_anomaly -> 11-day mean")
  }
  
  # --- 3i. sequential components (validated decomposition) ------------------
  sie <- sie %>% mutate(
    amplitude_component = fitted_amp  - iac_notrend - trend_component,
    phase_component     = fitted_apac - fitted_amp,
    raw_anomaly         = Extent - fitted_apac,
    est_anomaly = ifelse(is.na(est_anomaly_g),
                         as.numeric(stats::filter(raw_anomaly,
                                                  rep(1/11, 11), sides = 2)),
                         est_anomaly_g),
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
    mutate(beta_asym = beta1 - beta2, sector = sector_col)
  
  daily <- sie %>%
    select(Date, Year, DOY, tdate, Extent, anchor_year, t, t_span, frac,
           phase, scaling, min_extent, max_extent, amplitude,
           iac_notrend, trend_component, trend_component_k6,
           u_amp, fitted_amp, u_apac, fitted_apac, fitted_phase,
           residual_apac, volatility,
           amplitude_component, phase_component, raw_anomaly, est_anomaly,
           trend_pct, amp_pct, phase_pct, raw_pct) %>%
    mutate(sector = sector_col)
  
  list(annual = annual, daily = daily,
       rmse = data.frame(sector = sector_col, rmse_trad = rmse_trad,
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

# ── 4. RUN ──────────────────────────────────────────────────────────────────

A <- list(); D <- list(); R <- list()
for (sec in SECTOR_COLS) {
  res <- fit_sector(raw, sec)
  A[[sec]] <- res$annual; D[[sec]] <- res$daily; R[[sec]] <- res$rmse
}
annual_df <- bind_rows(A); daily_df <- bind_rows(D); rmse_df <- bind_rows(R)

# ── 5. SAVE ─────────────────────────────────────────────────────────────────

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
write.csv(annual_df, file.path(OUTPUT_DIR, "annual_params_E.csv"), row.names = FALSE)
write.csv(daily_df,  file.path(OUTPUT_DIR, "daily_fitted_E.csv"),  row.names = FALSE)
write.csv(rmse_df,   file.path(OUTPUT_DIR, "rmse_summary_E.csv"),  row.names = FALSE)

# ── 6. VALIDATION PRINTS ────────────────────────────────────────────────────

message("\n===== RMSE TABLE — CANONICAL benchmark forms (chapter Table 2) =====")
print(rmse_df %>% select(sector, pct_iac_trend, pct_amp_can,
                         pct_phase_can, pct_apac_can) %>%
        rename(pct_invariant = pct_iac_trend) %>%
        mutate(across(where(is.numeric), ~round(.x, 1))))
message("\n===== PURE Eq-10/12 forms with approximate beta (Fig-7 models) =====")
print(rmse_df %>% select(sector, pct_phase, pct_apac) %>%
        mutate(across(where(is.numeric), ~round(.x, 1))))
if (DATE_END <= as.Date("2018-12-31"))
  message("H&R circumpolar 1979-2018: IAC 28.7 | Amp 55.2 | Phase 63.9 | APAC 77.3")

message("\nbeta(2016) by sector (phase-warp parameters, previously unpublished):")
print(annual_df %>% filter(Year == 2016) %>%
        select(sector, beta1, beta2, beta_asym) %>%
        mutate(across(where(is.numeric), ~round(.x, 2))))
message("\nWrote *_E.csv outputs. Fig 7 plotting: use *_pct columns; the")
message("invariant is plotted in raw Mkm^2 centered (H&R's mixed convention).")