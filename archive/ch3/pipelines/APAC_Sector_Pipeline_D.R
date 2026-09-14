# =============================================================================
# APAC_Sector_Pipeline_D.R  --  TWO-LIMB PHASE (H&R Eq. 8-9, both extrema)
# -----------------------------------------------------------------------------
# Purpose: test whether the phase term has been crippled by model
# mis-specification, and produce H&R Figure 7a components for every
# sector-year.
#
# THE CENTRAL ISSUE
#
# H&R Eq. 7:   extent(t) = a_P[phase(t)] + alpha(t)
# H&R Eq. 11:  extent(t) = a_A[phase(t), min, max] + alpha(t)
#
# In both, the cycle is indexed by PHASE. Phase REPLACES day-of-year; it does
# not sit alongside it. Pipeline A/B fitted
#     Extent ~ s(tdate) + s(DOY) + s(phase)
# so s(DOY) already absorbs the whole cycle shape and s(phase) -- a monotone
# re-indexing of the same day counter -- has almost nothing left to explain.
# Measured phase share was 1-3% in every sector at every trend basis. That is
# a symptom of this, not a finding about sea ice.
#
# Set INCLUDE_DOY_IN_PHASE_MODELS <- FALSE for the H&R-faithful form.
# Run it BOTH ways and compare: that comparison is the diagnostic.
#
# SECOND ISSUE: THE BETA WARP
#
# phase = 365 * pbeta((t - t_min)/(t_max - t_min), b1, b2)
#
# With b=(1,1) this is a shift AND a uniform dilation (t_max - t_min varies
# 318-429 d, so the dilation is real). What b=(1,1) CANNOT do is warp the
# advance limb differently from the retreat limb. But that asymmetry is
# exactly H&R's 2016 result: "the phase contributed a small positive anomaly
# during the growth stage and a strongly negative anomaly during retreat".
# H&R Eq. 10/12 minimise over {b1(y), b2(y)} -- they estimate it.
#
# ESTIMATE_BETA <- TRUE turns on iterative estimation (backfitting: fit the
# spline given phase, then per year optimise b given the spline, repeat).
# NOTE: a rough Python prototype of this converged but did not clearly improve
# fit. Treat it as exploratory and check BETA_SUMMARY before relying on it.
#
# THE FIX THAT MATTERS -- TWO-LIMB PHASE
#
# H&R Eq. 8-9 define phase on min.extent.day(y) <= t <= max.extent.day(y):
# anchored at BOTH extrema. Pipelines A-C anchored one Beta at the minimum
# and warped the whole min-to-min cycle; the maximum date played no role
# (handoff finding #1). A year with an early maximum (2016 circumpolar: -21 d)
# could not have its retreat pulled early, and fitted betas stayed ~1.05.
#
# Pipeline D: each cycle year is an ADVANCE limb (min -> max) and a RETREAT
# limb (max -> next min), each with its own Beta warp:
#   advance:  phase = P_max * pbeta(u, ba1, ba2)
#   retreat:  phase = P_max + (365 - P_max) * pbeta(u, br1, br2)
# Binned-mean test, phase-only, beta fitted (variance explained vs traditional):
#   circumpolar  one-limb +3.4%  ->  two-limb +43.3%
#   Ross                 +13.2%  ->           +55.9%
#   Weddell              +21.3%  ->           +56.5%
#
# HANDCOCK ALIGNMENT (Sep 2026 email from author)
#   - s(tdate) uses bs="tp" (default) not "cc"     <- already correct in Pipeline D
#   - method = "REML" not "GCV.Cp"                  <- already correct in Pipeline D
#   - s(DOY, bs="cc", k=50, fx=FALSE) in IAC model  <- IAC_DOY_K bumped 25 -> 50
#   - knots = list(DOY = c(0, 365))                 <- changed from c(1, 365)
#   - as.numeric(Extent) type cast                  <- ADDED at load time
#
# OUTPUTS
#   annual_params_D.csv      annual scalars incl. estimated betas
#   daily_fitted_D.csv       daily series + Fig 7a components (sum exactly)
#   rmse_summary_D.csv       Table-1 style comparison, both model forms
#   fig7a_components.csv     tidy long format, every sector-year, for plotting
# =============================================================================

library(dplyr)
library(lubridate)
library(mgcv)
library(rugarch)
have_tidyr <- requireNamespace("tidyr", quietly = TRUE)
if (!have_tidyr) message("NOTE: tidyr not installed; fig7a_components.csv will be skipped.")

# ── 0. SETTINGS ──────────────────────────────────────────────────────────────

# THE KEY TOGGLE. Set TRUE to include s(DOY) alongside s(phase) in phase/APAC
# models. This is the form that produced Table 2 results (Phase 70.5%, APAC 80.9%)
# and reproduces H&R Table 1 within ~3 percentage points.
# Setting FALSE ("H&R-faithful": phase replaces DOY) breaks the phase RMSE badly
# without proper Beta estimation.
INCLUDE_DOY_IN_PHASE_MODELS <- TRUE

# Estimate the Beta warp per year instead of fixing b = (1,1)?
# TRUE is now the default: with b fixed at (1,1) the phase model is WORSE
# than the traditional cycle (Ross -130%), because re-indexing to a noisy
# minimum date smears the composite. Fitted b recovers +44% in Ross.
ESTIMATE_BETA <- FALSE   # diagnostic run: skip beta backfitting
BETA_ITERS    <- 4          # backfitting iterations
BETA_BOUNDS   <- c(0.4, 2.5) # keep the warp physically sensible
BETA_GRID     <- c(0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0)  # first-iteration start grid

TREND_BS <- "tp"
TREND_K  <- 12    # Handcock used default k=10 and got edf 8.98.
# Raised from 8 to unlock the 2014+ trend "hook" for 2016.
DOY_K    <- 100
PHASE_K  <- 100
IAC_DOY_K <- 50   # Handcock's spec (was 25).

# Repo root: local first, cluster fallback
.roots <- c(Sys.getenv("SEAICE_ROOT", unset = NA),
            path.expand("~/Research/repos/sea-ice-phase"),
            "/Users/fridaperez/Research/repos/sea-ice-phase",
            "/user/geog/falejandraperez/sea-ice-phase")
ROOT <- NA
for (r in .roots) if (!is.na(r) && dir.exists(file.path(r, "scripts"))) { ROOT <- r; break }
if (is.na(ROOT)) stop("Cannot locate sea-ice-phase repo. Set SEAICE_ROOT.")

INPUT_FILE <- file.path(ROOT, "scripts", "R", "observations",
                        "SIE_daily_sector_and_circumpolar_million_km2.csv")
OUTPUT_DIR <- file.path(ROOT, "scripts", "R", "Ch3", "data")

DATE_START <- as.Date("1979-01-01")
DATE_END   <- as.Date("2023-12-31")

SECTOR_COLS <- c("SIE_circumpolar")   # diagnostic run: circumpolar only
# Full production list:
# SECTOR_COLS <- c("SIE_Weddell", "SIE_Amundsen_Bellingshausen", "SIE_Ross",
#                  "SIE_East_Antarctica", "SIE_King_Haakon", "SIE_circumpolar")

message("=== Pipeline D: two-limb phase ===")
message("  s(DOY) in phase/APAC models : ", INCLUDE_DOY_IN_PHASE_MODELS,
        if (!INCLUDE_DOY_IN_PHASE_MODELS) "   <- H&R-faithful" else "   <- Pipeline A/B form")
message("  Estimate Beta warp          : ", ESTIMATE_BETA,
        if (!ESTIMATE_BETA) "   <- WARNING: phase model will be worse than traditional" else "")

trend_term <- sprintf('s(tdate, bs="%s", k=%d)', TREND_BS, TREND_K)
doy_term   <- sprintf('s(DOY, bs="cc", k=%d, fx=FALSE)', DOY_K)
phase_term <- sprintf('s(phase, bs="cc", k=%d, fx=FALSE)', PHASE_K)

# Formula builders -----------------------------------------------------------
f_phase_model <- function(resp) {
  if (INCLUDE_DOY_IN_PHASE_MODELS) {
    rhs <- paste(trend_term, "+", doy_term, "+", phase_term)
  } else {
    rhs <- paste(trend_term, "+", phase_term)
  }
  as.formula(paste(resp, "~", rhs))
}
knots_for <- function() {
  if (INCLUDE_DOY_IN_PHASE_MODELS) {
    list(DOY = c(0, 365), phase = c(0, 365))
  } else {
    list(phase = c(0, 365))
  }
}

centre_doy <- function(doy, ref) ((doy - ref + 182) %% 365) - 182 + ref

# ── 1. LOAD ──────────────────────────────────────────────────────────────────

raw <- read.csv(INPUT_FILE, stringsAsFactors = FALSE)
names(raw)[tolower(names(raw)) %in% c("date", "time")] <- "Date"
if (grepl("^\\d{4}-\\d{2}-\\d{2}", raw$Date[1])) {
  raw$Date <- as.Date(raw$Date)              # ISO yyyy-mm-dd
} else {
  raw$Date <- as.Date(raw$Date, format = "%m/%d/%y")   # legacy US
}
if (all(is.na(raw$Date))) stop("Date parsing failed.")

raw <- raw %>% filter(Date >= DATE_START, Date <= DATE_END) %>% arrange(Date)
raw$Year <- year(raw$Date); raw$DOY <- yday(raw$Date); raw$tdate <- as.numeric(raw$Date)

# Handcock's flagged bug (Sep 2026 email): ensure all sector Extent columns
# are numeric, not character. Silent failure otherwise.
for (col in SECTOR_COLS) {
  if (col %in% names(raw)) raw[[col]] <- as.numeric(raw[[col]])
}

message("Loaded ", nrow(raw), " rows: ", min(raw$Date), " to ", max(raw$Date))

# ── 2. FIT ONE SECTOR ────────────────────────────────────────────────────────

fit_sector <- function(raw_data, sector_col) {
  
  message("\n==== ", sector_col, " ====")
  
  sie <- raw_data %>%
    select(Date, Year, DOY, tdate, Extent = all_of(sector_col)) %>%
    filter(!is.na(Extent)) %>% arrange(Date)
  
  yearly_stats <- sie %>% group_by(Year) %>%
    summarise(min_extent = min(Extent), max_extent = max(Extent),
              amplitude_raw = max_extent - min_extent,
              min_doy_raw = DOY[which.min(Extent)],
              max_doy_raw = DOY[which.max(Extent)],
              min_date = Date[which.min(Extent)],
              max_date = Date[which.max(Extent)], .groups = "drop")
  
  # NOTE: scaling_factor is recomputed below on CYCLE-year extrema once
  # cyc_year is known. The calendar-year version here is a placeholder so
  # the pipe does not break; it is overwritten in 2a.
  sie <- sie %>% left_join(yearly_stats, by = "Year") %>%
    mutate(scaling_factor = (Extent - min_extent) / (amplitude_raw + 1e-10))
  
  # --- 2a. TWO-LIMB cycle clock (H&R Eq. 8-9, anchored at BOTH extrema) -----
  # Each cycle year y runs min_date(y) -> max_date(y) [advance] ->
  # min_date(y+1) [retreat]. Calendar Jan-Feb before min_date(y) belong to
  # cycle year y-1. Each day gets: cyc_year, limb (0 adv / 1 ret), u in [0,1].
  ys   <- yearly_stats %>% arrange(Year)
  mind <- setNames(as.numeric(ys$min_date), ys$Year)
  maxd <- setNames(as.numeric(ys$max_date), ys$Year)
  yrs_ok <- ys$Year[(ys$Year + 1) %in% ys$Year]
  
  dnum <- as.numeric(sie$Date)
  sie$cyc_year <- NA_integer_; sie$limb <- NA_integer_; sie$u <- NA_real_
  for (y in yrs_ok) {
    m0 <- mind[as.character(y)]; mx <- maxd[as.character(y)]; m1 <- mind[as.character(y + 1)]
    if (!(m0 < mx && mx < m1)) next
    ia <- dnum >= m0 & dnum < mx
    ir <- dnum >= mx & dnum < m1
    sie$cyc_year[ia] <- as.integer(y); sie$limb[ia] <- 0L; sie$u[ia] <- (dnum[ia] - m0) / (mx - m0)
    sie$cyc_year[ir] <- as.integer(y); sie$limb[ir] <- 1L; sie$u[ir] <- (dnum[ir] - mx) / (m1 - mx)
  }
  sie <- sie %>% filter(!is.na(cyc_year))
  sie$u <- pmin(pmax(sie$u, 1e-9), 1 - 1e-9)
  
  # --- Amplitude normalisation on CYCLE-year extrema --------------------------
  # Each cycle year y runs min(y) -> max(y) -> min(y+1). Its amplitude is
  # max(y) - min(y) and its base is min(y). Joining by CALENDAR year (as
  # H&R Eq. 5 literally states) scales Jan-Feb of year y+1 by year y+1's
  # extrema, producing a step in every component at the calendar boundary.
  cyc_stats <- yearly_stats %>%
    transmute(cyc_year = Year,
              cyc_min_extent = min_extent, cyc_max_extent = max_extent,
              cyc_amplitude  = amplitude_raw)
  sie <- sie %>%
    select(-any_of(c("min_extent", "max_extent", "amplitude_raw"))) %>%
    left_join(cyc_stats, by = "cyc_year") %>%
    rename(min_extent = cyc_min_extent, max_extent = cyc_max_extent,
           amplitude_raw = cyc_amplitude) %>%
    mutate(scaling_factor = (Extent - min_extent) / (amplitude_raw + 1e-10)) %>%
    filter(!is.na(scaling_factor))
  
  adv_len <- mean(maxd[as.character(yrs_ok)] - mind[as.character(yrs_ok)])
  cyc_len <- mean(mind[as.character(yrs_ok + 1)] - mind[as.character(yrs_ok)])
  P_MAX <- 365 * adv_len / cyc_len
  message("  P_max (climatological phase of maximum): ", round(P_MAX, 1),
          "   advance ", round(adv_len), " d of ", round(cyc_len), " d cycle")
  
  sie$t     <- ifelse(sie$limb == 0L, sie$u * adv_len, adv_len + sie$u * (cyc_len - adv_len))
  sie$t_min <- 0; sie$t_max <- cyc_len
  first_year <- min(sie$cyc_year)
  
  # --- 2b. phase from per-limb Beta CDFs -----------------------------------
  make_phase <- function(df, betas) {
    b <- betas[match(df$cyc_year, betas$Year), ]
    adv <- P_MAX * pbeta(df$u, b$ba1, b$ba2)
    ret <- P_MAX + (365 - P_MAX) * pbeta(df$u, b$br1, b$br2)
    ifelse(df$limb == 0L, adv, ret)
  }
  init_betas <- function(df) data.frame(Year = sort(unique(df$cyc_year)),
                                        ba1 = 1, ba2 = 1, br1 = 1, br2 = 1)
  
  # --- 2c. Beta estimation (backfitting), per response, per limb -----------
  estimate_beta <- function(sie, resp, label) {
    betas <- init_betas(sie)
    sie$phase <- make_phase(sie, betas)
    message("  Estimating two-limb Beta warp on ", label, " (", BETA_ITERS, " iterations)...")
    for (it in seq_len(BETA_ITERS)) {
      g_tmp <- gam(f_phase_model(resp), data = sie, method = "REML", knots = knots_for())
      pred_at <- function(newdf) as.numeric(predict(g_tmp, newdata = newdf))
      for (i in seq_len(nrow(betas))) {
        Y <- betas$Year[i]
        for (limb in c(0L, 1L)) {
          idx <- which(sie$cyc_year == Y & sie$limb == limb)
          if (length(idx) < 20) next
          dsub <- sie[idx, ]
          cols <- if (limb == 0L) c("ba1", "ba2") else c("br1", "br2")
          obj <- function(b) {
            if (any(b <= 0)) return(1e9)
            if (limb == 0L) {
              dsub$phase <- P_MAX * pbeta(dsub$u, b[1], b[2])
            } else {
              dsub$phase <- P_MAX + (365 - P_MAX) * pbeta(dsub$u, b[1], b[2])
            }
            mean((dsub[[resp]] - pred_at(dsub))^2, na.rm = TRUE)
          }
          start <- as.numeric(betas[i, cols])
          if (it == 1) {
            best_g <- obj(start)
            for (g1 in BETA_GRID) for (g2 in BETA_GRID) {
              v <- obj(c(g1, g2))
              if (is.finite(v) && v < best_g) { best_g <- v; start <- c(g1, g2) }
            }
          }
          opt <- tryCatch(optim(start, obj, method = "L-BFGS-B",
                                lower = BETA_BOUNDS[1], upper = BETA_BOUNDS[2],
                                control = list(maxit = 60)),
                          error = function(e) NULL)
          cur <- obj(as.numeric(betas[i, cols]))
          if (!is.null(opt) && opt$value < cur) {
            betas[i, cols] <- opt$par
          } else if (obj(start) < cur) {
            betas[i, cols] <- start
          }
        }
      }
      sie$phase <- make_phase(sie, betas)
      rmse_it <- sqrt(mean((sie[[resp]] - as.numeric(predict(g_tmp)))^2, na.rm = TRUE))
      message("    iter ", it, "  RMSE ", round(rmse_it, 5),
              "  adv b=(", round(mean(betas$ba1), 2), ",", round(mean(betas$ba2), 2), ")",
              "  ret b=(", round(mean(betas$br1), 2), ",", round(mean(betas$br2), 2), ")")
    }
    betas
  }
  
  betas_apac  <- init_betas(sie)
  betas_phase <- betas_apac
  if (ESTIMATE_BETA) {
    betas_apac  <- estimate_beta(sie, "scaling_factor", "scaling_factor (APAC, Eq. 12)")
    betas_phase <- estimate_beta(sie, "Extent",         "Extent (phase-only, Eq. 10)")
  }
  betas <- betas_apac
  
  # --- 2d. Model 1: traditional --------------------------------------------
  trad <- sie %>% group_by(DOY) %>%
    summarise(trad_mean = mean(Extent, na.rm = TRUE), .groups = "drop")
  sie <- sie %>% left_join(trad, by = "DOY")
  rmse_trad <- sqrt(mean((sie$Extent - sie$trad_mean)^2, na.rm = TRUE))
  
  # --- 2e. trend-free invariant (Fig 7a pale-blue reference) ---------------
  # Handcock spec: s(DOY, bs="cc", k=50, fx=FALSE), knots=list(DOY=c(0,365))
  g_iac0 <- gam(as.formula(paste0('Extent ~ s(DOY, bs="cc", k=', IAC_DOY_K, ', fx=FALSE)')),
                data = sie, method = "REML", knots = list(DOY = c(0, 365)))
  sie$iac_notrend      <- as.numeric(predict(g_iac0))
  sie$anomaly_from_iac <- sie$Extent - sie$iac_notrend
  
  # invariant WITH trend (Table 1 Model 2) -- Handcock's exact spec
  g_iac <- gam(as.formula(paste("Extent ~", trend_term, "+", doy_term)),
               data = sie, method = "REML", knots = list(DOY = c(0, 365)))
  sie$fitted_invariant <- as.numeric(predict(g_iac))
  rmse_iac <- sqrt(mean((sie$Extent - sie$fitted_invariant)^2, na.rm = TRUE))
  
  # --- 2f. Model 3: amplitude-adjusted (always trend + DOY) -----------------
  g_amp <- gam(as.formula(paste("scaling_factor ~", trend_term, "+", doy_term)),
               data = sie, method = "REML", knots = list(DOY = c(0, 365)))
  sie$fitted_amp <- as.numeric(predict(g_amp)) * sie$amplitude_raw + sie$min_extent
  rmse_amp <- sqrt(mean((sie$Extent - sie$fitted_amp)^2, na.rm = TRUE))
  
  # --- 2g. Model 4: phase-adjusted (beta fitted on Extent, H&R Eq. 10) -----
  sie$phase <- make_phase(sie, betas_phase)
  g_phase <- gam(f_phase_model("Extent"), data = sie,
                 method = "REML", knots = knots_for())
  sie$fitted_phase <- as.numeric(predict(g_phase))
  rmse_phase <- sqrt(mean((sie$Extent - sie$fitted_phase)^2, na.rm = TRUE))
  
  # --- 2h. Model 5: APAC (beta fitted on scaling_factor, H&R Eq. 12) -------
  sie$phase <- make_phase(sie, betas_apac)
  g_apac <- gam(f_phase_model("scaling_factor"), data = sie,
                method = "REML", knots = knots_for())
  sie$fitted_apac   <- as.numeric(predict(g_apac)) * sie$amplitude_raw + sie$min_extent
  sie$residual_apac <- sie$Extent - sie$fitted_apac
  rmse_apac <- sqrt(mean(sie$residual_apac^2, na.rm = TRUE))
  
  pct <- function(r) 100 * (1 - r^2 / rmse_trad^2)
  message(sprintf("  Trad %.4f | IAC %.4f (%.1f%%) | Amp %.4f (%.1f%%) | Phase %.4f (%.1f%%) | APAC %.4f (%.1f%%)",
                  rmse_trad, rmse_iac, pct(rmse_iac), rmse_amp, pct(rmse_amp),
                  rmse_phase, pct(rmse_phase), rmse_apac, pct(rmse_apac)))
  
  edf_trend <- summary(g_apac)$s.table["s(tdate)", "edf"]
  
  # --- 2i. GARCH DISABLED for diagnostic run --------------------------------
  # Original hybrid solver was hanging on ABS. Enable after diagnostic works.
  sie$volatility <- NA_real_
  sie$arma_mean  <- NA_real_
  # spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(2, 2)),
  #                    mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
  #                    distribution.model = "norm")
  # ok <- !is.na(sie$residual_apac)
  # gf <- tryCatch(ugarchfit(spec, sie$residual_apac[ok], solver = "hybrid"),
  #                error = function(e) { message("  GARCH failed: ", e$message); NULL })
  # if (!is.null(gf)) {
  #   sie$volatility[ok] <- as.numeric(sigma(gf))
  #   sie$arma_mean[ok]  <- as.numeric(fitted(gf))
  # }
  
  # --- 2j. H&R Figure 7a components ----------------------------------------
  # anomaly_from_iac = trend + amplitude + phase + residual_apac  (exact)
  # Decomposition is sequential: amplitude extracted before phase.
  # Trend in EXTENT units, from the invariant model (H&R Eq. 15):
  #   extent(t) = trend(t) + a[doy(t)]
  # Previously this used the s(tdate) term of the normalised APAC model
  # scaled by amplitude -- a trend in cycle SHAPE, not in extent, and near
  # zero within any single year. The extent trend carries the 2016-2018
  # decline that H&R's Fig 6 shows.
  iac_terms <- predict(g_iac, type = "terms")
  sie$trend_component <- as.numeric(iac_terms[, "s(tdate)"])
  
  sie <- sie %>% mutate(
    amplitude_component = fitted_amp  - iac_notrend - trend_component,
    phase_component     = fitted_apac - fitted_amp,
    raw_anomaly         = Extent - fitted_apac,
    est_anomaly         = arma_mean)
  
  sum_err <- mean(abs(sie$anomaly_from_iac -
                        (sie$trend_component + sie$amplitude_component +
                           sie$phase_component + sie$residual_apac)), na.rm = TRUE)
  message("  Fig 7a sum check, mean |error| = ", format(sum_err, digits = 3))
  if (sum_err > 1e-8) warning("Components do not sum for ", sector_col)
  
  # --- 2k. annual parameters ------------------------------------------------
  ref_min <- median(sie$min_doy_raw); ref_max <- median(sie$max_doy_raw)
  
  # Seasonal-mean SIE per year, for the RQ3 comparison: does an index that
  # correlates with the SIE ANOMALY correlate with timing, amplitude, or neither?
  annual <- sie %>% group_by(Year) %>%
    summarise(sie_annual = mean(Extent, na.rm = TRUE),
              sie_DJF = mean(Extent[month(Date) %in% c(12, 1, 2)], na.rm = TRUE),
              sie_MAM = mean(Extent[month(Date) %in% 3:5], na.rm = TRUE),
              sie_JJA = mean(Extent[month(Date) %in% 6:8], na.rm = TRUE),
              sie_SON = mean(Extent[month(Date) %in% 9:11], na.rm = TRUE),
              max_doy_fitted = DOY[which.max(fitted_phase)],
              min_doy_fitted = DOY[which.min(fitted_phase)],
              amplitude_fitted = max(fitted_amp) - min(fitted_amp),
              max_doy_raw = DOY[which.max(Extent)],
              min_doy_raw = DOY[which.min(Extent)],
              amplitude_raw_yr = max(Extent) - min(Extent),
              cycle_length = max(t_max) - max(t_min),
              .groups = "drop") %>%
    mutate(min_doy_raw_c = centre_doy(min_doy_raw, ref_min),
           max_doy_raw_c = centre_doy(max_doy_raw, ref_max)) %>%
    mutate(sie_annual_anom = sie_annual - mean(sie_annual),
           sie_DJF_anom = sie_DJF - mean(sie_DJF), sie_MAM_anom = sie_MAM - mean(sie_MAM),
           sie_JJA_anom = sie_JJA - mean(sie_JJA), sie_SON_anom = sie_SON - mean(sie_SON)) %>%
    mutate(max_doy_raw_anom   = max_doy_raw_c - median(max_doy_raw_c),
           min_doy_raw_anom   = min_doy_raw_c - median(min_doy_raw_c),
           amplitude_raw_anom = amplitude_raw_yr - mean(amplitude_raw_yr),
           amplitude_anom     = amplitude_fitted - mean(amplitude_fitted)) %>%
    left_join(betas_apac, by = "Year") %>%
    mutate(beta_asym_adv = ba1 - ba2, beta_asym_ret = br1 - br2) %>%
    left_join(betas_phase %>% rename(ba1_phase = ba1, ba2_phase = ba2,
                                     br1_phase = br1, br2_phase = br2), by = "Year") %>%
    mutate(P_max = P_MAX, sector = sector_col)
  
  daily_out <- sie %>%
    select(Date, Year, DOY, tdate, Extent, phase, cyc_year, limb, u, t, t_min, t_max,
           scaling_factor, iac_notrend, fitted_invariant, fitted_amp,
           fitted_phase, fitted_apac, residual_apac, anomaly_from_iac,
           volatility, arma_mean, trend_component, amplitude_component,
           phase_component, raw_anomaly, est_anomaly) %>%
    mutate(sector = sector_col)
  
  list(annual = annual, daily = daily_out, betas = betas,
       rmse = data.frame(sector = sector_col,
                         rmse_trad = rmse_trad, rmse_iac = rmse_iac,
                         rmse_amp = rmse_amp, rmse_phase = rmse_phase,
                         rmse_apac = rmse_apac,
                         pct_iac = pct(rmse_iac), pct_amp = pct(rmse_amp),
                         pct_phase = pct(rmse_phase), pct_apac = pct(rmse_apac),
                         edf_trend = edf_trend, sum_err = sum_err,
                         doy_in_phase = INCLUDE_DOY_IN_PHASE_MODELS,
                         beta_estimated = ESTIMATE_BETA))
}

# ── 3. RUN ───────────────────────────────────────────────────────────────────

A <- list(); D <- list(); R <- list()
for (sec in SECTOR_COLS) {
  res <- fit_sector(raw, sec)
  A[[sec]] <- res$annual; D[[sec]] <- res$daily; R[[sec]] <- res$rmse
}
annual_df <- bind_rows(A); daily_df <- bind_rows(D); rmse_df <- bind_rows(R)

# ── 4. FIGURE 7a COMPONENTS, TIDY LONG ───────────────────────────────────────
# One row per sector-year-day-component. Feeds an all-sectors Fig 7a.

fig7a <- NULL
if (have_tidyr) {
  fig7a <- daily_df %>%
    select(sector, Year, DOY, iac_notrend, trend_component, amplitude_component,
           phase_component, raw_anomaly, est_anomaly, anomaly_from_iac) %>%
    tidyr::pivot_longer(cols = c(trend_component, amplitude_component,
                                 phase_component, raw_anomaly, est_anomaly,
                                 anomaly_from_iac),
                        names_to = "component", values_to = "value")
}

# ── 5. SAVE ──────────────────────────────────────────────────────────────────

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
sfx <- if (INCLUDE_DOY_IN_PHASE_MODELS) "_D_withDOY" else "_D"
write.csv(annual_df, file.path(OUTPUT_DIR, paste0("annual_params", sfx, ".csv")), row.names = FALSE)
write.csv(daily_df,  file.path(OUTPUT_DIR, paste0("daily_fitted",  sfx, ".csv")), row.names = FALSE)
write.csv(rmse_df,   file.path(OUTPUT_DIR, paste0("rmse_summary",  sfx, ".csv")), row.names = FALSE)
if (!is.null(fig7a))
  write.csv(fig7a, file.path(OUTPUT_DIR, paste0("fig7a_components", sfx, ".csv")), row.names = FALSE)

# ── 6. THE DIAGNOSTIC ────────────────────────────────────────────────────────

message("\n================ TABLE 1 STYLE, ALL SECTORS ================")
print(rmse_df %>% select(sector, pct_iac, pct_amp, pct_phase, pct_apac) %>%
        mutate(across(where(is.numeric), ~round(.x, 1))))

message("\nH&R circumpolar 1979-2018: IAC 28.7, Amp 55.2, Phase 63.9, APAC 77.3")
message("Their key claim: phase > amplitude.\n")

cmp <- rmse_df %>% mutate(phase_beats_amp = pct_phase > pct_amp) %>%
  select(sector, pct_amp, pct_phase, phase_beats_amp)
print(cmp)

message("\n--- Nested phase contribution (APAC minus amplitude-only) ---")
nested <- rmse_df %>%
  mutate(phase_adds = round(pct_apac - pct_amp, 1),
         amp_adds   = round(pct_apac - pct_phase, 1)) %>%
  select(sector, phase_adds, amp_adds)
print(nested)
message("\nIn Pipeline B (with s(DOY)) phase added 1-3% only.")
message("If phase_adds is now much larger, s(DOY) was the problem.")

if (ESTIMATE_BETA) {
  message("\n--- BETA SUMMARY (APAC estimation; adv = min->max, ret = max->next min) ---")
  print(annual_df %>% group_by(sector) %>%
          summarise(adv_b1 = round(mean(ba1, na.rm = TRUE), 2), adv_b2 = round(mean(ba2, na.rm = TRUE), 2),
                    ret_b1 = round(mean(br1, na.rm = TRUE), 2), ret_b2 = round(mean(br2, na.rm = TRUE), 2),
                    sd_ret_asym = round(sd(beta_asym_ret, na.rm = TRUE), 3),
                    max_dev_from_1 = round(max(abs(c(ba1, ba2, br1, br2) - 1), na.rm = TRUE), 2),
                    n_at_bounds = sum(ba1 <= 0.41 | ba2 <= 0.41 | br1 <= 0.41 | br2 <= 0.41 |
                                        ba1 >= 2.49 | ba2 >= 2.49 | br1 >= 2.49 | br2 >= 2.49, na.rm = TRUE),
                    .groups = "drop"))
  message("\n2016 and 2023 betas (APAC estimation):")
  print(annual_df %>% filter(Year %in% c(2016, 2023)) %>%
          select(sector, Year, ba1, ba2, br1, br2, beta_asym_ret) %>%
          mutate(across(where(is.numeric), ~round(.x, 2))))
}

message("\nWrote files with suffix '", sfx, "'")
message("NEXT: flip INCLUDE_DOY_IN_PHASE_MODELS and rerun, then compare the")
message("two rmse_summary files. That comparison is the diagnostic.")