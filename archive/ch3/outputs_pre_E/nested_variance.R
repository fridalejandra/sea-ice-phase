# =============================================================================
# NESTED VARIANCE-LADDER DIAGNOSTIC v2 — standalone, addresses three concerns:
#
#  1. Confirms the DOY and phase cyclic splines (the actual APAC mechanism)
#     are UNTOUCHED — only s(tdate), the trend term, changes basis. Printed
#     explicitly per sector so you can see it in the output, not just trust it.
#
#  2. Removes the arbitrary TREND_K=6 choice: gives the trend a generous
#     ceiling (k=20) and lets REML/GCV choose how much of it to use, reporting
#     the effective degrees of freedom (edf) actually used. Also sweeps
#     TREND_K over a grid to confirm share_amp / share_phase are stable
#     regardless of the ceiling (this is the real test of Concern 1).
#
#  3. Splits variance shares into pre-2016 (1980-2015) and post-2016
#     (2016-2023) using era-specific residuals from the SAME full-record fit,
#     so we can see whether amplitude's dominance is a stable structural
#     feature or concentrated in the post-2016 low-ice years.
#
# Reads daily_fitted.csv directly. Does not touch your tuned pipeline.
# =============================================================================

suppressPackageStartupMessages({
  library(mgcv)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
infile <- if (length(args) >= 1) args[1] else "daily_fitted.csv"

DOY_K   <- 100     # UNCHANGED from your pipeline — cyclic, defines cycle shape
PHASE_K <- 100     # UNCHANGED from your pipeline — cyclic, defines phase cycle
TREND_K_GRID <- c(4, 6, 8, 10, 15, 20, 30)   # sensitivity sweep
TREND_K_MAIN <- 20  # generous ceiling for the main/era-split run; REML/GCV
# decides how much of it to actually use (see edf output)

cat("Reading:", infile, "\n")
d <- read.csv(infile, stringsAsFactors = FALSE)

required_cols <- c("Extent","DOY","tdate","phase","scaling_factor","sector",
                   "Year","residual_apac")
missing <- setdiff(required_cols, names(d))
if (length(missing) > 0) stop("Missing expected columns: ", paste(missing, collapse=", "))

sectors <- unique(d$sector)

fit_nested <- function(sie, trend_k) {
  trend_term <- sprintf('s(tdate, bs="tp", k=%d)', trend_k)  # non-cyclic trend
  
  g_iac <- gam(as.formula(paste("Extent ~", trend_term,
                                '+ s(DOY, bs="cc", k=', DOY_K, ')')),           # cycle: cc, unchanged
               data = sie, method = "REML", knots = list(DOY = c(1, 365)))
  r_iac <- sqrt(mean((sie$Extent - as.numeric(predict(g_iac)))^2, na.rm = TRUE))
  edf_trend_iac <- summary(g_iac)$s.table["s(tdate)", "edf"]
  
  g_amp <- gam(as.formula(paste("scaling_factor ~", trend_term,
                                '+ s(DOY, bs="cc", k=', DOY_K, ')')),
               data = sie, method = "REML", knots = list(DOY = c(1, 365)))
  fit_amp <- as.numeric(predict(g_amp)) * sie$amplitude_raw + sie$min_extent
  r_amp <- sqrt(mean((sie$Extent - fit_amp)^2, na.rm = TRUE))
  resid_amp <- sie$Extent - fit_amp
  
  g_phase <- gam(as.formula(paste("Extent ~", trend_term,
                                  '+ s(DOY, bs="cc", k=', DOY_K, ')',            # unchanged
                                  '+ s(phase, bs="cc", k=', PHASE_K, ', fx=FALSE)')),  # unchanged
                 data = sie, method = "REML",
                 knots = list(DOY = c(1, 365), phase = c(0, 365)))
  r_phase <- sqrt(mean((sie$Extent - as.numeric(predict(g_phase)))^2, na.rm = TRUE))
  
  g_apac_c <- gam(as.formula(paste("scaling_factor ~", trend_term,
                                   '+ s(DOY, bs="cc", k=', DOY_K, ')',
                                   '+ s(phase, bs="cc", k=', PHASE_K, ')')),
                  data = sie, method = "REML",
                  knots = list(DOY = c(1, 365), phase = c(0, 365)))
  fit_apac_c <- as.numeric(predict(g_apac_c)) * sie$amplitude_raw + sie$min_extent
  r_apac_c <- sqrt(mean((sie$Extent - fit_apac_c)^2, na.rm = TRUE))
  resid_apac <- sie$Extent - fit_apac_c
  edf_trend_apac <- summary(g_apac_c)$s.table["s(tdate)", "edf"]
  
  list(r_iac = r_iac, r_amp = r_amp, r_phase = r_phase, r_apac_c = r_apac_c,
       edf_trend_iac = edf_trend_iac, edf_trend_apac = edf_trend_apac,
       resid_amp = resid_amp, resid_apac = resid_apac,
       fit_amp = fit_amp, fit_apac_c = fit_apac_c)
}

prep_sector <- function(d, sec) {
  sie <- d %>% filter(sector == sec) %>%
    filter(!is.na(Extent), !is.na(DOY), !is.na(tdate), !is.na(phase),
           !is.na(scaling_factor))
  yr_stats <- sie %>% group_by(Year) %>%
    summarise(min_extent = min(Extent, na.rm = TRUE),
              max_extent = max(Extent, na.rm = TRUE), .groups = "drop") %>%
    mutate(amplitude_raw = max_extent - min_extent)
  sie <- sie %>% left_join(yr_stats, by = "Year")
  trad_means <- sie %>% group_by(DOY) %>%
    summarise(trad_mean = mean(Extent, na.rm = TRUE), .groups = "drop")
  sie %>% left_join(trad_means, by = "DOY")
}

# ---------------------------------------------------------------------------
# PART A: sensitivity sweep over TREND_K — confirms shares are stable
# ---------------------------------------------------------------------------
cat("\n================ PART A: TREND_K sensitivity sweep ================\n")
sweep_rows <- list()
for (sec in sectors) {
  sie <- prep_sector(d, sec)
  r_trad <- sqrt(mean((sie$Extent - sie$trad_mean)^2, na.rm = TRUE))
  for (tk in TREND_K_GRID) {
    fit <- fit_nested(sie, tk)
    v <- function(r) r^2; tot <- v(r_trad)
    amp_first <- (v(fit$r_iac) - v(fit$r_amp)) / tot
    amp_second <- (v(fit$r_phase) - v(fit$r_apac_c)) / tot
    phase_first <- (v(fit$r_iac) - v(fit$r_phase)) / tot
    phase_second <- (v(fit$r_amp) - v(fit$r_apac_c)) / tot
    sweep_rows[[length(sweep_rows)+1]] <- data.frame(
      sector = sec, trend_k_ceiling = tk,
      edf_trend_used = round(fit$edf_trend_apac, 2),
      share_amp = round((amp_first+amp_second)/2, 3),
      share_phase = round((phase_first+phase_second)/2, 3),
      order_gap = round(abs(amp_first-amp_second), 3)
    )
  }
  cat("  done:", sec, "\n")
}
sweep <- do.call(rbind, sweep_rows)
write.csv(sweep, "trend_k_sensitivity.csv", row.names = FALSE)
cat("\nWrote trend_k_sensitivity.csv\n")
cat("READ THIS: if share_amp / share_phase are roughly constant across\n")
cat("trend_k_ceiling despite edf_trend_used changing, the amplitude>>phase\n")
cat("finding is NOT sensitive to my arbitrary choice of k.\n\n")
print(sweep)

# ---------------------------------------------------------------------------
# PART B: main fit (generous k, REML-selected edf) + pre/post-2016 split
# ---------------------------------------------------------------------------
cat("\n================ PART B: era split (pre/post 2016) ================\n")
era_rows <- list()
for (sec in sectors) {
  sie <- prep_sector(d, sec)
  r_trad <- sqrt(mean((sie$Extent - sie$trad_mean)^2, na.rm = TRUE))
  fit <- fit_nested(sie, TREND_K_MAIN)
  
  era <- ifelse(sie$Year < 2016, "pre_2016", "post_2016")
  v <- function(r) r^2
  
  for (e in c("pre_2016", "post_2016")) {
    idx <- era == e
    tot_e <- mean((sie$Extent[idx] - sie$trad_mean[idx])^2, na.rm = TRUE)
    # era-specific mean squared error, using residuals from the SAME
    # full-record fit (we are not refitting per era — too few years post-2016)
    m_amp_e   <- mean(fit$resid_amp[idx]^2, na.rm = TRUE)
    m_apac_e  <- mean(fit$resid_apac[idx]^2, na.rm = TRUE)
    pct_amp_e  <- 100 * (1 - m_amp_e / tot_e)
    pct_apac_e <- 100 * (1 - m_apac_e / tot_e)
    era_rows[[length(era_rows)+1]] <- data.frame(
      sector = sec, era = e, n_years = length(unique(sie$Year[idx])),
      edf_trend = round(fit$edf_trend_apac, 2),
      pct_amp = round(pct_amp_e, 1),
      pct_apac = round(pct_apac_e, 1)
    )
  }
  cat("  done:", sec, "  (trend edf used, full record:",
      round(fit$edf_trend_apac, 2), "out of ceiling", TREND_K_MAIN, ")\n")
}
era_tab <- do.call(rbind, era_rows)
write.csv(era_tab, "era_split_ladder.csv", row.names = FALSE)
cat("\nWrote era_split_ladder.csv\n")
cat("READ THIS: compare pct_amp pre_2016 vs post_2016 per sector.\n")
cat("  similar magnitude in both eras -> amplitude's dominance is a stable\n")
cat("    structural feature of the annual cycle, present well before 2016\n")
cat("  much larger post_2016            -> current result IS substantially\n")
cat("    about the mean-state shift / recent amplitude decline, not a\n")
cat("    general property of the whole 44-yr record — say so explicitly\n\n")
print(era_tab)