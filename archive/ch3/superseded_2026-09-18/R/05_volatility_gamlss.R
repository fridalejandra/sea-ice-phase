# =============================================================================
# 05_volatility_gamlss.R — seasonal volatility of daily SIE, pre/post-2016
#
# Follows M. Handcock's suggestion (email, Sep 2026): model the SCALE of the
# daily anomaly with a cyclic spline in day-of-year (an "annual cycle in
# volatility"), with an indicator for post-2016 and a sensor term so retrieval
# effects are held constant within an instrument.
#
# Data: daily_fitted_E.csv from 1988-01-01 (SSM/I onward; the SMMR period has
# every-other-day sampling and its GARCH volatility is 5-10x higher for that
# reason alone — see fig11). Sensor factor: SSM/I 1988-2007, SSMIS 2008-.
#
# Two response variables. PRIMARY: the daily tendency dSIE = Extent(t) - Extent(t-1)
# (consecutive days only), which is what the 2020 paper's GARCH innovations
# measure — day-to-day change — and which removes level, trend and nearly all
# amplitude/phase variation by construction. CHECK: the APAC residual.
# (The unadjusted anomaly was tried first: its scale term absorbs year-to-year
# level differences even with a period intercept in the location, so it is
# not a day-to-day measure. Kept out.)
#
# Model A (per sector): y ~ pbc(DOY) + post2016 + sensor        [location]
#                       sigma ~ pbc(DOY) + sensor + post2016    [log-scale]
#   post2016 = one multiplicative change in volatility after 2016, all seasons.
# Shape comparison:     the same model fitted SEPARATELY to 1988-2015 and
#   2016-2023 (Handcock: let the seasonal volatility differ by period); the
#   ratio of the two fitted sigma curves by day of year, summarised by season.
# Uncertainty: year-block bootstrap (resample calendar years with replacement,
# refit), NBOOT reps, because daily residuals are autocorrelated and the
# model's own SEs assume independence.
#
# Outputs (results/ch3/tables/):
#   t34c_volatility_gamlss_post2016.csv   sector, response, exp(coef), boot CI, by-season ratios
#   t34c_volatility_seasonal_curves.csv   sector, response, DOY, sigma_pre, sigma_post
# =============================================================================
suppressPackageStartupMessages({ library(dplyr); library(gamlss) })

ROOT     <- path.expand("~/Research/repos/sea-ice-phase")
DAILY    <- file.path(ROOT, "data/ch3/daily_fitted.csv")
OUT_DIR  <- file.path(ROOT, "results/ch3/tables")
NBOOT    <- as.integer(Sys.getenv("NBOOT", "100"))   # NBOOT=0 skips the bootstrap
START    <- as.Date("1988-01-01")
BREAK    <- 2016
SENSOR_SWITCH <- 2008

daily <- read.csv(DAILY, stringsAsFactors = FALSE)
daily <- daily[daily$period == "FULL", ]
daily$Date <- as.Date(daily$Date)
d <- daily %>% filter(Date >= START, Year <= 2023) %>% arrange(sector, Date) %>%
  group_by(sector) %>%
  mutate(dSIE = ifelse(as.numeric(Date - lag(Date)) == 1, Extent - lag(Extent), NA_real_)) %>%
  ungroup() %>%
  mutate(post2016 = factor(Year >= BREAK, levels = c(FALSE, TRUE)),
         sensor   = factor(ifelse(Year >= SENSOR_SWITCH, "SSMIS", "SSMI"), levels = c("SSMI", "SSMIS")))
message("rows: ", nrow(d), "  ", min(d$Date), " to ", max(d$Date))

frame_of <- function(dd, resp)   # plain data.frame with only the modelled columns, NA rows dropped
  na.omit(data.frame(y = as.numeric(dd[[resp]]), DOY = as.numeric(dd$DOY), post2016 = dd$post2016,
                     sensor = dd$sensor, Year = dd$Year))
fit_one <- function(dd, resp) {
  dd <- frame_of(dd, resp)
  # location carries the period level and sensor too, otherwise the post-2016
  # mean offset of the unadjusted anomaly is booked as scale
  m <- gamlss(y ~ pbc(DOY) + post2016 + sensor, sigma.formula = ~ pbc(DOY) + sensor + post2016,
              family = NO, data = dd, trace = FALSE)
  co <- coef(m, what = "sigma")
  list(model = m, post = unname(co["post2016TRUE"]), sens = unname(co["sensorSSMIS"]), data = dd)
}
fit_period <- function(dd, resp) {  # one period only: no post2016 term; sensor only if both present
  dd <- frame_of(dd, resp)
  if (length(unique(dd$sensor)) > 1)
    m <- gamlss(y ~ pbc(DOY) + sensor, sigma.formula = ~ pbc(DOY) + sensor, family = NO, data = dd, trace = FALSE)
  else
    m <- gamlss(y ~ pbc(DOY), sigma.formula = ~ pbc(DOY), family = NO, data = dd, trace = FALSE)
  list(model = m, data = dd)
}
SEASON_OF <- function(doy) { m <- as.integer(format(as.Date(doy - 1, origin = "2001-01-01"), "%m"))
  c("DJF","DJF","MAM","MAM","MAM","JJA","JJA","JJA","SON","SON","SON","DJF")[m] }

rows <- list(); curves <- list()
for (sec in unique(d$sector)) {
  for (resp in c("dSIE", "residual_apac")) {
    dd <- d[d$sector == sec, ]
    f  <- fit_one(dd, resp)
    message(sprintf("  %-28s %-17s post2016 x%.3f  sensor x%.3f", sec, resp, exp(f$post), exp(f$sens)))

    # seasonal shape by period: separate fits, sigma curves at the SSMIS level
    fp <- fit_period(dd[dd$Year <  BREAK, ], resp)
    fq <- fit_period(dd[dd$Year >= BREAK, ], resp)
    nd_p <- data.frame(DOY = 1:365, sensor = factor("SSMIS", levels = c("SSMI", "SSMIS")))
    sig_pre  <- predict(fp$model, what = "sigma", newdata = nd_p, type = "response", data = fp$data)
    sig_post <- predict(fq$model, what = "sigma", newdata = data.frame(DOY = 1:365), type = "response", data = fq$data)
    cv <- data.frame(sector = sec, response = resp, DOY = 1:365, sigma_pre = sig_pre, sigma_post = sig_post)
    cv$season <- SEASON_OF(cv$DOY)
    curves[[paste(sec, resp)]] <- cv
    by_season <- cv %>% group_by(season) %>% summarise(ratio = mean(sigma_post) / mean(sigma_pre), .groups = "drop")
    aicA <- AIC(f$model); aicB <- NA_real_
    message(sprintf("     separate-period fits: post/pre sigma ratio  DJF %.2f  MAM %.2f  JJA %.2f  SON %.2f   (full-year %.2f)",
                    by_season$ratio[by_season$season=="DJF"], by_season$ratio[by_season$season=="MAM"],
                    by_season$ratio[by_season$season=="JJA"], by_season$ratio[by_season$season=="SON"],
                    mean(sig_post) / mean(sig_pre)))

    # year-block bootstrap of the post2016 coefficient
    boot <- NA_real_
    if (NBOOT > 0) {
      yrs <- unique(dd$Year); set.seed(1)
      boot <- replicate(NBOOT, {
        ys <- sample(yrs, length(yrs), replace = TRUE)
        db <- do.call(rbind, lapply(seq_along(ys), function(i) { x <- dd[dd$Year == ys[i], ]; x$Year <- 10000 + i; x }))
        db$post2016 <- factor(ys[match(db$Year - 10000, seq_along(ys))] >= BREAK, levels = c(FALSE, TRUE))
        # keep sensor as in the resampled years
        db$sensor <- factor(ifelse(ys[match(db$Year - 10000, seq_along(ys))] >= SENSOR_SWITCH, "SSMIS", "SSMI"), levels = c("SSMI", "SSMIS"))
        if (length(unique(db$post2016)) < 2 || length(unique(db$sensor)) < 2) return(NA_real_)
        tryCatch(fit_one(db, resp)$post, error = function(e) NA_real_)
      })
      boot <- boot[is.finite(boot)]
    }
    rows[[paste(sec, resp)]] <- data.frame(
      sector = sec, response = resp, n_days = nrow(dd),
      vol_ratio_post2016 = exp(f$post),
      boot_lo = if (length(boot) > 10) exp(quantile(boot, 0.025)) else NA,
      boot_hi = if (length(boot) > 10) exp(quantile(boot, 0.975)) else NA,
      n_boot = length(boot),
      vol_ratio_ssmis_vs_ssmi = exp(f$sens),
      ratio_fullyear_separate_fits = mean(cv$sigma_post) / mean(cv$sigma_pre),
      ratio_DJF = by_season$ratio[by_season$season=="DJF"], ratio_MAM = by_season$ratio[by_season$season=="MAM"],
      ratio_JJA = by_season$ratio[by_season$season=="JJA"], ratio_SON = by_season$ratio[by_season$season=="SON"])
  }
}
res <- do.call(rbind, rows); rownames(res) <- NULL
cur <- do.call(rbind, curves); rownames(cur) <- NULL
write.csv(res, file.path(OUT_DIR, "t34c_volatility_gamlss_post2016.csv"), row.names = FALSE)
write.csv(cur, file.path(OUT_DIR, "t34c_volatility_seasonal_curves.csv"), row.names = FALSE)
cat("\n== post-2016 multiplicative change in day-to-day volatility (season and sensor held fixed) ==\n")
print(res %>% mutate(across(where(is.numeric), ~ round(.x, 3))), row.names = FALSE)
message("wrote t34c_volatility_gamlss_post2016.csv, t34c_volatility_seasonal_curves.csv")
