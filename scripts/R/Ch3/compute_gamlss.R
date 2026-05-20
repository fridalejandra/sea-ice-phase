# compute_gamlss.R
# =================
# GAMLSS models for key atmosphere-ice pairs.
#
# For each pair, fits a GAMLSS model with:
#   mu    ~ atmospheric_index + s(Year)   — mean as function of index + time trend
#   sigma ~ s(Year)                        — variance as function of time
#
# This tests whether:
#   1. The atmospheric index predicts the mean of the ice variable
#   2. The variance of the ice variable has changed over time (post-2016)
#
# Three key pairs (run for both raw and APAC anomalies):
#   1. EA phase       ~ SAM RET      (r=+0.49, strongest result)
#   2. King Haakon amplitude ~ Nino34 annual  (r=-0.42, most robust)
#   3. Weddell amplitude ~ Nino34 RET  (r=+0.44, shoulder season signal)
#
# Output:
#   gamlss_results.csv     — model summaries, AIC, variance shift estimates
#   fig_gamlss_*.png       — one figure per pair showing mean + variance fits
#
# Requires: gamlss, dplyr, ggplot2, mgcv

library(gamlss)
library(dplyr)
library(ggplot2)
library(mgcv)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   <- "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
FIG_DIR    <- "/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch3/figures"
GDRIVE     <- "gdrive:My Drive/sea-ice-phase/results/Ch3_Figures"

ANNUAL_CSV <- file.path(DATA_DIR, "annual_params.csv")
INDEX_CSV  <- file.path(DATA_DIR, "master_index_detrended.csv")

YEAR_MIN   <- 1979
YEAR_MAX   <- 2023

# ── Load data ─────────────────────────────────────────────────────────────────
cat("Loading data...\n")
annual <- read.csv(ANNUAL_CSV)
annual <- annual[annual$Year >= YEAR_MIN & annual$Year <= YEAR_MAX, ]

idx    <- read.csv(INDEX_CSV)

# ── Helper: detrend a series ──────────────────────────────────────────────────
detrend_series <- function(years, values) {
  mask <- !is.na(values)
  if (sum(mask) < 5) return(values)
  fit  <- lm(values[mask] ~ years[mask])
  values - predict(fit, newdata = data.frame(`years[mask]` = years),
                   type = "response")
}

# ── Helper: prepare analysis dataset ─────────────────────────────────────────
prepare_data <- function(sector_col, apac_var, raw_var, idx_col) {
  # Pull ice variables for this sector
  sec <- annual[annual$sector == sector_col,
                c("Year", apac_var, raw_var)] %>%
    arrange(Year)

  # Detrend
  sec[[apac_var]] <- detrend_series(sec$Year, sec[[apac_var]])
  sec[[raw_var]]  <- detrend_series(sec$Year, sec[[raw_var]])

  # Merge with index
  df <- sec %>%
    inner_join(idx[, c("Year", idx_col)], by = "Year") %>%
    filter(!is.na(.data[[apac_var]]),
           !is.na(.data[[raw_var]]),
           !is.na(.data[[idx_col]])) %>%
    mutate(year_scaled = as.numeric(scale(Year)))

  cat(sprintf("  %s ~ %s: %d rows (%d–%d)\n",
              sector_col, idx_col, nrow(df),
              min(df$Year), max(df$Year)))
  df
}

# ── Helper: fit GAMLSS and return summary ─────────────────────────────────────
fit_gamlss <- function(df, response_var, idx_col, pair_label, var_label) {

  y   <- df[[response_var]]
  x   <- df[[idx_col]]
  yr  <- df$year_scaled
  yrs <- df$Year

  # --- Model 0: null (intercept only) ---
  m0 <- gamlss(y ~ 1,
               sigma.formula = ~1,
               family = NO(),
               trace = FALSE)

  # --- Model 1: index only in mu, constant sigma ---
  m1 <- gamlss(y ~ x,
               sigma.formula = ~1,
               family = NO(),
               trace = FALSE)

  # --- Model 2: index + time trend in mu, constant sigma ---
  m2 <- tryCatch(
    gamlss(y ~ x + pb(yrs),
           sigma.formula = ~1,
           family = NO(),
           trace = FALSE),
    error = function(e) {
      cat("    Model 2 failed, falling back to linear time\n")
      gamlss(y ~ x + yrs, sigma.formula = ~1,
             family = NO(), trace = FALSE)
    }
  )

  # --- Model 3: index in mu, time-varying sigma ---
  m3 <- tryCatch(
    gamlss(y ~ x + pb(yrs),
           sigma.formula = ~pb(yrs),
           family = NO(),
           trace = FALSE),
    error = function(e) {
      cat("    Model 3 failed, falling back to linear sigma\n")
      gamlss(y ~ x + yrs,
             sigma.formula = ~yrs,
             family = NO(), trace = FALSE)
    }
  )

  # --- Compare models by AIC ---
  aic_vals <- c(AIC(m0), AIC(m1), AIC(m2), AIC(m3))
  best_mod <- c("null","index_only","index+trend","index+trend+sigma_trend")[
    which.min(aic_vals)]

  cat(sprintf("    AIC: null=%.1f  index=%.1f  +trend=%.1f  +sigma=%.1f  → best: %s\n",
              aic_vals[1], aic_vals[2], aic_vals[3], aic_vals[4], best_mod))

  # --- Extract fitted values from best model ---
  best <- list(m0, m1, m2, m3)[[which.min(aic_vals)]]

  # Fitted mean and sigma over time
  mu_fit    <- fitted(m3, "mu")
  sigma_fit <- exp(fitted(m3, "sigma"))  # gamlss models log(sigma)

  # Pre vs post 2016 variance
  pre_sd  <- sd(y[yrs <  2016], na.rm = TRUE)
  post_sd <- sd(y[yrs >= 2016], na.rm = TRUE)
  var_ratio <- post_sd / pre_sd

  # Coefficient on index in m1
  coef_idx <- coef(m1)["x"]

  list(
    pair        = pair_label,
    variable    = var_label,
    response    = response_var,
    index       = idx_col,
    n           = nrow(df),
    aic_null    = round(aic_vals[1], 2),
    aic_index   = round(aic_vals[2], 2),
    aic_trend   = round(aic_vals[3], 2),
    aic_sigma   = round(aic_vals[4], 2),
    best_model  = best_mod,
    coef_index  = round(coef_idx, 4),
    pre_sd      = round(pre_sd,   4),
    post_sd     = round(post_sd,  4),
    var_ratio   = round(var_ratio, 3),
    df_plot     = data.frame(
      Year      = yrs,
      y         = y,
      index     = x,
      mu_fit    = mu_fit,
      sigma_fit = sigma_fit,
      period    = ifelse(yrs >= 2016, "2016+", "1979–2015")
    )
  )
}

# ── Helper: plot one pair ─────────────────────────────────────────────────────
plot_gamlss_pair <- function(res_raw, res_apac, pair_label, outfile) {

  # Combine raw and APAC plot data
  df_raw   <- res_raw$df_plot  %>% mutate(type = "Raw anomaly")
  df_apac  <- res_apac$df_plot %>% mutate(type = "APAC anomaly")
  df_all   <- bind_rows(df_raw, df_apac)

  # Colour by period
  period_cols <- c("1979–2015" = "#378ADD", "2016+" = "#D4537E")

  p <- ggplot(df_all, aes(x = Year)) +

    # ±1σ fitted band (time-varying sigma from m3)
    geom_ribbon(aes(ymin = mu_fit - sigma_fit,
                    ymax = mu_fit + sigma_fit),
                fill = "#BBBBBB", alpha = 0.35) +

    # Fitted mean line
    geom_line(aes(y = mu_fit), color = "#2C2C2A",
              linewidth = 1.2, linetype = "solid") +

    # Observed points coloured by period
    geom_point(aes(y = y, color = period), size = 2.0, alpha = 0.85) +

    # 2016 vertical reference line
    geom_vline(xintercept = 2016, color = "#D4537E",
               linewidth = 0.8, linetype = "dotted") +

    scale_color_manual(values = period_cols, name = NULL) +

    facet_wrap(~type, ncol = 2, scales = "free_y") +

    labs(
      title    = pair_label,
      subtitle = sprintf(
        "Raw: pre-2016 σ=%.3f, post-2016 σ=%.3f (ratio=%.2f)  |  APAC: pre σ=%.3f, post σ=%.3f (ratio=%.2f)",
        res_raw$pre_sd,  res_raw$post_sd,  res_raw$var_ratio,
        res_apac$pre_sd, res_apac$post_sd, res_apac$var_ratio),
      x = "Year",
      y = "Anomaly"
    ) +

    theme_classic(base_size = 11) +
    theme(
      strip.text       = element_text(face = "bold", size = 11),
      plot.title       = element_text(face = "bold", size = 12),
      plot.subtitle    = element_text(size = 8, color = "#5F5E5A"),
      legend.position  = "bottom",
      panel.grid.major = element_line(color = "#F0F0F0"),
    )

  ggsave(outfile, p, width = 11, height = 5, dpi = 300)
  cat(sprintf("  Saved → %s\n", outfile))

  # Sync to Google Drive
  system(sprintf('rclone copy "%s" "%s"', outfile, GDRIVE))
}

# ── Define key pairs ──────────────────────────────────────────────────────────
PAIRS <- list(
  list(
    label      = "East Antarctica phase ~ SAM retreat season",
    sector     = "SIE_East_Antarctica",
    apac_var   = "max_doy_anom",
    raw_var    = "max_doy_raw_anom",
    idx_col    = "SAM_RET",
    outfile    = file.path(FIG_DIR, "fig_gamlss_ea_phase_sam_ret.png")
  ),
  list(
    label      = "King Haakon amplitude ~ Niño3.4 annual",
    sector     = "SIE_King_Haakon",
    apac_var   = "amplitude_anom",
    raw_var    = "amplitude_raw_anom",
    idx_col    = "Nino34_annual",
    outfile    = file.path(FIG_DIR, "fig_gamlss_kh_amp_nino34_annual.png")
  ),
  list(
    label      = "Weddell amplitude ~ Niño3.4 retreat season",
    sector     = "SIE_Weddell",
    apac_var   = "amplitude_anom",
    raw_var    = "amplitude_raw_anom",
    idx_col    = "Nino34_RET",
    outfile    = file.path(FIG_DIR, "fig_gamlss_weddell_amp_nino34_ret.png")
  )
)

# ── Run ───────────────────────────────────────────────────────────────────────
all_results <- list()

for (pair in PAIRS) {
  cat(sprintf("\n=== %s ===\n", pair$label))

  df <- prepare_data(pair$sector, pair$apac_var,
                     pair$raw_var, pair$idx_col)

  cat("  Fitting GAMLSS — raw anomaly...\n")
  res_raw  <- fit_gamlss(df, pair$raw_var,  pair$idx_col,
                         pair$label, "raw")

  cat("  Fitting GAMLSS — APAC anomaly...\n")
  res_apac <- fit_gamlss(df, pair$apac_var, pair$idx_col,
                         pair$label, "apac")

  plot_gamlss_pair(res_raw, res_apac, pair$label, pair$outfile)

  all_results[[pair$label]] <- list(raw = res_raw, apac = res_apac)
}

# ── Save summary table ────────────────────────────────────────────────────────
cat("\nSaving summary table...\n")
summary_rows <- lapply(all_results, function(x) {
  bind_rows(
    data.frame(pair=x$raw$pair,  type="raw",  x$raw[ setdiff(names(x$raw),  c("df_plot","pair","variable","response","index"))]),
    data.frame(pair=x$apac$pair, type="apac", x$apac[setdiff(names(x$apac), c("df_plot","pair","variable","response","index"))])
  )
})
summary_df <- bind_rows(summary_rows)
out_csv    <- file.path(DATA_DIR, "gamlss_results.csv")
write.csv(summary_df, out_csv, row.names = FALSE)
cat(sprintf("Saved → %s\n", out_csv))

system(sprintf('rclone copy "%s" "%s"', out_csv, GDRIVE))

cat("\nDone.\n")
