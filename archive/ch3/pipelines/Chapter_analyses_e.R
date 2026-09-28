# =============================================================================
# chapter_analyses_E.R — §3.3.5 residual variability + §3.4 timing/amplitude/
# beta correlation tables, from the Pipeline-E outputs.
#
# Inputs : daily_fitted_E.csv, annual_params_E.csv
# Optional: a climate-index CSV (see INDEX_FILE below) with columns
#           Year, <index columns...> (annual or seasonal values). If absent,
#           the index correlations are skipped and everything else still runs.
#
# Outputs (to OUT_DIR):
#   s335_residual_variability.csv   decadal residual/volatility stats
#   s335_summary.txt                copy-paste numbers for the text
#   s34_internal_correlations.csv   per sector: seasonal SIE anomalies vs
#                                   timing / amplitude / beta-asymmetry
#   s34_index_correlations.csv      (only if INDEX_FILE exists)
# =============================================================================

library(dplyr)
library(tidyr)

ROOT    <- path.expand("~/Research/repos/sea-ice-phase")
OUT_DIR <- file.path(ROOT, "scripts/R/Ch3/data")
DAILY   <- file.path(OUT_DIR, "daily_fitted_E.csv")
ANNUAL  <- file.path(OUT_DIR, "annual_params_E.csv")
INDEX_FILE <- file.path(ROOT, "scripts/R/Ch3/data", "climate_indices.csv")  # optional

daily  <- read.csv(DAILY,  stringsAsFactors = FALSE)
annual <- read.csv(ANNUAL, stringsAsFactors = FALSE)
daily$Date <- as.Date(daily$Date)

# ── §3.3.5 RESIDUAL VARIABILITY ─────────────────────────────────────────────
d <- daily %>%
  mutate(decade = paste0(pmin(floor(Year / 10) * 10, 2020), "s"))

s335 <- d %>% group_by(sector, decade) %>%
  summarise(n = sum(!is.na(residual_apac)),
            sd_residual   = sd(residual_apac, na.rm = TRUE),
            mean_abs_res  = mean(abs(residual_apac), na.rm = TRUE),
            mean_volatility = mean(volatility, na.rm = TRUE),
            .groups = "drop")

overall <- d %>% group_by(sector) %>%
  summarise(sd_residual = sd(residual_apac, na.rm = TRUE),
            mean_volatility = mean(volatility, na.rm = TRUE),
            # trend in daily |residual| vs year (is unexplained variability growing?)
            rho_year = suppressWarnings(
              cor(Year, abs(residual_apac), method = "spearman",
                  use = "complete.obs")),
            .groups = "drop")

write.csv(s335, file.path(OUT_DIR, "s335_residual_variability.csv"),
          row.names = FALSE)

sink(file.path(OUT_DIR, "s335_summary.txt"))
cat("== §3.3.5 numbers ==\n\nOverall (full record):\n")
print(as.data.frame(overall), digits = 3)
cat("\nBy decade (sd of APAC residual, Mkm^2):\n")
print(s335 %>% select(sector, decade, sd_residual) %>%
        pivot_wider(names_from = decade, values_from = sd_residual) %>%
        as.data.frame(), digits = 3)
cat("\nLast decade vs first decade ratio (residual sd):\n")
rat <- s335 %>% group_by(sector) %>%
  summarise(ratio = sd_residual[decade == max(decade)] /
              sd_residual[decade == min(decade)])
print(as.data.frame(rat), digits = 3)
sink()
message("Wrote s335_residual_variability.csv + s335_summary.txt")

# ── §3.4 INTERNAL CORRELATIONS ──────────────────────────────────────────────
# Question: do seasonal SIE anomalies co-vary with TIMING (max_doy_anom),
# AMPLITUDE (amplitude_anom), or CYCLE-SHAPE DISTORTION (beta)?
ann <- annual %>% group_by(sector) %>%
  mutate(across(c(sie_annual, sie_DJF, sie_MAM, sie_JJA, sie_SON),
                ~ .x - mean(.x, na.rm = TRUE), .names = "{.col}_anom"),
         beta_warp = sqrt((beta1 - 1)^2 + (beta2 - 1)^2)) %>%
  ungroup()

targets <- c("max_doy_anom", "min_doy_anom", "amplitude_anom",
             "beta_asym", "beta_warp")
seasons <- c("sie_annual_anom", "sie_DJF_anom", "sie_MAM_anom",
             "sie_JJA_anom", "sie_SON_anom")

rows <- list()
for (sec in unique(ann$sector)) {
  a <- ann[ann$sector == sec, ]
  for (sn in seasons) for (tv in targets) {
    rows[[paste(sec, sn, tv)]] <- data.frame(
      sector = sec,
      season = sub("sie_", "", sub("_anom", "", sn)),
      target = tv,
      r = suppressWarnings(cor(a[[sn]], a[[tv]], use = "complete.obs")))
  }
}
s34 <- bind_rows(rows)

write.csv(s34, file.path(OUT_DIR, "s34_internal_correlations.csv"),
          row.names = FALSE)
message("Wrote s34_internal_correlations.csv")

cat("\n== §3.4 headline: |r| of seasonal SIE anomaly with timing vs amplitude ==\n")
print(s34 %>% filter(target %in% c("max_doy_anom", "amplitude_anom")) %>%
        pivot_wider(names_from = target, values_from = r) %>%
        mutate(across(where(is.numeric), ~round(.x, 2))) %>%
        as.data.frame())

# ── §3.4 EXTERNAL INDEX CORRELATIONS (optional) ─────────────────────────────
if (file.exists(INDEX_FILE)) {
  idx <- read.csv(INDEX_FILE, stringsAsFactors = FALSE)
  stopifnot("Year" %in% names(idx))
  index_cols <- setdiff(names(idx), "Year")
  ai <- ann %>% left_join(idx, by = "Year")
  out <- list()
  for (sec in unique(ai$sector)) {
    a <- ai[ai$sector == sec, ]
    for (ic in index_cols) for (tv in c(targets, seasons)) {
      out[[paste(sec, ic, tv)]] <- data.frame(
        sector = sec, index = ic, target = tv,
        r = suppressWarnings(cor(a[[ic]], a[[tv]], use = "complete.obs")))
    }
  }
  s34x <- bind_rows(out)
  write.csv(s34x, file.path(OUT_DIR, "s34_index_correlations.csv"),
            row.names = FALSE)
  message("Wrote s34_index_correlations.csv (indices: ",
          paste(index_cols, collapse = ", "), ")")
} else {
  message("No climate_indices.csv found — index correlations skipped. ",
          "Drop a CSV with Year + index columns at:\n  ", INDEX_FILE)
}