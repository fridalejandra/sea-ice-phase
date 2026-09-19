# =============================================================================
# 04_chapter_analyses.R — §3.3.5 residual variability + §3.4 timing/amplitude/
# beta correlation tables.
#
# FIXED 2026-09-18: previously read daily_fitted_E.csv / annual_params_E.csv,
# an old "Pipeline-E" data variant that no longer exists on disk (confirmed
# via `ls data/ch3/*.csv` -- only the canonical, unsuffixed files are there
# now, so the old version would fail outright with file-not-found). This now
# reads the canonical daily_fitted.csv / annual_params.csv -- the same files
# ch3_config.py's DAILY_CSV/ANNUAL_CSV point to -- filtered to period ==
# "FULL". Both files carry a FULL (1979-2025) and an HR2018 (1979-2018)
# window that overlap for 2016-2018; every other script in this pipeline
# filters to FULL for exactly that reason (same bug, same fix, as
# fig_07_abs_growth_season.py's period-filter fix earlier this chapter).
# Confirmed via the provenance audit (results/ch3/tables/_provenance_audit.csv)
# that this script is still the sole writer of both output tables below --
# nothing else in the pipeline has taken over either one.
#
# NOT independently verified in this pass: whether daily_fitted.csv still has
# residual_apac / volatility columns, and whether annual_params.csv still has
# beta1 / beta2 / sie_DJF / sie_MAM / sie_JJA / sie_SON under those exact
# names -- the annual columns were confirmed against a real uploaded copy
# earlier this chapter, the daily ones were not re-checked this pass. The
# stopifnot()-style checks below fail loudly and specifically if any are
# missing, rather than erroring deep inside a dplyr pipeline.
#
# Inputs : daily_fitted.csv, annual_params.csv (both filtered to period == "FULL")
# Optional: a climate-index CSV (see INDEX_FILE below) with columns
#           Year, <index columns...> (annual or seasonal values). If absent,
#           the index correlations are skipped and everything else still runs.
#           This block was already disabled (if (FALSE && ...)) with a
#           comment reading "index correlations now in ch3_stats.py" -- left
#           disabled as-is, not re-enabled, since that migration looks real
#           (nothing in the provenance audit contradicts it).
#
# Outputs (to OUT_DIR):
#   s335_residual_variability.csv   decadal residual/volatility stats
#   s335_summary.txt                copy-paste numbers for the text
#   s34_internal_correlations.csv   per sector: seasonal SIE anomalies vs
#                                   timing / amplitude / beta-asymmetry
#   s34_index_correlations.csv      (only if INDEX_FILE exists -- currently
#                                   dead code, see note above)
# =============================================================================

library(dplyr)
library(tidyr)

ROOT    <- path.expand("~/Research/repos/sea-ice-phase")
DATA_DIR <- file.path(ROOT, "data/ch3")
OUT_DIR  <- file.path(ROOT, "results/ch3/tables")
DAILY   <- file.path(DATA_DIR, "daily_fitted.csv")
ANNUAL  <- file.path(DATA_DIR, "annual_params.csv")
INDEX_FILE <- file.path(DATA_DIR, "climate_indices.csv")  # optional

daily  <- read.csv(DAILY,  stringsAsFactors = FALSE)
annual <- read.csv(ANNUAL, stringsAsFactors = FALSE)
daily$Date <- as.Date(daily$Date)

# period-contamination guard -- see header note above.
if ("period" %in% names(daily))  daily  <- daily  %>% filter(period == "FULL")
if ("period" %in% names(annual)) annual <- annual %>% filter(period == "FULL")

annual <- annual %>% filter(Year >= 1979)  # first (partial) year excluded per methods 2.1.1
daily  <- daily  %>% filter(Year >= 1979)

# Fail loudly and specifically if the canonical files don't have the columns
# this script assumes -- better than a cryptic dplyr error three steps in.
need_daily  <- c("sector", "Year", "residual_apac", "volatility")
need_annual <- c("sector", "Year", "sie_annual", "sie_DJF", "sie_MAM", "sie_JJA",
                 "sie_SON", "max_doy_raw_anom", "min_doy_raw_anom",
                 "amplitude_raw_anom", "beta1", "beta2")
miss_daily  <- setdiff(need_daily,  names(daily))
miss_annual <- setdiff(need_annual, names(annual))
if (length(miss_daily) > 0)
  stop("daily_fitted.csv is missing expected column(s): ", paste(miss_daily, collapse = ", "))
if (length(miss_annual) > 0)
  stop("annual_params.csv is missing expected column(s): ", paste(miss_annual, collapse = ", "))

# ── §3.3.5 RESIDUAL VARIABILITY ─────────────────────────────────────────────
d <- daily %>%
  mutate(decade = paste0(pmax(1980, pmin(floor(Year / 10) * 10, 2020)), "s"))

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

targets <- c("max_doy_raw_anom", "min_doy_raw_anom", "amplitude_raw_anom")  # raw observed scalars only; fitted = attribution, beta = held out
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
print(s34 %>% filter(target %in% c("max_doy_raw_anom", "amplitude_raw_anom")) %>%
        pivot_wider(names_from = target, values_from = r) %>%
        mutate(across(where(is.numeric), ~round(.x, 2))) %>%
        as.data.frame())

# ── §3.4 EXTERNAL INDEX CORRELATIONS (optional) ─────────────────────────────
if (FALSE && file.exists(INDEX_FILE)) {  # index correlations now in ch3_stats.py
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