# =============================================================================
# rolling_correlations_E.R — Fig S08 + non-stationarity tests
#
# Question: are the relationships among cycle parameters (timing, amplitude,
# warp) — and between those parameters and the large-scale modes — stationary
# over 1979-2023?
#
# Two complementary products:
#   1. ROLLING correlations (WINDOW-yr, overlapping) — the figure. Windows
#      overlap, so these are descriptive; no per-window p-values.
#   2. HALF-SPLIT test — the citable statistic: correlation in the first half
#      vs second half of the record, difference tested by Fisher z
#      (independent samples, clean inference).
#
# Outputs:
#   rolling_correlations_E.csv
#   halfsplit_tests_E.csv        <- cover-letter numbers live here
#   FIGS08_rolling_E.png         2x3 grid, one panel per sector
# =============================================================================

library(dplyr)

ROOT    <- path.expand("~/Research/repos/sea-ice-phase")
OUT_DIR <- file.path(ROOT, "scripts/R/Ch3/data")
ANNUAL  <- file.path(OUT_DIR, "annual_params_E.csv")
INDEX_FILE <- file.path(OUT_DIR, "climate_indices.csv")

WINDOW <- 15          # rolling window (years)
SPLIT_YEAR <- 2001    # first half = Year < SPLIT_YEAR; second half = >=

ann <- read.csv(ANNUAL, stringsAsFactors = FALSE) %>%
  mutate(beta_warp = sqrt((beta1 - 1)^2 + (beta2 - 1)^2))
if (file.exists(INDEX_FILE)) {
  ann <- ann %>% left_join(read.csv(INDEX_FILE), by = "Year")
}

# Variable pairs to track (add more rows freely)
PAIRS <- list(
  c("max_doy_anom",  "amplitude_anom"),   # timing-amplitude coupling
  c("beta_asym",     "amplitude_anom"),   # warp-amplitude coupling
  c("max_doy_anom",  "beta_asym")         # the two timing metrics
)
if ("SAM_ann" %in% names(ann)) PAIRS <- c(PAIRS, list(
  c("SAM_ann", "max_doy_anom"),           # SAM-timing coupling
  c("N34_ann", "amplitude_anom"),         # ENSO-amplitude coupling
  c("N34_ann", "beta_warp")               # ENSO-shape coupling (Ross story)
))

SECTORS <- unique(ann$sector)
LAB <- function(s) sub("SIE_", "", s)

# ── rolling correlations ────────────────────────────────────────────────────
roll <- list()
for (sec in SECTORS) {
  a <- ann[ann$sector == sec, ]
  a <- a[order(a$Year), ]
  for (p in PAIRS) {
    if (!all(p %in% names(a))) next
    for (i in seq_len(nrow(a) - WINDOW + 1)) {
      w <- a[i:(i + WINDOW - 1), ]
      roll[[length(roll) + 1]] <- data.frame(
        sector = sec, pair = paste(p, collapse = " ~ "),
        center_year = mean(w$Year),
        r = suppressWarnings(cor(w[[p[1]]], w[[p[2]]],
                                 use = "complete.obs")))
    }
  }
}
roll <- bind_rows(roll)
write.csv(roll, file.path(OUT_DIR, "rolling_correlations_E.csv"),
          row.names = FALSE)

# ── half-split Fisher-z tests ───────────────────────────────────────────────
fisher_z <- function(r) 0.5 * log((1 + r) / (1 - r))
hs <- list()
for (sec in SECTORS) {
  a <- ann[ann$sector == sec, ]
  h1 <- a[a$Year <  SPLIT_YEAR, ]
  h2 <- a[a$Year >= SPLIT_YEAR, ]
  for (p in PAIRS) {
    if (!all(p %in% names(a))) next
    r1 <- suppressWarnings(cor(h1[[p[1]]], h1[[p[2]]], use = "complete.obs"))
    r2 <- suppressWarnings(cor(h2[[p[1]]], h2[[p[2]]], use = "complete.obs"))
    n1 <- sum(complete.cases(h1[, p])); n2 <- sum(complete.cases(h2[, p]))
    z  <- (fisher_z(r1) - fisher_z(r2)) / sqrt(1/(n1 - 3) + 1/(n2 - 3))
    hs[[length(hs) + 1]] <- data.frame(
      sector = sec, pair = paste(p, collapse = " ~ "),
      r_first = round(r1, 2), r_second = round(r2, 2),
      n1 = n1, n2 = n2, delta_r = round(r2 - r1, 2),
      z = round(z, 2), p_value = round(2 * pnorm(-abs(z)), 4))
  }
}
hs <- bind_rows(hs) %>% arrange(p_value)
write.csv(hs, file.path(OUT_DIR, "halfsplit_tests_E.csv"), row.names = FALSE)

cat("\n===== NON-STATIONARITY: half-split shifts, most significant first =====\n")
print(head(as.data.frame(hs), 15), row.names = FALSE)
cat(sprintf("\n(split at %d; first half n~%d, second n~%d; Fisher z, two-tailed)\n",
            SPLIT_YEAR, hs$n1[1], hs$n2[1]))

# ── figure: 2x3 grid of rolling correlations ────────────────────────────────
core_pairs <- unique(roll$pair)[1:min(3, length(unique(roll$pair)))]
cols <- c("#E8140C", "#14148C", "#1B7A1B", "#FF8C00", "#7B1FA2", "#00838F")

png(file.path(OUT_DIR, "FIGS08_rolling_E.png"), width = 15, height = 9,
    units = "in", res = 300)
par(mfrow = c(2, 3), mar = c(4, 4, 2.5, 1))
for (i in seq_along(SECTORS)) {
  sec <- SECTORS[i]
  rr <- roll[roll$sector == sec & roll$pair %in% core_pairs, ]
  plot(NULL, xlim = range(rr$center_year), ylim = c(-1, 1),
       xlab = sprintf("Center of %d-yr window", WINDOW),
       ylab = "Rolling correlation r",
       main = sprintf("(%s)  %s", letters[i], LAB(sec)))
  abline(h = 0, lty = 2)
  # single-window 5% threshold, purely as visual reference
  abline(h = c(-1, 1) * qt(0.975, WINDOW - 2) /
           sqrt(WINDOW - 2 + qt(0.975, WINDOW - 2)^2), col = "grey70",
         lty = 3)
  for (j in seq_along(core_pairs)) {
    pr <- rr[rr$pair == core_pairs[j], ]
    lines(pr$center_year, pr$r, col = cols[j], lwd = 2.5)
  }
  if (i == 1) legend("bottomleft", legend = core_pairs, col = cols,
                     lwd = 2.5, cex = 0.7, bty = "n")
}
dev.off()
cat("Wrote FIGS08_rolling_E.png + rolling_correlations_E.csv + halfsplit_tests_E.csv\n")