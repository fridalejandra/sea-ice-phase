# =============================================================================
# ice_atmos_nonstationarity.R — has the atmosphere-ice coupling changed?
#
# Half-split (1979-2000 vs 2001-2023) Fisher-z tests + rolling correlations
# for INDEX x CYCLE-PARAMETER pairs, with an honest inference design:
#
#   PRIMARY family  — hypothesis-driven pairs (ENSO -> Pacific-sector cycle
#                     via the Amundsen Sea Low; SAM -> circumpolar cycle),
#                     small family, Bonferroni-corrected.
#   EXPLORATORY     — every other index x target x sector combination,
#                     Benjamini-Hochberg FDR q-values.
#
# Outputs:
#   iceatmos_halfsplit_primary.csv / _exploratory.csv
#   FIG_iceatmos_rolling.png   (rolling r for the primary pairs)
# =============================================================================

library(dplyr)

ROOT    <- path.expand("~/Research/repos/sea-ice-phase")
OUT_DIR <- file.path(ROOT, "scripts/R/Ch3/data")
SPLIT   <- 2001
WINDOW  <- 15

ann <- read.csv(file.path(OUT_DIR, "annual_params_E.csv")) %>%
  mutate(beta_warp = sqrt((beta1 - 1)^2 + (beta2 - 1)^2)) %>%
  left_join(read.csv(file.path(OUT_DIR, "climate_indices.csv")), by = "Year")

INDEXES <- grep("^(SAM|N34)_", names(ann), value = TRUE)
TARGETS <- c("max_doy_anom", "min_doy_anom", "amplitude_anom",
             "beta_asym", "beta_warp")

# PRIMARY family: (sector, index, target) — physical-pathway hypotheses.
PRIMARY <- rbind(
  expand.grid(sector = c("SIE_Ross", "SIE_Amundsen_Bellingshausen"),
              index  = c("N34_ann", "N34_SON"),
              target = c("amplitude_anom", "beta_warp", "max_doy_anom"),
              stringsAsFactors = FALSE),
  expand.grid(sector = "SIE_circumpolar",
              index  = c("SAM_ann", "SAM_MAM"),
              target = c("max_doy_anom", "amplitude_anom"),
              stringsAsFactors = FALSE))

fz <- function(r) 0.5 * log((1 + r) / (1 - r))

halfsplit <- function(a, ic, tv) {
  h1 <- a[a$Year <  SPLIT, ]; h2 <- a[a$Year >= SPLIT, ]
  r1 <- suppressWarnings(cor(h1[[ic]], h1[[tv]], use = "complete.obs"))
  r2 <- suppressWarnings(cor(h2[[ic]], h2[[tv]], use = "complete.obs"))
  n1 <- sum(complete.cases(h1[, c(ic, tv)]))
  n2 <- sum(complete.cases(h2[, c(ic, tv)]))
  z  <- (fz(r1) - fz(r2)) / sqrt(1/(n1 - 3) + 1/(n2 - 3))
  data.frame(r_first = round(r1, 2), r_second = round(r2, 2),
             n1 = n1, n2 = n2, z = round(z, 2),
             p = 2 * pnorm(-abs(z)))
}

# ── all combinations ────────────────────────────────────────────────────────
res <- list()
for (sec in unique(ann$sector)) {
  a <- ann[ann$sector == sec, ]
  for (ic in INDEXES) for (tv in TARGETS) {
    r <- halfsplit(a, ic, tv)
    r$sector <- sec; r$index <- ic; r$target <- tv
    res[[paste(sec, ic, tv)]] <- r
  }
}
res <- bind_rows(res)

is_primary <- with(res, paste(sector, index, target) %in%
                     with(PRIMARY, paste(sector, index, target)))
prim <- res[is_primary, ] %>%
  mutate(p_bonf = pmin(1, p * n()), sig = ifelse(p_bonf < 0.05, "**",
                                                 ifelse(p < 0.05, "*", ""))) %>%
  arrange(p)
expl <- res[!is_primary, ] %>%
  mutate(q_fdr = p.adjust(p, method = "BH")) %>% arrange(p)

write.csv(prim, file.path(OUT_DIR, "iceatmos_halfsplit_primary.csv"),
          row.names = FALSE)
write.csv(expl, file.path(OUT_DIR, "iceatmos_halfsplit_exploratory.csv"),
          row.names = FALSE)

cat(sprintf("\n===== PRIMARY family (n = %d tests, Bonferroni) =====\n",
            nrow(prim)))
print(prim %>% mutate(p = round(p, 4), p_bonf = round(p_bonf, 3)) %>%
        select(sector, index, target, r_first, r_second, z, p, p_bonf, sig),
      row.names = FALSE)
cat(sprintf("\n===== EXPLORATORY top 12 (of %d, BH-FDR) =====\n", nrow(expl)))
print(expl %>% head(12) %>%
        mutate(p = round(p, 4), q_fdr = round(q_fdr, 3)) %>%
        select(sector, index, target, r_first, r_second, z, p, q_fdr),
      row.names = FALSE)

# ── rolling correlations for the primary pairs, one panel per pair with
#    |z| >= 1.5 (uninteresting flat pairs skipped) ---------------------------
top <- prim[abs(prim$z) >= 1.5, ]
if (nrow(top) > 0) {
  nc <- min(3, nrow(top)); nr <- ceiling(nrow(top) / nc)
  png(file.path(OUT_DIR, "FIG_iceatmos_rolling.png"),
      width = 5.2 * nc, height = 4.2 * nr, units = "in", res = 300)
  par(mfrow = c(nr, nc), mar = c(4, 4, 3, 1))
  for (k in seq_len(nrow(top))) {
    sec <- top$sector[k]; ic <- top$index[k]; tv <- top$target[k]
    a <- ann[ann$sector == sec, ]; a <- a[order(a$Year), ]
    cy <- rr <- c()
    for (i in seq_len(nrow(a) - WINDOW + 1)) {
      w <- a[i:(i + WINDOW - 1), ]
      cy <- c(cy, mean(w$Year))
      rr <- c(rr, suppressWarnings(cor(w[[ic]], w[[tv]],
                                       use = "complete.obs")))
    }
    plot(cy, rr, type = "l", lwd = 3, col = "#C0392B", ylim = c(-1, 1),
         xlab = sprintf("Center of %d-yr window", WINDOW),
         ylab = "Rolling r",
         main = sprintf("%s\n%s ~ %s", sub("SIE_", "", sec), ic, tv))
    abline(h = 0, lty = 2); abline(v = SPLIT, col = "grey60", lty = 3)
    legend("topleft", bty = "n", cex = 0.85, legend = sprintf(
      "halves: %.2f -> %.2f (p=%.3g)", top$r_first[k], top$r_second[k],
      top$p[k]))
  }
  dev.off()
  cat("\nWrote FIG_iceatmos_rolling.png (", nrow(top), "panels )\n")
}