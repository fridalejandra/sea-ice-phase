# =============================================================================
# plot_fig7_E.R — Figure 7 panels from daily_fitted_E.csv
#
# Rebuilds the decomposition curves from the u-space columns with the
# retreat-limb stat blend (this is REQUIRED: the raw *_pct columns are
# normalized by each row's calendar-year amplitude, which steps on Jan 1;
# the blend removes that cliff, exactly as in the validated reproduction).
#
# Outputs:
#   FIG7a_circumpolar_E.png   single panel, H&R style/limits
#   FIG7_sectors_E.png        2x3 grid, all regions, auto y-limits
# =============================================================================

ROOT <- path.expand("~/Research/repos/sea-ice-phase")
DAILY <- file.path(ROOT, "scripts/R/Ch3/data/daily_fitted_E.csv")
OUT_DIR <- file.path(ROOT, "scripts/R/Ch3/data")
YEAR <- 2016
TREND_COL <- "trend_component"      # or "trend_component_k6"

SECTORS <- c("SIE_circumpolar", "SIE_Weddell", "SIE_King_Haakon",
             "SIE_East_Antarctica", "SIE_Ross", "SIE_Amundsen_Bellingshausen")
LABELS <- c(SIE_circumpolar = "Circumpolar", SIE_Weddell = "Weddell",
            SIE_King_Haakon = "King Haakon",
            SIE_East_Antarctica = "East Antarctica", SIE_Ross = "Ross",
            SIE_Amundsen_Bellingshausen = "Amundsen–Bellingshausen")

d_all <- read.csv(DAILY, stringsAsFactors = FALSE)
d_all$Date <- as.Date(d_all$Date)

components_for <- function(sector, year) {
  t0 <- as.Date(sprintf("%d-02-21", year))
  t1 <- as.Date(sprintf("%d-02-20", year + 1))
  g <- d_all[d_all$sector == sector & d_all$Date >= t0 & d_all$Date <= t1, ]
  g <- g[order(g$Date), ]
  if (nrow(g) < 300) stop("Not enough rows for ", sector, " ", year)
  cd <- as.numeric(g$Date - t0)
  
  # cycle-year stats (first rows carry year Y's stats) and next-year stats
  ampY <- g$amplitude[1];            minY <- g$min_extent[1]
  ampN <- tail(g$amplitude, 1);      minN <- tail(g$min_extent, 1)
  
  # retreat-limb blend from the day of the cycle maximum
  dmax <- cd[which.max(g$Extent)]
  w <- pmin(pmax((cd - dmax) / (365 - dmax), 0), 1)
  ampb <- ampY * (1 - w) + ampN * w
  minb <- minY * (1 - w) + minN * w
  famp  <- g$u_amp  * ampb + minb
  fapac <- g$u_apac * ampb + minb
  
  tv <- g[[TREND_COL]]
  raw <- (g$Extent - fapac) / ampY * 100
  est <- as.numeric(stats::filter(raw, rep(1/11, 11), sides = 2))
  data.frame(
    cycle_day = cd,
    invariant = g$iac_notrend - mean(g$iac_notrend),   # raw Mkm^2, centered
    trend     = tv / ampY * 100,                        # % of amplitude
    amplitude = (famp - g$iac_notrend - tv) / ampY * 100,
    phase     = (fapac - famp) / ampY * 100,
    raw       = raw,
    est       = est)
}

draw_panel <- function(cc, title, ylim = NULL) {
  if (is.null(ylim))
    ylim <- range(c(cc$invariant, cc$trend, cc$amplitude, cc$phase, cc$raw),
                  na.rm = TRUE) + c(-1, 1)
  plot(cc$cycle_day, cc$invariant, type = "l", col = "#A8D8EA", lwd = 5,
       xlim = c(0, 365), ylim = ylim, xaxt = "n",
       xlab = "Day of the cycle", ylab = "Anomaly for sea ice extent",
       main = title)
  axis(1, at = c(0, 100, 200, 300))
  abline(h = 0, lty = 2)
  lines(cc$cycle_day, cc$trend,     col = "#1B7A1B", lwd = 3)
  lines(cc$cycle_day, cc$amplitude, col = "#14148C", lwd = 3)
  lines(cc$cycle_day, cc$phase,     col = "#E8140C", lwd = 3.5)
  lines(cc$cycle_day, cc$raw,       col = "black",   lwd = 1)
  lines(cc$cycle_day, cc$est,       col = "#FFA500", lwd = 2)
}

add_legend <- function(cex = 0.85) {
  legend("bottom", ncol = 2, bty = "n", cex = cex,
         col = c("#A8D8EA", "#1B7A1B", "#14148C", "#E8140C",
                 "black", "#FFA500"),
         lwd = c(5, 3, 3, 3.5, 1, 2),
         legend = c("Invariant annual cycle", "Trend component",
                    "Amplitude component", "Phase component",
                    "Raw anomaly", "Estimated anomaly"))
}

# ── single-panel circumpolar (H&R style/limits) ────────────────────────────
png(file.path(OUT_DIR, "FIG7a_circumpolar_E.png"), width = 9.5, height = 7,
    units = "in", res = 300)
cc <- components_for("SIE_circumpolar", YEAR)
draw_panel(cc, sprintf("(a)  Circumpolar — %d", YEAR), ylim = c(-10.5, 7.5))
add_legend()
dev.off()
message("Wrote FIG7a_circumpolar_E.png")

# ── 2x3 sector grid ─────────────────────────────────────────────────────────
png(file.path(OUT_DIR, "FIG7_sectors_E.png"), width = 16, height = 9.5,
    units = "in", res = 300)
par(mfrow = c(2, 3), mar = c(4, 4, 2.5, 1))
for (i in seq_along(SECTORS)) {
  cc <- components_for(SECTORS[i], YEAR)
  draw_panel(cc, sprintf("(%s)  %s", letters[i], LABELS[SECTORS[i]]))
  if (i == 1) add_legend(cex = 0.7)
}
dev.off()
message("Wrote FIG7_sectors_E.png")