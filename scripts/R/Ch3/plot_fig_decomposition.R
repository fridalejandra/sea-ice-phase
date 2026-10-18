# =============================================================================
# plot_fig_decomposition.R -- the four terms of the decomposition for one year,
# manuscript version (Fig. 3.3 and Appendix Fig. A3), everything in 10^6 km^2.
#
# Replaces the fig07 panels of plot_fig7_sectors.R for the chapter. Same maths
# as plot_anatomy_slide_v2.R (cycle 21 Feb Y -> 20 Feb Y+1, retreat-limb blend,
# sum check), same Okabe-Ito palette as the slides, one unit throughout: the
# cycle is large (+-9 for the total) and the components small (+-2), so they
# are drawn in two panels instead of one mixed-unit axis.
#
# Figure 1  fig_decomposition_<YEAR>.png          circumpolar total
#   (a) observed extent, invariant annual cycle, fully adjusted cycle (10^6 km^2)
#   (b) the extent anomaly (observed - invariant) split into Trend, Amplitude,
#       Phase and the Raw Anomaly (shaded, with its 11-day mean)
# Figure 2  fig_decomposition_sectors_<YEAR>.png  panel (b) for the five sectors
#           and the total, 3 x 2, each on its own axis, one legend
# Each also written as *_presentation.png (transparent background).
#
# Run:  Rscript plot_fig_decomposition.R            # 2022
#       Rscript plot_fig_decomposition.R 2023
# =============================================================================

ROOT    <- "/Users/fridaperez/Research/repos/sea-ice-phase"
DAILY   <- file.path(ROOT, "data", "ch3", "daily_fitted.csv")
OUT_DIR <- file.path(ROOT, "results", "ch3", "figures")
YEAR    <- 2022
PERIOD  <- "FULL"
TREND_COL <- "trend_component"
SMOOTH_DAYS <- 11
TEXT_WIDTH_IN <- 6.5

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) YEAR <- as.integer(args[1])

# chapter order
SECTORS <- c("SIE_Weddell", "SIE_Amundsen_Bellingshausen", "SIE_Ross",
             "SIE_East_Antarctica", "SIE_King_Haakon", "SIE_circumpolar")
LABELS  <- c(SIE_Weddell = "Weddell", SIE_Amundsen_Bellingshausen = "Amundsen–Bellingshausen",
             SIE_Ross = "Ross", SIE_East_Antarctica = "East Antarctica",
             SIE_King_Haakon = "King Haakon", SIE_circumpolar = "Circumpolar total")

# ---- look (Okabe-Ito, as in plot_anatomy_slide_v2.R and ch3_style.py) ------
INK       <- "#222222"
AXIS_COL  <- "#595959"
COL_IAC   <- "#5BA4CF"
COL_OBS   <- INK
COL_FIT   <- "#7A7A7A"
COL_TREND <- "#009E73"
COL_AMP   <- "#E69F00"
COL_PHASE <- "#CC79A7"
COL_RAW   <- "#8A8D93"
COL_FILL  <- adjustcolor("#B8B8B8", alpha.f = 0.55)
FONT      <- "sans"

if (!file.exists(DAILY)) stop("Input not found: ", DAILY)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
d_raw <- read.csv(DAILY, stringsAsFactors = FALSE)
d_raw$Date <- as.Date(d_raw$Date)
if (!"period" %in% names(d_raw)) d_raw$period <- PERIOD
if (sum(d_raw$period == PERIOD) == 0)
  stop("period '", PERIOD, "' not found; available: ", paste(unique(d_raw$period), collapse = ", "))

# ---- one sector, one cycle -> components, all in 10^6 km^2 ------------------
components_for <- function(sector, year, period) {
  t0 <- as.Date(sprintf("%d-02-21", year))
  t1 <- as.Date(sprintf("%d-02-20", year + 1))
  g  <- d_raw[d_raw$period == period & d_raw$sector == sector &
                d_raw$Date >= t0 & d_raw$Date <= t1, ]
  g  <- g[order(g$Date), ]
  if (nrow(g) < 300)
    stop(sprintf("Only %d rows for %s %d (%s) - cycle incomplete?", nrow(g), sector, year, period))
  if (any(duplicated(g$Date)))
    stop(sprintf("Duplicate dates for %s %d (%s) - period filter did not take.", sector, year, period))
  cd <- as.numeric(g$Date - t0)

  ampY <- g$amplitude[1];       minY <- g$min_extent[1]
  ampN <- tail(g$amplitude, 1); minN <- tail(g$min_extent, 1)
  dmax  <- cd[which.max(g$Extent)]
  w     <- pmin(pmax((cd - dmax) / (365 - dmax), 0), 1)
  ampb  <- ampY * (1 - w) + ampN * w
  minb  <- minY * (1 - w) + minN * w
  famp  <- g$u_amp  * ampb + minb
  fapac <- g$u_apac * ampb + minb
  tv    <- g[[TREND_COL]]

  out <- data.frame(
    day       = cd,
    observed  = g$Extent,
    invariant = g$iac_notrend,
    adjusted  = fapac,
    trend     = tv,
    amplitude = famp - g$iac_notrend - tv,
    phase     = fapac - famp,
    raw       = g$Extent - fapac,
    anomaly   = g$Extent - g$iac_notrend)
  err <- max(abs(rowSums(out[c("trend", "amplitude", "phase", "raw")]) - out$anomaly), na.rm = TRUE)
  if (err > 1e-6 * max(1, max(abs(g$Extent), na.rm = TRUE)))
    warning(sprintf("components do not sum to the anomaly (max difference %.3g)", err))
  scale <- if (max(g$Extent, na.rm = TRUE) > 1000) 1e-6 else 1
  cols <- c("observed", "invariant", "adjusted", "trend", "amplitude", "phase", "raw", "anomaly")
  out[cols] <- out[cols] * scale
  out$raw_smooth <- as.numeric(stats::filter(out$raw, rep(1 / SMOOTH_DAYS, SMOOTH_DAYS), sides = 2))
  out$raw_smooth[is.na(out$raw_smooth)] <- out$raw[is.na(out$raw_smooth)]

  months <- seq(as.Date(sprintf("%d-03-01", year)), by = "month", length.out = 12)
  list(df = out, dmax = dmax, t0 = t0,
       tick_at = as.numeric(months - t0), tick_lab = format(months, "%b"))
}

# ---- drawing pieces ---------------------------------------------------------
month_axis <- function(cc, every = 2, labels = TRUE) {
  axis(1, at = cc$tick_at, labels = FALSE, col = AXIS_COL, col.ticks = AXIS_COL, tcl = -0.3)
  if (labels) {
    i <- seq(1, 12, by = every)
    axis(1, at = cc$tick_at[i], labels = cc$tick_lab[i], tick = FALSE, line = -0.5,
         col.axis = INK, cex.axis = 0.85)
  }
}
y_axis <- function() axis(2, col = AXIS_COL, col.ticks = AXIS_COL, col.axis = INK, las = 1, cex.axis = 0.85, tcl = -0.3)
panel_letter <- function(txt, cex = 0.95) mtext(txt, side = 3, adj = 0, line = 0.15, font = 2, cex = cex, col = INK)

draw_cycles <- function(cc) {                      # panel (a)
  x <- cc$df
  yl <- range(c(x$observed, x$invariant, x$adjusted), na.rm = TRUE); yl <- yl + c(-0.04, 0.10) * diff(yl)
  plot(NA, NA, xlim = c(0, 365), ylim = yl, xaxt = "n", yaxt = "n", bty = "l", xaxs = "i",
       xlab = "", ylab = "", col.lab = INK)
  abline(v = cc$dmax, col = "grey60", lty = 3, lwd = 1)
  lines(x$day, x$invariant, col = COL_IAC, lwd = 4.5, lend = "round")
  lines(x$day, x$adjusted,  col = COL_FIT, lwd = 1.3, lty = 2)
  lines(x$day, x$observed,  col = COL_OBS, lwd = 1.1)
  y_axis(); month_axis(cc, labels = FALSE)
  mtext(expression("Extent (" * 10^6 ~ km^2 * ")"), side = 2, line = 2.9, cex = 0.85, col = INK)
  legend("topleft", inset = c(0.0, 0.0), bty = "n", cex = 0.8, text.col = INK,
         col = c(COL_IAC, COL_FIT, COL_OBS), lwd = c(4.5, 1.3, 1.1), lty = c(1, 2, 1), seg.len = 2.2,
         legend = c("invariant annual cycle", sprintf("APAC fit, %d", YEAR), sprintf("observed, %d", YEAR)))
}

draw_components <- function(cc, lwd = 1.9, ylab = TRUE, month_labels = TRUE, every = 2, headroom = 0.12) {
  x <- cc$df
  yl <- range(c(x$trend, x$amplitude, x$phase, x$raw), na.rm = TRUE)
  yl <- yl + c(-headroom, headroom) * diff(yl)
  plot(NA, NA, xlim = c(0, 365), ylim = yl, xaxt = "n", yaxt = "n", bty = "l", xaxs = "i",
       xlab = "", ylab = "")
  polygon(c(x$day, rev(x$day)), c(x$raw, rep(0, nrow(x))), col = COL_FILL, border = NA)
  abline(h = 0, col = "grey70", lwd = 0.9)
  abline(v = cc$dmax, col = "grey60", lty = 3, lwd = 1)
  lines(x$day, x$raw_smooth, col = COL_RAW,   lwd = 1.2)
  lines(x$day, x$trend,      col = COL_TREND, lwd = lwd, lend = "round")
  lines(x$day, x$amplitude,  col = COL_AMP,   lwd = lwd, lend = "round")
  lines(x$day, x$phase,      col = COL_PHASE, lwd = lwd, lend = "round")
  y_axis(); month_axis(cc, every = every, labels = month_labels)
  if (ylab) {
    mtext("Contribution to the", side = 2, line = 3.7, cex = 0.85, col = INK)
    mtext(expression("extent anomaly (" * 10^6 ~ km^2 * ")"), side = 2, line = 2.6, cex = 0.85, col = INK)
  }
}
component_legend <- function(where = "bottomleft", cex = 0.8, inset = 0) {
  op <- par(xpd = NA); on.exit(par(op))
  # four entries, one row; the Raw Anomaly key is its shaded band with the 11-day line through it
  legend(where, inset = inset, bty = "n", cex = cex, ncol = 4, text.col = INK,
         x.intersp = 0.6, seg.len = 1.8,
         col    = c(COL_TREND, COL_AMP, COL_PHASE, COL_RAW),
         lwd    = c(1.9, 1.9, 1.9, 1.2),
         fill   = c(NA, NA, NA, COL_FILL), border = NA,
         legend = c("Trend", "Amplitude", "Phase", "Raw Anomaly"))
}

open_png <- function(f, w, h, bg) {
  if (capabilities("cairo")) png(f, width = w, height = h, units = "in", res = 300, bg = bg, type = "cairo")
  else                       png(f, width = w, height = h, units = "in", res = 300, bg = bg)
}

# ---- Figure 1: circumpolar, two panels ------------------------------------
cc <- components_for("SIE_circumpolar", YEAR, PERIOD)
# panel (b) for Figure 1: as draw_components, with room below the data for the legend
draw_components_with_legend <- function(cc) {
  x <- cc$df
  yl <- range(c(x$trend, x$amplitude, x$phase, x$raw), na.rm = TRUE)
  yl <- yl + c(-0.30, 0.14) * diff(yl)
  plot(NA, NA, xlim = c(0, 365), ylim = yl, xaxt = "n", yaxt = "n", bty = "l", xaxs = "i", xlab = "", ylab = "")
  polygon(c(x$day, rev(x$day)), c(x$raw, rep(0, nrow(x))), col = COL_FILL, border = NA)
  abline(h = 0, col = "grey70", lwd = 0.9)
  abline(v = cc$dmax, col = "grey60", lty = 3, lwd = 1)
  lines(x$day, x$raw_smooth, col = COL_RAW, lwd = 1.2)
  lines(x$day, x$trend,     col = COL_TREND, lwd = 1.9, lend = "round")
  lines(x$day, x$amplitude, col = COL_AMP,   lwd = 1.9, lend = "round")
  lines(x$day, x$phase,     col = COL_PHASE, lwd = 1.9, lend = "round")
  y_axis(); month_axis(cc)
  mtext("Contribution to the", side = 2, line = 3.7, cex = 0.85, col = INK)
  mtext(expression("extent anomaly (" * 10^6 ~ km^2 * ")"), side = 2, line = 2.6, cex = 0.85, col = INK)
  component_legend("bottomleft", cex = 0.8)
}
draw_fig1 <- function() {
  layout(matrix(1:2, ncol = 1), heights = c(1, 1.5))
  par(family = FONT, fg = AXIS_COL, mgp = c(2.4, 0.6, 0))
  par(mar = c(0.8, 4.9, 1.2, 1.0)); draw_cycles(cc); panel_letter("(a)")
  par(mar = c(3.0, 4.9, 1.0, 1.0)); draw_components_with_legend(cc); panel_letter("(b)")
  mtext("maximum", side = 3, at = cc$dmax, line = 0.15, cex = 0.72, col = "grey45")
}
for (bg in c("white", "transparent")) {
  f <- file.path(OUT_DIR, sprintf("fig_decomposition_%d%s.png", YEAR, if (bg == "white") "" else "_presentation"))
  open_png(f, TEXT_WIDTH_IN, 5.6, bg)
  draw_fig1(); invisible(dev.off()); message("Wrote ", f)
}

# ---- Figure 2: the six series, components only, 3 x 2 -----------------------
cc_all <- lapply(SECTORS, function(s) components_for(s, YEAR, PERIOD)); names(cc_all) <- SECTORS
draw_fig2 <- function() {
  layout(rbind(matrix(1:6, nrow = 3, byrow = TRUE), c(7, 7)), heights = c(1, 1, 1, 0.14))
  par(family = FONT, fg = AXIS_COL, mgp = c(2.4, 0.6, 0))
  for (i in seq_along(SECTORS)) {
    bottom <- i >= 5; left <- i %% 2 == 1
    par(mar = c(if (bottom) 2.8 else 1.2, if (left) 4.6 else 3.2, 1.4, 0.8))
    draw_components(cc_all[[SECTORS[i]]], lwd = 1.6, ylab = FALSE, month_labels = bottom, every = 3, headroom = 0.10)
    if (left) mtext(expression(10^6 ~ km^2), side = 2, line = 2.6, cex = 0.8, col = INK)
    panel_letter(sprintf("(%s)  %s", letters[i], LABELS[SECTORS[i]]), cex = 0.85)
  }
  par(mar = c(0, 0, 0, 0)); plot.new(); component_legend("center", cex = 0.85)
}
for (bg in c("white", "transparent")) {
  f <- file.path(OUT_DIR, sprintf("fig_decomposition_sectors_%d%s.png", YEAR, if (bg == "white") "" else "_presentation"))
  open_png(f, TEXT_WIDTH_IN, 7.4, bg)
  draw_fig2(); invisible(dev.off()); message("Wrote ", f)
}

message("mean |component| over the cycle, circumpolar (10^6 km^2):")
for (nm in c("anomaly", "trend", "amplitude", "phase", "raw"))
  message(sprintf("  %-10s %.2f", nm, mean(abs(cc$df[[nm]]), na.rm = TRUE)))
