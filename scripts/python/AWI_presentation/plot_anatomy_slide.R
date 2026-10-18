# =============================================================================
# plot_anatomy_slide.R -- slide version of the Ch. 3 anatomy figure
# (plot_fig7_sectors.R, panel a): one sector, one cycle.
#
# Same elements as the chapter figure, cleaned for a slide:
#   invariant annual cycle (light blue), trend (green), amplitude (amber),
#   phase (purple), raw anomaly (black line with grey fill).
# Transparent background, no title, no legend (direct labels on the right).
# Component colours match the schematic slide.
#
# components_for() is copied from plot_fig7_sectors.R with the same maths:
#   cycle = 21 Feb (YEAR) to 20 Feb (YEAR + 1), day 0 = the minimum
#   retreat-limb blend of amplitude and minimum (removes the 1 January step)
#   trend     = trend_component
#   amplitude = famp  - iac_notrend - trend
#   phase     = fapac - famp
#   raw       = Extent - fapac
#   extent anomaly = Extent - iac_notrend (= the four parts, checked)
#
# UNITS
#   "percent" (default, as in the chapter and Handcock & Raphael 2020 Fig 7a):
#       components in % of the annual amplitude; the invariant cycle in
#       10^6 km^2 centred on its mean. Mixed units on one axis, so the caption
#       must say so, e.g. "components in % of the annual amplitude; invariant
#       cycle in 10^6 km^2 (centred)".
#   "km2": everything in 10^6 km^2 (true scale; components look small next to
#       the cycle).
#
# Run (from anywhere):
#   Rscript plot_anatomy_slide.R                        # 2023, circumpolar, FULL fit
#   Rscript plot_anatomy_slide.R 2024                   # another year
#   Rscript plot_anatomy_slide.R 2016 SIE_Weddell       # year, sector
# or edit YEAR / SECTOR below and source() it in RStudio.
#
# Writes (in OUT_DIR): anatomy_<YEAR>.png (300 dpi, transparent) and .svg
# =============================================================================

ROOT    <- "/Users/fridaperez/Research/repos/sea-ice-phase"
DAILY   <- file.path(ROOT, "data", "ch3", "daily_fitted.csv")
OUT_DIR <- file.path(ROOT, "results", "ch3", "figures")

YEAR    <- 2023
SECTOR  <- "SIE_circumpolar"
PERIOD  <- "FULL"          # the chapter's full-record fit (to 2025)
TREND_COL <- "trend_component"
UNITS   <- "percent"       # "percent" (chapter convention) or "km2"

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) YEAR   <- as.integer(args[1])
if (length(args) >= 2) SECTOR <- args[2]
if (length(args) >= 3) DAILY  <- args[3]

# ---- look -------------------------------------------------------------------
INK        <- "#222222"     # use "#EEEEEE" on a dark slide
MUTED      <- "#6B6B6B"
COL_IAC    <- "#9CCBE3"     # invariant annual cycle: light blue (as in the chapter)
COL_TREND  <- "#009E73"     # trend: green
COL_AMP    <- "#E69F00"     # amplitude: amber
COL_PHASE  <- "#CC79A7"     # phase: reddish purple
COL_FILL   <- adjustcolor("#B8B8B8", alpha.f = 0.6)   # under the raw anomaly
FONT          <- "sans"     # e.g. "Helvetica" or "Arial"; match the other slide figures
SHOW_MAX_LINE <- TRUE       # faint line at the day of the cycle maximum
SHOW_EXTENT_ANOMALY <- FALSE  # dashed line: the sum of the four parts
SMOOTH_DAYS   <- 11
# How the raw anomaly is drawn:
#   "smooth": the 11-day mean (default; keeps single-day spikes off the slide)
#   "daily" : the daily values, spikes and all
#   "both"  : faint daily values under the 11-day mean
RAW_MODE      <- "smooth"

if (!file.exists(DAILY)) stop("Input not found: ", DAILY)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

d_raw <- read.csv(DAILY, stringsAsFactors = FALSE)
d_raw$Date <- as.Date(d_raw$Date)
if (!"period" %in% names(d_raw)) {
  message("no period column: treating file as single-period")
  d_raw$period <- PERIOD
}
if (sum(d_raw$period == PERIOD) == 0)
  stop("period '", PERIOD, "' not found; available: ",
       paste(unique(d_raw$period), collapse = ", "))

# ── one sector, one cycle -> components (from plot_fig7_sectors.R) ───────────
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

  tv  <- g[[TREND_COL]]
  out <- data.frame(
    day       = cd,
    invariant = g$iac_notrend - mean(g$iac_notrend),   # centred, as in the chapter figure
    trend     = tv,
    amplitude = famp - g$iac_notrend - tv,
    phase     = fapac - famp,
    raw       = g$Extent - fapac,
    anomaly   = g$Extent - g$iac_notrend)

  # the four parts must add up to the extent anomaly (they do, by construction)
  err <- max(abs(rowSums(out[c("trend", "amplitude", "phase", "raw")]) - out$anomaly),
             na.rm = TRUE)
  if (err > 1e-6 * max(1, max(abs(g$Extent), na.rm = TRUE)))
    warning(sprintf("components do not sum to the anomaly (max difference %.3g)", err))

  # to 10^6 km^2 if the file is in km^2
  scale <- if (max(g$Extent, na.rm = TRUE) > 1000) 1e-6 else 1
  cols <- c("invariant", "trend", "amplitude", "phase", "raw", "anomaly")
  out[cols] <- out[cols] * scale
  ampY <- ampY * scale

  # components as % of the annual amplitude (the chapter's convention)
  if (UNITS == "percent") {
    pc <- c("trend", "amplitude", "phase", "raw", "anomaly")
    out[pc] <- out[pc] / ampY * 100
  }
  out$raw_smooth <- as.numeric(stats::filter(out$raw, rep(1 / SMOOTH_DAYS, SMOOTH_DAYS), sides = 2))
  out$raw_smooth[is.na(out$raw_smooth)] <- out$raw[is.na(out$raw_smooth)]
  list(df = out, dmax = dmax, t0 = t0, scale = scale, ampY = ampY)
}

cc <- components_for(SECTOR, YEAR, PERIOD)
x  <- cc$df
message(sprintf("%s %d, fit '%s', %s; annual amplitude %.2f x 10^6 km^2", SECTOR, YEAR, PERIOD,
                if (cc$scale != 1) "km^2 converted to 10^6 km^2" else "units as in file (assumed 10^6 km^2)",
                cc$ampY))

# ── draw ─────────────────────────────────────────────────────────────────────
raw_plot <- if (RAW_MODE == "daily") x$raw else x$raw_smooth
LINES <- list(
  list(name = "invariant annual cycle", y = x$invariant, col = COL_IAC,   lwd = 7,   txt = COL_IAC),
  list(name = "trend",                  y = x$trend,     col = COL_TREND, lwd = 5,   txt = COL_TREND),
  list(name = "amplitude",              y = x$amplitude, col = COL_AMP,   lwd = 5,   txt = COL_AMP),
  list(name = "phase",                  y = x$phase,     col = COL_PHASE, lwd = 5,   txt = COL_PHASE),
  list(name = "raw anomaly",            y = raw_plot,    col = INK,       lwd = 1.6, txt = MUTED))

yvals <- c(x$invariant, x$trend, x$amplitude, x$phase, raw_plot)
if (SHOW_EXTENT_ANOMALY) yvals <- c(yvals, x$anomaly)
yr   <- range(yvals, na.rm = TRUE)
pad  <- 0.06 * diff(yr)
ylim <- yr + c(-pad, pad)

# keep direct labels from colliding, preserving their vertical order
spread <- function(ys, min_gap) {
  o <- order(ys); adj <- ys
  for (i in seq_along(o)[-1]) {
    a <- o[i - 1]; b <- o[i]
    if (adj[b] - adj[a] < min_gap) adj[b] <- adj[a] + min_gap
  }
  adj + (mean(ys) - mean(adj))
}

months <- as.Date(c(sprintf("%d-%02d-01", YEAR, c(3, 5, 7, 9, 11)),
                    sprintf("%d-01-01", YEAR + 1)))
tick_at  <- as.numeric(months - cc$t0)
tick_lab <- format(months, "%b")

ylab_txt <- if (UNITS == "percent") "Anomaly for sea ice extent" else
  expression("Anomaly for sea ice extent (" * 10^6 ~ km^2 * ")")

draw <- function() {
  par(mar = c(3.4, 5.2, 0.8, 16), family = FONT, fg = INK, col.axis = INK,
      col.lab = MUTED, cex.axis = 1.5, cex.lab = 1.5, xpd = FALSE)
  plot(NA, NA, xlim = c(0, 365), ylim = ylim, xaxt = "n", yaxt = "n", bty = "n", xaxs = "i",
       xlab = "", ylab = ylab_txt)
  polygon(c(x$day, rev(x$day)), c(raw_plot, rep(0, nrow(x))), col = COL_FILL, border = NA)
  if (RAW_MODE == "both") lines(x$day, x$raw, col = adjustcolor(INK, alpha.f = 0.3), lwd = 0.8)
  abline(h = 0, col = INK, lwd = 1.2, lty = 2)
  if (SHOW_MAX_LINE) abline(v = cc$dmax, col = "grey70", lty = 3, lwd = 1.6)
  for (l in LINES) lines(x$day, l$y, col = l$col, lwd = l$lwd, lend = "round")
  if (SHOW_EXTENT_ANOMALY) lines(x$day, x$anomaly, col = INK, lwd = 2.5, lty = 2)
  axis(1, at = tick_at, labels = tick_lab, col = "grey60", tcl = -0.4, lwd = 1)
  axis(2, col = "grey60", las = 1, lwd = 1)

  # direct labels in the right margin
  ends <- sapply(LINES, function(l) tail(l$y[!is.na(l$y)], 1))
  ypos <- spread(ends, 0.075 * diff(ylim))
  par(xpd = NA)
  for (i in seq_along(LINES))
    text(372, ypos[i], LINES[[i]]$name, col = LINES[[i]]$txt, adj = c(0, 0.5), cex = 1.4, font = 2)
}

f_png <- file.path(OUT_DIR, sprintf("anatomy_%d.png", YEAR))
f_svg <- file.path(OUT_DIR, sprintf("anatomy_%d.svg", YEAR))
W <- 13; H <- 5.8

if (capabilities("cairo")) {
  png(f_png, width = W, height = H, units = "in", res = 300, bg = "transparent", type = "cairo")
} else {
  png(f_png, width = W, height = H, units = "in", res = 300, bg = "transparent")
}
draw(); invisible(dev.off())
message("Wrote ", f_png)

if (capabilities("cairo")) {
  svg(f_svg, width = W, height = H, bg = "transparent")
  draw(); invisible(dev.off())
  message("Wrote ", f_svg)
} else {
  message("(no cairo: skipped the .svg; the .png is transparent only if your R png device supports it)")
}

if (RAW_MODE != "daily") message("CAPTION: raw anomaly shown as an ", SMOOTH_DAYS, "-day mean.")
if (UNITS == "percent")
  message("CAPTION: components in % of the annual amplitude; invariant annual cycle in 10^6 km^2, centred on its mean (as in Handcock and Raphael 2020, Fig. 7a).")

# sizes for the cycle, mean |value| over the cycle, in 10^6 km^2 (compare with Fig. 3)
k <- if (UNITS == "percent") cc$ampY / 100 else 1
message("mean |component| over the cycle (10^6 km^2):")
for (nm in c("anomaly", "trend", "amplitude", "phase", "raw"))
  message(sprintf("  %-10s %.2f", nm, mean(abs(x[[nm]]), na.rm = TRUE) * k))
