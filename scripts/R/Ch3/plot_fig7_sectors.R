# =============================================================================
# plot_fig7_sectors.R — §3.2 component anatomy, one cycle, all sectors,
#                       for each year in YEARS.
#
# Panel (a) for 2016 reproduces H&R (2020) Fig 7a using the paper's own fit
# window (1979–2018, period == "HR2018"). For years after 2018 there is no
# HR2018 fit, so panel (a) uses the full-record fit, and the title says which.
#
# Plotting convention (both panel (a) and the grid), as in the published
# figure: the invariant annual cycle is drawn in 10^6 km^2 centred on its
# mean; the components are drawn in PERCENT OF THE ANNUAL AMPLITUDE. Mixed
# units on one axis — say so in the caption. Each grid panel has its own
# y-axis, so a sector with a small anomaly is not flattened by a sector with a
# large one.
#
# The cycle runs 21 Feb (year Y) to 20 Feb (Y+1); day 0 is the minimum. If the
# record ends before 20 Feb Y+1 the cycle is INCOMPLETE and the script says so:
# the retreat-limb statistics for that year are over a shorter window.
#
# The retreat-limb blend is REQUIRED: the raw *_pct columns are normalised by
# each row's calendar-year amplitude, which steps on 1 January; the blend
# removes that discontinuity.
#
# Dominance, per sector, three ways:
#   mean_retreat  mean |x| from the cycle maximum to the end of the cycle
#                 (the interval H&R's attribution refers to)
#   peak          max |x| anywhere in the cycle
#   mean_full     mean |x| over the whole cycle (diluted by the advance, where
#                 phase ~ 0; reported for completeness, not for the text)
#
# Outputs, per year:
#   results/ch3/figures/fig07a_circumpolar_<YEAR>.png
#   results/ch3/figures/fig07_sectors_<YEAR>.png
#   results/ch3/tables/t32_component_magnitude_<YEAR>.csv
# and combined:
#   results/ch3/tables/t32_component_magnitude_all.csv
# =============================================================================

ROOT     <- "/Users/fridaperez/Research/repos/sea-ice-phase"
DAILY    <- file.path(ROOT, "data", "ch3", "daily_fitted.csv")
FIG_DIR  <- file.path(ROOT, "results", "ch3", "figures")
TAB_DIR  <- file.path(ROOT, "results", "ch3", "tables")
YEARS    <- c(2016, 2022, 2023)
PERIOD_HR   <- "HR2018"    # used for panel (a) when the year is inside it
PERIOD_FULL <- "FULL"      # sector grid always; panel (a) when year > 2018
TREND_COL   <- "trend_component"
GRID_IAC    <- TRUE        # invariant cycle on every grid panel
GRID_SHARED_Y <- FALSE     # FALSE = each panel its own axis (keep FALSE)

if (!file.exists(DAILY)) stop("Input not found: ", DAILY)
for (d in c(FIG_DIR, TAB_DIR)) dir.create(d, showWarnings = FALSE, recursive = TRUE)
message("DAILY = ", DAILY)

SECTORS <- c("SIE_circumpolar", "SIE_Weddell", "SIE_King_Haakon",
             "SIE_East_Antarctica", "SIE_Ross", "SIE_Amundsen_Bellingshausen")
LABELS  <- c(SIE_circumpolar = "Circumpolar", SIE_Weddell = "Weddell",
             SIE_King_Haakon = "King Haakon",
             SIE_East_Antarctica = "East Antarctica", SIE_Ross = "Ross",
             SIE_Amundsen_Bellingshausen = "Amundsen-Bellingshausen")

d_raw <- read.csv(DAILY, stringsAsFactors = FALSE)
d_raw$Date <- as.Date(d_raw$Date)
if (!"period" %in% names(d_raw)) {
  message("no period column: single-period file, using it everywhere")
  d_raw$period <- PERIOD_FULL; PERIOD_HR <- PERIOD_FULL
}
for (p in unique(c(PERIOD_HR, PERIOD_FULL))) {
  n <- sum(d_raw$period == p)
  if (n == 0) stop("period '", p, "' not in file; available: ",
                   paste(unique(d_raw$period), collapse = ", "))
  message(sprintf("period %-7s %d rows, %s to %s", p, n,
                  min(d_raw$Date[d_raw$period == p]), max(d_raw$Date[d_raw$period == p])))
}
hr_last_year <- as.integer(format(max(d_raw$Date[d_raw$period == PERIOD_HR]), "%Y"))

# ── one sector, one cycle → components ──────────────────────────────────────
components_for <- function(sector, year, period) {
  t0 <- as.Date(sprintf("%d-02-21", year))
  t1 <- as.Date(sprintf("%d-02-20", year + 1))
  g  <- d_raw[d_raw$period == period & d_raw$sector == sector &
                d_raw$Date >= t0 & d_raw$Date <= t1, ]
  g  <- g[order(g$Date), ]
  if (nrow(g) < 200)
    stop(sprintf("Only %d rows for %s %d (%s) — not enough of the cycle.", nrow(g), sector, year, period))
  if (any(duplicated(g$Date)))
    stop(sprintf("Duplicate dates for %s %d (%s) — period filter did not take.", sector, year, period))
  cd <- as.numeric(g$Date - t0)
  complete <- max(cd) >= 360
  
  ampY <- g$amplitude[1];       minY <- g$min_extent[1]
  ampN <- tail(g$amplitude, 1); minN <- tail(g$min_extent, 1)
  
  dmax  <- cd[which.max(g$Extent)]
  w     <- pmin(pmax((cd - dmax) / (365 - dmax), 0), 1)
  ampb  <- ampY * (1 - w) + ampN * w
  minb  <- minY * (1 - w) + minN * w
  famp  <- g$u_amp  * ampb + minb
  fapac <- g$u_apac * ampb + minb
  
  tv  <- g[[TREND_COL]]
  raw <- (g$Extent - fapac) / ampY * 100
  est <- as.numeric(stats::filter(raw, rep(1/11, 11), sides = 2))
  list(
    dmax = dmax, ampY = ampY, last_day = max(cd), complete = complete,
    df = data.frame(
      cycle_day     = cd,
      invariant_km2 = g$iac_notrend - mean(g$iac_notrend),   # 10^6 km^2, centred
      trend         = tv / ampY * 100,                        # % of amplitude
      amplitude     = (famp - g$iac_notrend - tv) / ampY * 100,
      phase         = (fapac - famp) / ampY * 100,
      raw           = raw,
      est           = est))
}

COL <- c(invariant = "#A8D8EA", trend = "#1B7A1B", amplitude = "#14148C",
         phase = "#E8140C", raw = "black", est = "#FFA500")
LWD <- c(invariant = 5, trend = 3, amplitude = 3, phase = 3.5, raw = 1, est = 2)
LAB <- c(invariant = "Invariant annual cycle", trend = "Trend component",
         amplitude = "Amplitude component", phase = "Phase component",
         raw = "Raw anomaly", est = "Estimated anomaly")
YLAB <- "Anomaly for sea ice extent"

draw_panel <- function(cc, title, ylim, show_iac) {
  plot(NA, NA, xlim = c(0, 365), ylim = ylim, xaxt = "n",
       xlab = "Day of the cycle", ylab = YLAB, main = title)
  axis(1, at = c(0, 100, 200, 300))
  abline(h = 0, lty = 2)
  if (show_iac) lines(cc$cycle_day, cc$invariant_km2, col = COL["invariant"], lwd = LWD["invariant"])
  for (k in c("trend", "amplitude", "phase", "raw", "est"))
    lines(cc$cycle_day, cc[[k]], col = COL[k], lwd = LWD[k])
}
add_legend <- function(keys, cex) {
  legend("bottom", ncol = 2, bty = "n", cex = cex,
         col = COL[keys], lwd = LWD[keys], legend = LAB[keys])
}
panel_cols <- c(if (GRID_IAC) "invariant_km2", "trend", "amplitude", "phase", "raw")
panel_ylim <- function(x) range(unlist(x$df[panel_cols]), na.rm = TRUE) + c(-1, 1)
grid_keys  <- c(if (GRID_IAC) "invariant", "trend", "amplitude", "phase", "raw", "est")
comps      <- c("trend", "amplitude", "phase", "raw")

# ── one year ────────────────────────────────────────────────────────────────
run_year <- function(YEAR) {
  message(sprintf("\n==================== %d ====================", YEAR))
  period_a <- if (YEAR <= hr_last_year) PERIOD_HR else PERIOD_FULL
  fit_lab  <- if (period_a == PERIOD_HR) sprintf("fit 1979-%d", hr_last_year) else "full-record fit"
  
  # panel (a)
  ca <- components_for("SIE_circumpolar", YEAR, period_a)
  inc <- if (ca$complete) "" else sprintf("  [cycle ends day %d]", ca$last_day)
  if (!ca$complete)
    message(sprintf("  NOTE: %d cycle is INCOMPLETE — record ends at cycle day %d of 365.", YEAR, ca$last_day))
  f1 <- file.path(FIG_DIR, sprintf("fig07a_circumpolar_%d.png", YEAR))
  png(f1, width = 9.5, height = 7, units = "in", res = 300)
  draw_panel(ca$df, sprintf("(a)  Circumpolar - %d  (%s)%s", YEAR, fit_lab, inc),
             ylim = panel_ylim(ca), show_iac = TRUE)
  add_legend(c("invariant", "trend", "amplitude", "phase", "raw", "est"), cex = 0.85)
  dev.off()
  message("Wrote ", f1)
  
  # grid
  cc_all <- lapply(SECTORS, function(s) components_for(s, YEAR, PERIOD_FULL))
  names(cc_all) <- SECTORS
  yl_shared <- range(unlist(lapply(cc_all, panel_ylim)))
  f2 <- file.path(FIG_DIR, sprintf("fig07_sectors_%d.png", YEAR))
  png(f2, width = 16, height = 9.5, units = "in", res = 300)
  par(mfrow = c(2, 3), mar = c(4, 4.4, 2.5, 1))
  for (i in seq_along(SECTORS)) {
    x <- cc_all[[SECTORS[i]]]
    draw_panel(x$df, sprintf("(%s)  %s - %d%s", letters[i], LABELS[SECTORS[i]], YEAR, inc),
               ylim = if (GRID_SHARED_Y) yl_shared else panel_ylim(x),
               show_iac = GRID_IAC)
    if (i == 1) add_legend(grid_keys, cex = 0.7)
  }
  dev.off()
  message("Wrote ", f2)
  
  # dominance
  summ <- do.call(rbind, lapply(SECTORS, function(s) {
    x   <- cc_all[[s]]$df
    ret <- x$cycle_day >= cc_all[[s]]$dmax
    mf  <- sapply(comps, function(k) mean(abs(x[[k]]),      na.rm = TRUE))
    mr  <- sapply(comps, function(k) mean(abs(x[[k]][ret]), na.rm = TRUE))
    pk  <- sapply(comps, function(k) max (abs(x[[k]]),      na.rm = TRUE))
    sg  <- sapply(comps, function(k) sign(mean(x[[k]][ret], na.rm = TRUE)))  # sign on retreat
    data.frame(year = YEAR, sector = LABELS[s], period = PERIOD_FULL,
               cycle_complete = cc_all[[s]]$complete, last_day = cc_all[[s]]$last_day,
               day_of_max = cc_all[[s]]$dmax,
               mean_retreat_trend = mr["trend"], mean_retreat_amp = mr["amplitude"],
               mean_retreat_phase = mr["phase"], mean_retreat_resid = mr["raw"],
               dominant_retreat = comps[which.max(mr)],
               sign_retreat_trend = sg["trend"], sign_retreat_amp = sg["amplitude"],
               sign_retreat_phase = sg["phase"],
               peak_trend = pk["trend"], peak_amp = pk["amplitude"],
               peak_phase = pk["phase"], peak_resid = pk["raw"],
               dominant_peak = comps[which.max(pk)],
               mean_full_trend = mf["trend"], mean_full_amp = mf["amplitude"],
               mean_full_phase = mf["phase"], mean_full_resid = mf["raw"],
               dominant_full = comps[which.max(mf)],
               row.names = NULL)
  }))
  summ[] <- lapply(summ, function(v) if (is.numeric(v)) round(v, 2) else v)
  
  cat(sprintf("\n-- %d: dominant component (%% of amplitude, %s fit) --\n", YEAR, PERIOD_FULL))
  print(summ[, c("sector", "day_of_max", "dominant_retreat", "dominant_peak", "cycle_complete")],
        row.names = FALSE)
  cat("\n-- retreat-limb mean |contribution|, with sign of the retreat mean --\n")
  print(summ[, c("sector", "mean_retreat_trend", "mean_retreat_amp", "mean_retreat_phase",
                 "mean_retreat_resid", "sign_retreat_amp", "sign_retreat_phase")], row.names = FALSE)
  cat("\n-- peak |contribution| --\n")
  print(summ[, c("sector", "peak_trend", "peak_amp", "peak_phase", "peak_resid")], row.names = FALSE)
  
  f3 <- file.path(TAB_DIR, sprintf("t32_component_magnitude_%d.csv", YEAR))
  write.csv(summ, f3, row.names = FALSE)
  message("Wrote ", f3)
  summ
}

all <- do.call(rbind, lapply(YEARS, run_year))
f_all <- file.path(TAB_DIR, "t32_component_magnitude_all.csv")
write.csv(all, f_all, row.names = FALSE)
message("\nWrote ", f_all)

cat("\n==== dominant component on the retreat limb, by year ====\n")
print(reshape(all[, c("year", "sector", "dominant_retreat")],
              idvar = "sector", timevar = "year", direction = "wide"), row.names = FALSE)