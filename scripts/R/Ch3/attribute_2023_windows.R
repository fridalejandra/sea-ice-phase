# =============================================================================
# plot_fig7_sectors.R — §3.5 component anatomy, one cycle, all sectors,
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
# THE RESIDUAL. The black line (`raw`) is Extent − fully adjusted APAC fit,
# i.e. the raw APAC anomaly of Sect. 2.2.4 / 3.5 -- the residual -- in % of
# amplitude. It is now also drawn as a filled band so it reads against the
# components, and it is checked against the `residual_apac` column of the
# daily file (if present): the two agree exactly on the ADVANCE limb (before
# the day of maximum, where the blend weight is zero) and differ on the retreat
# limb by the blend, which replaces the calendar-year amplitude (stepping on
# 1 January) with a ramp from this year's to next year's. The console prints
# the maximum difference on the advance limb (must be ~0), the size of the
# blend effect on the retreat limb, and the residual's SD for the cycle in
# both units.
#
# The cycle runs 21 Feb (year Y) to 20 Feb (Y+1); day 0 is the minimum. If the
# record ends before 20 Feb Y+1 the cycle is INCOMPLETE and the script says so:
# the retreat-limb statistics for that year are over a shorter window.
#
# The retreat-limb blend is REQUIRED: the raw *_pct columns are normalised by
# each row's calendar-year amplitude, which steps on 1 January; the blend
# removes that discontinuity.
#
# Layout (2026-09-21 facelift, to match Figs 3-8): no figure title; panel
# titles "(a) Weddell" bold, black, left-aligned; sector order as in the rest
# of the chapter (Weddell, A-B, Ross, EA, KH, circumpolar); one legend below
# the grid; L-shaped axes. The year is in the file name and the caption, not
# repeated on six panels.
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
COMBINED <- c(2016, 2023)  # the two-year figure (Fig. 10): rows of sectors, one block per year
PERIOD_HR   <- "HR2018"    # used for panel (a) when the year is inside it
PERIOD_FULL <- "FULL"      # sector grid always; panel (a) when year > 2018
TREND_COL   <- "trend_component"
GRID_IAC    <- TRUE        # invariant cycle on every grid panel
GRID_SHARED_Y <- FALSE     # FALSE = each panel its own axis (keep FALSE)
RESID_FILL  <- TRUE        # shade the residual between the black line and zero
GRID_EST    <- FALSE       # 11-day mean of the residual on the grid panels? It is the
# same series smoothed; keep it only in panel (a), the H&R
# reproduction, where it matches their figure.
JAN1_DAY    <- 314         # cycle day of 1 January (21 Feb + 314 d); the amplitude step

if (!file.exists(DAILY)) stop("Input not found: ", DAILY)
for (d in c(FIG_DIR, TAB_DIR)) dir.create(d, showWarnings = FALSE, recursive = TRUE)
message("DAILY = ", DAILY)

# chapter order (Figs 3-8): Weddell, A-B, Ross, EA, KH, circumpolar
SECTORS <- c("SIE_Weddell", "SIE_Amundsen_Bellingshausen", "SIE_Ross",
             "SIE_East_Antarctica", "SIE_King_Haakon", "SIE_circumpolar")
LABELS  <- c(SIE_circumpolar = "Circumpolar total", SIE_Weddell = "Weddell",
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
HAS_RESID <- "residual_apac" %in% names(d_raw)
message("residual_apac column in file: ", HAS_RESID,
        if (HAS_RESID) "  (will be checked against Extent - blended APAC)" else "")

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
  
  tv        <- g[[TREND_COL]]
  resid_km2 <- g$Extent - fapac                 # THE residual, 10^6 km^2 (blended scaling)
  raw       <- resid_km2 / ampY * 100           # same, % of the cycle-year amplitude
  est       <- as.numeric(stats::filter(raw, rep(1/11, 11), sides = 2))
  
  # check against the file's own residual column. The blend starts at dmax
  # (w = 0 before it), so identity is expected only on the ADVANCE limb; the
  # difference on the retreat limb is the blend itself and is reported too.
  resid_check <- NA_real_; blend_effect <- NA_real_
  if (HAS_RESID) {
    adv <- cd < dmax
    ret <- cd >= dmax & cd < JAN1_DAY
    resid_check  <- max(abs(resid_km2[adv] - g$residual_apac[adv]), na.rm = TRUE)
    blend_effect <- max(abs(resid_km2[ret] - g$residual_apac[ret]), na.rm = TRUE)
  }
  
  list(
    dmax = dmax, ampY = ampY, last_day = max(cd), complete = complete,
    resid_check = resid_check, blend_effect = blend_effect,
    resid_sd_km2 = sd(resid_km2, na.rm = TRUE),
    resid_sd_pct = sd(raw, na.rm = TRUE),
    resid_peak_date = g$Date[which.max(abs(resid_km2))],
    resid_peak_km2  = resid_km2[which.max(abs(resid_km2))],
    df = data.frame(
      date          = g$Date,
      cycle_day     = cd,
      invariant_km2 = g$iac_notrend - mean(g$iac_notrend),   # 10^6 km^2, centred
      trend         = tv / ampY * 100,                        # % of amplitude
      amplitude     = (famp - g$iac_notrend - tv) / ampY * 100,
      phase         = (fapac - famp) / ampY * 100,
      raw           = raw,                                    # residual, % of amplitude
      resid_km2     = resid_km2,                              # residual, 10^6 km^2
      est           = est))
}

COL <- c(invariant = "#A8D8EA", trend = "#1B7A1B", amplitude = "#14148C",
         phase = "#E8140C", raw = "black", est = "#FFA500")
LWD <- c(invariant = 5, trend = 3, amplitude = 3, phase = 3.5, raw = 1, est = 2)
LAB <- c(invariant = "Invariant annual cycle", trend = "Trend component",
         amplitude = "Amplitude component", phase = "Phase component",
         raw = "Raw APAC anomaly (residual)", est = "Residual, 11-day mean")
YLAB <- "Anomaly for sea ice extent"
FILL_COL <- adjustcolor("grey30", alpha.f = 0.22)

draw_panel <- function(cc, title, ylim, show_iac, xlab = "Day of the cycle", show_est = TRUE) {
  plot(NA, NA, xlim = c(0, 365), ylim = ylim, xaxt = "n", bty = "l", las = 1,
       xlab = xlab, ylab = YLAB, main = "")
  axis(1, at = c(0, 100, 200, 300))
  abline(h = 0, lty = 2, col = "grey40")
  if (show_iac) lines(cc$cycle_day, cc$invariant_km2, col = COL["invariant"], lwd = LWD["invariant"])
  # the residual as a band first, so the components sit on top of it
  if (RESID_FILL) {
    ok <- !is.na(cc$raw)
    polygon(c(cc$cycle_day[ok], rev(cc$cycle_day[ok])),
            c(cc$raw[ok], rep(0, sum(ok))), col = FILL_COL, border = NA)
  }
  for (k in c("trend", "amplitude", "phase", "raw", if (show_est) "est"))
    lines(cc$cycle_day, cc[[k]], col = COL[k], lwd = LWD[k])
  # bold, black, left-aligned panel title -- matches Figs 3-8
  title(main = title, adj = 0, font.main = 2, cex.main = 1.15, col.main = "black", line = 0.8)
}
add_legend <- function(keys, cex, where = "bottom", ncol = 2) {
  legend(where, ncol = ncol, bty = "n", cex = cex,
         col = COL[keys], lwd = LWD[keys],
         fill = ifelse(keys == "raw" & RESID_FILL, FILL_COL, NA),
         border = NA, legend = LAB[keys])
}
panel_cols <- c(if (GRID_IAC) "invariant_km2", "trend", "amplitude", "phase", "raw")
panel_ylim <- function(x) range(unlist(x$df[panel_cols]), na.rm = TRUE) + c(-1, 1)
grid_keys  <- c(if (GRID_IAC) "invariant", "trend", "amplitude", "phase", "raw", if (GRID_EST) "est")
comps      <- c("trend", "amplitude", "phase", "raw")

# ── one year ────────────────────────────────────────────────────────────────
run_year <- function(YEAR) {
  message(sprintf("\n==================== %d ====================", YEAR))
  period_a <- if (YEAR <= hr_last_year) PERIOD_HR else PERIOD_FULL
  fit_lab  <- if (period_a == PERIOD_HR) sprintf("fit 1979-%d", hr_last_year) else "full-record fit"
  
  # panel (a): the H&R reproduction, kept as its own file
  ca <- components_for("SIE_circumpolar", YEAR, period_a)
  inc <- if (ca$complete) "" else sprintf("  [cycle ends day %d]", ca$last_day)
  if (!ca$complete)
    message(sprintf("  NOTE: %d cycle is INCOMPLETE — record ends at cycle day %d of 365.", YEAR, ca$last_day))
  f1 <- file.path(FIG_DIR, sprintf("fig07a_circumpolar_%d.png", YEAR))
  png(f1, width = 9.5, height = 7, units = "in", res = 300)
  par(mar = c(4.2, 4.6, 2.5, 1), family = "sans")
  draw_panel(ca$df, sprintf("(a)  Circumpolar total, %d  (%s)%s", YEAR, fit_lab, inc),
             ylim = panel_ylim(ca), show_iac = TRUE)
  add_legend(c("invariant", "trend", "amplitude", "phase", "raw", "est"), cex = 0.85)
  dev.off()
  message("Wrote ", f1)
  
  # grid: 2 x 3 panels + one legend row underneath
  cc_all <- lapply(SECTORS, function(s) components_for(s, YEAR, PERIOD_FULL))
  names(cc_all) <- SECTORS
  yl_shared <- range(unlist(lapply(cc_all, panel_ylim)))
  f2 <- file.path(FIG_DIR, sprintf("fig07_sectors_%d.png", YEAR))
  png(f2, width = 16, height = 10, units = "in", res = 300)
  layout(matrix(c(1:6, 7, 7, 7), nrow = 3, byrow = TRUE), heights = c(1, 1, 0.14))
  par(mar = c(4.2, 4.6, 2.8, 1), family = "sans")
  for (i in seq_along(SECTORS)) {
    x <- cc_all[[SECTORS[i]]]
    draw_panel(x$df, sprintf("(%s)  %s", letters[i], LABELS[SECTORS[i]]),
               ylim = if (GRID_SHARED_Y) yl_shared else panel_ylim(x),
               show_iac = GRID_IAC, show_est = GRID_EST,
               xlab = if (i > 3) "Day of the cycle" else "")
  }
  par(mar = c(0, 0, 0, 0))
  plot.new()
  add_legend(grid_keys, cex = 1.0, where = "center", ncol = 3)
  dev.off()
  message("Wrote ", f2)
  
  # residual check + summary, per sector
  cat(sprintf("\n-- %d: the residual (Extent - blended APAC), by sector --\n", YEAR))
  rs <- do.call(rbind, lapply(SECTORS, function(s) {
    x <- cc_all[[s]]
    data.frame(sector = LABELS[s],
               sd_km2 = round(x$resid_sd_km2, 3), sd_pct_amp = round(x$resid_sd_pct, 1),
               peak_km2 = round(x$resid_peak_km2, 3), peak_date = as.character(x$resid_peak_date),
               max_diff_vs_csv_advance = if (HAS_RESID) signif(x$resid_check, 2) else NA,
               blend_effect_retreat    = if (HAS_RESID) signif(x$blend_effect, 2) else NA,
               row.names = NULL)
  }))
  print(rs, row.names = FALSE)
  if (HAS_RESID && any(rs$max_diff_vs_csv_advance > 1e-6, na.rm = TRUE))
    message("  WARNING: residual differs from residual_apac on the ADVANCE limb, where the blend is off -- ",
            "the CSV's residual_apac is not Extent - fitted APAC with the row's own amplitude/minimum.")
  
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
               resid_sd_km2 = cc_all[[s]]$resid_sd_km2,
               resid_sd_pct = cc_all[[s]]$resid_sd_pct,
               row.names = NULL)
  }))
  summ[] <- lapply(summ, function(v) if (is.numeric(v)) round(v, 3) else v)
  
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
  list(summ = summ, cc_all = cc_all)
}

res <- lapply(YEARS, run_year); names(res) <- YEARS
all <- do.call(rbind, lapply(res, `[[`, "summ"))

# ── combined figure: the years in COMBINED stacked, sectors across ──────────
# 4 rows x 3 cols for two years (two rows per year), one legend row at the
# bottom. Full-page figure. Each panel keeps its own y-axis.
if (length(COMBINED) > 0 && all(as.character(COMBINED) %in% names(res))) {
  nyr  <- length(COMBINED)
  npan <- nyr * length(SECTORS)
  fC <- file.path(FIG_DIR, sprintf("fig10_anatomy_%s.png", paste(COMBINED, collapse = "_")))
  png(fC, width = 13, height = 4.2 * nyr * 2 + 0.6, units = "in", res = 300)
  layout(matrix(c(1:npan, rep(npan + 1, 3)), nrow = nyr * 2 + 1, byrow = TRUE),
         heights = c(rep(1, nyr * 2), 0.12))
  par(mar = c(4.2, 4.6, 2.8, 1), family = "sans")
  k <- 0
  for (yr in COMBINED) {
    cc_all <- res[[as.character(yr)]]$cc_all
    for (i in seq_along(SECTORS)) {
      k <- k + 1
      x <- cc_all[[SECTORS[i]]]
      last_row <- (yr == tail(COMBINED, 1)) && i > 3
      draw_panel(x$df, sprintf("(%s)  %s, %d", letters[k], LABELS[SECTORS[i]], yr),
                 ylim = panel_ylim(x), show_iac = GRID_IAC, show_est = GRID_EST,
                 xlab = if (last_row) "Day of the cycle" else "")
    }
  }
  par(mar = c(0, 0, 0, 0)); plot.new()
  add_legend(grid_keys, cex = 1.0, where = "center", ncol = 3)
  dev.off()
  message("Wrote ", fC)
}
f_all <- file.path(TAB_DIR, "t32_component_magnitude_all.csv")
write.csv(all, f_all, row.names = FALSE)
message("\nWrote ", f_all)

cat("\n==== dominant component on the retreat limb, by year ====\n")
print(reshape(all[, c("year", "sector", "dominant_retreat")],
              idvar = "sector", timevar = "year", direction = "wide"), row.names = FALSE)