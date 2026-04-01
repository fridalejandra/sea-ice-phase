# ============================================================
# SECTORAL INVARIANT ANNUAL CYCLE + PHASE SCALAR + AMPLITUDE
# Input file columns expected:
# Date, SIE_circumpolar, SIE_Weddell, SIE_Amundsen_Bellingshausen,
# SIE_Ross, SIE_East_Antarctica, SIE_King_Haakon
# ============================================================
library(dplyr)
library(tidyr)
library(stringr)
library(lubridate)
library(ggplot2)
library(mgcv)

infile <- "~/Research/repos/sea-ice-phase/scripts/R/Sea_Ice_Sheets/SIE_daily_sector_and_circumpolar_million_km2.csv"
outdir <- "/Users/fridaperez/Desktop/sector_phase_outputs"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# -----------------------------
# 1. LOAD + RESHAPE DATA
# -----------------------------
df_wide <- read.csv(infile, stringsAsFactors = FALSE)

df_long <- df_wide %>%
  mutate(Date = mdy(Date)) %>%
  filter(!is.na(Date)) %>%
  pivot_longer(
    cols = starts_with("SIE_"),
    names_to = "sector",
    values_to = "Extent"
  ) %>%
  mutate(
    sector = str_remove(sector, "^SIE_"),
    Extent = as.numeric(Extent),
    Year = year(Date),
    Month = month(Date),
    Day = day(Date)
  ) %>%
  filter(!is.na(Extent)) %>%
  filter(!(Month == 2 & Day == 29)) %>%
  mutate(DOY = yday(Date)) %>%
  arrange(sector, Date)

print(df_long %>% count(sector))

# -----------------------------
# 2. COMPUTE IAC PER SECTOR
# -----------------------------
df_long <- df_long %>%
  group_by(sector) %>%
  group_modify(~{
    mod <- gam(Extent ~ s(DOY, bs = "cc", k = 25),
               data = .x, method = "REML")
    .x$IAC <- predict(mod, newdata = .x)
    .x
  }) %>%
  ungroup()

# -----------------------------
# 3. BUILD IAC TABLE PER SECTOR
# -----------------------------
iac_df <- df_long %>%
  group_by(sector, DOY) %>%
  summarise(
    IAC = mean(IAC, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(sector) %>%
  mutate(
    IAC_norm = (IAC - min(IAC, na.rm = TRUE)) /
      (max(IAC, na.rm = TRUE) - min(IAC, na.rm = TRUE))
  ) %>%
  ungroup()

write.csv(iac_df, file.path(outdir, "sector_IAC_daily.csv"), row.names = FALSE)

# -----------------------------
# 4. HELPER: CIRCULAR SHIFT
# -----------------------------
shift_vec <- function(x, lag_days) {
  n <- length(x)
  lag_days <- lag_days %% n
  if (lag_days == 0) return(x)
  c(tail(x, n - lag_days), head(x, lag_days))
}

# -----------------------------
# 5. ANNUAL PHASE SCALAR + AMPLITUDE
# -----------------------------
lags_to_test <- -40:40

phase_amp_sector <- df_long %>%
  group_by(sector, Year) %>%
  group_modify(~{
    sec_name <- .y$sector[[1]]
    
    dat <- .x %>%
      group_by(DOY) %>%
      summarise(Extent = mean(Extent, na.rm = TRUE), .groups = "drop") %>%
      arrange(DOY)
    
    if (nrow(dat) < 300) {
      return(data.frame(
        phase_scalar = NA_real_,
        sse_min = NA_real_,
        amplitude_range = NA_real_,
        min_doy = NA_real_,
        max_doy = NA_real_,
        min_extent = NA_real_,
        max_extent = NA_real_
      ))
    }
    
    dat2 <- dat %>%
      left_join(
        iac_df %>% filter(sector == sec_name) %>% select(DOY, IAC, IAC_norm),
        by = "DOY"
      )
    
    obs_min <- min(dat2$Extent, na.rm = TRUE)
    obs_max <- max(dat2$Extent, na.rm = TRUE)
    obs_rng <- obs_max - obs_min
    
    if (is.na(obs_rng) || obs_rng == 0) {
      return(data.frame(
        phase_scalar = NA_real_,
        sse_min = NA_real_,
        amplitude_range = NA_real_,
        min_doy = NA_real_,
        max_doy = NA_real_,
        min_extent = NA_real_,
        max_extent = NA_real_
      ))
    }
    
    obs_norm <- (dat2$Extent - obs_min) / obs_rng
    ref_norm <- dat2$IAC_norm
    
    sse_vals <- sapply(lags_to_test, function(L) {
      shifted_ref <- shift_vec(ref_norm, L)
      sum((obs_norm - shifted_ref)^2, na.rm = TRUE)
    })
    
    best_lag <- lags_to_test[which.min(sse_vals)]
    
    data.frame(
      phase_scalar = best_lag,
      sse_min = min(sse_vals, na.rm = TRUE),
      amplitude_range = obs_rng,
      min_doy = dat2$DOY[which.min(dat2$Extent)],
      max_doy = dat2$DOY[which.max(dat2$Extent)],
      min_extent = obs_min,
      max_extent = obs_max
    )
  }) %>%
  ungroup()

write.csv(
  phase_amp_sector,
  file.path(outdir, "sector_annual_phase_amplitude.csv"),
  row.names = FALSE
)

print(head(phase_amp_sector, 12))

# -----------------------------
# 6. CLEAN FIGURES
# -----------------------------

# A. PHASE BY SECTOR
p_phase <- ggplot(
  phase_amp_sector,
  aes(x = Year, y = phase_scalar)
) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray45", linewidth = 0.4) +
  geom_line(linewidth = 0.6, color = "black") +
  geom_point(size = 1.4, color = "black") +
  geom_smooth(method = "loess", se = FALSE, span = 0.35, color = "red", linewidth = 0.8) +
  facet_wrap(~sector, scales = "free_y", ncol = 2) +
  labs(
    title = "Sectoral phase variability relative to the invariant annual cycle",
    subtitle = "Negative = ahead of phase; Positive = behind phase",
    x = "Year",
    y = "Phase shift (days)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold")
  )

# B. AMPLITUDE BY SECTOR
p_amp <- ggplot(
  phase_amp_sector,
  aes(x = Year, y = amplitude_range)
) +
  geom_line(linewidth = 0.6, color = "black") +
  geom_point(size = 1.4, color = "black") +
  geom_smooth(method = "loess", se = FALSE, span = 0.35, color = "red", linewidth = 0.8) +
  facet_wrap(~sector, scales = "free_y", ncol = 2) +
  labs(
    title = "Sectoral amplitude variability",
    subtitle = "Annual max-min range in sea ice extent",
    x = "Year",
    y = expression("Amplitude range (million km"^2*")")
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold")
  )

# C. PHASE VS AMPLITUDE BY SECTOR
p_scatter <- ggplot(
  phase_amp_sector,
  aes(x = amplitude_range, y = phase_scalar)
) +
  geom_point(size = 1.6, alpha = 0.8, color = "black") +
  geom_smooth(method = "lm", se = FALSE, color = "red", linewidth = 0.8) +
  facet_wrap(~sector, scales = "free", ncol = 2) +
  labs(
    title = "Sectoral relationship between phase and amplitude",
    x = expression("Amplitude range (million km"^2*")"),
    y = "Phase shift (days)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold")
  )

##### TESTS
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)

# -----------------------------
# A. seasonal envelope
# -----------------------------
seasonal_env <- df_long %>%
  group_by(sector, DOY) %>%
  summarise(
    mean_extent = mean(Extent, na.rm = TRUE),
    p10 = quantile(Extent, 0.10, na.rm = TRUE),
    p90 = quantile(Extent, 0.90, na.rm = TRUE),
    .groups = "drop"
  )

# optional ordering
sector_levels <- c("Amundsen_Bellingshausen", "Ross", "Weddell",
                   "East_Antarctica", "King_Haakon", "circumpolar")

phase_amp_sector$sector <- factor(phase_amp_sector$sector, levels = sector_levels)
seasonal_env$sector <- factor(seasonal_env$sector, levels = sector_levels)

# -----------------------------
# B. phase panel
# -----------------------------
p_phase_combo <- ggplot(
  phase_amp_sector,
  aes(x = Year, y = phase_scalar)
) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray45", linewidth = 0.4) +
  geom_line(linewidth = 0.5, color = "black") +
  geom_point(size = 1.1, color = "black") +
  geom_smooth(method = "loess", se = FALSE, span = 0.35, color = "red", linewidth = 0.8) +
  facet_wrap(~sector, scales = "free_y", ncol = 3) +
  labs(
    title = "Sectoral phase variability",
    subtitle = "Negative = ahead of phase; Positive = behind phase",
    x = "Year",
    y = "Phase shift (days)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold")
  )

# -----------------------------
# C. envelope panel
# -----------------------------
p_env_combo <- ggplot(
  seasonal_env,
  aes(x = DOY, y = mean_extent)
) +
  geom_ribbon(aes(ymin = p10, ymax = p90), fill = "steelblue", alpha = 0.25) +
  geom_line(linewidth = 0.8, color = "black") +
  facet_wrap(~sector, scales = "free_y", ncol = 3) +
  scale_x_continuous(
    breaks = c(1, 91, 182, 274, 365),
    labels = c("Jan", "Apr", "Jul", "Oct", "Dec")
  ) +
  labs(
    title = "Sectoral seasonal envelope",
    subtitle = "Black = mean seasonal cycle; blue band = 10th–90th percentile range",
    x = "Month",
    y = expression("Sea ice extent (million km"^2*")")
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold")
  )

# -----------------------------
# D. combine
# -----------------------------
combined_fig <- p_phase_combo / p_env_combo +
  plot_annotation(
    title = "Seasonal structure of Antarctic sea ice by sector"
  )

print(combined_fig)

ggsave(
  file.path(outdir, "combined_phase_envelope_by_sector.png"),
  combined_fig,
  width = 13,
  height = 11,
  dpi = 300
)

# D. OPTIONAL: OVERLAID NORMALIZED CYCLES FOR EXAMPLE YEARS
# picks one early, one near-zero, one late year per sector could be added later

print(p_phase)
print(p_amp)
print(p_scatter)

ggsave(file.path(outdir, "sector_phase_timeseries.png"), p_phase, width = 11, height = 8, dpi = 300)
ggsave(file.path(outdir, "sector_amplitude_timeseries.png"), p_amp, width = 11, height = 8, dpi = 300)
ggsave(file.path(outdir, "sector_phase_vs_amplitude.png"), p_scatter, width = 11, height = 8, dpi = 300)




# -----------------------------
# 7. OPTIONAL SUMMARY TABLES
# -----------------------------
summary_tbl <- phase_amp_sector %>%
  group_by(sector) %>%
  summarise(
    mean_phase = mean(phase_scalar, na.rm = TRUE),
    sd_phase = sd(phase_scalar, na.rm = TRUE),
    mean_amp = mean(amplitude_range, na.rm = TRUE),
    sd_amp = sd(amplitude_range, na.rm = TRUE),
    mean_sse = mean(sse_min, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(summary_tbl, file.path(outdir, "sector_phase_amplitude_summary.csv"), row.names = FALSE)
print(summary_tbl)