# =============================================================================
# APAC Diagnostic Plots
# Neal's advice: understand the model before making claims with it
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# --- Load data ----------------------------------------------------------------
daily  <- read.csv("~/Research/repos/sea-ice-phase/scripts/R/Ch3/data/daily_fitted.csv")
annual <- read.csv("~/Research/repos/sea-ice-phase/scripts/R/Ch3/data/annual_params.csv")

daily$Date  <- as.Date(daily$Date)
daily$sector <- gsub("SIE_", "", daily$sector)
annual$sector <- gsub("SIE_", "", annual$sector)

sectors <- unique(daily$sector)

# Sector color palette
sec_colors <- c(
  "Weddell"               = "#1f78b4",
  "Amundsen_Bellingshausen" = "#e31a1c",
  "Ross"                  = "#33a02c",
  "East_Antarctica"       = "#ff7f00",
  "King_Haakon"           = "#6a3d9a",
  "circumpolar"           = "#000000"
)

# =============================================================================
# 1. FITTED VS OBSERVED AMPLITUDE
# =============================================================================
p_amp <- ggplot(annual, aes(x = amplitude_raw_yr, y = amplitude_fitted, color = sector)) +
  geom_point(alpha = 0.7, size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
  facet_wrap(~sector, scales = "free", ncol = 3) +
  scale_color_manual(values = sec_colors) +
  labs(
    title = "Fitted vs Observed Amplitude",
    subtitle = "Points should fall close to the 1:1 line if the model tracks observations well",
    x = "Raw observed amplitude (million km²)",
    y = "APAC fitted amplitude (million km²)"
  ) +
  theme_bw() +
  theme(legend.position = "none")

# =============================================================================
# 2. FITTED VS OBSERVED PHASE (DOY of maximum)
# =============================================================================
p_phase <- ggplot(annual, aes(x = max_doy_raw, y = max_doy_fitted, color = sector)) +
  geom_point(alpha = 0.7, size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
  facet_wrap(~sector, scales = "free", ncol = 3) +
  scale_color_manual(values = sec_colors) +
  labs(
    title = "Fitted vs Observed Phase (DOY of maximum)",
    subtitle = "Divergence from 1:1 line shows where fitted and raw phase disagree",
    x = "Raw observed DOY of maximum",
    y = "APAC fitted DOY of maximum"
  ) +
  theme_bw() +
  theme(legend.position = "none")

# =============================================================================
# 3. RESIDUAL HISTOGRAMS AT EACH MODEL STAGE
# =============================================================================
resid_long <- daily %>%
  select(Date, sector, residual_amp, residual_phase, residual_apac, raw_anomaly) %>%
  pivot_longer(
    cols = c(residual_amp, residual_phase, residual_apac, raw_anomaly),
    names_to = "model_stage",
    values_to = "residual"
  ) %>%
  mutate(model_stage = recode(model_stage,
                              raw_anomaly    = "1. TAC residual",
                              residual_amp   = "2. Amplitude model residual",
                              residual_phase = "3. Phase model residual",
                              residual_apac  = "4. APAC residual"
  ))

p_hist <- ggplot(resid_long %>% filter(!is.na(residual)), 
                 aes(x = residual, fill = model_stage)) +
  geom_histogram(bins = 60, alpha = 0.8, color = "white") +
  facet_grid(sector ~ model_stage, scales = "free") +
  labs(
    title = "Residual distributions at each model stage",
    subtitle = "Do residuals become more normal as we add amplitude and phase adjustment?",
    x = "Residual (million km²)",
    y = "Count"
  ) +
  theme_bw() +
  theme(legend.position = "none",
        strip.text.x = element_text(size = 7),
        strip.text.y = element_text(size = 7))

# =============================================================================
# 4. Q-Q PLOTS AT EACH MODEL STAGE
# =============================================================================
qq_plots <- list()
stages <- c("raw_anomaly", "residual_amp", "residual_phase", "residual_apac")
stage_labels <- c("TAC residual", "Amplitude residual", "Phase residual", "APAC residual")

for (i in seq_along(stages)) {
  d <- daily %>%
    filter(!is.na(.data[[stages[i]]])) %>%
    select(sector, value = all_of(stages[i]))
  
  qq_plots[[i]] <- ggplot(d, aes(sample = value, color = sector)) +
    stat_qq(alpha = 0.3, size = 0.5) +
    stat_qq_line(linewidth = 0.8) +
    facet_wrap(~sector, scales = "free", ncol = 3) +
    scale_color_manual(values = sec_colors) +
    labs(
      title = paste("Q-Q Plot:", stage_labels[i]),
      x = "Theoretical quantiles",
      y = "Sample quantiles"
    ) +
    theme_bw() +
    theme(legend.position = "none")
}

# =============================================================================
# 5. VOLATILITY SEASONAL CYCLE BY SECTOR
# =============================================================================
vol_seasonal <- daily %>%
  filter(!is.na(volatility)) %>%
  mutate(era = ifelse(Year >= 2016, "2016+", "1980-2015")) %>%
  group_by(sector, DOY, era) %>%
  summarise(mean_vol = mean(volatility, na.rm = TRUE), .groups = "drop")

p_vol <- ggplot(vol_seasonal, aes(x = DOY, y = mean_vol, color = era)) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~sector, scales = "free_y", ncol = 3) +
  scale_color_manual(values = c("1980-2015" = "steelblue", "2016+" = "firebrick")) +
  labs(
    title = "Mean seasonal volatility cycle by era",
    subtitle = "Has day-to-day variability changed post-2016?",
    x = "Day of year",
    y = "Mean volatility (million km²)",
    color = "Era"
  ) +
  theme_bw()

# =============================================================================
# 6. RESIDUAL TIME SERIES — HAS APAC RESIDUAL CHANGED POST-2016?
# =============================================================================
annual_resid <- daily %>%
  group_by(sector, Year) %>%
  summarise(
    mean_apac_resid = mean(residual_apac, na.rm = TRUE),
    sd_apac_resid   = sd(residual_apac, na.rm = TRUE),
    .groups = "drop"
  )

p_resid_ts <- ggplot(annual_resid, aes(x = Year, y = mean_apac_resid, color = sector)) +
  geom_line(linewidth = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = 2016, linetype = "dotted", color = "red") +
  facet_wrap(~sector, scales = "free_y", ncol = 3) +
  scale_color_manual(values = sec_colors) +
  labs(
    title = "Annual mean APAC residual over time",
    subtitle = "Red dotted line marks 2016. Is there a systematic shift in the residual?",
    x = "Year",
    y = "Mean APAC residual (million km²)"
  ) +
  theme_bw() +
  theme(legend.position = "none")

# =============================================================================
# PRINT ALL TO GUI
# =============================================================================
print(p_amp)
print(p_phase)
print(p_hist)
for (p in qq_plots) print(p)
print(p_vol)
print(p_resid_ts)

