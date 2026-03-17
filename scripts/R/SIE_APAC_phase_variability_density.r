## ============================================================
## Seasonal reference frames + phase-organization diagnostic
## ============================================================

library(tidyverse)
library(lubridate)
library(mgcv)
library(viridis)
library(hexbin)

theme_set(theme_bw(base_size = 12))

## ------------------------------------------------------------
## 1. Load and prepare data
## ------------------------------------------------------------

df <- read_csv(
  "/Users/fridaperez/Developer/repos/sea-ice-phase/scripts/R/Sea_Ice_Sheets/Bootstrap79-24.csv",
  col_types = cols(
    Date   = col_character(),
    Extent = col_double()
  )
) %>%
  mutate(
    Date   = mdy(Date),
    Year   = year(Date),
    DOY    = yday(Date),
    Extent = as.numeric(Extent)
  )

## ------------------------------------------------------------
## 2. Traditional Annual Cycle (TAC)
## ------------------------------------------------------------

tac <- df %>%
  group_by(DOY) %>%
  summarise(
    TAC = mean(Extent, na.rm = TRUE),
    .groups = "drop"
  )

df <- df %>% left_join(tac, by = "DOY")

## ------------------------------------------------------------
## 3. Invariant Annual Cycle (IAC)
## ------------------------------------------------------------

gam_iac <- gam(
  Extent ~ s(DOY, bs = "cc", k = 25),
  data = df
)

df$IAC <- predict(gam_iac, newdata = df)

## ------------------------------------------------------------
## 4. Phase + amplitude preprocessing (EXACT match to your code)
## ------------------------------------------------------------

# Annual minimum date
yearly_min <- df %>%
  group_by(Year) %>%
  summarise(
    Date1 = Date[which.min(Extent)],
    .groups = "drop"
  ) %>%
  mutate(
    Date2 = lag(Date1),
    Date3 = lead(Date1)
  )

df <- df %>% left_join(yearly_min, by = "Year")

# Relative time since minimum (KEEP ROWWISE)
df <- df %>%
  rowwise() %>%
  mutate(
    t = case_when(
      Year == min(Year) ~ 365 - as.numeric(Date3 - Date),
      Date >= Date1     ~ as.numeric(Date - Date1),
      TRUE              ~ as.numeric(Date - Date2)
    )
  ) %>%
  ungroup()

# Normalize to phase
t_stats <- df %>%
  group_by(Year) %>%
  summarise(
    t_min = min(t, na.rm = TRUE),
    t_max = max(t, na.rm = TRUE),
    .groups = "drop"
  )

df <- df %>%
  left_join(t_stats, by = "Year") %>%
  mutate(
    phase = 365 * (t - t_min) / (t_max - t_min)
  )

# Amplitude normalization
yearly_amp <- df %>%
  group_by(Year) %>%
  summarise(
    min_extent = min(Extent, na.rm = TRUE),
    max_extent = max(Extent, na.rm = TRUE),
    amplitude  = max_extent - min_extent,
    .groups = "drop"
  )

df <- df %>%
  left_join(yearly_amp, by = "Year") %>%
  mutate(
    scaling = (Extent - min_extent) / amplitude
  )

## ------------------------------------------------------------
## 5. APAC (EXACT formulation)
## ------------------------------------------------------------

gam_apac <- gam(
  scaling ~ s(phase, bs = "cc", k = 100),
  data = df
)

df$APAC <- predict(gam_apac, newdata = df) * df$amplitude + df$min_extent

## ------------------------------------------------------------
## 6. APAC-based daily variability
## ------------------------------------------------------------

df <- df %>%
  mutate(
    dep_APAC = Extent - APAC,
    abs_dep  = abs(dep_APAC)
  )

## Drop days with undefined phase (edge years)
df <- df %>%
  filter(!is.na(phase))

## ------------------------------------------------------------
## 7. Era split
## ------------------------------------------------------------

df <- df %>%
  mutate(
    era = if_else(Year <= 2016, "Pre-2016", "Post-2016")
  )

## ------------------------------------------------------------
## 8. DEBUG scatter (keep this)
## ------------------------------------------------------------

ggsave(
  "DEBUG_phase_vs_absdep.png",
  ggplot(df, aes(phase, abs_dep)) +
    geom_point(alpha = 0.01) +
    facet_wrap(~ era),
  width = 7,
  height = 4,
  dpi = 200
)

## ------------------------------------------------------------
## 9. Phase–variability density plot (CORE NEW FIGURE)
## ------------------------------------------------------------

fig_phase_density <- ggplot(
  df,
  aes(x = phase, y = abs_dep)
) +
  geom_hex(bins = 50) +
  facet_wrap(~ era, ncol = 1) +
  scale_fill_viridis_c(
    name = "Days",
    trans = "log10"
  ) +
  scale_x_continuous(
    breaks = c(0, 90, 180, 270, 365),
    labels = c("Min", "Advance", "Max", "Retreat", "Min")
  ) +
  labs(
    x = "Seasonal phase (APAC-aligned)",
    y = expression("|Daily APAC departure| (10"^6*" km"^2*")"),
    title = "Temporal organization of daily Antarctic sea ice variability",
    subtitle = "Daily variability organized by seasonal phase, before and after 2016"
  ) +
  theme(
    legend.position = "right",
    strip.background = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title.position = "plot"
  )

ggsave(
  "Fig_APAC_phase_variability_density_pre_post.png",
  fig_phase_density,
  width = 8,
  height = 6,
  dpi = 300
)

