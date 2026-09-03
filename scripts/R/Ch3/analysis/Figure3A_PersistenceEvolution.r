## ============================================================
## Figure 3A — Frequency of extreme daily variability
## Annual number of days with |APAC residuals| above 75th percentile
## FULLY STANDALONE SCRIPT
## ============================================================

library(tidyverse)
library(lubridate)
library(mgcv)

theme_set(theme_bw(base_size = 12))

## ------------------------------------------------------------
## Paths
## ------------------------------------------------------------

DATA_PATH <- "/Users/fridaperez/Developer/repos/sea-ice-phase/scripts/R/Sea_Ice_Sheets/SIE_daily_sector_and_circumpolar_million_km2.csv"
OUT_DIR   <- "/Users/fridaperez/Desktop/Clic_Ch3/Figure3A_frequency"

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

## ------------------------------------------------------------
## 1. Load and reshape data
## ------------------------------------------------------------

df <- read_csv(DATA_PATH)

df <- df %>%
  pivot_longer(
    cols = starts_with("SIE_"),
    names_to = "Sector",
    values_to = "Extent"
  ) %>%
  mutate(
    Sector = str_remove(Sector, "^SIE_"),
    Date   = mdy(Date),
    Year   = year(Date),
    DOY    = yday(Date),
    Extent = as.numeric(Extent)
  ) %>%
  filter(!is.na(Extent))

stopifnot(nrow(df) > 0)

## ------------------------------------------------------------
## 2. APAC model (UNCHANGED)
## ------------------------------------------------------------

min_dates <- df %>%
  group_by(Sector, Year) %>%
  summarise(Date1 = Date[which.min(Extent)], .groups = "drop") %>%
  group_by(Sector) %>%
  mutate(
    Date2 = lag(Date1),
    Date3 = lead(Date1)
  ) %>%
  ungroup()

df <- df %>% left_join(min_dates, by = c("Sector", "Year"))

df <- df %>%
  rowwise() %>%
  mutate(
    t = case_when(
      Year == min(Year, na.rm = TRUE) ~ 365 - as.numeric(Date3 - Date),
      Date >= Date1                   ~ as.numeric(Date - Date1),
      TRUE                            ~ as.numeric(Date - Date2)
    )
  ) %>%
  ungroup()

t_stats <- df %>%
  group_by(Sector, Year) %>%
  summarise(
    t_min = min(t, na.rm = TRUE),
    t_max = max(t, na.rm = TRUE),
    .groups = "drop"
  )

df <- df %>%
  left_join(t_stats, by = c("Sector", "Year")) %>%
  mutate(
    phase = 365 * (t - t_min) / (t_max - t_min)
  )

amp <- df %>%
  group_by(Sector, Year) %>%
  summarise(
    min_extent = min(Extent, na.rm = TRUE),
    max_extent = max(Extent, na.rm = TRUE),
    amplitude  = max_extent - min_extent,
    .groups = "drop"
  )

df <- df %>%
  left_join(amp, by = c("Sector", "Year")) %>%
  mutate(
    scaling = (Extent - min_extent) / amplitude
  )

df <- df %>%
  group_by(Sector) %>%
  group_modify(~{
    g <- gam(
      scaling ~ s(phase, bs = "cc", k = 100),
      data = .x
    )
    .x$APAC <- predict(g, newdata = .x) * .x$amplitude + .x$min_extent
    .x
  }) %>%
  ungroup()

df <- df %>%
  mutate(res_APAC = Extent - APAC)

## ------------------------------------------------------------
## 3. Remove Feb 29
## ------------------------------------------------------------

df <- df %>%
  filter(!(month(Date) == 2 & day(Date) == 29))

## ------------------------------------------------------------
## 4. Define high-variability threshold (75th percentile)
## ------------------------------------------------------------

thresholds <- df %>%
  group_by(Sector) %>%
  summarise(
    q75 = quantile(abs(res_APAC), 0.75, na.rm = TRUE),
    .groups = "drop"
  )

df <- df %>%
  left_join(thresholds, by = "Sector") %>%
  mutate(
    high_var = abs(res_APAC) > q75
  )

## ------------------------------------------------------------
## 5. Annual frequency of high-variability days
## ------------------------------------------------------------

annual_freq <- df %>%
  group_by(Sector, Year) %>%
  summarise(
    n_high_var_days = sum(high_var, na.rm = TRUE),
    .groups = "drop"
  )

## ------------------------------------------------------------
## 6. FIGURE 3A — Time evolution
## ------------------------------------------------------------

p3A <- ggplot(
  annual_freq,
  aes(x = Year, y = n_high_var_days)
) +
  geom_line(color = "grey70", linewidth = 0.8) +
  geom_point(color = "#2166ac", size = 2) +
  facet_wrap(~ Sector, scales = "free_y") +
  labs(
    title = "Time evolution of extreme daily Antarctic sea ice variability",
    subtitle = "Annual number of days with |APAC residuals| above the 75th percentile",
    x = "Year",
    y = "Number of high-variability days"
  ) +
  theme(
    panel.grid.minor = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold")
  )

ggsave(
  file.path(OUT_DIR, "Figure3A_annual_high_variability_days_75th.png"),
  p3A,
  width = 11,
  height = 7,
  dpi = 300
)
