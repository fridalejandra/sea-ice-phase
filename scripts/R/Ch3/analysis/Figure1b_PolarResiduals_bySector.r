## ============================================================
## Figure 1b — Polar view of daily APAC residuals
## FULLY STANDALONE SCRIPT (POSTER-READY)
## ============================================================

library(tidyverse)
library(lubridate)
library(mgcv)
library(zoo)
library(viridis)

theme_set(theme_bw(base_size = 11))

## ------------------------------------------------------------
## USER CONTROLS (EDIT HERE)
## ------------------------------------------------------------

YEARS_SHOW <- c(2001, 2014, 2016, 2018, 2020, 2022, 2024)

## ------------------------------------------------------------
## Paths
## ------------------------------------------------------------

DATA_PATH <- "/Users/fridaperez/Developer/repos/sea-ice-phase/scripts/R/Sea_Ice_Sheets/SIE_daily_sector_and_circumpolar_million_km2.csv"
OUT_DIR   <- "/Users/fridaperez/Desktop/Clic_Ch3/Figure1_polar_final"

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

## ------------------------------------------------------------
## 1. Load and reshape data (WIDE → LONG)
## ------------------------------------------------------------

df <- read_csv(DATA_PATH) %>%
  pivot_longer(
    cols = starts_with("SIE_"),
    names_to = "Sector",
    values_to = "Extent"
  ) %>%
  mutate(
    Sector = str_remove(Sector, "^SIE_"),
    Date   = mdy(Date),
    Year   = year(Date),
    Extent = as.numeric(Extent)
  ) %>%
  filter(!is.na(Extent))

stopifnot(nrow(df) > 0)

## ------------------------------------------------------------
## 2. APAC MODEL (UNCHANGED)
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
  mutate(phase = 365 * (t - t_min) / (t_max - t_min))

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
  mutate(scaling = (Extent - min_extent) / amplitude)

df <- df %>%
  group_by(Sector) %>%
  group_modify(~{
    g <- gam(scaling ~ s(phase, bs = "cc", k = 100), data = .x)
    .x$APAC <- predict(g, newdata = .x) * .x$amplitude + .x$min_extent
    .x
  }) %>%
  ungroup() %>%
  mutate(res_APAC = Extent - APAC)

## ------------------------------------------------------------
## 3. Remove Feb 29 + define DOY365
## ------------------------------------------------------------

df <- df %>%
  filter(!(month(Date) == 2 & day(Date) == 29)) %>%
  mutate(
    DOY_raw = yday(Date),
    DOY365  = if_else(leap_year(Date) & DOY_raw > 59, DOY_raw - 1L, DOY_raw)
  )

## ------------------------------------------------------------
## 4. Subset to SELECTED years + drop circumpolar
## ------------------------------------------------------------

df_polar <- df %>%
  filter(
    Year %in% YEARS_SHOW,
    Sector != "circumpolar"
  ) %>%
  mutate(
    Year  = factor(Year, levels = YEARS_SHOW),
    theta = 2 * pi * (DOY365 - 1) / 365
  )

## ------------------------------------------------------------
## 5. Close loops for polar paths
## ------------------------------------------------------------

df_polar <- df_polar %>%
  group_by(Sector, Year) %>%
  arrange(DOY365) %>%
  bind_rows(slice(., 1) %>% mutate(theta = 2 * pi)) %>%
  ungroup()

## ------------------------------------------------------------
## 6. Radial limits + rings
## ------------------------------------------------------------

r_lim   <- max(abs(df_polar$res_APAC), na.rm = TRUE)
r_rings <- pretty(c(-r_lim, r_lim), 5)

## ------------------------------------------------------------
## Sector labels
## ------------------------------------------------------------

sector_labs <- c(
  "Amundsen_Bellingshausen" = "ABS",
  "East_Antarctica"        = "EA",
  "King_Haakon"            = "KH",
  "Ross"                   = "ROSS",
  "Weddell"                = "WED"
)

## ------------------------------------------------------------
## 7. POLAR FIGURE (POSTER-TIGHT)
## ------------------------------------------------------------

p_polar <- ggplot(
  df_polar,
  aes(x = theta, y = res_APAC, group = interaction(Sector, Year))
) +
  geom_hline(
    yintercept = r_rings,
    color = "grey85",
    linewidth = 0.3
  ) +
  geom_path(
    aes(color = Year),
    alpha = 0.9,
    linewidth = 0.9
  ) +
  facet_wrap(
    ~ Sector,
    ncol = 3,
    labeller = labeller(Sector = sector_labs)
  ) +
  scale_x_continuous(
    limits = c(0, 2*pi),
    breaks = 2 * pi *
      ((yday(ymd(paste0("2001-", 1:12, "-15"))) - 1) / 365),
    labels = month.abb
  ) +
  scale_y_continuous(
    limits = c(-r_lim, r_lim),
    expand = expansion(mult = 0.08)
  ) +
  scale_color_viridis_d(option = "turbo", name = "Year") +
  coord_polar(start = -pi/2) +
  theme_void(base_size = 11) +
  theme(
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    strip.text = element_text(face = "bold", size = 11, margin = margin(b = 2)),
    panel.spacing = unit(0.5, "lines"),
    panel.grid.major.x = element_line(color = "grey70", linewidth = 0.4),
    axis.text.x = element_text(size = 8, color = "grey40"),
    axis.text.y = element_blank(),
    axis.ticks  = element_blank(),
    legend.title = element_text(size = 9),
    legend.text  = element_text(size = 8),
    plot.margin = margin(10, 40, 10, 10),
    legend.position = "right"
  )

ggsave(
  file.path(OUT_DIR, "Figure1b_polar_APAC_residuals_faceted.png"),
  p_polar,
  width  = 18,
  height = 12,
  dpi    = 300
)
