## ============================================================
## ABS vs Ross — Seasonal distribution of daily sea ice variability
## APAC-based, poster-safe, minimal
## ============================================================

library(tidyverse)
library(lubridate)
library(mgcv)
library(zoo)
library(viridis)

theme_set(theme_void(base_size = 12))

## ------------------------------------------------------------
## Paths
## ------------------------------------------------------------

DATA_PATH <- "/Users/fridaperez/Developer/repos/sea-ice-phase/scripts/R/Sea_Ice_Sheets/SIE_daily_sector_and_circumpolar_million_km2.csv"
OUT_DIR   <- "/Users/fridaperez/Desktop/Clic_Ch3/"

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

## ------------------------------------------------------------
## 1. Load, reshape, and KEEP ONLY ABS + Ross
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
  filter(
    !is.na(Extent),
    Sector %in% c("Amundsen_Bellingshausen", "Ross")
  )

stopifnot(nrow(df) > 0)

## ------------------------------------------------------------
## 2. Traditional Annual Cycle (for reference)
## ------------------------------------------------------------

df <- df %>% mutate(DOY = yday(Date))

tac <- df %>%
  group_by(Sector, DOY) %>%
  summarise(TAC = mean(Extent, na.rm = TRUE), .groups = "drop")

df <- df %>%
  left_join(tac, by = c("Sector", "DOY")) %>%
  mutate(anom_TAC = Extent - TAC)

## ------------------------------------------------------------
## 3. APAC model (UNCHANGED)
## ------------------------------------------------------------

min_dates <- df %>%
  group_by(Sector, Year) %>%
  summarise(Date1 = Date[which.min(Extent)], .groups = "drop") %>%
  group_by(Sector) %>%
  mutate(Date2 = lag(Date1), Date3 = lead(Date1)) %>%
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
  summarise(t_min = min(t), t_max = max(t), .groups = "drop")

df <- df %>%
  left_join(t_stats, by = c("Sector", "Year")) %>%
  mutate(phase = 365 * (t - t_min) / (t_max - t_min))

amp <- df %>%
  group_by(Sector, Year) %>%
  summarise(
    min_extent = min(Extent),
    max_extent = max(Extent),
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
## 4. Remove Feb 29, define DOY365
## ------------------------------------------------------------

df <- df %>%
  filter(!(month(Date) == 2 & day(Date) == 29)) %>%
  mutate(
    DOY_raw = yday(Date),
    DOY365  = if_else(leap_year(Date) & DOY_raw > 59,
                      DOY_raw - 1,
                      DOY_raw)
  )

stopifnot(max(df$DOY365) == 365)

## ------------------------------------------------------------
## 5. Variability metric (IQR of APAC residuals)
## ------------------------------------------------------------

df_plot <- df %>%
  group_by(Sector, DOY365) %>%
  summarise(
    variability = IQR(res_APAC, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(Sector, DOY365) %>%
  group_by(Sector) %>%
  mutate(
    variability_smooth = rollmean(
      variability, k = 7, fill = NA, align = "center"
    )
  ) %>%
  filter(!is.na(variability_smooth)) %>%
  ungroup()

## ------------------------------------------------------------
## 6. Recenter DOY at climatological minimum (Feb 20)
## ------------------------------------------------------------

SEA_ICE_MIN_DOY <- 51

df_plot <- df_plot %>%
  mutate(
    DOY_shift = if_else(
      DOY365 < SEA_ICE_MIN_DOY,
      DOY365 + 365,
      DOY365
    ),
    Sector = factor(
      Sector,
      levels = c("Amundsen_Bellingshausen", "Ross")
    )
  )

## ------------------------------------------------------------
## 7. Color limits
## ------------------------------------------------------------

lims <- quantile(
  df_plot$variability_smooth,
  probs = c(0.05, 0.95),
  na.rm = TRUE
)

sector_labels <- c(
  "Amundsen_Bellingshausen" = "ABS",
  "Ross"                   = "ROS"
)

## ------------------------------------------------------------
## 8. Final figure
## ------------------------------------------------------------

p_fig1 <- ggplot(
  df_plot,
  aes(x = DOY_shift, y = fct_rev(Sector), fill = variability_smooth)
) +
  geom_tile() +
  scale_y_discrete(labels = sector_labels) +
  scale_x_continuous(
    breaks = c(
      51, 79, 110, 140, 171, 201,
      232, 263, 293, 324, 354, 365 + 20
    ),
    labels = c(
      "Feb", "Mar", "Apr", "May", "Jun", "Jul",
      "Aug", "Sep", "Oct", "Nov", "Dec", "Jan"
    ),
    expand = c(0, 0)
  ) +
  scale_fill_viridis_c(
    name = "Daily variability\n(IQR)",
    option = "magma",
    limits = lims,
    oob = scales::squish,
    trans = "sqrt",
    guide = guide_colorbar(
      title.position = "left",
      title.vjust = 0.5,
      barheight = unit(45, "mm"),
      ticks.colour = "black",
      frame.colour = "black"
    )
  ) +
  theme(
    legend.position = "right",
    legend.title = element_text(angle = 90, colour = "black"),
    legend.text  = element_text(size = 9, colour = "black"),
    axis.text.x  = element_text(size = 9, colour = "black"),
    axis.text.y  = element_text(size = 11, colour = "black"),
    plot.margin  = margin(5, 5, 5, 5)
  )

ggsave(
  file.path(OUT_DIR, "Figure_ABS_Ross_seasonal_variability.png"),
  p_fig1,
  width = 6,
  height = 3,
  dpi = 300
)
