## ============================================================
## Antarctic SIE:
## Traditional, Invariant, APAC cycles
## + Sectoral persistence → amplitude feedback (Figure B)
## ============================================================

library(tidyverse)
library(lubridate)
library(mgcv)

theme_set(theme_bw(base_size = 12))

## ------------------------------------------------------------
## 1. Load data
## ------------------------------------------------------------

data_path <- "/Users/fridaperez/Developer/repos/sea-ice-phase/scripts/R/Sea_Ice_Sheets/"

df <- read_csv(
  data_path,
  col_types = cols(
    Date   = col_character(),
    Extent = col_double(),
    Sector = col_character()
  )
) %>%
  mutate(
    Date   = mdy(Date),
    Year   = year(Date),
    DOY    = yday(Date),
    Extent = as.numeric(Extent)
  ) %>%
  filter(!is.na(Date), !is.na(Extent))

## ------------------------------------------------------------
## 2. Traditional Annual Cycle (TAC)
## ------------------------------------------------------------

tac <- df %>%
  group_by(Sector, DOY) %>%
  summarise(
    TAC = mean(Extent, na.rm = TRUE),
    .groups = "drop"
  )

df <- df %>% left_join(tac, by = c("Sector", "DOY"))

## ------------------------------------------------------------
## 3. Invariant Annual Cycle (IAC) — per sector
## ------------------------------------------------------------

df <- df %>%
  group_by(Sector) %>%
  group_modify(~ {
    g <- gam(
      Extent ~ s(DOY, bs = "cc", k = 25),
      data = .x
    )
    .x$IAC <- predict(g, newdata = .x)
    .x
  }) %>%
  ungroup()

## ------------------------------------------------------------
## 4. Phase + amplitude preprocessing (per sector, per year)
## ------------------------------------------------------------

yearly_min <- df %>%
  group_by(Sector, Year) %>%
  summarise(
    Date1 = Date[which.min(Extent)],
    .groups = "drop"
  ) %>%
  group_by(Sector) %>%
  mutate(
    Date2 = lag(Date1),
    Date3 = lead(Date1)
  ) %>%
  ungroup()

df <- df %>% left_join(yearly_min, by = c("Sector", "Year"))

df <- df %>%
  group_by(Sector, Year) %>%
  rowwise() %>%
  mutate(
    t = case_when(
      Year == min(Year) ~ 365 - as.numeric(Date3 - Date),
      Date >= Date1     ~ as.numeric(Date - Date1),
      TRUE              ~ as.numeric(Date - Date2)
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

yearly_amp <- df %>%
  group_by(Sector, Year) %>%
  summarise(
    min_extent = min(Extent, na.rm = TRUE),
    max_extent = max(Extent, na.rm = TRUE),
    amplitude  = max_extent - min_extent,
    .groups = "drop"
  )

df <- df %>%
  left_join(yearly_amp, by = c("Sector", "Year")) %>%
  mutate(
    scaling = (Extent - min_extent) / amplitude
  )

## ------------------------------------------------------------
## 5. APAC — diagnostic, phase-only, per sector
## ------------------------------------------------------------

df <- df %>%
  group_by(Sector) %>%
  group_modify(~ {
    g <- gam(
      scaling ~ s(phase, bs = "cc", k = 100),
      data = .x
    )
    .x$APAC <- predict(g, newdata = .x) * .x$amplitude + .x$min_extent
    .x
  }) %>%
  ungroup()

## ------------------------------------------------------------
## 6. APAC departures (sign-based)
## ------------------------------------------------------------

df <- df %>%
  mutate(
    dep_APAC = Extent - APAC,
    sign_dep = sign(dep_APAC)
  )

## ------------------------------------------------------------
## 7. ±30 days around minimum (APAC phase)
## ------------------------------------------------------------

min_window <- df %>%
  filter(phase <= 30 | phase >= (365 - 30))

## ------------------------------------------------------------
## 8. Mean run length (sign-based)
## ------------------------------------------------------------

mean_run_length <- function(x) {
  x <- x[x != 0]
  if (length(x) == 0) return(NA_real_)
  mean(rle(x)$lengths)
}

mrl_sector_year <- min_window %>%
  group_by(Sector, Year) %>%
  summarise(
    MRL_min     = mean_run_length(sign_dep),
    min_extent = unique(min_extent),
    .groups = "drop"
  )

## ------------------------------------------------------------
## 9. Minimum amplitude anomaly (sector-relative)
## ------------------------------------------------------------

mrl_sector_year <- mrl_sector_year %>%
  group_by(Sector) %>%
  mutate(
    min_anom = min_extent - mean(min_extent, na.rm = TRUE)
  ) %>%
  ungroup()

## ------------------------------------------------------------
## 10. Era split
## ------------------------------------------------------------

mrl_sector_year <- mrl_sector_year %>%
  mutate(
    era = if_else(Year <= 2016, "Pre-2016", "Post-2016")
  )

## ------------------------------------------------------------
## 11. Correlation: persistence ↔ amplitude (Figure B data)
## ------------------------------------------------------------

figB_data <- mrl_sector_year %>%
  group_by(Sector, era) %>%
  summarise(
    r_MRL_amp = cor(MRL_min, min_anom,
                    use = "complete.obs"),
    n_years = sum(!is.na(MRL_min) & !is.na(min_anom)),
    .groups = "drop"
  )

print(figB_data)

## ------------------------------------------------------------
## 12. FIGURE B — Sectoral sensitivity
## ------------------------------------------------------------

figB <- ggplot(figB_data,
               aes(x = Sector,
                   y = r_MRL_amp,
                   color = era)) +
  geom_hline(yintercept = 0,
             linetype = "dashed",
             color = "grey50") +
  geom_point(size = 3,
             position = position_dodge(width = 0.4)) +
  scale_color_manual(
    values = c("Pre-2016" = "#1f78b4",
               "Post-2016" = "#e31a1c")
  ) +
  labs(
    x = NULL,
    y = "Correlation between persistence and minimum depth",
    color = NULL,
    title = "Sectoral sensitivity of Antarctic sea-ice minima to daily persistence",
    subtitle = "Persistence measured as sign-based mean run length (±30 days around minimum)"
  ) +
  theme(
    legend.position = "top",
    plot.title.position = "plot"
  )

ggsave(
  "/Users/fridaperez/Desktop/Clic_Ch3/FigB_persistence_vs_amplitude_by_sector.png",
  figB,
  width = 9,
  height = 4.5,
  dpi = 300
)
