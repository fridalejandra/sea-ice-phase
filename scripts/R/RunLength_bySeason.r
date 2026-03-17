## ============================================================
## Figure 2A / 2B — Persistence of daily Antarctic sea ice variability
## Absolute APAC residuals, run-length statistics
## OPTION B: All seasons (supporting) + JJA-only (main)
## ============================================================

library(tidyverse)
library(lubridate)
library(mgcv)

theme_set(theme_bw(base_size = 12))

## ------------------------------------------------------------
## Paths
## ------------------------------------------------------------

DATA_PATH <- "/Users/fridaperez/Developer/repos/sea-ice-phase/scripts/R/Sea_Ice_Sheets/SIE_daily_sector_and_circumpolar_million_km2.csv"
OUT_DIR   <- "/Users/fridaperez/Desktop/Clic_Ch3/Figure2_persistence"

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

## ------------------------------------------------------------
## 1. Load and reshape data (SECTORS FIXED)
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
    
    ## Explicit sector recoding (NO circumpolar)
    Sector = case_when(
      Sector == "Amundsen_Bellingshausen" ~ "Amundsen–Bellingshausen",
      Sector == "Ross"                    ~ "Ross Sea",
      Sector == "King_Haakon"             ~ "King Haakon",
      Sector == "Weddell"                 ~ "Weddell Sea",
      Sector == "East_Antarctica"         ~ "East Antarctica",
      TRUE ~ NA_character_
    ),
    
    ## Lock order for faceting
    Sector = factor(
      Sector,
      levels = c(
        "Amundsen–Bellingshausen",
        "Ross Sea",
        "King Haakon",
        "Weddell Sea",
        "East Antarctica"
      )
    ),
    
    Date   = mdy(Date),
    Year   = year(Date),
    DOY    = yday(Date),
    Extent = as.numeric(Extent)
  ) %>%
  filter(!is.na(Extent), !is.na(Sector))

stopifnot(nrow(df) > 0)
stopifnot(!any(is.na(df$Sector)))

unique(df$Sector)


## ------------------------------------------------------------
## 2. APAC model
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
## 3. Remove Feb 29 + define seasons
## ------------------------------------------------------------

df <- df %>%
  filter(!(month(Date) == 2 & day(Date) == 29)) %>%
  mutate(
    Season = case_when(
      month(Date) %in% c(12, 1, 2) ~ "DJF",
      month(Date) %in% c(3, 4, 5)  ~ "MAM",
      month(Date) %in% c(6, 7, 8)  ~ "JJA",
      TRUE                         ~ "SON"
    )
  )

## ------------------------------------------------------------
## 4. Define high-variability threshold (75th percentile, all seasons)
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
## 5. Identify run lengths
## ------------------------------------------------------------

runs <- df %>%
  arrange(Sector, Date) %>%
  group_by(Sector) %>%
  mutate(
    run_id = cumsum(high_var != lag(high_var, default = FALSE))
  ) %>%
  filter(high_var) %>%
  group_by(Sector, run_id) %>%
  summarise(
    run_length = n(),
    Season     = first(Season),
    .groups = "drop"
  )

## ------------------------------------------------------------
## 6. Cap run lengths for plotting
## ------------------------------------------------------------

MAX_RUN <- 20

runs <- runs %>%
  mutate(run_length = pmin(run_length, MAX_RUN))

runs_JJA <- runs %>%
  filter(Season == "JJA")

## ------------------------------------------------------------
## 7A. Supporting figure — All seasons
## ------------------------------------------------------------

p_all <- ggplot(
  runs,
  aes(x = run_length, fill = Season)
) +
  geom_histogram(
    aes(y = after_stat(count / sum(count))),
    binwidth = 1,
    color = "grey30",
    alpha = 0.85,
    boundary = 0
  ) +
  facet_wrap(~ Sector, scales = "free_y", ncol = 2)+
  scale_x_continuous(
    breaks = seq(1, MAX_RUN, by = 2),
    limits = c(1, MAX_RUN)
  ) +
  scale_fill_manual(
    values = c(
      "DJF" = "#d73027",
      "MAM" = "#fc8d59",
      "JJA" = "#4575b4",
      "SON" = "#91bfdb"
    )
  ) +
  labs(
    title = "Seasonal structure of high-variability persistence",
    x = "Run length (days)",
    y = "Probability",
    fill = "Season"
  ) +
  theme(
    panel.grid.minor = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(size= 14, face = "bold"),
    axis.text = element_text(size = 12),
    legend.position = "bottom"
  
  )

ggsave(
  file.path(OUT_DIR, "Figure2A_run_length_all_seasons_75th.png"),
  p_all,
  width = 10.5,
  height = 9.5,
  dpi = 400
)

## ------------------------------------------------------------
## 7B. Main figure — JJA only
## ------------------------------------------------------------

p_jja <- ggplot(
  runs_JJA,
  aes(x = run_length)
) +
  geom_histogram(
    aes(y = after_stat(count / sum(count))),
    binwidth = 1,
    fill = "#4575b4",
    color = "grey30",
    alpha = 0.9,
    boundary = 0
  ) +
  facet_wrap(~ Sector, scales = "free_y") +
  scale_x_continuous(
    breaks = seq(1, MAX_RUN, by = 2),
    limits = c(1, MAX_RUN)
  ) +
  labs(
    title = "Winter persistence of high daily sea ice variability (JJA)",
    x = "Run length (days)",
    y = "Probability"
  ) +
  theme(
    panel.grid.minor = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )

ggsave(
  file.path(OUT_DIR, "Figure2B_run_length_JJA_75th.png"),
  p_jja,
  width = 11,
  height = 7,
  dpi = 300
)
