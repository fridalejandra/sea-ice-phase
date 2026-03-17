## ============================================================
## TAC anomaly vs APAC residual
## Daily SIE variability by DOY
## One year, one figure per sector
## ============================================================

library(tidyverse)
library(lubridate)
library(mgcv)
DATA_PATH <- "/Users/fridaperez/Developer/repos/sea-ice-phase/scripts/R/Sea_Ice_Sheets/SIE_daily_sector_and_circumpolar_million_km2.csv"
YEAR_TO_PLOT <- 2014
OUT_DIR <- "/Users/fridaperez/Desktop/Clic_Ch3"

theme_set(theme_bw(base_size = 12))

dir.create(OUT_DIR, showWarnings = FALSE)



## ------------------------------------------------------------
## 1. Load and reshape data (WIDE → LONG)
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
## 2. Traditional Annual Cycle (TAC) anomaly
## ------------------------------------------------------------

tac <- df %>%
  group_by(Sector, DOY) %>%
  summarise(
    TAC = mean(Extent, na.rm = TRUE),
    .groups = "drop"
  )

df <- df %>%
  left_join(tac, by = c("Sector", "DOY")) %>%
  mutate(
    anom_TAC = Extent - TAC
  )

## ------------------------------------------------------------
## 3. APAC (phase + amplitude adjusted, same logic as your script)
## ------------------------------------------------------------

## --- seasonal minimum dates ---
min_dates <- df %>%
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

df <- df %>% left_join(min_dates, by = c("Sector", "Year"))

## --- construct phase time coordinate ---
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

## --- amplitude scaling ---
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

## --- APAC GAM (per sector, pooled years) ---
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

df <- df %>%
  mutate(
    res_APAC = Extent - APAC
  )

## ------------------------------------------------------------
## 4. Subset to single year
## ------------------------------------------------------------

df_y <- df %>%
  filter(Year == YEAR_TO_PLOT)

stopifnot(nrow(df_y) > 0)

## ------------------------------------------------------------
## 5. Plot: TAC anomaly vs APAC residual (|.|) by DOY
## ------------------------------------------------------------
for (s in unique(df_y$Sector)) {
  
  d <- df_y %>% filter(Sector == s)
  
  if (nrow(d) == 0) next
  
  # Annotation y-position (below envelopes)
  y_annotate <- -1.1 * max(
    c(abs(d$anom_TAC), abs(d$res_APAC)),
    na.rm = TRUE
  )
  
  p <- ggplot(d, aes(x = DOY)) +
    
    # Traditional anomaly envelope (TAC)
    geom_ribbon(
      aes(
        ymin = -abs(anom_TAC),
        ymax =  abs(anom_TAC)
      ),
      fill  = "grey70",
      alpha = 0.6
    ) +
    
    # APAC residual envelope
    geom_ribbon(
      aes(
        ymin = -abs(res_APAC),
        ymax =  abs(res_APAC)
      ),
      fill  = "#e31a1c",
      alpha = 0.5
    ) +
    
    # Month annotations
    annotate("text", x =  15, y = y_annotate, label = "Jan", size = 3) +
    annotate("text", x =  75, y = y_annotate, label = "Mar", size = 3) +
    annotate("text", x = 135, y = y_annotate, label = "May", size = 3) +
    annotate("text", x = 195, y = y_annotate, label = "Jul", size = 3) +
    annotate("text", x = 255, y = y_annotate, label = "Sep", size = 3) +
    annotate("text", x = 315, y = y_annotate, label = "Nov", size = 3) +
    
    labs(
      x = "Day of year",
      y = expression("|Daily departure| (10"^6*" km"^2*")"),
      title = paste("Daily Antarctic sea ice variability –", s),
      subtitle = paste(
        YEAR_TO_PLOT,
        "Traditional anomaly (gray) vs APAC residual (red)"
      )
    ) +
    
    theme(
      plot.title.position = "plot"
    )
  
  ggsave(
    filename = file.path(
      OUT_DIR,
      paste0("Fig_TAC_vs_APAC_", s, "_", YEAR_TO_PLOT, ".png")
    ),
    plot  = p,
    width = 8,
    height = 4,
    dpi   = 300
  )
}


