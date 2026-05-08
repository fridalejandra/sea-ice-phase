## ============================================================
## Canonical Seasonal Cycle Decomposition for Antarctic SIE
## IAC (Invariant) + APAC (Amplitude–Phase Adjusted)
## ============================================================

library(tidyverse)
library(lubridate)
library(mgcv)

## ------------------------------------------------------------
## 0. Paths and options
## ------------------------------------------------------------

data_path <- "Bootstrap79-24.csv"

## ------------------------------------------------------------
## 1. Read and prepare data
## ------------------------------------------------------------

df <- read_csv(data_path, show_col_types = FALSE) %>%
  rename(Date = Dat) %>%
  mutate(
    Date   = as.Date(Date),
    Year   = year(Date),
    DOY    = yday(Date),
    Extent = as.numeric(Extent),
    tdate  = as.numeric(Date)
  ) %>%
  filter(Year >= 1978)

stopifnot(all(!is.na(df$Extent)))

## ------------------------------------------------------------
## 2. Traditional Annual Cycle (TAC) – optional context
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
## One smooth seasonal curve, fixed in calendar space

gam_iac <- gam(
  Extent ~ s(DOY, bs = "cc", k = 25),
  data   = df,
  method = "REML"
)

df$IAC <- predict(gam_iac)

## ------------------------------------------------------------
## 4. Phase construction (Handcock-style)
## ------------------------------------------------------------

## 4.1 Annual minimum date (anchor)
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

## 4.2 Relative time t (days since min)
df <- df %>%
  rowwise() %>%
  mutate(
    t = case_when(
      Year == min(Year)        ~ 365 - as.numeric(Date3 - Date),
      Year == min(Year) + 1 &
        Date < Date1           ~ 365 - as.numeric(Date1 - Date),
      Date >= Date1            ~ as.numeric(Date - Date1),
      TRUE                     ~ as.numeric(Date - Date2)
    )
  ) %>%
  ungroup()

## 4.3 Normalize t to [0, 365] within each year
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
    phase = 365 * pbeta(
      (t - t_min) / (t_max - t_min + 1e-10),
      shape1 = 1,
      shape2 = 1
    )
  )

## ------------------------------------------------------------
## 5. Amplitude normalization
## ------------------------------------------------------------

yearly_amp <- df %>%
  group_by(Year) %>%
  summarise(
    min_extent = min(Extent, na.rm = TRUE),
    max_extent = max(Extent, na.rm = TRUE),
    amplitude  = max_extent - min_extent,
    .groups = "drop"
  )

df <- df %>% left_join(yearly_amp, by = "Year")

df <- df %>%
  mutate(
    scaling = (Extent - min_extent) / amplitude
  )

## ------------------------------------------------------------
## 6. Amplitude–Phase Adjusted Annual Cycle (APAC)
## ------------------------------------------------------------
## Smooth canonical cycle in phase space, then reconstruct

gam_apac <- gam(
  scaling ~ s(phase, bs = "cc", k = 100),
  data   = df,
  method = "REML"
)

df$APAC <- predict(gam_apac) * df$amplitude + df$min_extent

## ------------------------------------------------------------
## 7. Residual definitions (FINAL, EXPLICIT)
## ------------------------------------------------------------

df <- df %>%
  mutate(
    res_TAC  = Extent - TAC,   # optional, context only
    res_IAC  = Extent - IAC,
    res_APAC = Extent - APAC
  )

## ------------------------------------------------------------
## 8. Output objects
## ------------------------------------------------------------

## This data frame is the canonical product
## Everything else (figures, persistence, poster) uses this

saveRDS(df, "SIE_IAC_APAC_canonical.rds")
