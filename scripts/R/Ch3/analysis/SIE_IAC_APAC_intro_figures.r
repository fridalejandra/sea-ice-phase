## ============================================================
## Figure 1 & Figure 2
## Traditional, Invariant, and APAC reference frames
## ============================================================

library(tidyverse)
library(lubridate)
library(mgcv)

theme_set(theme_bw(base_size = 12))

## ------------------------------------------------------------
## 1. Load and prepare data
## ------------------------------------------------------------

df <- read_csv(
  "/Users/fridaperez/Developer/repos/sea-ice-phase/scripts/R/Sea_Ice_Sheets/Bootstrap79-24.csv",
  col_types = cols(
    Date = col_character(),
    Extent = col_double()
  )
) %>%
  mutate(
    Date   = mdy(Date),
    Year   = year(Date),
    DOY    = yday(Date),
    Extent = as.numeric(Extent)
  )


range(df$Date)
range(df$DOY)
summary(df$Extent)

## ------------------------------------------------------------
## 2. Traditional Annual Cycle (TAC)
## (EXACT match to your script)
## ------------------------------------------------------------

tac <- df %>%
  group_by(DOY) %>%
  summarise(
    TAC = mean(Extent, na.rm = TRUE),
    .groups = "drop"
  )

df <- df %>% left_join(tac, by = "DOY")


summary(df$DOY)
summary(df$Extent)

sum(is.na(df$DOY))
sum(is.na(df$Extent))

## ------------------------------------------------------------
## 3. Invariant Annual Cycle (IAC)
## (EXACT match: no REML)
## ------------------------------------------------------------

gam_iac <- gam(
  Extent ~ s(DOY, bs = "cc", k = 25),
  data = df
)

df$IAC <- predict(gam_iac, newdata = df)

## ------------------------------------------------------------
## 4. Phase + amplitude preprocessing (EXACT logic)
## ------------------------------------------------------------

# Annual minimum date
yearly_min <- df %>%
  group_by(Year) %>%
  summarise(Date1 = Date[which.min(Extent)], .groups = "drop") %>%
  mutate(
    Date2 = lag(Date1),
    Date3 = lead(Date1)
  )

df <- df %>% left_join(yearly_min, by = "Year")

# Relative time since minimum
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

# Normalize to phase (linear beta, same as your code)
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
## 5. APAC (diagnostic, matches your formulation)
## NO REML, NO Extent leakage
## ------------------------------------------------------------

gam_apac <- gam(
  scaling ~ s(phase, bs = "cc", k = 100),
  data = df
)

df$APAC <- predict(gam_apac, newdata = df) * df$amplitude + df$min_extent

## ------------------------------------------------------------
## 6. Reference-dependent departures (Figure 2 objects)
## ------------------------------------------------------------

df <- df %>%
  mutate(
    dep_TAC  = Extent - TAC,
    dep_IAC  = Extent - IAC,
    dep_APAC = Extent - APAC
  )

## ------------------------------------------------------------
## 7. Subset for figures
## ------------------------------------------------------------

df_fig <- df %>%
  filter(Year >= 2021, Year <= 2023)

## ------------------------------------------------------------
## 8. FIGURE 1 — Seasonal reference frames
## ------------------------------------------------------------

# fig1 <- ggplot(df_fig, aes(x = Date)) +
#   geom_line(aes(y = Extent, color = "Observed SIE"),
#             linewidth = 0.4) +
#   geom_line(aes(y = TAC, color = "Traditional annual cycle"),
#             linewidth = 1.0, linetype = "dashed") +
#   geom_line(aes(y = IAC, color = "Invariant annual cycle (IAC)"),
#             linewidth = 1.1) +
#   geom_line(aes(y = APAC, color = "Amplitude–phase adjusted annual cycle (APAC)"),
#             linewidth = 1.1) +
#   scale_color_manual(
#     values = c(
#       "Observed SIE" = "black",
#       "Traditional annual cycle" = "grey50",
#       "Invariant annual cycle (IAC)" = "#1f78b4",
#       "Amplitude–phase adjusted annual cycle (APAC)" = "#e31a1c"
#     )
#   ) +
#   labs(
#     x = NULL,
#     y = expression("Sea ice extent (10"^6*" km"^2*")"),
#     color = NULL,
#     title = "Observed Antarctic sea ice extent and alternative seasonal references",
#     subtitle = "2015–2017"
#   ) +
#   theme(
#     legend.position = "top",
#     plot.title.position = "plot"
#   )
# 
# ggsave("Fig1_SIE_TAC_IAC_APAC.png", fig1,
#        width = 9, height = 4, dpi = 300)
## sanity checks (keep these)
names(df_fig)
colSums(is.na(df_fig[, c("Date", "Extent", "TAC", "IAC", "APAC")]))

## restrict to common valid window for clean plotting
df_plot <- df_fig |>
  dplyr::filter(
    !is.na(Date),
    !is.na(Extent),
    !is.na(TAC),
    !is.na(IAC),
    !is.na(APAC)
  )

### POSTER FIGURE ###
fig1 <- ggplot(df_plot, aes(x = Date)) +
  
  ## background references first
  geom_line(
    aes(y = TAC, color = "Traditional annual cycle"),
    linewidth = 0.9,
    linetype = "dashed"
  ) +
  geom_line(
    aes(y = IAC, color = "Invariant annual cycle (IAC)"),
    linewidth = 1.2
  ) +
  geom_line(
    aes(y = APAC, color = "Amplitude-phase adjusted annual cycle (APAC)"),
    linewidth = 1.2
  ) +
  
  ## observations on top
  geom_line(
    aes(y = Extent, color = "Observed SIE"),
    linewidth = 0.5,
    alpha = 0.9
  ) +
  
  scale_color_manual(
    values = c(
      "Observed SIE" = "black",
      "Traditional annual cycle" = "grey65",
      "Invariant annual cycle (IAC)" = "#1f78b4",
      "Amplitude-phase adjusted annual cycle (APAC)" = "#e31a1c"
    )
  ) +
  
  labs(
    x = NULL,
    y = expression("Sea ice extent (10"^6*" km"^2*")"),
    color = NULL
  ) +
  
  theme(
    legend.position = "top",
    plot.title.position = "plot",
    text = element_text(size = 16),
    legend.text = element_text(size = 14),
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    legend.key.width = unit(1.4, "cm")
  )
ggsave(
  "Fig1_SIE_TAC_IAC_APAC.png",
  fig1,
  width = 12,
  height = 5,
  dpi = 600
)

ggsave(
  "Fig1_SIE_TAC_IAC_APAC.pdf",
  fig1,
  width = 12,
  height = 5
)

## ------------------------------------------------------------
## 9. FIGURE 2 — Reference-dependent departures
## ------------------------------------------------------------

fig2 <- ggplot(df_fig, aes(x = Date)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  geom_line(aes(y = dep_TAC, color = "Departure from TAC"),
            linewidth = 0.4) +
  geom_line(aes(y = dep_IAC, color = "Departure from IAC"),
            linewidth = 0.45) +
  geom_line(aes(y = dep_APAC, color = "Departure from APAC"),
            linewidth = 0.45) +
  scale_color_manual(
    values = c(
      "Departure from TAC"  = "grey50",
      "Departure from IAC"  = "#1f78b4",
      "Departure from APAC" = "#e31a1c"
    )
  ) +
  labs(
    x = NULL,
    y = expression("SIE departure (10"^6*" km"^2*")"),
    color = NULL,
    title = "Daily SIE departures depend on the seasonal reference",
    subtitle = "Observed SIE minus seasonal cycle"
  ) +
  theme(
    legend.position = "top",
    plot.title.position = "plot"
  )

ggsave("Fig2_SIE_departures_TAC_IAC_APAC.png", fig2,
       width = 9, height = 4, dpi = 300)
