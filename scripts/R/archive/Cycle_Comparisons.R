library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)
library(ggplot2)

# ------------------------
# User settings
# ------------------------
DATA_PATH <- "/Users/fridaperez/Developer/repos/sea-ice-phase/scripts/R/Sea_Ice_Sheets/SIE_daily_sector_and_circumpolar_million_km2.csv"
YEAR_TO_PLOT <- 2016
OUT_DIR <- "/Users/fridaperez/Desktop/Clic_Ch3"

# ------------------------
# Load and prepare circumpolar data ONLY
# ------------------------
df_csv <- read_csv(DATA_PATH) %>%
  select(Date, SIE_circumpolar) %>%
  rename(Extent = SIE_circumpolar) %>%
  mutate(
    Date   = mdy(Date),
    Year   = year(Date),
    DOY    = yday(Date),
    tdate  = as.numeric(Date),
    Extent = as.numeric(Extent)
  ) %>%
  filter(!is.na(Extent))

stopifnot(all(c("Date", "Year", "DOY", "Extent") %in% names(df_csv)))

# ------------------------
# Calculate minimum extent day per year
# ------------------------
yearly_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(
    Date1 = Date[which.min(Extent)],
    .groups = "drop"
  ) %>%
  mutate(
    Date2 = lag(Date1),
    Date3 = lead(Date1)
  )

df_csv <- df_csv %>%
  left_join(yearly_stats, by = "Year")

# ------------------------
# Calculate phase time variable t
# ------------------------
df_csv <- df_csv %>%
  rowwise() %>%
  mutate(
    t = case_when(
      Year == min(Year) ~ 365 - as.numeric(Date3 - Date),
      Date >= Date1     ~ as.numeric(Date - Date1),
      Date <  Date1     ~ as.numeric(Date - Date2)
    )
  ) %>%
  ungroup()

# ------------------------
# Calculate t_min and t_max by year
# ------------------------
t_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(
    t_min = min(t, na.rm = TRUE),
    t_max = max(t, na.rm = TRUE),
    .groups = "drop"
  )

df_csv <- df_csv %>%
  left_join(t_stats, by = "Year")

# ------------------------
# Phase-adjusted DOY (APAC phase)
# ------------------------
df_csv <- df_csv %>%
  rowwise() %>%
  mutate(
    phase = 365 * pbeta(
      (t - t_min) / (t_max - t_min + 1e-10),
      shape1 = 1,
      shape2 = 1
    )
  ) %>%
  ungroup()

# ------------------------
# Amplitude scaling
# ------------------------
yearly_amp <- df_csv %>%
  group_by(Year) %>%
  summarise(
    max_extent = max(Extent, na.rm = TRUE),
    min_extent = min(Extent, na.rm = TRUE),
    .groups = "drop"
  )

df_csv <- df_csv %>%
  left_join(yearly_amp, by = "Year") %>%
  mutate(
    amplitude      = max_extent - min_extent,
    scaling_factor = (Extent - min_extent) / amplitude
  )

# ------------------------
# APAC GAM (NO IAC)
# ------------------------
gam_model_apac <- gam(
  scaling_factor ~
    s(tdate, bs = "cc", k = 150) +
    s(DOY,   bs = "cc", k = 100) +
    s(phase, bs = "cc", k = 100),
  data = df_csv
)

df_csv$Predicted_APAC <- predict(gam_model_apac, newdata = df_csv)
df_csv$Predicted_APAC_Extent <-
  df_csv$Predicted_APAC * df_csv$amplitude + df_csv$min_extent

# ------------------------
# Traditional climatological seasonal cycle (TAC)
# ------------------------
clim_cycle <- df_csv %>%
  group_by(DOY) %>%
  summarise(
    Clim = mean(Extent, na.rm = TRUE),
    .groups = "drop"
  )

df_csv <- df_csv %>%
  left_join(clim_cycle, by = "DOY")

# ------------------------
# Prepare plotting dataframe (LONG FORMAT)
# ------------------------
df_plot_long <- df_csv %>%
  filter(Year == YEAR_TO_PLOT) %>%
  select(
    Date,
    Extent,
    Clim,
    Predicted_APAC_Extent
  ) %>%
  pivot_longer(
    cols = c(Extent, Clim, Predicted_APAC_Extent),
    names_to = "Type",
    values_to = "Value"
  )

stopifnot(nrow(df_plot_long) > 0)

# ------------------------
# Poster-ready plot
# ------------------------
p_ref <- ggplot(df_plot_long, aes(x = Date, y = Value, color = Type)) +
  geom_line(linewidth = 1.2) +
  
  scale_color_manual(
    name = "Reference frame",
    values = c(
      "Extent" = "#1f78b4",
      "Clim" = "#33a02c",
      "Predicted_APAC_Extent" = "#e31a1c"
    ),
    labels = c(
      "Extent" = "Observed SIE",
      "Clim" = "Traditional climatology",
      "Predicted_APAC_Extent" = "APAC\n(phase + amplitude adjusted)"
    )
  )+
  
  scale_x_date(
    limits = as.Date(c(
      paste0(YEAR_TO_PLOT, "-01-01"),
      paste0(YEAR_TO_PLOT, "-12-31")
    )),
    date_labels = "%b",
    date_breaks = "1 month",
    expand = c(0, 0)
  )+
  
  labs(
    x = NULL,
    y = expression("Sea ice extent (10"^6*" km"^2*")")
  ) +
  
  theme_minimal(base_size = 18) +
  theme(
    axis.text.x  = element_text(size = 16),
    axis.text.y  = element_text(size = 16),
    axis.title.y = element_text(size = 18),
    legend.position = "bottom",
    legend.title = element_text(size = 16),
    legend.text  = element_text(size = 15),
    panel.grid.minor = element_blank(),
    plot.margin = margin(t = 8, r = 12, b = 22, l = 12)
  )

# ------------------------
# Save figure
# ------------------------
ggsave(
  filename = file.path(OUT_DIR, paste0("Figure_reference_frames_", YEAR_TO_PLOT, ".png")),
  plot = p_ref,
  width = 7.5,
  height = 7.5,
  dpi = 600
)

