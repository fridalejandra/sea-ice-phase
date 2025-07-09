library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)
library(ggplot2)

# Load the CSV file
df_csv <- read.csv("/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv")

# Convert 'Year', 'Month', 'Day' to Date
df_csv$Date <- as.Date(with(df_csv, paste(Year, Month, Day, sep = "-")), "%Y-%m-%d")
df_csv$tdate <- as.numeric(df_csv$Date)
df_csv$DOY <- yday(df_csv$Date)
df_csv$Extent <- as.numeric(df_csv$Extent)

# Filter for valid dates
df_csv <- df_csv %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2023-12-31'))

# Calculate min extent day per year
yearly_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(Date1 = Date[which.min(Extent)], .groups = 'drop')

# Add Date2 (min date of last year) and Date3 (min date of next year)
yearly_stats <- yearly_stats %>%
  mutate(Date2 = lag(Date1),  # Date for last year
         Date3 = lead(Date1)) # Date for next year

# Merge yearly_stats back into df_csv
df_csv <- df_csv %>%
  left_join(yearly_stats, by = "Year")

# Calculate t-values
df_csv <- df_csv %>%
  rowwise() %>%
  mutate(t = case_when(
    Year == 1978 ~ 365 - as.numeric(Date3 - Date),
    Year == 1979 & Date < Date1 ~ 365 - as.numeric(Date1 - Date),
    Date >= Date1 ~ as.numeric(Date - Date1),
    Date < Date1 ~ as.numeric(Date - Date2)
  )) %>%
  ungroup()

# Calculate t_min and t_max by year
t_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(t_min = min(t, na.rm = TRUE), 
            t_max = max(t, na.rm = TRUE)) %>%
  ungroup()

df_csv <- df_csv %>%
  left_join(t_stats, by = "Year")

# Calculate the phase-adjusted day of the year
df_csv <- df_csv %>%
  rowwise() %>%
  mutate(phase = 365 * pbeta((t - t_min) / (t_max - t_min + 1e-10), shape1 = 1, shape2 = 1)) %>%
  ungroup()

# Calculate yearly max, min extent, amplitude, and scaling factor
yearly_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(max_extent = max(Extent, na.rm = TRUE), 
            min_extent = min(Extent, na.rm = TRUE))

df_csv <- df_csv %>%
  left_join(yearly_stats, by = "Year") %>%
  mutate(amplitude = max_extent - min_extent,
         scaling_factor = (Extent - min_extent) / amplitude)

# Build GAM for Amplitude-Phase Adjusted Cycle (APAC)
gam_model_apac <- gam(scaling_factor ~ 
                        s(tdate, bs = "cc", k = 150) + 
                        s(DOY, bs = "cc", k = 100) + 
                        s(phase, bs = "cc", k = 100), 
                      data = df_csv)
df_csv$Predicted_APAC <- predict(gam_model_apac, newdata = df_csv)
df_csv$Predicted_APAC_Extent <- df_csv$Predicted_APAC * df_csv$amplitude + df_csv$min_extent

# Build GAM for Invariant Annual Cycle (IAC)
gam_model_iac <- gam(Extent ~ s(DOY, bs = "cc", k = 25), data = df_csv)
df_csv$Predicted_Invariant <- predict(gam_model_iac, newdata = df_csv)

# Filter data for the year 2022
df_2022 <- df_csv %>% filter(Year == 2022)

# Reshape df_2022 for plotting
df_2022_long <- df_2022 %>%
  select(Date, Extent, Predicted_APAC_Extent, Predicted_Invariant) %>%
  pivot_longer(
    cols = c(Extent, Predicted_APAC_Extent, Predicted_Invariant),
    names_to = "Type",
    values_to = "Value"
  )

# Calculate residual (IAC - APAC) for 2022
df_2022_res <- df_2022 %>%
  mutate(Residual = Predicted_Invariant - Predicted_APAC_Extent) %>%
  select(Date, Residual)

# Combine the residual with the existing long-format data:
df_2022_long_faceted <- df_2022_long %>%
  mutate(Group = "Main") %>%  # Assign main panel for observed and predicted series
  bind_rows(
    df_2022_res %>%
      mutate(Type = "Residual",    # Rename type to "Residual"
             Value = Residual,      # Set Value to the computed residual
             Group = "Residual")
  ) %>%
  mutate(Group = factor(Group, levels = c("Main", "Residual")))  # Ensure residual is at the bottom

# Plot with faceting: main panel at top, residual at the bottom, no title
ggplot(df_2022_long_faceted, aes(x = Date, y = Value, color = Type)) +
  geom_line(size = 1) +
  facet_wrap(~ Group, ncol = 1, scales = "free_y") +
  labs(
    x = "Date",
    y = "Extent (million square kilometers)"
  ) +
  scale_color_manual(
    name = "Legend",
    values = c(
      "Extent" = "blue",
      "Predicted_APAC_Extent" = "red",
      "Predicted_Invariant" = "purple",
      "Residual" = "black"
    ),
    labels = c(
      "Extent" = "Observed",
      "Predicted_APAC_Extent" = "APAC",
      "Predicted_Invariant" = "IAC",
      "Residual" = "Residual"
    )
  ) +
  scale_x_date(
    date_labels = "%b",  # Abbreviated month names
    date_breaks = "1 month"  # Monthly breaks
  ) +
  theme_minimal() +
  theme(
    plot.title = element_blank(),  # Remove title
    legend.position = "bottom"
  )

