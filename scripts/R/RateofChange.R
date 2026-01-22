# Load necessary libraries
library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)

# Load the CSV file
df_csv <- read.csv("/Users/fridaperez/Desktop/Sea_Ice_Sheets/S_seaice_extent_daily_v3.0.csv")

# Convert 'Year', 'Month', 'Day' to Date
df_csv$Date <- as.Date(with(df_csv, paste(Year, Month, Day, sep="-")), "%Y-%m-%d")
df_csv <- df_csv %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2023-12-31'))

# Calculate day of year (DOY) for aggregation
df_csv$DOY <- yday(df_csv$Date)

# Make the Extent values numeric
df_csv$Extent <- as.numeric(df_csv$Extent)

# Traditional Annual Cycle
average_annual_cycle <- df_csv %>% group_by(DOY) %>% summarize(mean_extent = mean(Extent, na.rm = TRUE))
df_csv <- df_csv %>% left_join(average_annual_cycle, by = "DOY") %>% rename(Predicted_Traditional = mean_extent)

# Invariant Annual Cycle using cyclic cubic splines
gam_model <- gam(Extent ~ s(DOY, bs = "cc", k = 25), data = df_csv)
df_csv$Predicted_Invariant <- predict(gam_model, newdata = df_csv)

# Group data by day of year for plotting averages across all years
average_cycles <- df_csv %>%
  group_by(DOY) %>%
  summarize(
    Traditional = mean(Predicted_Traditional, na.rm = TRUE),
    Invariant = mean(Predicted_Invariant, na.rm = TRUE)
  )

# Calculate the rate of change for Traditional and Invariant cycles
# Traditional cycle rate of change
average_cycles <- average_cycles %>%
  mutate(
    Traditional_rate_of_change = c(NA, diff(Traditional)),
    Invariant_rate_of_change = c(NA, diff(Invariant))
  )

# Plot Traditional vs Invariant Annual Cycle (Averages over all years)
ggplot(average_cycles, aes(x = DOY)) +
  geom_line(aes(y = Traditional, color = 'Traditional'), linetype = 'dashed', size = 1) +
  geom_line(aes(y = Invariant, color = 'Invariant'), linetype = 'solid', size = 1) +
  scale_x_continuous(
    breaks = c(1, 91, 182, 274, 365),
    labels = c("Jan", "Apr", "Jul", "Oct", "Dec")
  ) +
  labs(x = 'Month', y = 'Sea Ice Extent (millions of square kilometers)', color = 'Model') +
  theme_minimal() +
  scale_color_manual(values = c('Traditional' = 'blue', 'Invariant' = 'red')) +
  theme(
    legend.position = 'bottom',
    legend.title = element_text(size = 14),          # Increase legend title font size
    legend.text = element_text(size = 14),           # Increase legend text font size
    axis.title = element_text(size = 16),            # Increase axis title font size
    axis.text = element_text(size = 14),             # Increase axis text font size
    plot.title = element_blank()                     # Remove plot title
  )

# Plotting rate of change for Traditional and Invariant cycles
ggplot(average_cycles, aes(x = DOY)) +
  geom_line(aes(y = Traditional_rate_of_change, color = 'Traditional Rate of Change'), linetype = "dashed") +
  geom_line(aes(y = Invariant_rate_of_change, color = 'Invariant Rate of Change'), linetype = "dotdash") +
  labs(title = "Rate of Change in Sea Ice Extent (Traditional vs. Invariant Cycle)",
       x = "Day of Year", y = "Rate of Change in SIE") +
  theme_minimal() +
  scale_color_manual(name = "Cycle Type", values = c("Traditional Rate of Change" = "blue", "Invariant Rate of Change" = "red")) +
  theme(
    legend.position = 'bottom',
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14)
  )
