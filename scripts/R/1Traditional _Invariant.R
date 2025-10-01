# Load necessary libraries
library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)

# Load the CSV file
df_csv <- read.csv("/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv")

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
gam_model <- gam(Extent ~ s(DOY, bs = "cc", k = 25), data = df_csv) #k=14? 
df_csv$Predicted_Invariant <- predict(gam_model, newdata = df_csv)

# Group data by day of year for plotting averages across all years
average_cycles <- df_csv %>%
  group_by(DOY) %>%
  summarize(
    Traditional = mean(Predicted_Traditional, na.rm = TRUE),
    Invariant = mean(Predicted_Invariant, na.rm = TRUE)
  )

# Plot Traditional vs Invariant Annual Cycle (Averages over all years)
ggplot(average_cycles, aes(x = DOY)) +
  geom_line(aes(y = Traditional, color = 'Traditional'), linetype = 'dashed', size = 1) +
  geom_line(aes(y = Invariant, color = 'Invariant'), linetype = 'dotdash', size = 1) +
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
