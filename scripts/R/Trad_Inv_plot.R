# Load necessary libraries
library(dplyr)
library(ggplot2)
library(mgcv)
library(lubridate)

# Load the CSV file
df_csv <- read.csv("/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv")

# Convert 'Year', 'Month', 'Day' to Date
df_csv$Date <- as.Date(with(df_csv, paste(Year, Month, Day, sep = "-")), "%Y-%m-%d")
df_csv <- df_csv %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2018-12-31'))

# Make the dates numeric and calculate DOY (day of year)
df_csv$tdate <- as.numeric(df_csv$Date)
df_csv$DOY <- yday(df_csv$Date)

# Make the Extent values numeric
df_csv$Extent <- as.numeric(df_csv$Extent)

# 1. Traditional Annual Cycle: average sea ice extent for each day of the year (DOY)
average_annual_cycle <- df_csv %>%
  group_by(DOY) %>%
  summarise(mean_extent = mean(Extent, na.rm = TRUE))

# Add the traditional predictions to df_csv for RMSE calculation
df_csv <- df_csv %>%
  left_join(average_annual_cycle, by = "DOY") %>%
  mutate(Predicted_Traditional = mean_extent)

# 2. Invariant Annual Cycle using cyclic cubic splines (smooth annual cycle)
gam_model <- gam(Extent ~ s(tdate, bs = "cc", k = 14) + s(DOY, bs = "cc", k = 25), data = df_csv)
predicted_invariant <- predict(gam_model, newdata = data.frame(tdate = df_csv$tdate, DOY = df_csv$DOY))

# Add the predicted invariant cycle to the data
df_csv$Predicted_Invariant <- predicted_invariant

# Summarize the Invariant cycle by DOY
invariant_annual_cycle <- df_csv %>%
  group_by(DOY) %>%
  summarise(mean_invariant = mean(Predicted_Invariant, na.rm = TRUE))

# Calculate RMSE for Traditional and Invariant cycles
traditional_rmse <- sqrt(mean((df_csv$Predicted_Traditional - df_csv$Extent)^2, na.rm = TRUE))
invariant_rmse <- sqrt(mean((df_csv$Predicted_Invariant - df_csv$Extent)^2, na.rm = TRUE))

# Print RMSE values
print(paste("Traditional RMSE:", traditional_rmse))
print(paste("Invariant RMSE:", invariant_rmse))

# Create a plot for Traditional and Invariant Annual Cycles and add RMSE values
ggplot() +
  geom_line(data = average_annual_cycle, aes(x = DOY, y = mean_extent, color = 'Traditional'), size = 1) +  # Traditional cycle
  geom_line(data = invariant_annual_cycle, aes(x = DOY, y = mean_invariant, color = 'Invariant'), linetype = 'dashed', size = 1) +  # Invariant cycle
  scale_x_continuous(breaks = c(1, 61, 121, 182, 243, 304, 365), 
                     labels = c('Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov', 'Dec')) +  # X-axis as months
  labs(title = 'Traditional vs Invariant Annual Cycle of Sea Ice Extent',
       x = 'Month',
       y = 'Sea Ice Extent (in millions of square kilometers)',
       color = 'Cycle') +
  theme_minimal() +
  scale_color_manual(values = c('Traditional' = 'blue', 'Invariant' = 'red')) +
  theme(legend.position = 'bottom') +
  annotate("text", x = 50, y = max(average_annual_cycle$mean_extent), label = paste("Traditional RMSE:", round(traditional_rmse, 2)), color = 'blue') +
  annotate("text", x = 50, y = max(average_annual_cycle$mean_extent) - 0.5, label = paste("Invariant RMSE:", round(invariant_rmse, 2)), color = 'red')

