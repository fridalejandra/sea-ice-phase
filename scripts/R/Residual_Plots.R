# Load necessary libraries
library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)
library(ggplot2)

# CSV file
xpath_csv <- '/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv'
df_csv <- read.csv(xpath_csv)

# Convert 'Year', 'Month', 'Day' to Date and make a column 'Date'
df_csv$Date <- as.Date(with(df_csv, paste(Year, Month, Day, sep="-")), "%Y-%m-%d")
# Filter dates
df_csv <- df_csv %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2018-12-31'))

# Make the dates numeric 
df_csv$tdate <- as.numeric(df_csv$Date)

# Calculate day of year (DOY)
df_csv$DOY <- yday(df_csv$Date)

# Make the Extent values numeric
df_csv$Extent <- as.numeric(df_csv$Extent)

######### TRADITIONAL CYCLE ##########
average_annual_cycle <- df_csv %>% group_by(DOY) %>% summarize(mean_extent = mean(Extent, na.rm = TRUE))
df_csv <- df_csv %>% left_join(average_annual_cycle, by = "DOY") %>% rename(Predicted_Traditional = mean_extent)
traditional_rmse <- sqrt(mean((df_csv$Predicted_Traditional - df_csv$Extent)^2, na.rm = TRUE))

######### INVARIANT CYCLE - CUBIC SPLINES #########
# Fitting the model with cubic splines and cyclic component for DOY
gam_model <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 50, fx=FALSE), data = df_csv, method="REML", knots=list(DOY=c(0,365)))
df_csv$Predicted_Invariant <- predict(gam_model, newdata = df_csv)
invariant_rmse <- sqrt(mean((df_csv$Predicted_Invariant - df_csv$Extent)^2, na.rm = TRUE))

# Calculate residuals for the invariant cycle
df_csv <- df_csv %>%
  mutate(Residual_Invariant = Extent - Predicted_Invariant)

# Unique years in the dataset
unique_years <- unique(df_csv$Year)

# Loop over each year and create plots
for (year in unique_years) {
  df_year <- df_csv %>% filter(Year == year)
  
  plot_year <- ggplot(df_year, aes(x = Date, y = Residual_Invariant)) +
    geom_line(color = "blue", size = 1) +
    labs(title = paste("Residuals of Sea Ice Extent in", year),
         x = "Date",
         y = "Residual Sea Ice Extent (million sq km)") +
    theme_minimal()
  
  # Save the plot
  ggsave(paste0("residuals_sea_ice_extent_", year, ".png"), plot = plot_year, width = 12, height = 6)
  
  # Display the plot
  print(plot_year)
  
  # Plot residuals for all years in one plot
  ggplot(df_csv, aes(x = Date, y = Residual_Invariant, color = as.factor(Year))) +
    geom_line(size = 1) +
    labs(title = "Residuals of Sea Ice Extent (All Years)",
         x = "Date",
         y = "Residual Sea Ice Extent (million sq km)",
         color = "Year") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
}

# Filter for specific years, e.g., 2014 and 2016
df_subset <- df_csv %>% filter(Year %in% c(1980, 2016))

# Plot residuals for the selected years
ggplot(df_subset, aes(x = Date, y = Residual_Invariant, color = as.factor(Year))) +
  geom_line(size = 1) +
  labs(title = "Residuals of Sea Ice Extent in 2014-2020",
       x = "Date",
       y = "Residual Sea Ice Extent (million sq km)",
       color = "Year") +
  theme_minimal() +
  theme(legend.position = "bottom")
