# Load necessary libraries
library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)
library(ggplot2)

# CSV file
xpath_csv <- '/Users/fridaperez/Developer/repos/phase_project/SIE/S_seaice_extent_daily_v3.0.csv'
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
#knots = list(DOY = c(0, 365)): Specifies the knot placement for the DOY variable, indicating that it is cyclic with knots at 0 and 365 to capture the yearly cycle.
# method = "REML": Uses restricted maximum likelihood estimation for smoothing parameter selection.
gam_model <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 50, fx=FALSE), data = df_csv, method="REML", knots=list(DOY=c(0,365)))
df_csv$Predicted_Invariant <- predict(gam_model, newdata = df_csv)
invariant_rmse <- sqrt(mean((df_csv$Predicted_Invariant - df_csv$Extent)^2, na.rm = TRUE))

# Calculate residuals for the invariant cycle
df_csv <- df_csv %>%
  mutate(Residual_Invariant = Extent - Predicted_Invariant)

# Filter for years 2014 and 2016
df_2014 <- df_csv %>% filter(Year == 2014)
df_2016 <- df_csv %>% filter(Year == 2016)

# Plot residuals for 2014
plot_2014 <- ggplot(df_2014, aes(x = Date, y = Residual_Invariant)) +
  geom_line(color = "red", size = 1) +
  labs(title = "Residuals of Sea Ice Extent in 2014",
       x = "Date",
       y = "Residual Sea Ice Extent (million sq km)") +
  theme_minimal()

# Save the plot for 2014
ggsave("residuals_sea_ice_extent_2014.png", plot = plot_2014, width = 12, height = 6)

# Display the plot for 2014
print(plot_2014)

# Plot residuals for 2016
plot_2016 <- ggplot(df_2016, aes(x = Date, y = Residual_Invariant)) +
  geom_line(color = "red", size = 1) +
  labs(title = "Residuals of Sea Ice Extent in 2016",
       x = "Date",
       y = "Residual Sea Ice Extent (million sq km)") +
  theme_minimal()

# Save the plot for 2016
ggsave("residuals_sea_ice_extent_2016.png", plot = plot_2016, width = 12, height = 6)

# Display the plot for 2016
print(plot_2016)
