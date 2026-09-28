library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)

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

# Print RMSE values
print(traditional_rmse)
print(invariant_rmse)
# Load necessary libraries
library(ggplot2)

# Data preparation for plotting
average_annual_cycle <- df_csv %>% 
  group_by(DOY) %>% 
  summarize(mean_extent = mean(Extent, na.rm = TRUE))

invariant_cycle <- df_csv %>%
  group_by(DOY) %>%
  summarize(mean_invariant = mean(Predicted_Invariant, na.rm = TRUE))

# Plot
ggplot() +
  geom_line(data = average_annual_cycle, aes(x = DOY, y = mean_extent, color = "Traditional Cycle"), size = 1) +
  geom_line(data = invariant_cycle, aes(x = DOY, y = mean_invariant, color = "Invariant Cycle"), size = 1) +
  labs(title = "Modeled Sea Ice Extent Cycles",
       subtitle = paste("Traditional RMSE:", round(traditional_rmse, 2), 
                        " - Invariant RMSE:", round(invariant_rmse, 2)),
       x = "Day of Year (DOY)",
       y = "Sea Ice Extent (million sq km)") +
  scale_color_manual(values = c("Traditional Cycle" = "blue", 
                                "Invariant Cycle" = "red")) +
  theme_minimal() +
  theme(legend.title = element_blank())

# Display the plot
ggsave("modeled_sea_ice_extent_cycles.png", width = 12, height = 6)
print(ggplot() +
        geom_line(data = average_annual_cycle, aes(x = DOY, y = mean_extent, color = "Traditional Cycle"), size = 1) +
        geom_line(data = invariant_cycle, aes(x = DOY, y = mean_invariant, color = "Invariant Cycle"), size = 1) +
        labs(title = "Modeled Sea Ice Extent Cycles",
             subtitle = paste("Traditional RMSE:", round(traditional_rmse, 2), 
                              " - Invariant RMSE:", round(invariant_rmse, 2)),
             x = "Day of Year (DOY)",
             y = "Sea Ice Extent (million sq km)") +
        scale_color_manual(values = c("Traditional Cycle" = "blue", 
                                      "Invariant Cycle" = "red")) +
        theme_minimal() +
        theme(legend.title = element_blank()))
