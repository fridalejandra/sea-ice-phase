# 
# PROGRAM NAME:         IAC_NSIDC - 07.12.2024.R
#
# AURTHOR:              Frida Perez.
#
# PROJECT:              A quantitative study on modeling the annual cycle of daily Antarctic 
#                       sea ice extent 
#
# DATE FIRST CREATED:   Jul 12 2024.
#
# DATE LAST MODIFIED:   Jul 12 2024.
#
# MODIFIED BY:          Frida Perez.
#
# COMMENTS:             Type of Data Used: Daily sea ice extent data
#                       Instruments Used: Advanced Microwave Scanning Radiometer – 
#                       Earth Observing System (AMSR-E), Bootstrap Algorithm

          #   Set the working directory to a specific path
setwd("E:/HOkuku/Precision Consulting/Independent - Frida Perez")

          #   Install necessary packages if not already installed (Enable next code line to install)
#      install.packages(c("ggplot2", "dplyr", "lubridate", "readxl", "splines", "caret"))

      # Load the libraries
library(ggplot2)
library(dplyr)
library(lubridate)
library(readxl)
library(tidyverse)
library(mgcv)
library(splines)
library(caret)

          #   Import the data.
    # Import the Excel file
sea_ice_data <- read_excel("S_seaice_extent_daily_v4.0.xlsx")

    #   Inspect the data
head(sea_ice_data)
summary(sea_ice_data)
str(sea_ice_data)

    #   Check for missing values
sum(is.na(sea_ice_data$Extent))

          #   Convert Year, Month, Day into a Date object
sea_ice_data$Date <- as.Date(with(sea_ice_data, paste(Year, Month, Day, sep="-")), "%Y-%m-%d")

    #   Create a day of the year variable
sea_ice_data$DayOfYear <- as.numeric(format(sea_ice_data$Date, "%j"))

          # Visualize the Data
    #   Time Series Plot
ggplot(sea_ice_data, aes(x = Date, y = Extent)) +
  geom_line() +
  labs(title = "Daily Antarctic Sea Ice Extent Over Time",
       x = "Date", y = "Sea Ice Extent") +
  theme_minimal()

    #   Seasonal Plot
ggplot(sea_ice_data, aes(x = DayOfYear, y = Extent)) +
  geom_line(stat = "summary", fun = "mean") +
  labs(title = "Average Daily Antarctic Sea Ice Extent by Day of Year",
       x = "Day of Year", y = "Sea Ice Extent") +
  theme_minimal()

    #   Monthly Boxplot
ggplot(sea_ice_data, aes(x = factor(Month), y = Extent)) +
  geom_boxplot() +
  labs(title = "Monthly Variation in Antarctic Sea Ice Extent",
       x = "Month", y = "Sea Ice Extent") +
  theme_minimal()

    #   Yearly Trend
sea_ice_data %>%
  group_by(Year) %>%
  summarize(YearlyMeanExtent = mean(Extent, na.rm = TRUE)) %>%
  ggplot(aes(x = Year, y = YearlyMeanExtent)) +
  geom_line() +
  labs(title = "Yearly Average Antarctic Sea Ice Extent",
       x = "Year", y = "Average Sea Ice Extent") +
  theme_minimal()

          #   Summary Statistics
    # Summary statistics for sea ice extent
summary(sea_ice_data$Extent)

    # Summary statistics by month
sea_ice_data %>%
  group_by(Month) %>%
  summarize(MeanExtent = mean(Extent, na.rm = TRUE),
            SDExtent = sd(Extent, na.rm = TRUE))

          #   Fit the GAM
gam_model <- gam(Extent ~ s(DayOfYear, bs="cc"), data=sea_ice_data, family=gaussian())

    #   Summary of the model
summary(gam_model)

    #   Plot the smooth term
plot(gam_model, shade=TRUE)

                #   TRADITIONAL CYCLE VERSUS INVARIANT CYCLE - CUBIC SPLINES
          #   Perform cross-validation to compare the two models more robustly
    #   Define a control function for cross-validation
train_control <- trainControl(method = "cv", number = 10)

    #   Traditional model cross-validation
traditional_model <- train(Extent ~ DayOfYear, data = sea_ice_data, method = "lm", trControl = train_control)
traditional_rmse <- traditional_model$results$RMSE

    #   Cubic spline model cross-validation
spline_model <- train(Extent ~ bs(DayOfYear, df = 12), data = sea_ice_data, method = "lm", trControl = train_control)
    #   Summary of the model
summary(spline_model)

spline_rmse <- spline_model$results$RMSE


  #   Add predictions to the original dataframe
sea_ice_data$Predicted_Traditional <- predict(traditional_model, newdata = sea_ice_data)
sea_ice_data$Predicted_Spline <- predict(spline_model, newdata = sea_ice_data)

    #   Print cross-validated RMSE
print(traditional_rmse)
print(spline_rmse)

    #   Calculate residuals
sea_ice_data$Traditional_residuals <- residuals(traditional_model)
sea_ice_data$Spline_residuals <- residuals(spline_model)

    # Plot residuals
ggplot(sea_ice_data, aes(x = DayOfYear, y = Traditional_residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals of Traditional Model",
       x = "Day of Year",
       y = "Traditional Residuals")

ggplot(sea_ice_data, aes(x = DayOfYear, y = Spline_residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals of Cubic Spline Model",
       x = "Day of Year",
       y = "Cubic Spline Residuals")

          # Modeled Sea Ice Extent Cycles Plots
    # Data preparation for plotting
average_annual_cycle <- sea_ice_data %>% 
  group_by(DayOfYear) %>% 
  summarize(mean_extent = mean(Extent, na.rm = TRUE))

invariant_cycle <- sea_ice_data %>%
  group_by(DayOfYear) %>%
  summarize(mean_invariant = mean(Predicted_Spline, na.rm = TRUE))

    # Plot
ggplot() +
  geom_line(data = average_annual_cycle, aes(x = DayOfYear, y = mean_extent, color = "Traditional Cycle"), size = 1) +
  geom_line(data = invariant_cycle, aes(x = DayOfYear, y = mean_invariant, color = "Invariant Cycle"), size = 1) +
  labs(title = "Modeled Sea Ice Extent Cycles",
       subtitle = paste("Traditional RMSE:", round(traditional_rmse, 2), 
                        " - Invariant RMSE:", round(spline_rmse, 2)),
       x = "Day of Year",
       y = "Sea Ice Extent (million sq km)") +
  scale_color_manual(values = c("Traditional Cycle" = "blue", 
                                "Invariant Cycle" = "red")) +
  theme_minimal() +
  theme(legend.title = element_blank())

# Display the plot
ggsave("modeled_sea_ice_extent_cycles.png", width = 12, height = 6)
print(ggplot() +
        geom_line(data = average_annual_cycle, aes(x = DayOfYear, y = mean_extent, color = "Traditional Cycle"), size = 1) +
        geom_line(data = invariant_cycle, aes(x = DayOfYear, y = mean_invariant, color = "Invariant Cycle"), size = 1) +
        labs(title = "Modeled Sea Ice Extent Cycles",
             subtitle = paste("Traditional RMSE:", round(traditional_rmse, 2), 
                              " - Invariant RMSE:", round(spline_rmse, 2)),
             x = "Day of Year (DayOfYear)",
             y = "Sea Ice Extent (million sq km)") +
        scale_color_manual(values = c("Traditional Cycle" = "blue", 
                                      "Invariant Cycle" = "red")) +
        theme_minimal() +
        theme(legend.title = element_blank()))