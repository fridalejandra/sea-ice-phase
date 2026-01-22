# Load necessary libraries
library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)
library(ggplot2)
library(patchwork)

# Load the CSV file
df_csv <- read.csv("/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv")

# Convert 'Year', 'Month', 'Day' to Date
df_csv$Date <- as.Date(with(df_csv, paste(Year, Month, Day, sep="-")), "%Y-%m-%d")
df_csv <- df_csv %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2023-12-31'))  # Updated to end of 2023

# Calculate day of year (DOY) and Year
df_csv$DOY <- yday(df_csv$Date)
df_csv$Year <- year(df_csv$Date)

# Make the dates numeric
df_csv$tdate <- as.numeric(df_csv$Date)

# Make the Extent values numeric
df_csv$Extent <- as.numeric(df_csv$Extent)

# Calculate min extent day per year
yearly_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(Date1 = Date[which.min(Extent)], .groups = 'drop')

# Add Date2 (min date of last year) and Date3 (min date of next year)
yearly_stats <- yearly_stats %>%
  mutate(Date2 = lag(Date1),   # Date for last year
         Date3 = lead(Date1))  # Date for next year

# Merge yearly_stats back into df_csv
df_csv <- df_csv %>%
  left_join(yearly_stats, by = "Year")

# Calculate t-values based on the logic provided
df_csv <- df_csv %>%
  rowwise() %>%
  mutate(t = case_when(
    Year == 1978  ~ 365 - as.numeric(Date3 - Date),  # Special case for 1978
    Year == 1979 & Date < Date1 ~ 365 - as.numeric(Date1 - Date),  # Special case for 1979
    Date >= Date1 ~ as.numeric(Date - Date1),  # For current year
    Date < Date1 ~ as.numeric(Date - Date2)    # For last year
  )) %>%
  ungroup()

# Calculate t_min and t_max by year
t_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(t_min = min(t, na.rm = TRUE), 
            t_max = max(t, na.rm = TRUE)) %>%
  ungroup()

# Merge t_min and t_max back into the main dataset
df_csv <- df_csv %>%
  left_join(t_stats, by = "Year")

# Calculate the phase-adjusted day of the year using t, t_min, and t_max
df_csv <- df_csv %>%
  rowwise() %>%
  mutate(phase = 365 * pbeta((t - t_min) / (t_max - t_min + 1e-10), 
                             shape1 = 1, shape2 = 1))  # Using Beta distribution CDF for phase adjustment

# Remove rows where Year is 1978 and save the result to df_csv2
df_csv2 <- df_csv %>% 
  filter(Year != 1978)

# Build the GAM model with phase adjustment
gam_model_phase_adjusted <- gam(Extent ~ s(tdate, bs = "cc", k = 75) + 
                                  s(DOY, bs = "cc", k = 100) + 
                                  s(phase, bs = "cc", k = 100, fx = FALSE), 
                                data = df_csv2)

# Predict the phase-adjusted sea ice extent
df_csv2$Predicted_Phase_Adjusted <- predict(gam_model_phase_adjusted, newdata = df_csv2)

# Calculate RMSE for the phase-adjusted annual cycle
phase_adjusted_rmse <- sqrt(mean((df_csv2$Predicted_Phase_Adjusted - df_csv2$Extent)^2, na.rm = TRUE))
print(phase_adjusted_rmse)

# Calculate retreat timing: Find the day after the annual max when extent starts declining
retreat_timing_gam <- df_csv %>%
  group_by(Year) %>%
  filter(Extent == max(Extent, na.rm = TRUE)) %>%   # Identify the max extent day
  summarise(Max_DOY = DOY, Max_Extent = Extent, .groups = 'drop') %>%
  left_join(df_csv, by = "Year") %>%
  filter(DOY > Max_DOY & Extent < Max_Extent) %>%   # Find the first day after max when extent declines
  group_by(Year) %>%
  filter(DOY == min(DOY, na.rm = TRUE)) %>%         # First declining day
  summarise(Retreat_DOY = DOY, .groups = 'drop')

# Calculate advance timing: Find the first day after February when extent begins to increase
advance_timing_gam <- df_csv %>%
  filter(DOY >= 50 & DOY <= 68) %>%  # Filter to the relevant DOY range for advance
  group_by(Year) %>%
  filter(Extent == min(Extent, na.rm = TRUE)) %>%  # Identify the min extent day within range
  summarise(Min_DOY = DOY, Min_Extent = Extent, .groups = 'drop') %>%
  left_join(df_csv, by = "Year") %>%
  filter(DOY > Min_DOY & Extent > Min_Extent) %>%  # Find the first day after min when extent increases
  group_by(Year) %>%
  filter(DOY == min(DOY, na.rm = TRUE)) %>%        # First advancing day
  summarise(Advance_DOY = DOY, .groups = 'drop')

# Combine retreat and advance timing into one table, filtering for complete cases
timing_table <- retreat_timing_gam %>%
  left_join(advance_timing_gam, by = "Year") %>%
  rename(Retreat_Timing_DOY = Retreat_DOY, Advance_Timing_DOY = Advance_DOY) %>%
  drop_na()  # Remove rows with any NA values to ensure matching row counts

# Output the timing table
print(timing_table)

# Retreat Timing Curvilinear Trend Model
retreat_gam <- gam(Retreat_Timing_DOY ~ s(Year, k = 10, bs = "tp"), data = timing_table)

# Advance Timing Curvilinear Trend Model
advance_gam <- gam(Advance_Timing_DOY ~ s(Year, k = 10, bs = "tp"), data = timing_table)

# Predict retreat and advance values for visualization
timing_table$Retreat_DOY_Predicted <- predict(retreat_gam)
timing_table$Advance_DOY_Predicted <- predict(advance_gam)


# Retreat Timing Plot with Curvilinear Trend and Annotation
retreat_plot_curvilinear <- ggplot(timing_table, aes(x = Year, y = Retreat_Timing_DOY)) +
  geom_line(color = "#4682B4", size = 1) +
  geom_line(aes(y = Retreat_DOY_Predicted), color = "darkblue", size = 1, linetype = "dashed") +
  labs(title = "Retreat Timing", y = "Day of Year (DOY)") +
  theme_minimal(base_size = 14) +
  theme(
    strip.text = element_text(size = 12, face = "bold"),
    panel.grid.major = element_line(color = "grey80"),
    panel.grid.minor = element_line(color = "grey90")
  )

# Advance Timing Plot with Curvilinear Trend and Annotation
advance_plot_curvilinear <- ggplot(timing_table, aes(x = Year, y = Advance_Timing_DOY)) +
  geom_line(color = "#FF6347", size = 1) +
  geom_line(aes(y = Advance_DOY_Predicted), color = "red", size = 1, linetype = "dashed") +
  labs(title = "Advance Timing", y = "Day of Year (DOY)", x = "Year") +
  theme_minimal(base_size = 14) +
  theme(
    strip.text = element_text(size = 12, face = "bold"),
    panel.grid.major = element_line(color = "grey80"),
    panel.grid.minor = element_line(color = "grey90")
  )

# Combine the curvilinear trend plots vertically
combined_plot_curvilinear <- retreat_plot_curvilinear / advance_plot_curvilinear

# Display the combined plot
combined_plot_curvilinear #+ plot_annotation(title = "Curvilinear Trends in Sea Ice Retreat and Advance Timing with 2016 Annotation")

############### Correlation ##################

# Load necessary libraries
library(mgcv)
library(ggplot2)
library(patchwork)

# Fit a curvilinear regression (GAM) for Retreat Timing DOY
retreat_gam <- gam(Retreat_Timing_DOY ~ s(Year, bs = "tp", k = 10), data = timing_table)
advance_gam <- gam(Advance_Timing_DOY ~ s(Year, bs = "tp", k = 10), data = timing_table)

# Predict values for plotting
timing_table$Retreat_DOY_Predicted <- predict(retreat_gam)
timing_table$Advance_DOY_Predicted <- predict(advance_gam)

# Calculate correlation between Year and predicted values from the GAM models
retreat_gam_correlation <- cor(timing_table$Year, timing_table$Retreat_DOY_Predicted)
advance_gam_correlation <- cor(timing_table$Year, timing_table$Advance_DOY_Predicted)

# Output the GAM correlation values
cat("GAM Correlation between Year and Retreat Timing DOY:", retreat_gam_correlation, "\n")
cat("GAM Correlation between Year and Advance Timing DOY:", advance_gam_correlation, "\n")

# Retreat Timing Plot with Curvilinear Trend
retreat_plot <- ggplot(timing_table, aes(x = Year, y = Retreat_Timing_DOY)) +
  geom_point(color = "#4682B4") +
  geom_line(aes(y = Retreat_DOY_Predicted), color = "darkblue", size = 1) +
  annotate("text", x = min(timing_table$Year) + 1, y = max(timing_table$Retreat_Timing_DOY) - 2,
           label = paste("Correlation:", round(retreat_gam_correlation, 2)),
           color = "darkblue", size = 4, hjust = 0) +
  labs(title = "Retreat Timing DOY vs. Year (Curvilinear Trend)", y = "Retreat Timing (DOY)", x = "Year") +
  theme_minimal()

# Advance Timing Plot with Curvilinear Trend
advance_plot <- ggplot(timing_table, aes(x = Year, y = Advance_Timing_DOY)) +
  geom_point(color = "#FF6347") +
  geom_line(aes(y = Advance_DOY_Predicted), color = "red", size = 1) +
  annotate("text", x = min(timing_table$Year) + 1, y = max(timing_table$Advance_Timing_DOY) - 2,
           label = paste("Correlation:", round(advance_gam_correlation, 2)),
           color = "red", size = 4, hjust = 0) +
  labs(title = "Advance Timing DOY vs. Year (Curvilinear Trend)", y = "Advance Timing (DOY)", x = "Year") +
  theme_minimal()

# Combine the plots
combined_plot <- retreat_plot / advance_plot
combined_plot + plot_annotation(title = "Curvilinear Trends in Sea Ice Retreat and Advance Timing with Correlations")






# Fit GAM models for smoother trends in retreat and advance timings
retreat_gam <- gam(Retreat_Timing_DOY ~ s(Year, bs = "tp", k = 10), data = timing_table)
advance_gam <- gam(Advance_Timing_DOY ~ s(Year, bs = "tp", k = 10), data = timing_table)

# Predict values for visualization (smoothed trends)
timing_table <- timing_table %>%
  mutate(Retreat_DOY_Predicted = predict(retreat_gam),
         Advance_DOY_Predicted = predict(advance_gam))

# Calculate correlation between Year and observed values (Retreat_Timing_DOY and Advance_Timing_DOY)
retreat_correlation_observed <- cor(timing_table$Year, timing_table$Retreat_Timing_DOY, use = "complete.obs")
advance_correlation_observed <- cor(timing_table$Year, timing_table$Advance_Timing_DOY, use = "complete.obs")

# Plot retreat timing with curvilinear trend and observed correlation annotation
retreat_plot <- ggplot(timing_table, aes(x = Year, y = Retreat_Timing_DOY)) +
  geom_point(color = "#4682B4") +
  geom_line(aes(y = Retreat_DOY_Predicted), color = "darkblue", size = 1) +
  annotate("text", x = min(timing_table$Year) + 1, y = max(timing_table$Retreat_Timing_DOY) - 5,
           label = paste("Observed Correlation:", round(retreat_correlation_observed, 2)),
           color = "darkblue", size = 4, hjust = 0) +
  labs(title = "Retreat Timing DOY vs. Year (Curvilinear Trend)", y = "Retreat Timing (DOY)", x = "Year") +
  theme_minimal()

# Plot advance timing with curvilinear trend and observed correlation annotation
advance_plot <- ggplot(timing_table, aes(x = Year, y = Advance_Timing_DOY)) +
  geom_point(color = "#FF6347") +
  geom_line(aes(y = Advance_DOY_Predicted), color = "red", size = 1) +
  annotate("text", x = min(timing_table$Year) + 1, y = max(timing_table$Advance_Timing_DOY) - 5,
           label = paste("Observed Correlation:", round(advance_correlation_observed, 2)),
           color = "red", size = 4, hjust = 0) +
  labs(title = "Advance Timing DOY vs. Year (Curvilinear Trend)", y = "Advance Timing (DOY)", x = "Year") +
  theme_minimal()

# Combine the plots
combined_plot <- retreat_plot / advance_plot +
  plot_annotation(title = "Curvilinear Trends in Sea Ice Retreat and Advance Timing with Observed Correlations")

# Display the combined plot
print(combined_plot)






# Calculate the correlation between advance and retreat timings
advance_retreat_correlation <- cor(timing_table$Advance_Timing_DOY, timing_table$Retreat_Timing_DOY, use = "complete.obs")

# Output the correlation value
cat("Correlation between Advance and Retreat Timings:", advance_retreat_correlation, "\n")



# Shift the advance timing by one year to match it with the previous year's retreat timing
timing_table <- timing_table %>%
  arrange(Year) %>%
  mutate(Next_Advance_Timing_DOY = lead(Advance_Timing_DOY))

# Calculate the correlation between retreat timing and next year's advance timing
retreat_advance_correlation <- cor(timing_table$Retreat_Timing_DOY, timing_table$Next_Advance_Timing_DOY, use = "complete.obs")

# Output the correlation value
cat("Correlation between Retreat Timing and Following Year's Advance Timing:", retreat_advance_correlation, "\n")


