# Load required libraries
library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)
library(zoo)


# Load the CSV file
df_csv <- read.csv("/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv")

# Convert 'Year', 'Month', 'Day' to Date
df_csv$Date <- as.Date(with(df_csv, paste(Year, Month, Day, sep="-")), "%Y-%m-%d")
df_csv <- df_csv %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2023-12-31')) 
# Explanation: Filter data to include only the relevant date range (1978–2018).

# Make the dates numeric 
df_csv$tdate <- as.numeric(df_csv$Date) 
# Explanation: Convert Date to numeric for use in modeling (GAM).

# Calculate day of year (DOY)
df_csv$DOY <- yday(df_csv$Date)
# Explanation: Calculate the day of the year (1–365), which is essential for seasonal analysis.

# Make the Extent values numeric
df_csv$Extent <- as.numeric(df_csv$Extent)
# Explanation: Ensure Extent is numeric for analysis and modeling.

# Calculate min extent day per year
yearly_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(Date1 = Date[which.min(Extent)], .groups = 'drop')
# Explanation: Find the day of the year with the minimum sea ice extent for each year.

# Add Date2 (min date of last year) and Date3 (min date of next year)
yearly_stats <- yearly_stats %>%
  mutate(Date2 = lag(Date1),   # Date for last year
         Date3 = lead(Date1))  # Date for next year
# Explanation: Include information on the previous and next year’s minimum dates.

# Merge yearly_stats back into df_csv
df_csv <- df_csv %>%
  left_join(yearly_stats, by = "Year")
# Explanation: Add the computed yearly stats (Date1, Date2, Date3) back into the main dataset.

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
# Explanation: Compute the "t" variable for each year, representing the number of days since the phase reference point.

# Calculate t_min and t_max by year
t_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(t_min = min(t, na.rm = TRUE), 
            t_max = max(t, na.rm = TRUE)) %>%
  ungroup()
# Explanation: Calculate the minimum and maximum t-values for each year.

# Merge t_min and t_max back into the main dataset
df_csv <- df_csv %>%
  left_join(t_stats, by = "Year")
# Explanation: Add t_min and t_max back into the dataset for use in phase calculations.

# Calculate the phase-adjusted day of the year using t, t_min, and t_max
df_csv <- df_csv %>%
  rowwise() %>%
  mutate(phase = 365 * pbeta((t - t_min) / (t_max - t_min + 1e-10), 
                             shape1 = 1, shape2 = 1)) 
# Explanation: Use a Beta distribution cumulative density function (CDF) to calculate a phase-adjusted DOY.

# Calculate max and min sea ice extent per year
yearly_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(max_extent = max(Extent, na.rm = TRUE), 
            min_extent = min(Extent, na.rm = TRUE))
# Explanation: Compute the maximum and minimum sea ice extent for each year.

# Merge yearly stats back into the main dataset
df_csv <- df_csv %>%
  left_join(yearly_stats, by = "Year")

# Calculate the amplitude (max - min) and scaling factor for each year
df_csv <- df_csv %>%
  mutate(amplitude = max_extent - min_extent,
         scaling_factor = (Extent - min_extent) / amplitude)
# Explanation: Normalize the extent by calculating a scaling factor based on amplitude.

# Build the GAM model with amplitude adjustment
gam_model_amplitude_adjusted <- gam(scaling_factor ~ s(tdate, bs = "cc",k=150)+ s(DOY, bs = "cc", k = 100)+s(phase, bs = "cc", k = 100), 
                                    data = df_csv)
# Explanation: Fit a generalized additive model (GAM) to predict amplitude-adjusted scaling factors as functions of DOY and phase.

# Predict the amplitude-adjusted sea ice extent
df_csv$Predicted_Amplitude_Adjusted <- predict(gam_model_amplitude_adjusted, newdata = df_csv)
df_csv$Predicted_Amplitude_Adjusted_Extent <- df_csv$Predicted_Amplitude_Adjusted * df_csv$amplitude + df_csv$min_extent
# Explanation: Predict sea ice extent using the GAM model and convert scaling factors back to extent values.

# Calculate RMSE for the amplitude-adjusted annual cycle
amplitude_adjusted_rmse <- sqrt(mean((df_csv$Predicted_Amplitude_Adjusted_Extent - df_csv$Extent)^2, na.rm = TRUE))
print(amplitude_adjusted_rmse)
# Explanation: Evaluate the performance of the amplitude-adjusted cycle using RMSE.

# Add phase timing for advance and retreat
df_csv <- df_csv %>%
  mutate(advance_phase_timing = ifelse(phase < mean(phase[Extent == max_extent], na.rm = TRUE) - 5, "Ahead of Phase",
                                       ifelse(phase > mean(phase[Extent == max_extent], na.rm = TRUE) + 5, "Behind Phase", "On Phase")),
         retreat_phase_timing = ifelse(phase < mean(phase[Extent == min_extent], na.rm = TRUE) - 5, "Ahead of Phase",
                                       ifelse(phase > mean(phase[Extent == min_extent], na.rm = TRUE) + 5, "Behind Phase", "On Phase")))
# Explanation: Assign phase timing labels (ahead, behind, on) based on deviations from the long-term average.

# Add freezing and melting durations
yearly_durations <- df_csv %>%
  group_by(Year) %>%
  summarise(
    freezing_duration = max(DOY[Extent == max_extent]) - min(DOY[Extent == min_extent]),
    melting_duration = 365 - freezing_duration,
    rate_advance = amplitude / freezing_duration,
    rate_retreat = amplitude / melting_duration
  )
df_csv <- df_csv %>%
  left_join(yearly_durations, by = "Year")
# Explanation: Calculate freezing/melting durations and rates of advance/retreat.

# Extract amplitude and phase components
df_csv$Amplitude_Component <- predict(gam_model_amplitude_adjusted, newdata = df_csv, type = "terms")[, "s(DOY)"]
df_csv$Phase_Component <- predict(gam_model_amplitude_adjusted, newdata = df_csv, type = "terms")[, "s(phase)"]
# Explanation: Decompose GAM predictions into amplitude and phase components.

# Final DataFrame
summary_metrics <- df_csv %>%
  group_by(Year) %>%
  summarise(
    Min_Extent = unique(min_extent),
    Max_Extent = unique(max_extent),
    Freezing_Duration = unique(freezing_duration),
    Melting_Duration = unique(melting_duration),
    Rate_Advance = unique(rate_advance),
    Rate_Retreat = unique(rate_retreat)
  )
# Explanation: Create a summary DataFrame with metrics for each year.

# View the summary
print(summary_metrics)
# -----------------------------------
# Expand summary_metrics calculations
# -----------------------------------

library(dplyr)
library(ggplot2)

# 1. Timing of Minimum and Maximum Extent
timing_extents <- df_csv %>%
  group_by(Year) %>%
  summarise(
    DOY_Min = DOY[which.min(Extent)],
    DOY_Max = DOY[which.max(Extent)]
  )

summary_metrics <- summary_metrics %>%
  left_join(timing_extents, by = "Year")

# 2. Phase at Timing of Advance and Retreat
phase_anomalies <- df_csv %>%
  group_by(Year) %>%
  summarise(
    Phase_At_Advance = phase[which.max(Extent)],
    Phase_At_Retreat = phase[which.min(Extent)]
  ) %>%
  mutate(
    Anom_Phase_Advance = Phase_At_Advance - mean(Phase_At_Advance, na.rm = TRUE),
    Anom_Phase_Retreat = Phase_At_Retreat - mean(Phase_At_Retreat, na.rm = TRUE)
  )

summary_metrics <- summary_metrics %>%
  left_join(phase_anomalies, by = "Year")

# 3. Ice Cover Duration and Melt Period
summary_metrics <- summary_metrics %>%
  mutate(
    Ice_Cover_Duration = ifelse(DOY_Min > DOY_Max,
                                365 - (DOY_Min - DOY_Max),
                                DOY_Max - DOY_Min),
    Melt_Period = 365 - Ice_Cover_Duration
  )

# 4. DOY Anomalies for Min, Max, and Midpoint
summary_metrics <- summary_metrics %>%
  mutate(
    DOY_Anom_Min = DOY_Min - mean(DOY_Min, na.rm = TRUE),
    DOY_Anom_Max = DOY_Max - mean(DOY_Max, na.rm = TRUE),
    Midpoint_DOY = (DOY_Min + DOY_Max)/2,
    DOY_Anom_Midpoint = Midpoint_DOY - mean(Midpoint_DOY, na.rm = TRUE)
  )

# -----------------------------------
# Presentation-Ready Plots
# -----------------------------------

# 5.1 Rate of Advance and Retreat Over Time
ggplot(summary_metrics, aes(x = Year)) +
  geom_line(aes(y = Rate_Advance), color = "blue", size = 1) +
  geom_line(aes(y = Rate_Retreat), color = "red", size = 1) +
  geom_smooth(aes(y = Rate_Advance), method = "lm", color = "darkblue", se = FALSE, linetype = "dashed") +
  geom_smooth(aes(y = Rate_Retreat), method = "lm", color = "darkred", se = FALSE, linetype = "dashed") +
  labs(title = "Rate of Advance and Retreat (1978–2023)",
       y = "Rate (10⁶ km²/day)",
       x = "Year") +
  theme_minimal(base_size = 14)

# 5.2 Phase Timing at Advance and Retreat
ggplot(summary_metrics, aes(x = Year)) +
  geom_line(aes(y = Anom_Phase_Advance), color = "darkgreen", size = 1) +
  geom_line(aes(y = Anom_Phase_Retreat), color = "purple", size = 1) +
  geom_hline(yintercept = 0, linetype = "dotted") +
  labs(title = "Phase Anomalies at Advance and Retreat",
       y = "Phase Anomaly (days)",
       x = "Year") +
  theme_minimal(base_size = 14)

# 5.3 Change in Melt Period Duration Over Time
ggplot(summary_metrics, aes(x = Year, y = Melt_Period)) +
  geom_line(color = "tomato", size = 1) +
  geom_smooth(method = "lm", color = "darkred", se = FALSE, linetype = "dashed") +
  labs(title = "Melt Period Duration Over Time",
       y = "Melt Period (days)",
       x = "Year") +
  theme_minimal(base_size = 14)

# 5.4 DOY Anomalies of Min and Max Extent
ggplot(summary_metrics, aes(x = Year)) +
  geom_line(aes(y = DOY_Anom_Min), color = "navy", size = 1) +
  geom_line(aes(y = DOY_Anom_Max), color = "orange", size = 1) +
  geom_hline(yintercept = 0, linetype = "dotted") +
  labs(title = "Anomalies in Day of Year for Min and Max Extent",
       y = "DOY Anomaly (days)",
       x = "Year") +
  theme_minimal(base_size = 14)

# -----------------------------------
# Optional: Trendline Summaries
# -----------------------------------

# Quick trend statistics
summary(lm(Rate_Advance ~ Year, data = summary_metrics))
summary(lm(Rate_Retreat ~ Year, data = summary_metrics))
summary(lm(Melt_Period ~ Year, data = summary_metrics))
summary(lm(Anom_Phase_Advance ~ Year, data = summary_metrics))
summary(lm(Anom_Phase_Retreat ~ Year, data = summary_metrics))



write.csv(summary_metrics, "/Users/fridaperez/Desktop/summary_phase_metrics.csv", row.names = FALSE)

