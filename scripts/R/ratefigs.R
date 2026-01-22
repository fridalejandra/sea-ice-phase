library(tidyverse)
library(mgcv)
library(lubridate)
library(ggplot2)

# Load the CSV file
df_csv <- read.csv("/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv")

# Convert 'Year', 'Month', 'Day' to Date
df_csv$Date <- as.Date(with(df_csv, paste(Year, Month, Day, sep="-")), "%Y-%m-%d")
df_csv <- df_csv %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2023-12-31'))

# Calculate day of year (DOY)
df_csv$DOY <- yday(df_csv$Date)

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

# Calculate max and min sea ice extent per year
yearly_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(max_extent = max(Extent, na.rm = TRUE), 
            min_extent = min(Extent, na.rm = TRUE))

# Merge yearly stats back into the main dataset
df_csv <- df_csv %>%
  left_join(yearly_stats, by = "Year")

# Calculate the amplitude (max - min) and scaling factor for each year
df_csv <- df_csv %>%
  mutate(amplitude = max_extent - min_extent,
         scaling_factor = (Extent - min_extent) / amplitude)

# Build the GAM model with amplitude adjustment
gam_model_amplitude_adjusted <- gam(scaling_factor ~ s(DOY, bs = "cc", k = 100) + s(phase, bs = "cc", k = 100), 
                                    data = df_csv)

# Predict the amplitude-adjusted sea ice extent
df_csv$Predicted_Amplitude_Adjusted <- predict(gam_model_amplitude_adjusted, newdata = df_csv)

# Convert the predicted scaling factor back to sea ice extent
df_csv$Predicted_Amplitude_Adjusted_Extent <- df_csv$Predicted_Amplitude_Adjusted * df_csv$amplitude + df_csv$min_extent

# Calculate rate of change (derivative) of SIE
df_csv <- df_csv %>%
  arrange(Date) %>%
  group_by(Year) %>%
  mutate(rate_of_change = c(NA, diff(Predicted_Amplitude_Adjusted_Extent))) %>%
  ungroup()

# Calculate maximum and minimum extent dates and values for each year
melting_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(
    max_extent = max(Extent, na.rm = TRUE),
    min_extent = min(Extent, na.rm = TRUE),
    max_date = Date[which.max(Extent)],
    min_date = Date[which.min(Extent)]
  ) %>%
  mutate(
    DOY_max = yday(max_date),
    DOY_min = yday(min_date),
    amplitude = max_extent - min_extent,               # Total ice lost during melting season
    duration = as.numeric(min_date - max_date),        # Duration in days from max to min
    avg_rate_of_change = amplitude / duration          # Average daily rate of change
  )

# Calculate average DOY for max and min across all years
avg_DOY <- melting_stats %>%
  summarise(
    avg_DOY_max = mean(DOY_max, na.rm = TRUE),
    avg_DOY_min = mean(DOY_min, na.rm = TRUE)
  )

# Add columns indicating whether DOY_max and DOY_min are ahead or behind the average
melting_stats <- melting_stats %>%
  mutate(
    max_timing = ifelse(DOY_max < avg_DOY$avg_DOY_max, "Ahead", "Behind"),
    min_timing = ifelse(DOY_min < avg_DOY$avg_DOY_min, "Ahead", "Behind")
  )

# Filter for the years of interest and select relevant columns for the table
years_of_interest <- c(2016, 2013, 2021, 2022)
selected_years_stats <- melting_stats %>%
  filter(Year %in% years_of_interest) %>%
  select(Year, DOY_max, max_timing, DOY_min, min_timing, duration, avg_rate_of_change)

# Print the table
print(selected_years_stats)

# Plot the rate of change for each year starting from the day after Julian day of minimum extent
for (year in years_of_interest) {
  # Filter data for the specific year starting from DOY_min + 1 onward
  DOY_min <- selected_years_stats %>% filter(Year == year) %>% pull(DOY_min)
  df_year <- df_csv %>% filter(Year == year & DOY > DOY_min)  # Start from DOY_min + 1
  
  # Plot with a horizontal line at zero
  plot <- ggplot(df_year, aes(x = DOY, y = rate_of_change)) +
    geom_line(color = "blue") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    labs(title = paste("Rate of Change in Sea Ice Extent (SIE) for", year),
         x = "Julian Day (Starting from DOY_min + 1)",
         y = "Rate of Change (million square km per day)") +
    theme_minimal()
  
  print(plot)  # Display each plot
}

# Plot the rate of change for each year starting from Julian day of maximum extent
for (year in years_of_interest) {
  # Filter data for the specific year starting from DOY_max onward
  DOY_max <- selected_years_stats %>% filter(Year == year) %>% pull(DOY_max)
  df_year <- df_csv %>% filter(Year == year & DOY >= DOY_max)  # Start from DOY_max
  
  # Plot with a horizontal line at zero
  plot <- ggplot(df_year, aes(x = DOY, y = rate_of_change)) +
    geom_line(color = "blue") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    labs(title = paste("Rate of Change in Sea Ice Extent (SIE) for", year),
         x = "Julian Day (Starting from DOY_max)",
         y = "Rate of Change (million square km per day)") +
    theme_minimal()
  
  print(plot)  # Display each plot
  
  # Calculate maximum and minimum extent dates and values for each year
  melting_stats <- df_csv %>%
    group_by(Year) %>%
    summarise(
      max_extent = max(Extent, na.rm = TRUE),
      min_extent = min(Extent, na.rm = TRUE),
      max_date = Date[which.max(Extent)],
      min_date = Date[which.min(Extent)]
    ) %>%
    mutate(
      DOY_max = yday(max_date),
      DOY_min = yday(min_date),
      amplitude = max_extent - min_extent,               # Total ice lost during melting season
      duration = as.numeric(min_date - max_date),        # Duration in days from max to min
      avg_rate_of_change = amplitude / duration          # Average daily rate of change
    )
  
  # Calculate average DOY for max and min across all years
  avg_DOY <- melting_stats %>%
    summarise(
      avg_DOY_max = mean(DOY_max, na.rm = TRUE),
      avg_DOY_min = mean(DOY_min, na.rm = TRUE)
    )
  
  # Add columns indicating whether DOY_max and DOY_min are ahead or behind the average and by how many days
  melting_stats <- melting_stats %>%
    mutate(
      max_timing = ifelse(DOY_max < avg_DOY$avg_DOY_max, "Ahead", "Behind"),
      max_days_diff = DOY_max - avg_DOY$avg_DOY_max,
      min_timing = ifelse(DOY_min < avg_DOY$avg_DOY_min, "Ahead", "Behind"),
      min_days_diff = DOY_min - avg_DOY$avg_DOY_min
    )
  
  # Filter for the years of interest and select relevant columns for the table
  years_of_interest <- c(2016, 2013, 2021, 2022)
  selected_years_stats <- melting_stats %>%
    filter(Year %in% years_of_interest) %>%
    select(Year, DOY_max, max_timing, max_days_diff, DOY_min, min_timing, min_days_diff, duration, avg_rate_of_change)
  
  # Print the table
  print(selected_years_stats)
}
# Calculate average duration and average rate of change across all years
avg_duration <- mean(melting_stats$duration, na.rm = TRUE)
avg_rate_of_change_overall <- mean(melting_stats$avg_rate_of_change, na.rm = TRUE)

# Add columns indicating the difference from the average duration and average rate of change
melting_stats <- melting_stats %>%
  mutate(
    duration_diff = duration - avg_duration,  # Difference in days from average duration
    rate_of_change_diff = avg_rate_of_change - avg_rate_of_change_overall  # Difference in rate from average
  )

# Update selected_years_stats to include the new columns
selected_years_stats <- melting_stats %>%
  filter(Year %in% years_of_interest) %>%
  select(
    Year, DOY_max, max_timing, max_days_diff, DOY_min, min_timing, min_days_diff,
    duration, duration_diff, avg_rate_of_change, rate_of_change_diff
  )

# Print the updated table
print(selected_years_stats)

# Specify the file path where you want to save the CSV
write.csv(selected_years_stats, "/Users/fridaperez/Desktop/selected_years_stats.csv", row.names = FALSE)

avg_duration_overall <- mean(melting_stats$duration, na.rm = TRUE)
avg_rate_of_change_overall <- mean(melting_stats$avg_rate_of_change, na.rm = TRUE)


# Calculate average duration and rate of change for each year
annual_stats <- melting_stats %>%
  group_by(Year) %>%
  summarize(
    avg_duration = mean(duration, na.rm = TRUE),
    avg_rate_of_change = mean(avg_rate_of_change, na.rm = TRUE)
  )

# Add difference columns for each year relative to overall averages
annual_stats <- annual_stats %>%
  mutate(
    duration_diff = avg_duration - avg_duration_overall,  # Difference from the overall average
    rate_of_change_diff = avg_rate_of_change - avg_rate_of_change_overall
  )

# Print the annual stats for analysis
print(annual_stats)

# Export annual stats as CSV
write.csv(annual_stats, "/Users/fridaperez/Desktop/annual_stats.csv", row.names = FALSE)


# Assuming df_csv contains your sea ice data with columns 'Year', 'Extent', and 'Date'
library(tidyverse)
library(lubridate)

# Calculate minimum and maximum extent dates and values for each year
melting_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(
    max_extent = max(Extent, na.rm = TRUE),
    min_extent = min(Extent, na.rm = TRUE),
    max_date = Date[which.max(Extent)],
    min_date = Date[which.min(Extent)]
  ) %>%
  mutate(
    DOY_max = yday(max_date),
    DOY_min = yday(min_date),
    amplitude = max_extent - min_extent,               # Total ice lost during melting season
    duration = as.numeric(min_date - max_date),        # Duration in days from max to min
    avg_rate_of_change = amplitude / duration          # Average daily rate of change
  )

# Optional: print to check the structure
print(melting_stats)







# Load necessary libraries
library(tidyverse)
library(mgcv)
library(ggplot2)

# Assuming `melting_stats` is recreated as per previous instructions and contains 'Year', 'avg_rate_of_change', 'min_extent', and 'DOY_min'

# Create a new dataframe with lagged avg_rate_of_change to represent the previous year's rate
melting_stats_lagged <- melting_stats %>%
  mutate(
    prev_rate_of_change = lag(avg_rate_of_change),   # Previous year's rate of retreat
    next_min_extent = lead(min_extent),              # Next year's min extent
    next_DOY_min = lead(DOY_min)                     # Next year's min DOY
  ) %>%
  filter(!is.na(prev_rate_of_change) & !is.na(next_min_extent) & !is.na(next_DOY_min))  # Remove rows with NA values

# Calculate IQR for each relevant variable
iqr_prev_rate <- IQR(melting_stats_lagged$prev_rate_of_change, na.rm = TRUE)
iqr_min_extent <- IQR(melting_stats_lagged$next_min_extent, na.rm = TRUE)
iqr_DOY_min <- IQR(melting_stats_lagged$next_DOY_min, na.rm = TRUE)

# Define filtering limits (1.5 times the IQR is a common threshold)
lower_bound_rate = quantile(melting_stats_lagged$prev_rate_of_change, 0.25, na.rm = TRUE) - 1.5 * iqr_prev_rate
upper_bound_rate = quantile(melting_stats_lagged$prev_rate_of_change, 0.75, na.rm = TRUE) + 1.5 * iqr_prev_rate

lower_bound_extent = quantile(melting_stats_lagged$next_min_extent, 0.25, na.rm = TRUE) - 1.5 * iqr_min_extent
upper_bound_extent = quantile(melting_stats_lagged$next_min_extent, 0.75, na.rm = TRUE) + 1.5 * iqr_min_extent

lower_bound_DOY = quantile(melting_stats_lagged$next_DOY_min, 0.25, na.rm = TRUE) - 1.5 * iqr_DOY_min
upper_bound_DOY = quantile(melting_stats_lagged$next_DOY_min, 0.75, na.rm = TRUE) + 1.5 * iqr_DOY_min

# Identify and print DOY of outliers
outliers <- melting_stats_lagged %>%
  filter(
    prev_rate_of_change < lower_bound_rate | prev_rate_of_change > upper_bound_rate |
      next_min_extent < lower_bound_extent | next_min_extent > upper_bound_extent |
      next_DOY_min < lower_bound_DOY | next_DOY_min > upper_bound_DOY
  )
print("Outliers and their DOY values:")
print(outliers %>% select(Year, next_DOY_min, prev_rate_of_change, next_min_extent))

# Filter out the outliers for visualization
filtered_data <- melting_stats_lagged %>%
  filter(
    prev_rate_of_change >= lower_bound_rate & prev_rate_of_change <= upper_bound_rate,
    next_min_extent >= lower_bound_extent & next_min_extent <= upper_bound_extent,
    next_DOY_min >= lower_bound_DOY & next_DOY_min <= upper_bound_DOY
  )

# Correlations for filtered data
cor_filtered <- filtered_data %>%
  summarise(
    cor_rate_min_extent = cor(prev_rate_of_change, next_min_extent, use = "complete.obs"),
    cor_rate_DOY_min = cor(prev_rate_of_change, next_DOY_min, use = "complete.obs")
  )
print("Correlations for Filtered Data:")
print(cor_filtered)


# Quadratic regression for Previous Rate of Retreat vs. Next Year's Minimum Extent
ggplot(filtered_data, aes(x = prev_rate_of_change, y = next_min_extent)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "blue") +  # Quadratic fit
  labs(
    title = "Quadratic Fit: Previous Year's Rate of Retreat vs. Next Year's Minimum Extent",
    x = "Previous Rate of Retreat (10⁶ km²/day)",
    y = "Next Year's Minimum Extent (10⁶ km²)"
  ) +
  theme_minimal()

# Quadratic regression for Previous Rate of Retreat vs. Next Year's Minimum DOY
ggplot(filtered_data, aes(x = prev_rate_of_change, y = next_DOY_min)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "blue") +  # Quadratic fit
  labs(
    title = "Quadratic Fit: Previous Year's Rate of Retreat vs. Next Year's Minimum DOY",
    x = "Previous Rate of Retreat (10⁶ km²/day)",
    y = "Next Year's Minimum DOY"
  ) +
  theme_minimal()


# Loess fit for Previous Rate of Retreat vs. Next Year's Minimum Extent
ggplot(filtered_data, aes(x = prev_rate_of_change, y = next_min_extent)) +
  geom_point() +
  geom_smooth(method = "loess", color = "blue") +
  labs(
    title = "Loess Fit: Previous Year's Rate of Retreat vs. Next Year's Minimum Extent",
    x = "Previous Rate of Retreat (10⁶ km²/day)",
    y = "Next Year's Minimum Extent (10⁶ km²)"
  ) +
  theme_minimal()

# Loess fit for Previous Rate of Retreat vs. Next Year's Minimum DOY
ggplot(filtered_data, aes(x = prev_rate_of_change, y = next_DOY_min)) +
  geom_point() +
  geom_smooth(method = "loess", color = "blue") +
  labs(
    title = "Loess Fit: Previous Year's Rate of Retreat vs. Next Year's Minimum DOY",
    x = "Previous Rate of Retreat (10⁶ km²/day)",
    y = "Next Year's Minimum DOY"
  ) +
  theme_minimal()

# Fit a linear model for each relationship
model_extent <- lm(next_min_extent ~ prev_rate_of_change, data = filtered_data)
model_DOY <- lm(next_DOY_min ~ prev_rate_of_change, data = filtered_data)

# Plot residuals for Previous Rate of Retreat vs. Next Year's Minimum Extent
ggplot(filtered_data, aes(x = prev_rate_of_change, y = resid(model_extent))) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Residuals of Linear Fit: Previous Year's Rate of Retreat vs. Next Year's Minimum Extent",
    x = "Previous Rate of Retreat (10⁶ km²/day)",
    y = "Residuals"
  ) +
  theme_minimal()

# Plot residuals for Previous Rate of Retreat vs. Next Year's Minimum DOY
ggplot(filtered_data, aes(x = prev_rate_of_change, y = resid(model_DOY))) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Residuals of Linear Fit: Previous Year's Rate of Retreat vs. Next Year's Minimum DOY",
    x = "Previous Rate of Retreat (10⁶ km²/day)",
    y = "Residuals"
  ) +
  theme_minimal()
# Load necessary libraries
library(tidyverse)
library(ggplot2)

# Assuming `filtered_data` is already created as per previous instructions

# Calculate correlation coefficients
cor_coef_DOY <- cor(filtered_data$prev_rate_of_change, filtered_data$next_DOY_min, use = "complete.obs")
cor_coef_extent <- cor(filtered_data$prev_rate_of_change, filtered_data$next_min_extent, use = "complete.obs")

# Plot 1: Previous Rate of Retreat vs. Next Year's Minimum DOY with linear fit and correlation annotation
ggplot(filtered_data, aes(x = prev_rate_of_change, y = next_DOY_min)) +
  geom_point() +
  geom_smooth(method = "lm", color = "blue") +  # Linear fit
  labs(
    title = "Previous Year's Rate of Retreat vs. Next Year's Minimum DOY",
    x = "Previous Rate of Retreat (10⁶ km²/day)",
    y = "Next Year's Minimum DOY"
  ) +
  annotate("text", x = min(filtered_data$prev_rate_of_change), y = max(filtered_data$next_DOY_min),
           label = paste("Correlation:", round(cor_coef_DOY, 3)), hjust = 0, vjust = 1.5, size = 5, color = "red") +
  theme_minimal()

# Plot 2: Previous Rate of Retreat vs. Next Year's Minimum Extent with linear fit and correlation annotation
ggplot(filtered_data, aes(x = prev_rate_of_change, y = next_min_extent)) +
  geom_point() +
  geom_smooth(method = "lm", color = "blue") +  # Linear fit
  labs(
    title = "Previous Year's Rate of Retreat vs. Next Year's Minimum Extent",
    x = "Previous Rate of Retreat (10⁶ km²/day)",
    y = "Next Year's Minimum Extent (10⁶ km²)"
  ) +
  annotate("text", x = min(filtered_data$prev_rate_of_change), y = max(filtered_data$next_min_extent),
           label = paste("Correlation:", round(cor_coef_extent, 3)), hjust = 0, vjust = 1.5, size = 5, color = "red") +
  theme_minimal()
