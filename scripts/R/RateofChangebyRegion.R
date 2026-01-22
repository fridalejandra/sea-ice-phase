# Load necessary libraries
library(dplyr)
library(tidyverse)
library(readxl)
library(mgcv)
library(lubridate)

# Define the file path
file_path <- "/Users/fridaperez/Developer/repos/phase_project/SIE/Extent_Plots/S_Sea_Ice_Index_Regional_Daily_Data_G02135_v3.0.xlsx"

# List of sheets (sectors) to analyze
sheets <- c("Indian-Extent-km^2", "Pacific-Extent-km^2", "Ross-Extent-km^2", "Weddell-Extent-km^2", "Bell-Amundsen-Extent-km^2")

# Initialize an empty list to store data for each sector
sector_data <- list()

# Loop over each sheet to load and process data for each sector
for (sheet in sheets) {
  
  # Load the data
  df_excel <- read_excel(file_path, sheet = sheet)
  
  # Fill down the 'month' column
  df_excel <- df_excel %>% fill(month)
  
  # Reshape the data from wide to long format
  df_long <- df_excel %>%
    pivot_longer(cols = starts_with("19") | starts_with("20"), 
                 names_to = "Year", 
                 values_to = "Extent", 
                 names_transform = list(Year = as.integer))
  
  # Ensure 'day' column is numeric
  df_long$day <- as.numeric(df_long$day)
  
  # Generate the Date column
  df_long <- df_long %>%
    mutate(Date = as.Date(paste(Year, month, day, sep = "-"), format = "%Y-%B-%d"))
  
  # Filter dates to specified range
  df_long <- df_long %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2023-12-31'))
  
  # Calculate day of year (DOY) and adjust to make Julian day 50 as day 0
  df_long <- df_long %>%
    mutate(
      DOY = (yday(Date) - 50) %% 365,  # Shift DOY by 50 and wrap around
      Extent = as.numeric(Extent),
      Sector = sheet  # Add a column for the sector name
    )
  
  # Traditional Annual Cycle
  average_annual_cycle <- df_long %>%
    group_by(DOY) %>%
    summarize(mean_extent = mean(Extent, na.rm = TRUE))
  
  df_long <- df_long %>%
    left_join(average_annual_cycle, by = "DOY") %>%
    rename(Predicted_Traditional = mean_extent)
  
  # Invariant Annual Cycle using cyclic cubic splines
  gam_model <- gam(Extent ~ s(DOY, bs = "cc", k = 25), data = df_long)
  df_long$Predicted_Invariant <- predict(gam_model, newdata = df_long)
  
  # Calculate the rate of change for Traditional and Invariant cycles
  df_long <- df_long %>%
    arrange(Date) %>%
    mutate(
      Traditional_rate_of_change = c(diff(Predicted_Traditional), NA),
      Invariant_rate_of_change = c(diff(Predicted_Invariant), NA)
    )
  
  # Exclude the last 10 days in December to avoid the wraparound effect
  df_long <- df_long %>%
    filter(!(DOY %in% c(355:365)))
  
  # Store the processed data for each sector in a list
  sector_data[[sheet]] <- df_long
}

# Combine all sectors into a single data frame
combined_data <- bind_rows(sector_data)

# Plotting the daily rate of change for each sector
ggplot(combined_data, aes(x = DOY)) +
  geom_line(aes(y = Traditional_rate_of_change, color = 'Traditional Rate of Change'), linetype = 'dashed') +
  geom_line(aes(y = Invariant_rate_of_change, color = 'Invariant Rate of Change'), linetype = 'dotdash') +
  geom_hline(yintercept = 0, linetype = "solid", color = "gray") +  # Add a horizontal line at zero
  facet_wrap(~ Sector, scales = 'free_y') +  # Create a separate panel for each sector
  labs(
    title = "Daily Rate of Change in Sea Ice Extent (Traditional vs Invariant) by Sector",
    x = "Day of Year (Starting at Julian Day 50)",
    y = NULL  # Remove y-axis title
  ) +
  scale_x_continuous(breaks = c(0, 91, 182, 274, 365), labels = c("Feb", "May", "Aug", "Nov", "Jan")) +
  theme_minimal() +
  scale_color_manual(values = c('Traditional Rate of Change' = 'blue', 'Invariant Rate of Change' = 'red')) +
  theme(
    legend.position = 'bottom',
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    axis.title.x = element_text(size = 16),
    axis.text.y = element_blank(),  # Remove y-axis values
    axis.text.x = element_text(size = 14)
  )


# Subset data to focus on December
december_data <- combined_data %>%
  filter(DOY >= 320)  # Adjust to start around early December for full context

# Plot to inspect the rates of change in December
ggplot(december_data, aes(x = DOY)) +
  geom_line(aes(y = Traditional_rate_of_change, color = 'Traditional Rate of Change'), linetype = 'dashed') +
  geom_line(aes(y = Invariant_rate_of_change, color = 'Invariant Rate of Change'), linetype = 'dotdash') +
  geom_hline(yintercept = 0, linetype = "solid", color = "gray") +
  labs(
    title = "Daily Rate of Change in Sea Ice Extent (December)",
    x = "Day of Year",
    y = "Rate of Change in SIE"
  ) +
  scale_x_continuous(breaks = c(320, 330, 340, 350, 360), labels = c("Dec 1", "Dec 10", "Dec 20", "Dec 30", "Jan")) +
  theme_minimal() +
  scale_color_manual(values = c('Traditional Rate of Change' = 'blue', 'Invariant Rate of Change' = 'red'))

# Set a threshold for identifying extreme changes in rate of change
rate_change_threshold <- 20000  # Adjust this threshold as needed

# Filter out data points where the rate of change exceeds the threshold
combined_data_filtered <- combined_data %>%
  filter(abs(Traditional_rate_of_change) <= rate_change_threshold &
           abs(Invariant_rate_of_change) <= rate_change_threshold)

# Re-plot with filtered data to see if the December lines are removed
ggplot(combined_data_filtered, aes(x = DOY)) +
  geom_line(aes(y = Traditional_rate_of_change, color = 'Traditional Rate of Change'), linetype = 'dashed') +
  geom_line(aes(y = Invariant_rate_of_change, color = 'Invariant Rate of Change'), linetype = 'dotdash') +
  geom_hline(yintercept = 0, linetype = "solid", color = "gray") +
  facet_wrap(~ Sector, scales = 'free_y') +
  labs(
    title = "Daily Rate of Change in Sea Ice Extent (Traditional vs Invariant) by Sector",
    x = "Day of Year (Starting at Julian Day 50)",
    y = NULL
  ) +
  scale_x_continuous(breaks = c(0, 91, 182, 274, 365), labels = c("Feb", "May", "Aug", "Nov", "Jan")) +
  theme_minimal() +
  scale_color_manual(values = c('Traditional Rate of Change' = 'blue', 'Invariant Rate of Change' = 'red')) +
  theme(
    legend.position = 'bottom',
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    axis.title.x = element_text(size = 16),
    axis.text.y = element_blank(),
    axis.text.x = element_text(size = 14)
  )

