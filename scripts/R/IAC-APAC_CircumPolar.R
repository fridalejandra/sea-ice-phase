# Load necessary libraries
library(readxl)
library(dplyr)
library(tidyr)
library(lubridate)
library(mgcv)
library(ggplot2)

# Define the file path
file_path <- "/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv"

# List of sheets (sectors) to analyze
sheets <- c("Indian-Extent-km^2", "Pacific-Extent-km^2", "Ross-Extent-km^2", 
            "Weddell-Extent-km^2", "Bell-Amundsen-Extent-km^2")

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
    pivot_longer(
      cols = starts_with("19") | starts_with("20"), 
      names_to = "Year", 
      values_to = "Extent", 
      names_transform = list(Year = as.integer)
    )
  
  # Ensure 'day' column is numeric
  df_long$day <- as.numeric(df_long$day)
  
  # Generate the Date column
  df_long <- df_long %>%
    mutate(Date = as.Date(paste(Year, month, day, sep = "-"), format = "%Y-%B-%d"))
  
  # Filter dates to the specified range
  df_long <- df_long %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2023-12-31'))
  
  # Calculate day of year (DOY) and adjust to make Julian day 50 as day 0
  df_long <- df_long %>%
    mutate(
      DOY = (yday(Date) - 50) %% 365,  # Shift DOY by 50 and wrap around
      Extent = as.numeric(Extent),
      Sector = sheet  # Add a column for the sector name
    )
  
  # Ensure Sector is a factor
  df_long$Sector <- as.factor(df_long$Sector)
  
  # --- Invariant Annual Cycle ---
  # Calculate the traditional annual cycle
  average_annual_cycle <- df_long %>%
    group_by(DOY, Sector) %>%
    summarize(mean_extent = mean(Extent, na.rm = TRUE), .groups = "drop")
  
  df_long <- df_long %>%
    left_join(average_annual_cycle, by = c("DOY", "Sector")) %>%
    rename(Predicted_Traditional = mean_extent)
  
  # Dynamically adjust the GAM model
  if (length(unique(df_long$Sector)) > 1) {
    # GAM with Sector if multiple sectors exist
    gam_model_invariant <- gam(Extent ~ s(DOY, bs = "cc", k = 25) + Sector, data = df_long)
  } else {
    # GAM without Sector if only one sector exists
    gam_model_invariant <- gam(Extent ~ s(DOY, bs = "cc", k = 25), data = df_long)
  }
  
  df_long$Predicted_Invariant <- predict(gam_model_invariant, newdata = df_long)
  
  # --- Amplitude-Phase Adjusted Annual Cycle ---
  # Calculate min extent day per year for each sector
  yearly_stats <- df_long %>%
    group_by(Year, Sector) %>%
    summarise(Date1 = Date[which.min(Extent)], .groups = "drop")
  
  # Add Date2 (min date of last year) and Date3 (min date of next year)
  yearly_stats <- yearly_stats %>%
    group_by(Sector) %>%
    mutate(Date2 = lag(Date1), Date3 = lead(Date1))
  
  # Merge yearly stats back into the main dataset
  df_long <- df_long %>%
    left_join(yearly_stats, by = c("Year", "Sector"))
  
  # Calculate t-values based on logic
  df_long <- df_long %>%
    rowwise() %>%
    mutate(t = case_when(
      Year == min(Year) ~ 365 - as.numeric(Date3 - Date),
      Date >= Date1 ~ as.numeric(Date - Date1),
      Date < Date1 ~ as.numeric(Date - Date2)
    )) %>%
    ungroup()
  
  # Calculate t_min and t_max for each year and sector
  t_stats <- df_long %>%
    group_by(Year, Sector) %>%
    summarise(t_min = min(t, na.rm = TRUE), t_max = max(t, na.rm = TRUE), .groups = "drop")
  
  df_long <- df_long %>%
    left_join(t_stats, by = c("Year", "Sector"))
  
  # Calculate phase using Beta distribution CDF
  df_long <- df_long %>%
    rowwise() %>%
    mutate(phase = 365 * pbeta((t - t_min) / (t_max - t_min + 1e-10), shape1 = 1, shape2 = 1)) %>%
    ungroup()
  
  # Calculate max and min sea ice extent per year
  yearly_stats <- df_long %>%
    group_by(Year, Sector) %>%
    summarise(
      max_extent = max(Extent, na.rm = TRUE),
      min_extent = min(Extent, na.rm = TRUE),
      .groups = "drop"
    )
  
  df_long <- df_long %>%
    left_join(yearly_stats, by = c("Year", "Sector")) %>%
    mutate(
      amplitude = max_extent - min_extent,
      scaling_factor = (Extent - min_extent) / amplitude
    )
  
  # GAM for amplitude-phase adjusted cycle
  gam_model_apac <- gam(scaling_factor ~ s(DOY, bs = "cc", k = 100) + s(phase, bs = "cc", k = 100) + Sector, data = df_long)
  df_long$Predicted_Amplitude_Adjusted <- predict(gam_model_apac, newdata = df_long)
  
  # Convert the predicted scaling factor back to sea ice extent
  df_long$Predicted_Amplitude_Adjusted_Extent <- df_long$Predicted_Amplitude_Adjusted * df_long$amplitude + df_long$min_extent
  
  # --- Calculate the anomaly residual: IAC - APAC ---
  df_long <- df_long %>%
    mutate(Anomaly_Residual = Predicted_Invariant - Predicted_Amplitude_Adjusted_Extent)
  
  # Store processed data in the list
  sector_data[[sheet]] <- df_long
}

# Combine data from all sectors into one data frame
combined_data <- bind_rows(sector_data)

# --- Plot the anomaly residuals for all sectors ---
ggplot(combined_data, aes(x = DOY, y = Anomaly_Residual, color = Sector)) +
  geom_line(size = 1) +
  facet_wrap(~ Sector) +
  labs(
    title = "Anomaly Residuals (IAC - APAC) by Sector",
    x = "Day of Year (DOY)",
    y = "Anomaly Residual (Extent)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )

