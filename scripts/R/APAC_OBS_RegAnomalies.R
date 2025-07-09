# Load required libraries
library(dplyr)
library(lubridate)
library(mgcv)
library(ggplot2)

# Directory containing CSV files
input_dir <- "/Users/fridaperez/Desktop/SIE_Long_Regions/"
csv_files <- list.files(input_dir, pattern = "-Extent-km\\^2_long_format\\.csv$", full.names = TRUE)

if (length(csv_files) == 0) {
  stop("No matching files found in the directory.")
}

# Process each region separately
for (file in csv_files) {
  cat("\nProcessing file:", basename(file), "\n")
  df_csv <- read.csv(file)
  
  # Debug: Check column names
  cat("Debug: Column names in the dataset:\n")
  print(colnames(df_csv))
  
  # Ensure required columns exist
  if (!all(c("date", "Extent") %in% colnames(df_csv))) {
    stop(paste("Missing required columns in file:", basename(file)))
  }
  
  # Convert `date` to Date type and extract time-related fields
  df_csv$date <- as.Date(df_csv$date)
  df_csv <- df_csv %>%
    mutate(
      Year = year(date),
      Month = month(date),
      Day = day(date),
      DOY = yday(date),
      tdate = as.numeric(date),
      Extent = as.numeric(Extent) # Ensure Extent is numeric
    )
  
  # Calculate yearly min extent day and merge
  yearly_stats <- df_csv %>%
    group_by(Year) %>%
    summarise(Date1 = date[which.min(Extent)], .groups = 'drop') %>%
    mutate(Date2 = lag(Date1), Date3 = lead(Date1))
  
  df_csv <- df_csv %>%
    left_join(yearly_stats, by = "Year")
  
  # Compute t-values for phase calculation
  df_csv <- df_csv %>%
    rowwise() %>%
    mutate(t = case_when(
      Year == 1978 ~ 365 - as.numeric(Date3 - date),
      Year == 1979 & date < Date1 ~ 365 - as.numeric(Date1 - date),
      date >= Date1 ~ as.numeric(date - Date1),
      date < Date1 ~ as.numeric(date - Date2)
    )) %>%
    ungroup()
  
  # Calculate t_min and t_max by year
  t_stats <- df_csv %>%
    group_by(Year) %>%
    summarise(t_min = min(t, na.rm = TRUE), t_max = max(t, na.rm = TRUE), .groups = 'drop')
  
  df_csv <- df_csv %>%
    left_join(t_stats, by = "Year")
  
  # Phase adjustment using Beta distribution CDF
  df_csv <- df_csv %>%
    rowwise() %>%
    mutate(phase = 365 * pbeta((t - t_min) / (t_max - t_min + 1e-10), shape1 = 1, shape2 = 1)) %>%
    ungroup()
  
  # Calculate max and min sea ice extent per year
  yearly_stats <- df_csv %>%
    group_by(Year) %>%
    summarise(max_extent = max(Extent, na.rm = TRUE), min_extent = min(Extent, na.rm = TRUE))
  
  df_csv <- df_csv %>%
    left_join(yearly_stats, by = "Year")
  
  # Compute amplitude and scaling factor
  df_csv <- df_csv %>%
    mutate(amplitude = max_extent - min_extent,
           scaling_factor = (Extent - min_extent) / amplitude)
  
  # Build the GAM model for amplitude-phase adjusted extent
  cat("Building GAM model for APAC prediction...\n")
  gam_model_apac <- gam(scaling_factor ~ 
                          s(tdate, bs = "cc", k = 150) + 
                          s(DOY, bs = "cc", k = 100) + 
                          s(phase, bs = "cc", k = 100),
                        data = df_csv)
  
  # Predict APAC extent
  df_csv$Predicted_Amplitude_Adjusted <- predict(gam_model_apac, newdata = df_csv)
  df_csv$Predicted_Amplitude_Adjusted_Extent <- df_csv$Predicted_Amplitude_Adjusted * df_csv$amplitude + df_csv$min_extent
  
  # Debug: Check if APAC calculation exists
  cat("Debug: Checking APAC predictions...\n")
  print(head(df_csv %>% select(DOY, Predicted_Amplitude_Adjusted_Extent)))
  
  # Compute max_day_observed and max_day_apac
  max_day_observed <- df_csv %>%
    filter(Year == 2022 & abs(Extent - max_extent) < 1e-6) %>%
    summarise(DOY = first(DOY)) %>%
    pull(DOY)
  
  # Compute max_day_apac with debugging
  apac_max <- max(df_csv$Predicted_Amplitude_Adjusted_Extent, na.rm = TRUE)
  
  if (is.na(apac_max)) {
    cat("Warning: No valid maximum found for APAC. Skipping file:", basename(file), "\n")
    next
  }
  
  # Use tolerance to find DOY for APAC max
  max_day_apac <- df_csv %>%
    filter(Year == 2022 & abs(Predicted_Amplitude_Adjusted_Extent - apac_max) < 1e-6) %>%
    summarise(DOY = first(DOY)) %>%
    pull(DOY)
  
  # Debug: Check DOY max for APAC
  cat("Debug: DOY max for APAC:", max_day_apac, "\n")
  
  # Debug: Check max days
  cat("Debug: Checking max days for observed and APAC...\n")
  cat("Max day observed:", max_day_observed, "\n")
  cat("Max day APAC:", max_day_apac, "\n")
  
  if (length(max_day_observed) == 0 | length(max_day_apac) == 0) {
    cat("Warning: Missing maximum extent for observed or APAC. Skipping file:", basename(file), "\n")
    next
  }
  
  start_day <- min(max_day_observed, max_day_apac) + 1  # Start day after the earlier max
  
  # Find minimum day for 2023
  min_day <- df_csv %>%
    filter(Year == 2023 & abs(Extent - min_extent) < 1e-6) %>%
    summarise(DOY = first(DOY)) %>%
    pull(DOY)
  
  # Debug: Check min day
  cat("Debug: Checking min day for 2023...\n")
  print(min_day)
  
  if (length(min_day) == 0) {
    cat("Warning: Missing minimum extent for 2023. Skipping file:", basename(file), "\n")
    next
  }
  
  end_day <- min_day[1]
  
  # Filter for the range of interest
  df_filtered <- df_csv %>%
    filter((Year == 2022 & DOY >= start_day) | (Year == 2023 & DOY <= end_day))
  
  # Calculate anomalies
  df_filtered <- df_filtered %>%
    mutate(
      anomaly_observed = Extent - mean(Extent, na.rm = TRUE),
      anomaly_apac = Predicted_Amplitude_Adjusted_Extent - mean(Predicted_Amplitude_Adjusted_Extent, na.rm = TRUE)
    )
  
  # Debug: Check anomalies
  cat("Debug: Checking anomalies...\n")
  print(head(df_filtered %>% select(Year, DOY, anomaly_observed, anomaly_apac)))
  
  # Plot anomalies
  anomaly_plot <- ggplot(df_filtered, aes(x = as.Date(date))) +
    geom_line(aes(y = anomaly_observed, color = "Observed"), size = 1) +
    geom_line(aes(y = anomaly_apac, color = "APAC"), size = 1, linetype = "dashed") +
    labs(
      title = paste("Sea Ice Extent Anomalies -", basename(file)),
      subtitle = paste("From DOY", start_day, "of 2022 to DOY", end_day, "of 2023"),
      x = "Date",
      y = "Anomaly (million km²)",
      color = "Legend"
    ) +
    scale_color_manual(values = c("Observed" = "steelblue", "APAC" = "firebrick")) +
    theme_minimal()
  
  print(anomaly_plot)
}
