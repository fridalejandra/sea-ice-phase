library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)
library(ggplot2)

# Directory containing CSV files
input_dir <- "/Users/fridaperez/Desktop/SIE_Long_Regions/"

# List all CSV files in the directory
csv_files <- list.files(input_dir, pattern = "-Extent-km\\^2_long_format\\.csv$", full.names = TRUE)

if (length(csv_files) == 0) {
  stop("No matching files found in the directory.")
}

for (file in csv_files) {
  cat("Processing file:", basename(file), "\n")
  
  # Load the CSV file
  df_csv <- read.csv(file)
  
  # Check for required columns
  if (!all(c("date", "Extent") %in% colnames(df_csv))) {
    stop(paste("File", basename(file), "is missing required columns ('date' or 'Extent')."))
  }
  
  # Convert 'date' to Date format
  df_csv$date <- as.Date(df_csv$date, format = "%Y-%m-%d")
  
  # Extract Year and DOY
  df_csv <- df_csv %>%
    mutate(
      Year = year(date),
      DOY = yday(date),
      Extent = as.numeric(Extent) # Ensure Extent is numeric
    )
  
  # Invariant Annual Cycle (IAC)
  gam_model_iac <- gam(Extent ~ s(DOY, bs = "cc", k = 25), data = df_csv)
  df_csv$Predicted_Invariant <- predict(gam_model_iac, newdata = df_csv)
  
  # Min extent day per year
  yearly_stats <- df_csv %>%
    group_by(Year) %>%
    summarise(Date1 = date[which.min(Extent)], .groups = 'drop')
  
  yearly_stats <- yearly_stats %>%
    mutate(
      Date2 = lag(Date1),  # Previous year
      Date3 = lead(Date1)  # Next year
    )
  
  # Merge yearly stats
  df_csv <- df_csv %>%
    left_join(yearly_stats, by = "Year")
  
  # Calculate t-values
  df_csv <- df_csv %>%
    rowwise() %>%
    mutate(t = case_when(
      Year == 1978 ~ 365 - as.numeric(Date3 - date),
      Year == 1979 & date < Date1 ~ 365 - as.numeric(Date1 - date),
      date >= Date1 ~ as.numeric(date - Date1),
      date < Date1 ~ as.numeric(date - Date2)
    )) %>%
    ungroup()
  
  # Calculate t_min and t_max
  t_stats <- df_csv %>%
    group_by(Year) %>%
    summarise(
      t_min = min(t, na.rm = TRUE),
      t_max = max(t, na.rm = TRUE),
      .groups = 'drop'
    )
  
  df_csv <- df_csv %>%
    left_join(t_stats, by = "Year")
  
  # Phase calculation
  df_csv <- df_csv %>%
    rowwise() %>%
    mutate(
      phase = 365 * pbeta((t - t_min) / (t_max - t_min + 1e-10), 1, 1)
    ) %>%
    ungroup()
  
  # Calculate max, min extent, amplitude, and scaling factor
  yearly_stats <- df_csv %>%
    group_by(Year) %>%
    summarise(
      max_extent = max(Extent, na.rm = TRUE),
      min_extent = min(Extent, na.rm = TRUE),
      .groups = 'drop'
    )
  
  df_csv <- df_csv %>%
    left_join(yearly_stats, by = "Year") %>%
    mutate(
      amplitude = max_extent - min_extent,
      scaling_factor = (Extent - min_extent) / amplitude
    )
  
  # APAC Model
  gam_model_apac <- gam(scaling_factor ~ s(t, bs = "cc", k = 150) +
                          s(DOY, bs = "cc", k = 100) +
                          s(phase, bs = "cc", k = 100),
                        data = df_csv)
  
  df_csv <- df_csv %>%
    mutate(
      Predicted_APAC = predict(gam_model_apac, newdata = .),
      Predicted_APAC_Extent = Predicted_APAC * amplitude + min_extent
    )
  
  # Filter for 2022 data
  df_2022 <- df_csv %>% filter(Year == 2022)
  
  if (nrow(df_2022) > 0) {
    # Plot
    plot_cycles <- ggplot(df_2022, aes(x = date)) +
      geom_line(aes(y = Extent, color = "Observed"), size = 1) +
      geom_line(aes(y = Predicted_Invariant, color = "Invariant"), size = 1, linetype = "dashed") +
      geom_line(aes(y = Predicted_APAC_Extent, color = "APAC"), size = 1, linetype = "dotdash") +
      labs(
        title = paste("Sea Ice Cycles - 2022 -", basename(file)),
        x = "Date",
        y = "Sea Ice Extent (million km²)",
        color = "Legend"
      ) +
      scale_color_manual(
        values = c(
          "Observed" = "blue",
          "Invariant" = "purple",
          "APAC" = "red"
        )
      ) +
      scale_x_date(
        date_labels = "%b",  # Show only abbreviated month names
        date_breaks = "1 month"  # Ensure one tick per month
      ) +
      theme_minimal()
    
    print(plot_cycles)
  } else {
    warning(paste("No data for the year 2022 in file", basename(file)))
  }
}
retreat_timing <- df_csv %>%
  group_by(Year) %>%
  summarise(
    Max_DOY = DOY[which.max(Extent)],
    Retreat_DOY = DOY[which(Extent < Extent[which.max(Extent)][1] & DOY > DOY[which.max(Extent)][1])][1],
    .groups = 'drop'
  )