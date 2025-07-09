# Load necessary libraries
library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)
library(ggplot2)
library(stats)

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

# Calculate the standardized invariant annual cycle using a cubic spline
gam_model <- gam(Extent ~ s(DOY, bs = "cc", k = 50, fx = FALSE),
                 data = df_csv, method = "REML")

df_csv$Predicted_Invariant <- predict(gam_model, newdata = df_csv)

# Step 1: Calculate the annual maximum and minimum extent for each year
yearly_max_min <- df_csv %>%
  group_by(Year) %>%
  summarize(max_extent = max(Extent, na.rm = TRUE),
            min_extent = min(Extent, na.rm = TRUE))

# Step 2: Merge these values back into the main dataframe
df_csv <- df_csv %>%
  left_join(yearly_max_min, by = "Year")

# Check if the merge was successful and the columns are present
if(!("max_extent" %in% names(df_csv) && "min_extent" %in% names(df_csv))) {
  stop("max_extent and/or min_extent columns not found. Check the join step.")
}

# Step 3: Calculate the day of the year for the observed maximum and minimum extent for each year
phase_shifts <- df_csv %>%
  group_by(Year) %>%
  summarize(max_DOY = DOY[which.max(Extent)],
            min_DOY = DOY[which.min(Extent)])

# Step 4: Merge the phase shift information back into the main dataframe
df_csv <- df_csv %>%
  left_join(phase_shifts, by = "Year")

# Debugging: Print out the first few rows to check max_DOY, min_DOY, max_extent, and min_extent
print(head(df_csv[, c("Year", "DOY", "Extent", "max_DOY", "min_DOY", "max_extent", "min_extent")]))

# Step 5: Correctly calculate the Beta distribution parameters (β1 and β2) for each year
df_csv <- df_csv %>%
  mutate(beta_param = pmax(0, pmin(1, (DOY - min_DOY) / (max_DOY - min_DOY))),
         phase_shift = 365 * pbeta(beta_param, shape1 = 1, shape2 = 2))  # Example Beta(2,2)

# Debugging: Print out the first few rows to check beta_param and phase_shift
print(head(df_csv[, c("Year", "DOY", "beta_param", "phase_shift")]))

# Step 6: Adjust DOY by the calculated phase shift
df_csv <- df_csv %>%
  mutate(DOY_Adjusted = (DOY + phase_shift) %% 365)

# Debugging: Print out the first few rows to check DOY_Adjusted
print(head(df_csv[, c("Year", "DOY", "DOY_Adjusted")]))

# Step 7: Fit the GAM model with the phase-adjusted DOY
df_csv$Predicted_Phase_Adjusted <- predict(gam_model, newdata = df_csv %>% mutate(DOY = DOY_Adjusted))

# Debugging: Print out the first few rows to check Predicted_Phase_Adjusted
print(head(df_csv[, c("Year", "DOY_Adjusted", "Predicted_Phase_Adjusted")]))

# Step 8: Calculate RMSE for the phase-adjusted cycle
phase_rmse <- sqrt(mean((df_csv$Predicted_Phase_Adjusted - df_csv$Extent)^2, na.rm = TRUE))

# Print RMSE for the phase-adjusted model
print(paste("Phase-Adjusted RMSE:", round(phase_rmse, 2)))

