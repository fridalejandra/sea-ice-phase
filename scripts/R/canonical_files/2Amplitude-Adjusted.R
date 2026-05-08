# Load necessary libraries
library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)

# Load the CSV file
df_csv <- read.csv("/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv")

# Convert 'Year', 'Month', 'Day' to Date
df_csv$Date <- as.Date(with(df_csv, paste(Year, Month, Day, sep="-")), "%Y-%m-%d")
df_csv <- df_csv %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2018-12-31'))

# Calculate day of year (DOY) and Year
df_csv$DOY <- yday(df_csv$Date)
df_csv$Year <- year(df_csv$Date)

# Make the dates numeric 
df_csv$tdate <- as.numeric(df_csv$Date)

# Make the Extent values numeric
df_csv$Extent <- as.numeric(df_csv$Extent)

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
# We model the scaling factor (Extent normalized by yearly max-min) as a function of DOY
gam_model_amplitude_adjusted <- gam(scaling_factor ~s(tdate, bs = "cc",k=20)+ s(DOY, bs = "cc", k = 100), 
                                    data = df_csv)

# Predict the amplitude-adjusted sea ice extent
df_csv$Predicted_Amplitude_Adjusted <- predict(gam_model_amplitude_adjusted, newdata = df_csv)

# Convert the predicted scaling factor back to sea ice extent
df_csv$Predicted_Amplitude_Adjusted_Extent <- df_csv$Predicted_Amplitude_Adjusted * df_csv$amplitude + df_csv$min_extent

# Calculate RMSE for the amplitude-adjusted annual cycle
amplitude_adjusted_rmse <- sqrt(mean((df_csv$Predicted_Amplitude_Adjusted_Extent - df_csv$Extent)^2, na.rm = TRUE))

# Print RMSE values
print(amplitude_adjusted_rmse)



