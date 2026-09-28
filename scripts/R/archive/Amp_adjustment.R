# Load necessary libraries
library(dplyr)
library(mgcv)
library(lubridate)

# CSV file
xpath_csv <- '/Users/fridaperez/Developer/repos/phase_project/SIE/S_seaice_extent_daily_v3.0.csv'
df_csv <- read.csv(xpath_csv)

# Convert 'Year', 'Month', 'Day' to Date and create a 'Date' column
df_csv$Date <- as.Date(with(df_csv, paste(Year, Month, Day, sep="-")), "%Y-%m-%d")

# Filter the data to the desired date range
df_csv <- df_csv %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2018-12-31'))

# Convert dates to numeric
df_csv$tdate <- as.numeric(df_csv$Date)

# Calculate day of year (DOY)
df_csv$DOY <- yday(df_csv$Date)

# Ensure extent values are numeric
df_csv$Extent <- as.numeric(df_csv$Extent)

# Calculate the annual maximum and minimum extent for each year
yearly_max_min <- df_csv %>%
  group_by(Year) %>%
  summarize(max_extent = max(Extent, na.rm = TRUE),
            min_extent = min(Extent, na.rm = TRUE))

# Merge these values back into the main dataframe
df_csv <- df_csv %>%
  left_join(yearly_max_min, by = "Year")

# Calculate the standardized invariant annual cycle using a cubic spline
gam_model <- gam(Extent ~ s(DOY, bs = "cc", k = 50, fx = FALSE), 
                 data = df_csv, method = "REML")

df_csv$Predicted_Invariant <- predict(gam_model, newdata = df_csv)

# Calculate the amplitude-adjusted annual cycle
df_csv <- df_csv %>%
  mutate(Predicted_Amplitude_Adjusted = ((Predicted_Invariant - min(Predicted_Invariant)) / 
                                           (max(Predicted_Invariant) - min(Predicted_Invariant))) * 
           (max_extent - min_extent) + min_extent)

# Calculate RMSE for the amplitude-adjusted cycle
amplitude_rmse <- sqrt(mean((df_csv$Predicted_Amplitude_Adjusted - df_csv$Extent)^2, na.rm = TRUE))

# Print RMSE
print(paste("Amplitude-Adjusted RMSE:", round(amplitude_rmse, 2)))

# View the first few rows to ensure everything is calculated correctly
print(head(df_csv[, c("DOY", "Extent", "Predicted_Invariant", "Predicted_Amplitude_Adjusted")]))
