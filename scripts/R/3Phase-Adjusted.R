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

# Calculate day of year (DOY)
df_csv$DOY <- yday(df_csv$Date)

# Remove rows where Year is 1978 and save the result to df_csv2
df_csv2 <- df_csv %>% 
  filter(Year != 1978)


# Build the GAM model with phase adjustment
gam_model_phase_adjusted <- gam(Extent ~s(tdate, bs = "cc",k=75)+ s(DOY, bs = "cc", k = 100)+ s(phase, bs = "cc", k = 100,fx=FALSE), data = df_csv2)


# Predict the phase-adjusted sea ice extent
df_csv2$Predicted_Phase_Adjusted <- predict(gam_model_phase_adjusted, newdata = df_csv2)

# Calculate RMSE for the phase-adjusted annual cycle
phase_adjusted_rmse <- sqrt(mean((df_csv2$Predicted_Phase_Adjusted - df_csv2$Extent)^2, na.rm = TRUE))

# Print RMSE values
print(phase_adjusted_rmse)



