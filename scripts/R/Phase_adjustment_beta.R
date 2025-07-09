library(dplyr)
library(mgcv)
library(lubridate)

# Read data
xpath_csv <- '/Users/fridaperez/Developer/repos/phase_project/SIE/S_seaice_extent_daily_v3.0.csv'
df <- read.csv(xpath_csv)
## Pre-processing ##
# Convert 'Year', 'Month', 'Day' to Date and make a column 'Date'
df$Date <- as.Date(with(df, paste(Year, Month, Day, sep="-")), "%Y-%m-%d")

# Filter dates
df <- df %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2018-12-31'))

# Make the dates numeric 
df$tdate <- as.numeric(df$Date)

# Calculate day of year (DOY)
df$DOY <- yday(df$Date)

# Make the Extent values numeric
df$Extent <- as.numeric(df$Extent)

# Compute extent and normalize DOY between min and max extent days
annual_stats <- df %>% group_by(Year) %>%
  summarize(max_DOY = which.max(Extent), min_DOY = which.min(Extent))

df <- df %>% left_join(annual_stats, by = "Year")

# Apply Beta distribution adjustment
df <- df %>%
  mutate(
    beta_param = (DOY - min_DOY) / (max_DOY - min_DOY),
    phase_adjusted_DOY = 365 * pbeta(beta_param, shape1 = 2, shape2 = 2)
  )

# Fit GAM model with phase-adjusted DOY
gam_model <- gam(Extent ~ s(phase_adjusted_DOY, bs = "cc", k = 50, fx = FALSE), data = df)
df$Predicted_Phase_Adjusted <- predict(gam_model, newdata = df)

# Calculate RMSE
phase_rmse <- sqrt(mean((df$Predicted_Phase_Adjusted - df$Extent)^2, na.rm = TRUE))
print(paste("Phase-Adjusted RMSE:", round(phase_rmse, 3)))
