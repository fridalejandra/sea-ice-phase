library(ncdf4)
library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)

# Open file
nc_file <- nc_open("/Volumes/WorkDrive/SIA_invariant_data/Calculated_Sea_Ice_Area_with_Lat_Lon.nc")


# Pixel
lat_index <- 198
lon_index <- 447

# Load 1 pixel for all timesteps
sea_ice_area_pixel <- ncvar_get(
  nc_file, 
  varid = "Sea_Ice_Area",
  start = c(lat_index, lon_index, 1),  
  count = c(1, 1, -1)                 # Read 1 latitude, 1 longitude, and all timesteps
)

# Convert to dates
time <- ncvar_get(nc_file, "time")
time_origin <- as.Date("2012-07-13")  
dates <- time_origin + time

# Create a data frame for the pixel
df_pixel <- data.frame(
  Date = dates,
  Extent = sea_ice_area_pixel
)

# Preprocessing Steps
# Filter dates within the range 2013-01-01 to 2023-12-31
df_pixel <- df_pixel %>% filter(Date >= as.Date('2013-01-01') & Date <= as.Date('2023-12-31'))

# Calculate day of year (DOY) 
df_pixel <- df_pixel %>% mutate(DOY = yday(Date))

df_pixel$Extent <- as.numeric(df_pixel$Extent)

# TAC
average_annual_cycle <- df_pixel %>%
  group_by(DOY) %>%
  summarize(mean_extent = mean(Extent, na.rm = TRUE))

df_pixel <- df_pixel %>%
  left_join(average_annual_cycle, by = "DOY") %>%
  rename(Predicted_Traditional = mean_extent)

# IAC
gam_model <- gam(Extent ~ s(DOY, bs = "cc", k = 25), data = df_pixel)
df_pixel <- df_pixel %>%
  mutate(Predicted_Invariant = predict(gam_model, newdata = df_pixel))

# Group data by DOY 
average_cycles <- df_pixel %>%
  group_by(DOY) %>%
  summarize(
    Traditional = mean(Predicted_Traditional, na.rm = TRUE),
    Invariant = mean(Predicted_Invariant, na.rm = TRUE)
  )

#  Traditional vs Invariant Annual Cycle
ggplot(average_cycles, aes(x = DOY)) +
  geom_line(aes(y = Traditional, color = 'Traditional'), linetype = 'dashed', size = 1) +
  geom_line(aes(y = Invariant, color = 'Invariant'), linetype = 'dotdash', size = 1) +
  scale_x_continuous(
    breaks = seq(0, 365, by = 50), 
    labels = seq(0, 365, by = 50)  
  ) +
  labs(x = 'Day of Year (DOY)', y = 'Sea Ice Area (km²)', color = 'Model') +
  theme_minimal() +
  scale_color_manual(values = c('Traditional' = 'blue', 'Invariant' = 'red')) +
  theme(
    legend.position = 'bottom',
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    plot.title = element_blank()
  )


nc_close(nc_file)

