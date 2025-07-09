library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)
library(ggplot2)
library(patchwork)

# Load the CSV file
df_csv <- read.csv("/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv")

# Convert 'Year', 'Month', 'Day' to Date
df_csv$Date <- as.Date(with(df_csv, paste(Year, Month, Day, sep="-")), "%Y-%m-%d")

# Define years of interest
years_of_interest <- c(2013, 2016, 2021, 2022)

# Filter dataset
df_csv <- df_csv %>% filter(Year %in% years_of_interest)

# Calculate DOY
df_csv$DOY <- yday(df_csv$Date)

# Convert Extent to numeric
df_csv$Extent <- as.numeric(df_csv$Extent)

# Compute daily rate of change
df_csv <- df_csv %>%
  group_by(Year) %>%
  arrange(Date) %>%
  mutate(Rate_of_Change = Extent - lag(Extent)) %>%
  ungroup()

# Get min extent day per year
yearly_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(Date_min = Date[which.min(Extent)], DOY_min = DOY[which.min(Extent)], .groups = 'drop')

# Merge yearly stats back into df_csv
df_csv <- df_csv %>%
  left_join(yearly_stats, by = "Year")

# Calculate phase-adjusted DOY using APAC Model
df_csv <- df_csv %>%
  group_by(Year) %>%
  mutate(t = as.numeric(Date - Date_min)) %>%
  ungroup()

# Fit the APAC model
gam_model_apac <- gam(Extent ~ s(DOY, bs = "cc", k = 150) + s(t, bs = "cc", k = 100), 
                      data = df_csv)

# Predict APAC sea ice extent
df_csv$Predicted_APAC_Extent <- predict(gam_model_apac, newdata = df_csv)

# Calculate rate of change for APAC model
df_csv <- df_csv %>%
  group_by(Year) %>%
  arrange(Date) %>%
  mutate(Rate_of_Change_APAC = Predicted_APAC_Extent - lag(Predicted_APAC_Extent)) %>%
  ungroup()

# Define actual minimum DOY for each year based on data
min_doy <- yearly_stats$DOY_min
names(min_doy) <- yearly_stats$Year

# Create individual facet plots for each year except 2022
plot_list <- list()
for (i in seq_along(years_of_interest)) {
  year <- years_of_interest[i]
  
  df_year <- df_csv %>% filter(Year == year)
  
  # Shift x-axis to start at the actual minimum DOY for the year
  df_year <- df_year %>% filter(DOY >= min_doy[as.character(year)])
  
  p <- ggplot(df_year, aes(x = DOY, y = Rate_of_Change_APAC)) +
    geom_line(color = "blue", size = 1) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(
      title = paste(year),
      x = "Day of Year (DOY Min +1)",
      y = "Rate of Change (x 10^6 km²/day)"
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 14, face = "bold"),
      axis.title.y = element_text(size = 12, face = "bold")  # Bold Y-axis
    )
  
  plot_list[[as.character(year)]] <- p
}

### Create 2022 facet with both APAC and Traditional Model
df_2022 <- df_csv %>% filter(Year == 2022) %>% filter(DOY >= min_doy["2022"])

# Compute the Traditional rate of change as the mean across all years
df_traditional <- df_csv %>%
  group_by(DOY) %>%
  summarise(Rate_of_Change_Traditional = mean(Rate_of_Change, na.rm = TRUE), .groups = 'drop')

# **Filter traditional rate of change to match the DOY range of 2022**
df_traditional_filtered <- df_traditional %>% filter(DOY >= min_doy["2022"])

p_2022 <- ggplot() +
  geom_line(data = df_traditional_filtered, aes(x = DOY, y = Rate_of_Change_Traditional), color = "black", size = 1) +  # Traditional model (black)
  geom_line(data = df_2022, aes(x = DOY, y = Rate_of_Change_APAC), color = "blue", size = 1) +  # APAC model (blue, on top)
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "2022",
    x = "Day of Year (DOY Min +1)",
    y = "Rate of Change (x 10^6 km²/day)"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold")  # Bold Y-axis
  )

# Combine all facets with letters (a, b, c, etc.)
final_plot <- (plot_list[["2013"]] + plot_list[["2016"]]) / 
  (plot_list[["2021"]] + p_2022) +
  plot_annotation(tag_levels = 'a')  # Add facet labels

# Display plot
print(final_plot)


