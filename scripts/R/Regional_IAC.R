# Load necessary libraries
library(dplyr)
library(tidyverse)
library(readxl)
library(mgcv)
library(lubridate)


# Define the file path
file_path <- "/Users/fridaperez/Developer/repos/phase_project/SIE/Extent_Plots/S_Sea_Ice_Index_Regional_Daily_Data_G02135_v3.0.xlsx"

df_excel <- read_excel(file_path, sheet = "Indian-Area-km^2") #"Indian-Area-km^2,"Pacific-Area-km^2,"Ross-Area-km^2","Weddell-Area-km^2,'Bell-Amundsen-Area-km^2'"

# Fill down the 'month' column to propagate the value to all rows within each month
df_excel <- df_excel %>% fill(month)

# Reshape the data from wide to long format
df_long <- df_excel %>%
  pivot_longer(cols = starts_with("19") | starts_with("20"), 
               names_to = "Year", 
               values_to = "Area", 
               names_transform = list(Year = as.integer))

# Ensure 'day' column is numeric
df_long$day <- as.numeric(df_long$day)

# Generate the Date column
df_long <- df_long %>%
  mutate(Date = as.Date(paste(Year, month, day, sep = "-"), format = "%Y-%B-%d"))

# Inspect the long-format data
str(df_long)

df_long <- df_long %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2023-12-31'))

# Make the dates numeric 
df_long$tdate <- as.numeric(df_long$Date)

# Calculate day of year (DOY)
df_long$DOY <- yday(df_long$Date)

# Make the Extent values numeric
df_long$Area <- as.numeric(df_long$Area)

# Traditional Annual Cycle
average_annual_cycle <- df_long %>% group_by(DOY) %>% summarize(mean_extent = mean(Area, na.rm = TRUE))
df_long <- df_long %>% left_join(average_annual_cycle, by = "DOY") %>% rename(Predicted_Traditional = mean_extent)
traditional_rmse <- sqrt(mean((df_long$Predicted_Traditional - df_long$Area)^2, na.rm = TRUE))

# Invariant Annual Cycle using cyclic cubic splines
gam_model <- gam(Area ~ s(tdate, bs = "cc",k=14)+s(DOY, bs = "cc", k = 25), data = df_long)
df_long$Predicted_Invariant <- predict(gam_model, newdata = df_long)
invariant_rmse <- sqrt(mean((df_long$Predicted_Invariant - df_long$Area)^2, na.rm = TRUE))

# Print RMSE values
print(traditional_rmse)
print(invariant_rmse)

# Calculate the average invariant cycle across all years
average_invariant_cycle <- df_long %>% group_by(DOY) %>% summarize(mean_invariant_extent = mean(Predicted_Invariant, na.rm = TRUE))

# Ensure Traditional and Invariant averages are in the same dataframe for plotting
average_cycles <- average_annual_cycle %>% 
  left_join(average_invariant_cycle, by = "DOY") %>% 
  rename(Traditional = mean_extent, Invariant = mean_invariant_extent)

# Plot the fixed average Traditional and Invariant annual cycles
ggplot(average_cycles, aes(x = DOY)) +
  geom_line(aes(y = Traditional, color = 'Traditional'), linetype = 'dashed', size = 1) +  # Traditional model
  geom_line(aes(y = Invariant, color = 'Invariant'), linetype = 'dotdash', size = 1) +  # Invariant model
  labs(#title = 'Indian',
       x = 'Day of Year',
       y = 'Sea Ice Extent (in millions of square kilometers)',
       color = 'Model') +
  theme_minimal() +
  scale_color_manual(values = c('Traditional' = 'blue', 'Invariant' = 'red')) +
  theme(legend.position = 'bottom')

# Calculate the difference between observed extent and invariant predicted extent (residuals)
df_long <- df_long %>%
  mutate(Difference_Invariant_Observed = Area - Predicted_Invariant)

# Optional: Summarize the residuals by averaging them across all years
average_difference <- df_long %>%
  group_by(DOY) %>%
  summarize(mean_difference = mean(Difference_Invariant_Observed, na.rm = TRUE))

# Filter the data for the selected years (2008, 2004, 2016)
df_selected_years <- df_long %>% filter(Year %in% c(2014,2016, 2022, 2023))

# Plot residuals (Observed - Invariant) for the selected years
ggplot(df_selected_years, aes(x = DOY, y = Difference_Invariant_Observed, group = Year, color = factor(Year))) +
  geom_line(alpha = 0.6, size = 1) +  # Adjust size for better visibility
  labs(#title = 'Difference Between Observed and Invariant Annual Cycle (2004, 2008, 2016)',
       x = 'Day of Year',
       y = 'Difference (Observed - Invariant)',
       color = 'Year') +
  theme_minimal() +
  theme(legend.position = "right") +
  scale_color_viridis_d()  # Optional: use a color palette for better visualization


# Alternatively, you can plot the average difference over all years
ggplot(average_difference, aes(x = DOY, y = mean_difference)) +
  geom_line(color = 'purple', size = 1) +
  labs(title = 'Average Difference Between Observed and Invariant Annual Cycle',
       x = 'Day of Year',
       y = 'Average Difference (Observed - Invariant)') +
  theme_minimal()


