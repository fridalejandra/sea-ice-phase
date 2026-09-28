# Load necessary libraries
library(dplyr)
library(readxl)
library(lubridate)
library(ggplot2)
library(mgcv)
library(patchwork)

# Define the file path and sheet name
file_path <- "/Users/fridaperez/Developer/repos/phase_project/SIE/Extent_Plots/S_Sea_Ice_Index_Regional_Daily_Data_G02135_v3.0.xlsx"
sheet <- "Weddell-Extent-km^2"  # Replace with desired region

# Load the data for the specified region
df <- read_excel(file_path, sheet = sheet)

# Fill down the 'month' column and convert 'day' column to numeric
df <- df %>%
  fill(month) %>%
  mutate(day = as.numeric(day))

# Reshape data from wide to long format
df_long <- df %>%
  pivot_longer(cols = starts_with("19") | starts_with("20"), 
               names_to = "Year", 
               values_to = "Extent", 
               names_transform = list(Year = as.integer)) %>%
  mutate(Date = as.Date(paste(Year, month, day, sep = "-"), "%Y-%B-%d"),
         DOY = yday(Date)) %>%
  filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2023-12-31'))

# Calculate retreat timing
retreat_timing <- df_long %>%
  group_by(Year) %>%
  filter(!is.na(Extent) & Extent == max(Extent, na.rm = TRUE)) %>%
  summarise(Max_DOY = first(DOY), Max_Extent = Extent, .groups = 'drop') %>%
  left_join(df_long, by = "Year") %>%
  filter(DOY > Max_DOY & Extent < Max_Extent) %>%
  group_by(Year) %>%
  filter(DOY == min(DOY, na.rm = TRUE)) %>%
  summarise(Retreat_DOY = DOY, .groups = 'drop')

# Calculate advance timing
advance_timing <- df_long %>%
  filter(DOY >= 50 & DOY <= 68) %>%
  group_by(Year) %>%
  filter(Extent == min(Extent, na.rm = TRUE)) %>%
  summarise(Min_DOY = DOY, Min_Extent = Extent, .groups = 'drop') %>%
  left_join(df_long, by = "Year") %>%
  filter(DOY > Min_DOY & Extent > Min_Extent) %>%
  group_by(Year) %>%
  filter(DOY == min(DOY, na.rm = TRUE)) %>%
  summarise(Advance_DOY = DOY, .groups = 'drop')

# Combine retreat and advance timing into one table, filtering for complete cases
timing_table <- retreat_timing %>%
  left_join(advance_timing, by = "Year") %>%
  drop_na()

# Fit GAM models with higher flexibility for curvilinear trends
retreat_gam <- gam(Retreat_DOY ~ s(Year, bs = "tp", k = 15), data = timing_table)
advance_gam <- gam(Advance_DOY ~ s(Year, bs = "tp", k = 15), data = timing_table)

# Predict values for visualization
timing_table <- timing_table %>%
  mutate(Retreat_DOY_Predicted = predict(retreat_gam),
         Advance_DOY_Predicted = predict(advance_gam))

# Calculate correlations between observed and predicted values
retreat_correlation <- cor(timing_table$Retreat_DOY, timing_table$Retreat_DOY_Predicted)
advance_correlation <- cor(timing_table$Advance_DOY, timing_table$Advance_DOY_Predicted)

# Plot retreat timing with curvilinear trend and correlation annotation
retreat_plot <- ggplot(timing_table, aes(x = Year, y = Retreat_DOY)) +
  geom_point(color = "#4682B4") +
  geom_line(aes(y = Retreat_DOY_Predicted), color = "darkblue", size = 1) +
  annotate("text", x = min(timing_table$Year) + 1, y = max(timing_table$Retreat_DOY) - 5,
           label = paste("Correlation:", round(retreat_correlation, 2)),
           color = "darkblue", size = 4, hjust = 0) +
  labs(title = paste("Retreat Timing -", sheet), y = "Retreat DOY", x = "Year") +
  theme_minimal()

# Plot advance timing with curvilinear trend and correlation annotation
advance_plot <- ggplot(timing_table, aes(x = Year, y = Advance_DOY)) +
  geom_point(color = "#FF6347") +
  geom_line(aes(y = Advance_DOY_Predicted), color = "red", size = 1) +
  annotate("text", x = min(timing_table$Year) + 1, y = max(timing_table$Advance_DOY) - 5,
           label = paste("Correlation:", round(advance_correlation, 2)),
           color = "red", size = 4, hjust = 0) +
  labs(title = paste("Advance Timing -", sheet), y = "Advance DOY", x = "Year") +
  theme_minimal()

# Combine the plots
combined_plot <- retreat_plot / advance_plot +
  plot_annotation(title = paste("Curvilinear Trends in Sea Ice Retreat and Advance Timing -", sheet))

# Display the combined plot
print(combined_plot)

