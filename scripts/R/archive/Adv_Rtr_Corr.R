# Load necessary libraries
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
df_csv <- df_csv %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2023-12-31'))  # Updated to end of 2023

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

# Remove rows where Year is 1978 and save the result to df_csv2
df_csv2 <- df_csv %>% 
  filter(Year != 1978)

# Calculate retreat timing: Find the day after the annual max when extent starts declining
retreat_timing_gam <- df_csv %>%
  group_by(Year) %>%
  filter(Extent == max(Extent, na.rm = TRUE)) %>%   # Identify the max extent day
  summarise(Max_DOY = DOY, Max_Extent = Extent, .groups = 'drop') %>%
  left_join(df_csv, by = "Year") %>%
  filter(DOY > Max_DOY & Extent < Max_Extent) %>%   # Find the first day after max when extent declines
  group_by(Year) %>%
  filter(DOY == min(DOY, na.rm = TRUE)) %>%         # First declining day
  summarise(Retreat_Timing_DOY = DOY, .groups = 'drop')

# Calculate advance timing: Find the first day after February when extent begins to increase
advance_timing_gam <- df_csv %>%
  filter(DOY >= 50 & DOY <= 68) %>%  # Filter to the relevant DOY range for advance
  group_by(Year) %>%
  filter(Extent == min(Extent, na.rm = TRUE)) %>%  # Identify the min extent day within range
  summarise(Min_DOY = DOY, Min_Extent = Extent, .groups = 'drop') %>%
  left_join(df_csv, by = "Year") %>%
  filter(DOY > Min_DOY & Extent > Min_Extent) %>%  # Find the first day after min when extent increases
  group_by(Year) %>%
  filter(DOY == min(DOY, na.rm = TRUE)) %>%        # First advancing day
  summarise(Advance_Timing_DOY = DOY, .groups = 'drop')

# Combine retreat and advance timing into one table
timing_table <- retreat_timing_gam %>%
  left_join(advance_timing_gam, by = "Year") %>%
  rename(Retreat_Timing_DOY = Retreat_Timing_DOY, Advance_Timing_DOY = Advance_Timing_DOY) %>%
  drop_na()  # Remove rows with any NA values to ensure matching row counts

# Analyze Advance vs. Retreat Timing (Same Year and Lagged)

# Direct Correlation between Advance and Retreat Timing (same year)
advance_retreat_correlation <- cor(timing_table$Advance_Timing_DOY, timing_table$Retreat_Timing_DOY, method = "pearson")

# Output the correlation result
cat("Correlation between Advance and Retreat Timing DOY (same year):", advance_retreat_correlation, "\n")

# Lag Analysis: Compare Advance Timing DOY of one year with Retreat Timing DOY of the following year
# Create lagged column for Retreat Timing DOY
timing_table <- timing_table %>%
  arrange(Year) %>%
  mutate(Retreat_Timing_DOY_Lagged = lead(Retreat_Timing_DOY))

# Calculate correlation between Advance Timing DOY (current year) and Retreat Timing DOY (next year)
advance_lagged_retreat_correlation <- cor(timing_table$Advance_Timing_DOY, timing_table$Retreat_Timing_DOY_Lagged, use = "complete.obs", method = "pearson")

# Output the lagged correlation result
cat("Correlation between Advance Timing DOY (current year) and Retreat Timing DOY (next year):", advance_lagged_retreat_correlation, "\n")

# Visualize the relationships

# Same-year Advance vs Retreat Timing Plot
advance_retreat_plot <- ggplot(timing_table, aes(x = Advance_Timing_DOY, y = Retreat_Timing_DOY)) +
  geom_point(color = "#4682B4") +
  geom_smooth(method = "lm", se = FALSE, color = "darkblue", linetype = "dashed") +
  annotate("text", x = min(timing_table$Advance_Timing_DOY) + 2, y = max(timing_table$Retreat_Timing_DOY) - 2,
           label = paste("Correlation:", round(advance_retreat_correlation, 2)),
           color = "darkblue", size = 4, hjust = 0) +
  labs(title = "Same-Year Relationship: Advance vs Retreat Timing", x = "Advance Timing (DOY)", y = "Retreat Timing (DOY)") +
  theme_minimal()

# Lagged Advance vs Retreat Timing Plot
advance_lagged_retreat_plot <- ggplot(timing_table, aes(x = Advance_Timing_DOY, y = Retreat_Timing_DOY_Lagged)) +
  geom_point(color = "#FF6347") +
  geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "dashed") +
  annotate("text", x = min(timing_table$Advance_Timing_DOY) + 2, y = max(timing_table$Retreat_Timing_DOY_Lagged, na.rm = TRUE) - 2,
           label = paste("Lagged Correlation:", round(advance_lagged_retreat_correlation, 2)),
           color = "red", size = 4, hjust = 0) +
  labs(title = "Lagged Relationship: Advance Timing (Current Year) vs Retreat Timing (Next Year)", x = "Advance Timing (DOY)", y = "Retreat Timing (Next Year, DOY)") +
  theme_minimal()

# Combine the lag analysis plots
combined_advance_retreat_plot <- advance_retreat_plot / advance_lagged_retreat_plot
combined_advance_retreat_plot + plot_annotation(title = "Relationship Between Sea Ice Advance and Retreat Timing")
