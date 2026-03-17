# Load necessary libraries
library(dplyr)
library(tidyverse)
library(readxl)
library(mgcv)
library(lubridate)

# Define the file path
file_path <- "/Users/fridaperez/Developer/repos/phase_project/SIE/Extent_Plots/S_Sea_Ice_Index_Regional_Daily_Data_G02135_v3.0.xlsx"

# Load multiple sheets and combine into one dataframe, adding a 'Sector' column
sectors <- c("Bell-Amundsen-Extent-km^2", "Indian-Extent-km^2", "Pacific-Extent-km^2", "Ross-Extent-km^2", "Weddell-Extent-km^2")
df_list <- lapply(sectors, function(sheet) {
  data <- read_excel(file_path, sheet = sheet)
  data$Sector <- sheet
  return(data)
})
df_long <- bind_rows(df_list)

# Process each sector's data
df_long <- df_long %>% fill(month)
df_long <- df_long %>%
  pivot_longer(cols = starts_with("19") | starts_with("20"), 
               names_to = "Year", 
               values_to = "Extent", 
               names_transform = list(Year = as.integer)) %>%
  mutate(day = as.numeric(day),
         Date = as.Date(paste(Year, month, day, sep = "-"), format = "%Y-%B-%d")) %>%
  filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2023-12-31')) %>%
  mutate(tdate = as.numeric(Date), DOY = yday(Date), Extent = as.numeric(Extent))

# Calculate traditional and invariant cycles for each sector
df_long <- df_long %>%
  group_by(Sector) %>%
  group_modify(~ {
    # Traditional annual cycle (average by day of year)
    traditional_cycle <- .x %>% group_by(DOY) %>% summarize(Traditional = mean(Extent, na.rm = TRUE))
    .x <- .x %>% left_join(traditional_cycle, by = "DOY")
    
    # Invariant annual cycle using GAM model
    gam_model <- gam(Extent ~ s(DOY, bs = "cc", k = 25), data = .x)
    .x$Predicted_Invariant <- predict(gam_model, newdata = .x)
    
    return(.x)
  }) %>%
  ungroup()

# Average cycles for plotting
average_cycles <- df_long %>%
  group_by(Sector, DOY) %>%
  summarize(
    Traditional = mean(Traditional, na.rm = TRUE),
    Invariant = mean(Predicted_Invariant, na.rm = TRUE),
    .groups = 'drop'
  )

# Set common y-axis limits for all facets
y_limits <- range(c(average_cycles$Traditional, average_cycles$Invariant), na.rm = TRUE)

# Plot with both traditional and invariant cycles and y-axis adjustment
ggplot(average_cycles, aes(x = DOY)) +
  geom_line(aes(y = Traditional, color = 'Traditional'), linetype = 'dashed', size = 1) +
  geom_line(aes(y = Invariant, color = 'Invariant'), linetype = 'dotdash', size = 1) +
  scale_x_continuous(
    breaks = c(1, 91, 182, 274, 365),
    labels = c("Jan", "Apr", "Jul", "Oct", "Dec")
  ) +
  scale_y_continuous(
    limits = y_limits,
    labels = scales::label_number(scale = 1e-6, suffix = " x10^6", accuracy = 0.1)
  ) +
  labs(
    x = 'Month',
    y = expression(bold(Extent ~ (x10^6 ~ km^2))),
    color = 'Cycle Type'
  ) +
  theme_minimal(base_size = 14) +
  scale_color_manual(values = c('Traditional' = 'blue', 'Invariant' = 'red')) +
  facet_wrap(~ Sector, scales = "fixed") +
  theme(
    legend.position = 'bottom',
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    axis.title = element_text(size = 18, face = "bold"),
    axis.text = element_text(size = 14),
    strip.text = element_text(size = 16, face = "bold")
  )
