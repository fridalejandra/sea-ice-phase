# Load necessary libraries
library(mgcv)
library(ggplot2)
library(dplyr)

# Load the data
xpath_csv <- '/Users/fridaperez/Developer/repos/phase_project/SIE/S_seaice_extent_daily_v3.0.csv'
xpath_nc <- '/Users/fridaperez/Desktop/Bootstrap79-24.csv'

df_1 <- read.csv(xpath_csv)
df_2 <- read.csv(xpath_nc)

# Assuming df_1 already has 'Year', 'Month', and 'Day' columns
df_1$Date <- as.Date(with(df_1, paste(Year, Month, Day, sep = "-")), "%Y-%m-%d")

# Assuming df_2's 'Date' column is in mm/dd/yy format
df_2 <- df_2 %>%
  mutate(
    Month = as.integer(sub("^([0-9]+)/.*", "\\1", Date)),
    Day = as.integer(sub("^[0-9]+/([0-9]+)/.*", "\\1", Date)),
    Year = as.integer(sub(".*?([0-9]+)$", "\\1", Date)),
    Date = as.Date(paste(Year, Month, Day, sep = "-"), "%Y-%m-%d")
  )

# Filter data to the desired range
filter_dates <- function(df) {
  df %>%
    filter(Date >= as.Date("1978-01-01") & Date <= as.Date("2018-12-31")) %>%
    mutate(DOY = as.numeric(format(Date, "%j")))
}

df_1 <- filter_dates(df_1)
df_2 <- filter_dates(df_2)

process_data <- function(df) {
  # Traditional Annual Cycle
  avg_cycle <- df %>% group_by(DOY) %>% summarize(Mean_Extent = mean(Extent, na.rm = TRUE))
  df$Predicted_Traditional <- avg_cycle$Mean_Extent[match(df$DOY, avg_cycle$DOY)]
  traditional_rmse <- sqrt(mean((df$Extent - df$Predicted_Traditional)^2, na.rm = TRUE))
  
  # Fit the model using cyclic cubic splines for the Invariant Cycle
  gam_model <- gam(Extent ~ s(DOY, bs = "cc", k = 30), data = df)
  df$Predicted_Invariant <- predict(gam_model, newdata = list(DOY = df$DOY))
  invariant_rmse <- sqrt(mean((df$Extent - df$Predicted_Invariant)^2, na.rm = TRUE))
  
  list(df = df, traditional_rmse = traditional_rmse, invariant_rmse = invariant_rmse, gam_model = gam_model)
}

results_1 <- process_data(df_1)
results_2 <- process_data(df_2)

# Example of accessing results
cat("DF1 - Traditional RMSE:", results_1$traditional_rmse, "\n")
cat("DF1 - Invariant RMSE:", results_1$invariant_rmse, "\n")
cat("DF2 - Traditional RMSE:", results_2$traditional_rmse, "\n")
cat("DF2 - Invariant RMSE:", results_2$invariant_rmse, "\n")

# Plot example (can be modified for either dataframe)
plot_cycle <- function(df, title) {
  ggplot(df, aes(x = DOY, y = Extent)) +
    geom_point(alpha = 0.5) +
    geom_line(aes(y = Predicted_Traditional), color = "blue", linetype = "dashed", size = 1.2, label = "Traditional") +
    geom_line(aes(y = Predicted_Invariant), color = "red", size = 1.2, label = "Invariant") +
    ggtitle(title) +
    xlab("Day of Year") +
    ylab("Sea Ice Extent") +
    theme_minimal() +
    theme(legend.position = "bottom") +
    scale_color_manual(values = c("blue", "red"))
}

plot_cycle(results_1$df, "Comparison of Traditional and Invariant Cycles (DF1)")
