## Load Libraries ##
library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)
library(ggplot2)
library(stats)

## LOAD DATA ##
xpath_csv <- '/Users/fridaperez/Developer/repos/phase_project/SIE/S_seaice_extent_daily_v3.0.csv'
df_csv <- read.csv(xpath_csv)

## PRE-PROCESSING ##
df_csv$Date <- as.Date(with(df_csv, paste(Year, Month, Day, sep="-")), "%Y-%m-%d")
df_csv <- df_csv %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2018-12-31'))
df_csv$tdate <- as.numeric(df_csv$Date)
df_csv$DOY <- yday(df_csv$Date)
df_csv$Extent <- as.numeric(df_csv$Extent)

## TRADITIONAL ANNUAL CYCLE ##
average_annual_cycle <- df_csv %>% group_by(DOY) %>% summarize(mean_extent = mean(Extent, na.rm = TRUE))
df_csv <- df_csv %>% left_join(average_annual_cycle, by = "DOY") %>% rename(Predicted_Traditional = mean_extent)
traditional_rmse <- sqrt(mean((df_csv$Predicted_Traditional - df_csv$Extent)^2, na.rm = TRUE))

## INVARIANT ANNUAL CYCLE ##
gam_model <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 50, fx=FALSE), data = df_csv, method="REML", knots=list(DOY=c(0,365)))
df_csv$Predicted_Invariant <- predict(gam_model, newdata = df_csv)
invariant_rmse <- sqrt(mean((df_csv$Predicted_Invariant - df_csv$Extent)^2, na.rm = TRUE))

## AMPLITUDE ADJUSTMENT ##
# pre-processing
# Calculate the annual maximum and minimum extent for each year
yearly_max_min <- df_csv %>%
  group_by(Year) %>%
  summarize(max_extent = max(Extent, na.rm = TRUE),
            min_extent = min(Extent, na.rm = TRUE))

# Merge these values back to df
df_csv <- df_csv %>%
  left_join(yearly_max_min, by = "Year")

# Calculate the  invariant annual cycle again
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

## PHASE ADJUSTMENT ##
# Cross-correlation to find the best lag
cc_results <- ccf(df_csv$Extent, df_csv$Predicted_Invariant, lag.max = 365, plot = FALSE)
best_lag <- cc_results$lag[which.max(cc_results$acf)]

# Apply best lag for phase adjustment
df_csv$DOY_Adjusted <- (df_csv$DOY + best_lag) %% 365

# Re-calculate predictions with phase-adjusted DOY
df_csv$Predicted_Phase_Adjusted <- predict(gam_model, newdata = df_csv %>% mutate(DOY = DOY_Adjusted))

# Calculate RMSE for the phase-adjusted cycle
phase_rmse <- sqrt(mean((df_csv$Predicted_Phase_Adjusted - df_csv$Extent)^2, na.rm = TRUE))

 ## APAC ##
# Re-calculate predictions with phase-adjusted DOY and compute amplitude adjustment
df_csv$Predicted_APAC <- ((df_csv$Predicted_Phase_Adjusted - min(df_csv$Predicted_Phase_Adjusted)) /
                         (max(df_csv$Predicted_Phase_Adjusted) - min(df_csv$Predicted_Phase_Adjusted))) *
                         (df_csv$max_extent - df_csv$min_extent) + df_csv$min_extent

## Calculate RMSE for APAC ##
apac_rmse <- sqrt(mean((df_csv$Predicted_APAC - df_csv$Extent)^2, na.rm = TRUE))

## Print All RMSEs ##
print(paste("Traditional RMSE:", round(traditional_rmse, 3)))
print(paste("Invariant RMSE:", round(invariant_rmse, 3)))
print(paste("Amplitude Adjusted RMSE:", round(amplitude_rmse, 3))) #improved slightly
print(paste("Phase Adjusted RMSE:", round(phase_rmse, 3)))
print(paste("APAC:", round(apac_rmse, 3)))

## Plot APAC, Invariant, and Traditional Annual Cycles
## Why does the APAC have a thicker line? high variations? 
## I was attempting a plot like Figure 2, but found this interesting and not sure on the diagnosis. 
library(ggplot2)
library(dplyr)

df_long <- df_csv %>%
  gather(key = "Model", value = "Prediction", Predicted_Traditional, Predicted_Invariant, Predicted_APAC)

ggplot(df_long, aes(x = DOY, y = Prediction, color = Model)) +
  geom_line() + 
  facet_wrap(~Model, scales = "free_y") +  
  labs(title = "Sea Ice Extent Predictions by Model",
       x = "Day of the Year",
       y = "Sea Ice Extent (millions of km²)") +
  theme_minimal() +
  scale_color_manual(values = c("Predicted_Traditional" = "blue", "Predicted_Invariant" = "green",
                                "Predicted_APAC" = "red"))  # Adjust color coding as necessary


