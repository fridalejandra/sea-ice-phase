amplitude_factor <- sd(df_csv$Extent, na.rm = TRUE) / sd(df_csv$Predicted_k150, na.rm = TRUE)
df_csv$Predicted_Amplitude_Adjusted <- df_csv$Predicted_k150 * amplitude_factor

# Cross-correlation for phase shift
cross_corr <- ccf(df_csv$Extent, df_csv$Predicted_k150, plot = FALSE)
phase_shift <- cross_corr$lag[which.max(cross_corr$acf)]

# Adjust DOY for phase shift
df_csv$DOY_Adjusted <- df_csv$DOY + phase_shift
df_csv$DOY_Adjusted <- ifelse(df_csv$DOY_Adjusted > 365, df_csv$DOY_Adjusted - 365, df_csv$DOY_Adjusted)
df_csv$DOY_Adjusted <- ifelse(df_csv$DOY_Adjusted < 1, df_csv$DOY_Adjusted + 365, df_csv$DOY_Adjusted)

# Recalculate with phase adjustment
df_csv$Predicted_Phase_Adjusted <- predict(gam_model_k150, newdata = df_csv)
df_csv$Predicted_Amplitude_Phase_Adjusted <- df_csv$Predicted_Phase_Adjusted * amplitude_factor
amplitude_phase_rmse <- sqrt(mean((df_csv$Predicted_Amplitude_Phase_Adjusted - df_csv$Extent)^2, na.rm = TRUE))
print(amplitude_phase_rmse)

ggplot(df_csv, aes(x = Date)) +
  geom_line(aes(y = Extent), color = "black") +
  geom_line(aes(y = Predicted_Amplitude_Phase_Adjusted), color = "red") +
  labs(title = "Actual vs Predicted Sea Ice Extent", y = "Sea Ice Extent", x = "Date")
