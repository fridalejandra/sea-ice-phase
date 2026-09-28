# Using thin-plate regression splines
gam_model_tp <- gam(Extent ~ s(tdate) + s(DOY, bs = "tp", k = 50, fx=FALSE), data = df_csv, method="REML")

# Using cubic regression splines
gam_model_cr <- gam(Extent ~ s(tdate) + s(DOY, bs = "cr", k = 50, fx=FALSE), data = df_csv, method="REML")

# Predicting and calculating RMSE for each model
df_csv$Predicted_tp <- predict(gam_model_tp, newdata = df_csv)
df_csv$Predicted_cr <- predict(gam_model_cr, newdata = df_csv)

rmse_tp <- sqrt(mean((df_csv$Predicted_tp - df_csv$Extent)^2, na.rm = TRUE))
rmse_cr <- sqrt(mean((df_csv$Predicted_cr - df_csv$Extent)^2, na.rm = TRUE))

print(rmse_tp)
print(rmse_cr)
