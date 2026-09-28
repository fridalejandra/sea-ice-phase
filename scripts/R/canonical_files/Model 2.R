# Fitting GAM with different numbers of knots (k)
gam_model_k20 <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 20, fx=FALSE), 
                     data = df_csv, method="REML", knots=list(DOY=c(0,365)))

gam_model_k50 <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 50, fx=FALSE), 
                     data = df_csv, method="REML", knots=list(DOY=c(0,365)))

gam_model_k100 <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 100, fx=FALSE), 
                      data = df_csv, method="REML", knots=list(DOY=c(0,365)))

gam_model_k150 <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 150, fx=FALSE), 
                      data = df_csv, method="REML", knots=list(DOY=c(0,365)))

gam_model_k200 <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 200, fx=FALSE), 
                      data = df_csv, method="REML", knots=list(DOY=c(0,365)))

gam_model_k250 <- gam(Extent ~ s(tdate) + s(DOY, bs = "cc", k = 250, fx=FALSE), 
                      data = df_csv, method="REML", knots=list(DOY=c(0,365)))

# Predicting and calculating RMSE for each model
df_csv$Predicted_k20 <- predict(gam_model_k20, newdata = df_csv)
df_csv$Predicted_k50 <- predict(gam_model_k50, newdata = df_csv)
df_csv$Predicted_k100 <- predict(gam_model_k100, newdata = df_csv)
df_csv$Predicted_k150 <- predict(gam_model_k150, newdata = df_csv)
df_csv$Predicted_k200 <- predict(gam_model_k200, newdata = df_csv)
df_csv$Predicted_k250 <- predict(gam_model_k250, newdata = df_csv)

rmse_k20 <- sqrt(mean((df_csv$Predicted_k20 - df_csv$Extent)^2, na.rm = TRUE))
rmse_k50 <- sqrt(mean((df_csv$Predicted_k50 - df_csv$Extent)^2, na.rm = TRUE))
rmse_k100 <- sqrt(mean((df_csv$Predicted_k100 - df_csv$Extent)^2, na.rm = TRUE))
rmse_k150 <- sqrt(mean((df_csv$Predicted_k150 - df_csv$Extent)^2, na.rm = TRUE))
rmse_k200 <- sqrt(mean((df_csv$Predicted_k200 - df_csv$Extent)^2, na.rm = TRUE))
rmse_k250 <- sqrt(mean((df_csv$Predicted_k250 - df_csv$Extent)^2, na.rm = TRUE))

print(rmse_k20)
print(rmse_k50)
print(rmse_k100)
print(rmse_k150)
print(rmse_k200)
print(rmse_k250)
