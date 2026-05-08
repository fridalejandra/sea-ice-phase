library(dplyr)
library(tidyverse)
library(mgcv)
library(lubridate)

########################################################
############## PRE-PROCESSING ##########################

# 1. Load the CSV file
df_csv <- read.csv("/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv")

# 2. Convert 'Year', 'Month', 'Day' to Date
df_csv$Date <- as.Date(with(df_csv, paste(Year, Month, Day, sep="-")), "%Y-%m-%d")
df_csv <- df_csv %>% filter(Date >= as.Date('1978-01-01') & Date <= as.Date('2018-12-31'))

# 3. Make the dates numeric : 'tdate' = the same dates, but turned into a continuous
# day counter since 1970-01-01 (e.g., 5678, 7890, 14500 …).
df_csv$tdate <- as.numeric(df_csv$Date)

# 4. Calculate day of year (DOY)
df_csv$DOY <- yday(df_csv$Date)

# 5. Make the Extent values numeric
df_csv$Extent <- as.numeric(df_csv$Extent)

############## Calculating Phase ########################
#1. Calculate min extent day per year, so mostly in February
#   Date1 = Date_min
yearly_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(Date1 = Date[which.min(Extent)], .groups = 'drop')

#2. Add Date2 (min date of last year) and Date3 (min date of next year)
#   Date2 = Date_min - 1
#   Date3 = Date_min + 1
yearly_stats <- yearly_stats %>%
  mutate(Date2 = lag(Date1),   # Date for last year
         Date3 = lead(Date1))  # Date for next year

#3. Merge yearly_stats back into df_csv
df_csv <- df_csv %>%
  left_join(yearly_stats, by = "Year")

#4. Calculate t-values based on the logic provided
df_csv <- df_csv %>%
  rowwise() %>%
  mutate(t = case_when(
    Year == 1978  ~ 365 - as.numeric(Date3 - Date),  # If the first date is 1978-10-01 and the next min is ~1979-02-20 (~142 days ahead), then,t = 365 − 142 = 223
    Year == 1979 & Date < Date1 ~ 365 - as.numeric(Date1 - Date),  
    # Special case for early 1979:
    # ----------------------------
    # Normally, for dates before this year's minimum (Date < Date1), we would measure
    # "days since last year's minimum" using Date2. But the record only begins in late 1978,
    # so we don’t have a reliable 1978 minimum (Date2 is missing or incomplete).
    #
    # Instead, we wrap around from the upcoming 1979 minimum:
    #   t = 365 - (Date1 - Date)
    # e.g., if Date = 1979-01-15 and Date1 = 1979-02-20, then Date1 - Date = 36 days,
    # so t = 365 - 36 = 329 days since the (unobserved) previous minimum.
    #
    # This ensures that t stays on a ~0–365 scale, even though the data set starts mid-cycle.
    Date >= Date1 ~ as.numeric(Date - Date1),  # For current year
    Date < Date1 ~ as.numeric(Date - Date2)    # For last year
  )) %>%
  ungroup()

#5.  Calculate t_min and t_max by year
#    For each year, find the range of t-values:
#    t_min ≈ 0 (the annual minimum day)
#    t_max ≈ length of that year’s min-to-min cycle (~363–367 days)
#
#    These bounds allow us to normalize each year’s cycle onto [0, 1],
#    so that differences in calendar length don’t distort the phase scaling.

t_stats <- df_csv %>%
  group_by(Year) %>%
  summarise(t_min = min(t, na.rm = TRUE), 
            t_max = max(t, na.rm = TRUE)) %>%
  ungroup()

#6. Merge t_min and t_max back into the main data set
df_csv <- df_csv %>%
  left_join(t_stats, by = "Year")


#7. Here is where the real phase adjustment comes in. The former was a lot of pre-processing.
# Phase adjustment:
# -----------------
# Map each year's min→min cycle onto a common 0–365 "phase" axis.
# First normalize (t - t_min)/(t_max - t_min) so each year spans [0,1],
# then rescale by 365 to get phase days.
#
# We use the Beta CDF (pbeta) as the mapping function. With shape1=1, shape2=1
# this reduces to the uniform case (a straight linear stretch), but the Beta 
# formulation allows flexible skew/stretching if desired in other contexts.
#
# The key point: this puts every year on the same 0–365 phase scale, so the GAM 
# can align seasonal features consistently across years and any residual 
# departures reflect real timing differences rather than cycle-length quirks.

df_csv <- df_csv %>%
  rowwise() %>%
  mutate(phase = 365 * pbeta((t - t_min) / (t_max - t_min + 1e-10), # +1e-10 prevents division by zero in pathological cases
                             shape1 = 1, shape2 = 1))  # Using Beta distribution CDF for phase adjustment

#8. One-pass calendar prep + window filter 
df_csv <- df_csv %>%
  mutate(
    Date = as.Date(ISOdate(as.numeric(Year), as.numeric(Month), as.numeric(Day))),
    Year = lubridate::year(Date),
    DOY  = lubridate::yday(Date)
  ) %>%
  filter(Date >= as.Date("1978-01-01"),
         Date <= as.Date("2018-12-31"))

############## Calculating AMPLITUDE ########################
#1. Per-year extrema (calendar year) 
yearly_stats <- df_csv %>%
  group_by(Year) %>%
  summarize(
    max_extent = max(Extent, na.rm = TRUE),
    min_extent = min(Extent, na.rm = TRUE),
    .groups = "drop"
  )

#2. Attach extrema back to rows 
df_csv <- df_csv %>%
  left_join(yearly_stats, by = "Year")

#3.Calculate the amplitude (max - min) to get how much sea ice (extent) grows and shrinks between min and max
# Amplitude normalization
# Then rescale daily Extent values to a 0–1 "scaling factor":
# scaling_factor = (Extent - min) / amplitude
#
# Purpose:
# - Removes inter-annual differences in absolute ice cover (magnitude).
# - Puts each year's cycle on the same relative scale (0 = annual min, 1 = annual max).
# - Allows the GAM to learn the SHAPE of the seasonal cycle, not its absolute size.
# After fitting, predictions are back-transformed into physical units by
# multiplying by the amplitude and adding back the year's minimum.

df_csv <- df_csv %>%
  mutate(amplitude = max_extent - min_extent,
         scaling_factor = (Extent - min_extent) / amplitude)

############## Fitting the GAM to Amplitude and Phase  ########################
# ---------------------------------------------------------------
# APAC (practical implementation): shape on normalized scale (0–1),
# with phase alignment + calendar DOY + slow drift in time.
# Response is the amplitude-normalized series ("scaling_factor").
# ---------------------------------------------------------------

# 1) Fit the model 
gam_apac_full <- gam(
  scaling_factor ~ 
    s(tdate, bs = "cc", k = 150) +   # slow drift across the whole record (numeric date)
    s(DOY,   bs = "cc", k = 100) +   # calendar-seasonal features (on DOY scale)
    s(phase, bs = "cc", k = 100),    # phase-aligned annual shape (min→min normalized to 0–365)
  data = df_csv
  # , method = "REML"                # (optional but recommended: stabler smoothing selection)
  # , knots  = list(DOY = c(0,365),  # (optional: explicitly pin cyclic boundaries)
  #                 phase = c(0,365))
)

# 2) Predict the model on the normalized (0–1) scale
#    This is the fitted relative shape, BEFORE converting back to km^2.
df_csv$APAC_shape_norm <- predict(gam_apac_full, newdata = df_csv)

# 3) Back-transform to physical units:
#    E_hat = shape_hat * amplitude_y + min_extent_y
df_csv$APAC_extent_hat <- df_csv$APAC_shape_norm * df_csv$amplitude + df_csv$min_extent

# 4) Quick skill metric in observed units (million km^2)
apac_rmse <- sqrt(mean((df_csv$APAC_extent_hat - df_csv$Extent)^2, na.rm = TRUE))
print(apac_rmse)

# -------------------------
# What each term is doing:
# -------------------------
# - scaling_factor:
#     (Extent - min_y) / (max_y - min_y). Amplitude removed; response is 0–1.
#
# - s(phase, ...):
#     Learns the common seasonal SHAPE on a phase axis where each year’s
#     min→min cycle is normalized to 0–365 (APAC’s phase adjustment).
#
# - s(DOY, ...):
#     Lets the model keep calendar-tied features that persist on DOY
#     (e.g., small departures not fully explained by phase alignment).
#
# - s(tdate, ...):
#     Captures slow multi-year drift (trend-like behavior) on the numeric
#     date axis. This is NOT cyclic; it soaks up long-term changes in the
#     normalized series beyond the seasonal shape.
#
# - Back-transform:
#     Put the 0–1 prediction back into km^2 with the year’s amplitude
#     (max - min) and baseline (min).

# ============================================================
# Diagnostics from APAC:
#   - anomalies (km^2 and normalized)
#   - trend (smooth in time on anomalies)
#   - volatility (by DOY and rolling window)
#   - daily rate of change (observed & APAC)
# ============================================================

############## Getting into the variation  ########################


############## Volatility ########################################


############## Daily Rate of Change ##############################



############## Anomaly ###########################################



