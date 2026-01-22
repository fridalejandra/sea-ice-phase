# ============================
# Script A: APAC residuals (corrected) + standardized y-axis
# ============================

library(dplyr)
library(tidyr)
library(readxl)
library(mgcv)
library(lubridate)
library(ggplot2)
library(stringr)

# ---- Inputs ----
file_path <- "/Users/fridaperez/Developer/repos/phase_project/SIE/Extent_Plots/S_Sea_Ice_Index_Regional_Daily_Data_G02135_v3.0.xlsx"

sheets <- c("Indian-Extent-km^2", "Pacific-Extent-km^2", "Ross-Extent-km^2",
            "Weddell-Extent-km^2", "Bell-Amundsen-Extent-km^2")

selected_years <- c(2013, 2016, 2021, 2022)

out_dir <- "APAC_residual_plots_standardized"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Helpers ----
sanitize_sheet <- function(x) {
  x %>%
    str_replace_all("[\\^\\(\\)]+", "") %>%
    str_replace_all("[^A-Za-z0-9\\-]+", "_")
}

# Compute APAC residuals for one sector sheet
compute_apac_residuals <- function(sheet_name) {
  
  df_excel <- read_excel(file_path, sheet = sheet_name)
  
  # Ensure 'month' and 'day' columns exist; fill down month
  df_excel <- df_excel %>% fill(month)
  df_excel$day <- as.numeric(df_excel$day)
  
  df_long <- df_excel %>%
    pivot_longer(
      cols = starts_with("19") | starts_with("20"),
      names_to = "Year",
      values_to = "Extent",
      names_transform = list(Year = as.integer)
    ) %>%
    mutate(
      Extent = as.numeric(Extent),
      Sector = sheet_name,
      Date = as.Date(paste(Year, month, day, sep = "-"), format = "%Y-%B-%d")
    ) %>%
    filter(Date >= as.Date("1978-01-01"), Date <= as.Date("2023-12-31")) %>%
    arrange(Year, Date) %>%
    mutate(
      # Keep DOY in 1..365 space for cyclic splines
      DOY_raw = yday(Date),
      DOY = ((DOY_raw - 50 - 1) %% 365) + 1
    )
  
  # Yearly min/max + date of minimum (Date1)
  yearly_stats <- df_long %>%
    group_by(Year) %>%
    summarise(
      Date1 = Date[which.min(Extent)],
      max_extent = max(Extent, na.rm = TRUE),
      min_extent = min(Extent, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(Year) %>%
    mutate(
      Date2 = lag(Date1),
      Date3 = lead(Date1)
    )
  
  df_long <- df_long %>% left_join(yearly_stats, by = "Year")
  
  # Compute t with safe handling for edges
  # (For first year in record, Date2 is NA; for last year, Date3 is NA)
  df_long <- df_long %>%
    arrange(Year, Date) %>%
    rowwise() %>%
    mutate(
      t = case_when(
        # First year: use next year's min date
        is.na(Date2) & !is.na(Date3) ~ as.numeric(Date - Date1),
        # Last year: use current year's min date
        is.na(Date3) & !is.na(Date2) ~ as.numeric(Date - Date1),
        # Otherwise:
        Date >= Date1 ~ as.numeric(Date - Date1),
        Date < Date1  ~ as.numeric(Date - Date2)
      )
    ) %>%
    ungroup()
  
  # Avoid division by zero for amplitude (rare but safeguard)
  df_long <- df_long %>%
    mutate(
      amplitude = max_extent - min_extent,
      amplitude = if_else(is.na(amplitude) | amplitude == 0, NA_real_, amplitude),
      scaling_factor = (Extent - min_extent) / amplitude
    ) %>%
    filter(!is.na(scaling_factor), !is.na(DOY), !is.na(t))
  
  # Fit APAC GAM
  # Note: bs="cc" requires specifying knots for cyclic variable.
  # mgcv can infer with correct domain, but knots makes it safer.
  gam_model_apac <- gam(
    scaling_factor ~ s(DOY, bs = "cc", k = 100) + s(t, bs = "cc", k = 100),
    data = df_long,
    method = "REML",
    knots = list(DOY = c(0.5, 365.5))  # ensures wrap for DOY
  )
  
  df_long <- df_long %>%
    mutate(
      Predicted_Amplitude_Adjusted = predict(gam_model_apac, newdata = df_long),
      Predicted_Amplitude_Adjusted_Extent = Predicted_Amplitude_Adjusted * amplitude + min_extent,
      Residual_APAC = Extent - Predicted_Amplitude_Adjusted_Extent
    )
  
  return(df_long)
}

# ---- 1) Compute APAC residuals for all sectors ----
all_apac <- lapply(sheets, compute_apac_residuals) %>% bind_rows()

# ---- 2) Compute standardized y-limits across sectors (for selected years only) ----
apac_sel <- all_apac %>%
  filter(Year %in% selected_years) %>%
  arrange(Sector, Year, Date)

# Robust global limits: use 1st-99th percentile to avoid a single spike dominating
ylims <- quantile(apac_sel$Residual_APAC, probs = c(0.01, 0.99), na.rm = TRUE)
ylims <- c(floor(ylims[1] / 50000) * 50000, ceiling(ylims[2] / 50000) * 50000)

message("Standardized y-limits (1-99%): ", paste(ylims, collapse = " to "))

# ---- 3) Plot per sector, corrected grouping/sorting, standardized y-axis ----
for (sheet in sheets) {
  
  df_plot <- apac_sel %>%
    filter(Sector == sheet) %>%
    arrange(Year, DOY)
  
  p <- ggplot(df_plot, aes(x = DOY, y = Residual_APAC, color = factor(Year), group = Year)) +
    geom_hline(yintercept = 0, linewidth = 0.4, alpha = 0.6) +
    geom_line(linewidth = 1) +
    coord_cartesian(ylim = ylims) +
    labs(
      title = paste("APAC Residuals of Sea ice extent in", sheet, "(Selected Years)"),
      x = "Day of Year (shifted; 1–365)",
      y = "Residual sea ice extent (km²)",
      color = "Year"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
  
  fname <- file.path(out_dir, paste0("residuals_", sanitize_sheet(sheet), "_APAC_selected_years.png"))
  ggsave(fname, plot = p, width = 12, height = 6, dpi = 200)
  print(p)
}

message("Done. Plots saved to: ", out_dir)

