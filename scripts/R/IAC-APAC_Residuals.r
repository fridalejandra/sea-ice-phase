# ============================
#  IAC vs APAC + residual and fitted differences
# ============================

library(dplyr)
library(tidyr)
library(readxl)
library(mgcv)
library(lubridate)
library(ggplot2)
library(stringr)

file_path <- "/Users/fridaperez/Developer/repos/phase_project/SIE/Extent_Plots/S_Sea_Ice_Index_Regional_Daily_Data_G02135_v3.0.xlsx"

sheets <- c("Indian-Extent-km^2", "Pacific-Extent-km^2", "Ross-Extent-km^2",
            "Weddell-Extent-km^2", "Bell-Amundsen-Extent-km^2")

selected_years <- c(2013, 2016, 2021, 2022)

out_dir <- "IAC_APAC_comparisons_standardized"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

sanitize_sheet <- function(x) {
  x %>%
    str_replace_all("[\\^\\(\\)]+", "") %>%
    str_replace_all("[^A-Za-z0-9\\-]+", "_")
}

compute_iac_apac <- function(sheet_name) {
  
  df_excel <- read_excel(file_path, sheet = sheet_name) %>% fill(month)
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
      DOY_raw = yday(Date),
      DOY = ((DOY_raw - 50 - 1) %% 365) + 1
    )
  
  yearly_stats <- df_long %>%
    group_by(Year) %>%
    summarise(
      Date1 = Date[which.min(Extent)],
      max_extent = max(Extent, na.rm = TRUE),
      min_extent = min(Extent, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(Year) %>%
    mutate(Date2 = lag(Date1), Date3 = lead(Date1))
  
  df_long <- df_long %>% left_join(yearly_stats, by = "Year")
  
  df_long <- df_long %>%
    arrange(Year, Date) %>%
    rowwise() %>%
    mutate(
      t = case_when(
        is.na(Date2) & !is.na(Date3) ~ as.numeric(Date - Date1),
        is.na(Date3) & !is.na(Date2) ~ as.numeric(Date - Date1),
        Date >= Date1 ~ as.numeric(Date - Date1),
        Date < Date1  ~ as.numeric(Date - Date2)
      )
    ) %>%
    ungroup() %>%
    mutate(
      amplitude = max_extent - min_extent,
      amplitude = if_else(is.na(amplitude) | amplitude == 0, NA_real_, amplitude),
      scaling_factor = (Extent - min_extent) / amplitude
    ) %>%
    filter(!is.na(scaling_factor), !is.na(DOY), !is.na(t))
  
  # ---- IAC model: scaling_factor ~ s(DOY) ----
  gam_iac <- gam(
    scaling_factor ~ s(DOY, bs = "cc", k = 100),
    data = df_long,
    method = "REML",
    knots = list(DOY = c(0.5, 365.5))
  )
  
  df_long <- df_long %>%
    mutate(
      Pred_IAC = predict(gam_iac, newdata = df_long),
      Predicted_IAC_Extent = Pred_IAC * amplitude + min_extent
    )
  
  # ---- APAC model: scaling_factor ~ s(DOY) + s(t) ----
  gam_apac <- gam(
    scaling_factor ~ s(DOY, bs = "cc", k = 100) + s(t, bs = "cc", k = 100),
    data = df_long,
    method = "REML",
    knots = list(DOY = c(0.5, 365.5))
  )
  
  df_long <- df_long %>%
    mutate(
      Pred_APAC = predict(gam_apac, newdata = df_long),
      Predicted_APAC_Extent = Pred_APAC * amplitude + min_extent
    ) %>%
    mutate(
      Residual_IAC = Extent - Predicted_IAC_Extent,
      Residual_APAC = Extent - Predicted_APAC_Extent,
      Residual_Shrink = Residual_IAC - Residual_APAC,
      Cycle_Diff_IAC_APAC = Predicted_IAC_Extent - Predicted_APAC_Extent
    )
  
  df_long
}

# ---- Compute for all sectors ----
all_mod <- lapply(sheets, compute_iac_apac) %>% bind_rows()

mod_sel <- all_mod %>%
  filter(Year %in% selected_years) %>%
  arrange(Sector, Year, Date)

# ---- Standardize y-limits for each plot type across sectors ----
# Use robust 1–99% for each metric
get_ylims <- function(x) {
  q <- quantile(x, probs = c(0.01, 0.99), na.rm = TRUE)
  c(floor(q[1] / 50000) * 50000, ceiling(q[2] / 50000) * 50000)
}

yl_res_iac <- get_ylims(mod_sel$Residual_IAC)
yl_res_apac <- get_ylims(mod_sel$Residual_APAC)
yl_shrink <- get_ylims(mod_sel$Residual_Shrink)
yl_cyclediff <- get_ylims(mod_sel$Cycle_Diff_IAC_APAC)

message("Y-lims Residual_IAC: ", paste(yl_res_iac, collapse = " to "))
message("Y-lims Residual_APAC: ", paste(yl_res_apac, collapse = " to "))
message("Y-lims Residual_Shrink: ", paste(yl_shrink, collapse = " to "))
message("Y-lims Cycle_Diff_IAC_APAC: ", paste(yl_cyclediff, collapse = " to "))

# ---- Plot function ----
plot_metric <- function(df, metric_col, ylims, title_prefix, fname_suffix) {
  ggplot(df, aes(x = DOY, y = .data[[metric_col]], color = factor(Year), group = Year)) +
    geom_hline(yintercept = 0, linewidth = 0.4, alpha = 0.6) +
    geom_line(linewidth = 1) +
    coord_cartesian(ylim = ylims) +
    labs(
      title = paste(title_prefix, df$Sector[1]),
      x = "Day of Year (shifted; 1–365)",
      y = metric_col,
      color = "Year"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
}

# ---- Per sector outputs ----
for (sheet in sheets) {
  
  df_sector <- mod_sel %>%
    filter(Sector == sheet) %>%
    arrange(Year, DOY)
  
  p1 <- plot_metric(df_sector, "Residual_IAC", yl_res_iac,
                    "IAC Residuals (Selected Years) —", "resid_IAC")
  
  p2 <- plot_metric(df_sector, "Residual_APAC", yl_res_apac,
                    "APAC Residuals (Selected Years) —", "resid_APAC")
  
  p3 <- plot_metric(df_sector, "Residual_Shrink", yl_shrink,
                    "Residual shrink (IAC − APAC) —", "resid_shrink")
  
  p4 <- plot_metric(df_sector, "Cycle_Diff_IAC_APAC", yl_cyclediff,
                    "Fitted cycle difference (IAC − APAC) —", "cycle_diff")
  
  base <- sanitize_sheet(sheet)
  
  ggsave(file.path(out_dir, paste0(base, "_Residual_IAC.png")), p1, width = 12, height = 6, dpi = 200)
  ggsave(file.path(out_dir, paste0(base, "_Residual_APAC.png")), p2, width = 12, height = 6, dpi = 200)
  ggsave(file.path(out_dir, paste0(base, "_Residual_Shrink_IAC_minus_APAC.png")), p3, width = 12, height = 6, dpi = 200)
  ggsave(file.path(out_dir, paste0(base, "_Cycle_Diff_IAC_minus_APAC.png")), p4, width = 12, height = 6, dpi = 200)
  
  print(p1); print(p2); print(p3); print(p4)
}

message("Done. Outputs saved to: ", out_dir)
