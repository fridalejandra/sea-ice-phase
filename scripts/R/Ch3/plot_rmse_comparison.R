# =============================================================================
# plot_rmse_comparison.R
# Circumpolar RMSE comparison across 3 time periods.
# =============================================================================

library(dplyr)
library(ggplot2)
library(tidyr)
library(patchwork)

BASE_DIR <- "/Users/fridaperez/Research/repos/sea-ice-phase"
SCEN_DIR <- file.path(BASE_DIR, "scripts/R/Ch3/data/rmse_scenarios")
FIG_DIR  <- file.path(BASE_DIR, "scripts/R/Ch3/figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- LOAD SCENARIO RESULTS ---- #
scenarios <- list(
  "1979\u20132018 replication" = "v3_1979_2018",
  "1979\u20132018"             = "v4_1979_2018",
  "1979\u20132025"             = "v4_1979_2025"
)

read_rmse <- function(label, folder) {
  f <- file.path(SCEN_DIR, folder, "rmse_summary.csv")
  if (!file.exists(f)) { warning("Not found: ", f); return(NULL) }
  read.csv(f, stringsAsFactors = FALSE) %>%
    filter(sector == "SIE_circumpolar") %>%
    select(rmse_trad, rmse_iac, rmse_amp, rmse_phase, rmse_apac,
           pct_imp_iac, pct_imp_amp, pct_imp_phase, pct_imp_apac) %>%
    mutate(scenario = label)
}

df_all <- bind_rows(mapply(read_rmse, names(scenarios), scenarios,
                           SIMPLIFY = FALSE, USE.NAMES = FALSE))
if (nrow(df_all) == 0) stop("No scenario data found.")

model_levels <- c("Traditional", "Invariant", "Amplitude", "Phase", "APAC")

scen_levels <- c(
  "1979\u20132018 replication",
  "1979\u20132018",
  "1979\u20132025"
)

df_all$scenario <- factor(df_all$scenario, levels = scen_levels)

df_rmse <- df_all %>%
  pivot_longer(
    cols      = c(rmse_trad, rmse_iac, rmse_amp, rmse_phase, rmse_apac),
    names_to  = "model_raw", values_to = "rmse"
  ) %>%
  mutate(
    model = recode(model_raw,
                   rmse_trad = "Traditional", rmse_iac = "Invariant",
                   rmse_amp  = "Amplitude",   rmse_phase = "Phase",
                   rmse_apac = "APAC"),
    model = factor(model, levels = model_levels)
  )

df_pct <- df_all %>%
  mutate(pct_imp_trad = 0) %>%
  pivot_longer(
    cols      = c(pct_imp_trad, pct_imp_iac, pct_imp_amp, pct_imp_phase, pct_imp_apac),
    names_to  = "model_raw", values_to = "pct_imp"
  ) %>%
  mutate(
    model = recode(model_raw,
                   pct_imp_trad  = "Traditional", pct_imp_iac   = "Invariant",
                   pct_imp_amp   = "Amplitude",   pct_imp_phase = "Phase",
                   pct_imp_apac  = "APAC"),
    model = factor(model, levels = model_levels)
  ) %>%
  select(scenario, model, pct_imp)

df_plot <- df_rmse %>% left_join(df_pct, by = c("scenario", "model"))

scen_colours <- c(
  "1979\u20132018 replication" = "#2C6EA5",
  "1979\u20132018"             = "#1D9E75",
  "1979\u20132025"             = "#D85A30"
)

base_theme <- theme_bw(base_size = 11) +
  theme(
    text               = element_text(family = "Helvetica"),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.position    = "bottom",
    legend.title       = element_blank(),
    plot.title         = element_text(size = 11, face = "bold"),
    plot.tag           = element_text(size = 12, face = "bold")
  )

# ---- PANEL A: RMSE values ---- #
p_rmse <- ggplot(df_plot, aes(x = model, y = rmse, fill = scenario)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65, alpha = 0.9) +
  scale_fill_manual(values = scen_colours) +
  scale_y_continuous(
    name   = expression(RMSE~(10^6~km^2)),
    limits = c(0, NA),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(title = "Circumpolar Antarctic sea ice extent", x = NULL, tag = "(a)") +
  base_theme

# ---- PANEL B: % improvement ---- #
p_pct <- ggplot(
  df_plot %>% filter(model != "Traditional"),
  aes(x = model, y = pct_imp, fill = scenario)
) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65, alpha = 0.9) +
  scale_fill_manual(values = scen_colours) +
  scale_y_continuous(
    name   = "% improvement in MSE vs. traditional",
    limits = c(0, 100),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(x = "Annual cycle model", tag = "(b)") +
  base_theme

p_combined <- p_rmse / p_pct +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom",
        text = element_text(family = "Helvetica"))

ggsave(file.path(FIG_DIR, "RMSE_comparison_circumpolar.png"),
       p_combined, width = 9, height = 8, dpi = 300)
message("Saved: RMSE_comparison_circumpolar.png")