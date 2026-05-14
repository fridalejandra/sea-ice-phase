# =============================================================================
# plot_rmse_sectors.R
# RMSE improvement by sector, Bootstrap v4 1979-2025.
# =============================================================================

library(dplyr)
library(ggplot2)
library(tidyr)
library(patchwork)

BASE_DIR <- "/Users/fridaperez/Research/repos/sea-ice-phase"
FIG_DIR  <- file.path(BASE_DIR, "scripts/R/Ch3/figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

f <- file.path(BASE_DIR, "scripts/R/Ch3/data/rmse_summary.csv")
if (!file.exists(f)) stop("rmse_summary.csv not found in Ch3/data/.")
df <- read.csv(f, stringsAsFactors = FALSE)

sector_labels <- c(
  "SIE_circumpolar"             = "Circumpolar",
  "SIE_Weddell"                 = "Weddell",
  "SIE_Amundsen_Bellingshausen" = "Amundsen-Bellingshausen",
  "SIE_Ross"                    = "Ross",
  "SIE_East_Antarctica"         = "East Antarctica",
  "SIE_King_Haakon"             = "King Haakon VII"
)

sector_order <- c("Circumpolar", "Weddell", "Amundsen-Bellingshausen",
                  "Ross", "East Antarctica", "King Haakon VII")

model_levels <- c("Traditional", "Invariant", "Amplitude", "Phase", "APAC")

model_colours <- c(
  "Traditional" = "grey60",
  "Invariant"   = "#4E79A7",
  "Amplitude"   = "#1D9E75",
  "Phase"       = "#E15759",
  "APAC"        = "#D85A30"
)

df <- df %>%
  mutate(sector_label = factor(sector_labels[sector], levels = sector_order))

df_rmse <- df %>%
  select(sector_label, rmse_trad, rmse_iac, rmse_amp, rmse_phase, rmse_apac) %>%
  pivot_longer(starts_with("rmse"), names_to = "model_raw", values_to = "rmse") %>%
  mutate(
    model = recode(model_raw,
                   rmse_trad = "Traditional", rmse_iac = "Invariant",
                   rmse_amp  = "Amplitude",   rmse_phase = "Phase",
                   rmse_apac = "APAC"),
    model = factor(model, levels = model_levels)
  )

df_pct <- df %>%
  select(sector_label, pct_imp_iac, pct_imp_amp, pct_imp_phase, pct_imp_apac) %>%
  pivot_longer(starts_with("pct"), names_to = "model_raw", values_to = "pct_imp") %>%
  mutate(
    model = recode(model_raw,
                   pct_imp_iac   = "Invariant", pct_imp_amp   = "Amplitude",
                   pct_imp_phase = "Phase",     pct_imp_apac  = "APAC"),
    model = factor(model, levels = model_levels)
  )

base_theme <- theme_bw(base_size = 11) +
  theme(
    text                = element_text(family = "Helvetica"),
    panel.grid.minor    = element_blank(),
    panel.grid.major.x  = element_blank(),
    strip.background    = element_rect(fill = "grey96", colour = "grey80"),
    strip.text          = element_text(size = 10, face = "bold"),
    legend.position     = "bottom",
    legend.title        = element_blank(),
    plot.title          = element_text(size = 11, face = "bold"),
    plot.tag            = element_text(size = 12, face = "bold"),
    axis.text.x         = element_text(size = 9)
  )

# ---- PANEL A: RMSE by sector ---- #
p_rmse <- ggplot(df_rmse, aes(x = model, y = rmse, fill = model)) +
  geom_col(width = 0.7, alpha = 0.9) +
  scale_fill_manual(values = model_colours, guide = "none") +
  scale_y_continuous(
    name   = expression(RMSE~(10^6~km^2)),
    limits = c(0, NA),
    expand = expansion(mult = c(0, 0.1))
  ) +
  facet_wrap(~ sector_label, nrow = 2, scales = "free_y") +
  labs(title = "Annual cycle model performance by sector", x = NULL, tag = "(a)") +
  base_theme

# ---- PANEL B: % improvement by sector ---- #
p_pct <- ggplot(df_pct, aes(x = model, y = pct_imp, fill = model)) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.1f%%", pct_imp)),
            vjust = -0.4, size = 2.6, colour = "grey30",
            family = "Helvetica") +
  scale_fill_manual(
    values = model_colours[c("Invariant","Amplitude","Phase","APAC")],
    guide  = "none"
  ) +
  scale_y_continuous(
    name   = "% improvement in MSE vs. traditional",
    limits = c(0, 100),
    expand = expansion(mult = c(0, 0.12))
  ) +
  facet_wrap(~ sector_label, nrow = 2) +
  labs(x = "Annual cycle model", tag = "(b)") +
  base_theme

p_combined <- p_rmse / p_pct

ggsave(file.path(FIG_DIR, "RMSE_by_sector.png"),
       p_combined, width = 12, height = 10, dpi = 300)
message("Saved: RMSE_by_sector.png")