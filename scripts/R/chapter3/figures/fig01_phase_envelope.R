library(dplyr)
library(ggplot2)
library(readr)
library(patchwork)

BASE <- "~/Research/repos/sea-ice-phase/scripts/R/"
DATA <- file.path(BASE, "chapter3/data")
FIGS <- file.path(BASE, "chapter3/figures")


dir.create(FIGS, recursive = TRUE, showWarnings = FALSE)

phase_df <- read_csv(file.path(DATA, "sector_annual_phase_amplitude.csv"), show_col_types = FALSE)

phase_df <- phase_df %>% filter(sector != "circumpolar")

sector_levels <- c(
  "Amundsen_Bellingshausen",
  "Ross",
  "Weddell",
  "East_Antarctica",
  "King_Haakon"
)

sector_labels <- c(
  "Amundsen_Bellingshausen" = "Amundsen–Bellingshausen",
  "Ross" = "Ross Sea",
  "Weddell" = "Weddell Sea",
  "East_Antarctica" = "East Antarctica",
  "King_Haakon" = "King Haakon"
)

phase_df$sector <- factor(phase_df$sector, levels = sector_levels)

phase_df <- phase_df %>%
  mutate(
    phase_group = case_when(
      phase_scalar < 0 ~ "Ahead of phase",
      phase_scalar > 0 ~ "Behind phase",
      TRUE ~ "Neutral"
    )
  )

p_phase <- ggplot(phase_df, aes(Year, phase_scalar)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 0.6, color = "black") +
  geom_point(aes(color = phase_group), size = 1.6) +
  geom_smooth(method = "loess", se = FALSE, span = 0.35,
              color = "red", linewidth = 0.8) +
  facet_wrap(~sector, scales = "free_y", ncol = 3,
             labeller = as_labeller(sector_labels)) +
  scale_color_manual(
    values = c(
      "Ahead of phase" = "blue",
      "Behind phase" = "red",
      "Neutral" = "gray30"
    )
  ) +
  labs(
    title = "Sectoral variability in seasonal timing",
    x = "Year",
    y = "Phase shift (days)",
    color = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )

print(p_phase)

ggsave(
  file.path(FIGS, "fig01_phase_timeseries.png"),
  p_phase,
  width = 11,
  height = 7,
  dpi = 300
)




#### fig 2 #####

env_df <- read_csv(file.path(DATA, "sector_seasonal_envelope.csv"), show_col_types = FALSE)

env_df <- env_df %>% filter(sector != "circumpolar")

sector_levels <- c(
  "Amundsen_Bellingshausen",
  "Ross",
  "Weddell",
  "East_Antarctica",
  "King_Haakon"
)

sector_labels <- c(
  "Amundsen_Bellingshausen" = "Amundsen–Bellingshausen",
  "Ross" = "Ross Sea",
  "Weddell" = "Weddell Sea",
  "East_Antarctica" = "East Antarctica",
  "King_Haakon" = "King Haakon"
)

env_df$sector <- factor(env_df$sector, levels = sector_levels)

p_env <- ggplot(env_df, aes(DOY)) +
  geom_ribbon(aes(ymin = p10, ymax = p90),
              fill = "steelblue", alpha = 0.35) +
  geom_line(aes(y = IAC), linewidth = 0.8, color = "black") +
  facet_wrap(~sector, scales = "free_y", ncol = 3,
             labeller = as_labeller(sector_labels)) +
  scale_x_continuous(
    breaks = c(1, 91, 182, 274, 365),
    labels = c("Jan", "Apr", "Jul", "Oct", "Dec")
  ) +
  labs(
    title = "Sectoral seasonal structure and interannual variability",
    x = "Month",
    y = expression("Sea ice extent (million km"^2*")")
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold")
  )

print(p_env)

ggsave(
  file.path(FIGS, "fig02_seasonal_envelope.png"),
  p_env,
  width = 11,
  height = 7,
  dpi = 300
)