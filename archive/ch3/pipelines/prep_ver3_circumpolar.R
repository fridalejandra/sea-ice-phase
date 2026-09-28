# =============================================================================
# prep_v3_circumpolar.R
# Reformats the NSIDC Bootstrap v3 circumpolar SIE file into the same
# format as SIE_daily_sector_and_circumpolar_million_km2.csv so it can
# be read by APAC_Sector_Pipeline.R
#
# Input:  s_seaice_extent_daily_v3.0.csv  (Year, Month, Day, Extent, ...)
# Output: SIE_circumpolar_v3.csv          (Date, SIE_circumpolar)
# =============================================================================

library(dplyr)

OBS_DIR <- "/Users/fridaperez/Research/repos/sea-ice-phase/scripts/R/observations"

raw <- read.csv(
  file.path(OBS_DIR, "s_seaice_extent_daily_v3.0.csv"),
  skip        = 2,          # skip 2-row header
  header      = FALSE,
  strip.white = TRUE,
  stringsAsFactors = FALSE
)

# Keep only Year, Month, Day, Extent columns
raw <- raw[, 1:4]
colnames(raw) <- c("Year", "Month", "Day", "Extent")

# Remove any non-numeric rows (e.g. leftover header fragments)
raw <- raw %>%
  filter(!is.na(suppressWarnings(as.numeric(Year)))) %>%
  mutate(across(everything(), as.numeric))

# Build Date in MM/DD/YY format to match pipeline expectation
raw <- raw %>%
  mutate(
    Date = format(
      as.Date(paste(Year, Month, Day, sep = "-")),
      "%m/%d/%y"
    ),
    SIE_circumpolar = Extent
  ) %>%
  select(Date, SIE_circumpolar) %>%
  filter(!is.na(SIE_circumpolar) & SIE_circumpolar > 0)

out_file <- file.path(OBS_DIR, "SIE_circumpolar_v3.csv")
write.csv(raw, out_file, row.names = FALSE)
message("Written: ", out_file, "  (", nrow(raw), " rows)")
message("   Date range: ", raw$Date[1], " → ", raw$Date[nrow(raw)])