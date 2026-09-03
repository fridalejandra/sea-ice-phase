# =============================================================================
# run_rmse_scenarios.R
# Runs APAC_Sector_Pipeline.R for 3 scenarios by patching the config
# section before evaluating, without modifying the original pipeline file.
#
# Scenarios:
#   1. v3 Bootstrap,  1979-01-01 to 2018-12-31  (H&R 2020 replication)
#   2. v4 Bootstrap,  1979-01-01 to 2018-12-31  (same period, updated data)
#   3. v4 Bootstrap,  1979-01-01 to 2025-12-31  (full updated record)
# =============================================================================

BASE_DIR <- "/Users/fridaperez/Research/repos/sea-ice-phase"
OBS_DIR  <- file.path(BASE_DIR, "scripts/R/observations")
PIPELINE <- file.path(BASE_DIR, "scripts/R/Ch3/APAC_Sector_Pipeline.R")
OUT_BASE <- file.path(BASE_DIR, "scripts/R/Ch3/data/rmse_scenarios")

scenarios <- list(
  list(
    label      = "v3_1979_2018",
    input_file = file.path(OBS_DIR, "SIE_circumpolar_v3.csv"),
    date_start = "1979-01-01",
    date_end   = "2018-12-31",
    sectors    = "SIE_circumpolar"
  ),
  list(
    label      = "v4_1979_2018",
    input_file = file.path(OBS_DIR, "SIE_daily_sector_and_circumpolar_million_km2.csv"),
    date_start = "1979-01-01",
    date_end   = "2018-12-31",
    sectors    = "SIE_circumpolar"
  ),
  list(
    label      = "v4_1979_2025",
    input_file = file.path(OBS_DIR, "SIE_daily_sector_and_circumpolar_million_km2.csv"),
    date_start = "1979-01-01",
    date_end   = "2025-12-31",
    sectors    = "SIE_circumpolar"
  )
)

# Read the pipeline code once
pipeline_lines <- readLines(PIPELINE)

for (sc in scenarios) {
  
  message("\n", strrep("=", 60))
  message("Running scenario: ", sc$label)
  message(strrep("=", 60))
  
  out_dir <- file.path(OUT_BASE, sc$label)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Patch config lines
  patched <- pipeline_lines
  patched[grepl("^INPUT_FILE\\s*<-", patched)] <-
    sprintf('INPUT_FILE <- "%s"', sc$input_file)
  patched[grepl("^OUTPUT_DIR\\s*<-", patched)] <-
    sprintf('OUTPUT_DIR <- "%s"', out_dir)
  patched[grepl("^DATE_START\\s*<-", patched)] <-
    sprintf('DATE_START <- as.Date("%s")', sc$date_start)
  patched[grepl("^DATE_END\\s*<-", patched)] <-
    sprintf('DATE_END <- as.Date("%s")', sc$date_end)
  
  # Replace the entire SECTOR_COLS <- c(...) block with a single line
  start_idx <- which(grepl("^SECTOR_COLS\\s*<-\\s*c\\(", patched))
  if (length(start_idx) > 0) {
    # Find the closing ) of the c(...) block
    end_idx <- start_idx
    for (i in (start_idx + 1):length(patched)) {
      if (trimws(patched[i]) == ")") { end_idx <- i; break }
    }
    # Replace the whole block with one line, blank out the rest
    patched[start_idx] <- sprintf('SECTOR_COLS <- c("%s")', sc$sectors)
    if (end_idx > start_idx) patched[(start_idx + 1):end_idx] <- ""
  }
  
  tryCatch(
    eval(parse(text = paste(patched, collapse = "\n")), envir = new.env()),
    error = function(e) message("Scenario ", sc$label, " failed: ", e$message)
  )
  
  message("Scenario complete: ", sc$label)
  message("   Output: ", out_dir)
}

message("\n", strrep("=", 60))
message("All scenarios complete.")
message("RMSE files: ", OUT_BASE)