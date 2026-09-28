# =============================================================================
# make_indices.R — build climate_indices.csv for chapter_analyses_E.R
#
# Downloads monthly SAM (Marshall, BAS) and Nino 3.4 anomalies (NOAA PSL),
# builds annual + seasonal means, writes Year + index columns.
#
# Season convention (standard climate practice):
#   DJF of year Y = Dec(Y-1) + Jan(Y) + Feb(Y);  MAM/JJA/SON within year Y.
#   Annual = calendar-year mean.
#
# To add another index (e.g. DMI/IOD), append a fetch_monthly() call and
# bind its seasonal columns the same way.
# =============================================================================

OUT <- path.expand(
  "~/Research/repos/sea-ice-phase/data/ch3/climate_indices.csv")

# Generic parser: keeps lines that start with a 4-digit year followed by 12
# monthly values; `na_vals` are sentinel missing codes.
fetch_monthly <- function(url, na_vals = c(-99.99, -999, -9999)) {
  txt <- readLines(url, warn = FALSE)
  rows <- grep("^\\s*(19|20)\\d{2}(\\s+-?\\d)", txt, value = TRUE)
  m <- do.call(rbind, lapply(strsplit(trimws(rows), "\\s+"), function(p) {
    v <- suppressWarnings(as.numeric(p))
    if (length(v) < 13 || is.na(v[1])) return(NULL)
    v[1:13]
  }))
  df <- as.data.frame(m)
  names(df) <- c("Year", month.abb)
  for (mm in month.abb) df[[mm]][df[[mm]] %in% na_vals] <- NA
  df
}

seasonalize <- function(mon, prefix) {
  # DJF(Y) = Dec(Y-1), Jan(Y), Feb(Y)
  dec_prev <- mon$Dec[match(mon$Year - 1, mon$Year)]
  data.frame(
    Year = mon$Year,
    ann = rowMeans(mon[month.abb], na.rm = TRUE),
    DJF = rowMeans(cbind(dec_prev, mon$Jan, mon$Feb), na.rm = TRUE),
    MAM = rowMeans(mon[c("Mar", "Apr", "May")], na.rm = TRUE),
    JJA = rowMeans(mon[c("Jun", "Jul", "Aug")], na.rm = TRUE),
    SON = rowMeans(mon[c("Sep", "Oct", "Nov")], na.rm = TRUE)
  ) |> setNames(c("Year", paste0(prefix, c("_ann", "_DJF", "_MAM",
                                           "_JJA", "_SON"))))
}

message("Downloading SAM (Marshall / BAS)...")
sam_mon <- fetch_monthly(
  "https://legacy.bas.ac.uk/met/gjma/newsam.1957.2007.txt")
message("  SAM years: ", min(sam_mon$Year), "-", max(sam_mon$Year))

message("Downloading Nino 3.4 anomalies (NOAA PSL)...")
n34_mon <- fetch_monthly(
  "https://psl.noaa.gov/data/correlation/nina34.anom.data")
message("  Nino3.4 years: ", min(n34_mon$Year), "-", max(n34_mon$Year))

idx <- merge(seasonalize(sam_mon, "SAM"), seasonalize(n34_mon, "N34"),
             by = "Year", all = TRUE)
idx <- idx[idx$Year >= 1979 & idx$Year <= 2023, ]

write.csv(idx, OUT, row.names = FALSE)
message("Wrote ", OUT, "  (", nrow(idx), " years, ",
        ncol(idx) - 1, " index columns)")
print(head(idx, 3))