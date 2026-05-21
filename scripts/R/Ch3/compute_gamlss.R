# compute_gamlss_expanded.R
# ==========================
# Expanded GAMLSS: mean shifts, volatility changes, structural breaks
# Monthly + annual amplitude, all sectors x indices

library(gamlss)
library(dplyr)
library(ggplot2)
library(tidyr)
if (!requireNamespace("strucchange", quietly=TRUE))
  install.packages("strucchange", repos="https://cran.r-project.org")
library(strucchange)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  <- "~/Research/repos/sea-ice-phase/scripts/R/Ch3/data"
FIG_DIR   <- "~/Research/repos/sea-ice-phase/scripts/R/Ch3/figures"
INDEX_DIR <- "~/Research/repos/sea-ice-phase/data/indices"
dir.create(FIG_DIR, showWarnings=FALSE, recursive=TRUE)

YEAR_MIN <- 1979
YEAR_MAX <- 2023

SECTOR_LABELS <- c(
  "SIE_Weddell"                 = "Weddell",
  "SIE_Amundsen_Bellingshausen" = "ABS",
  "SIE_Ross"                    = "Ross",
  "SIE_East_Antarctica"         = "East Antarctica",
  "SIE_King_Haakon"             = "King Haakon"
)
SECTOR_COLORS <- c(
  "Weddell"         = "#2196F3",
  "ABS"             = "#F44336",
  "Ross"            = "#4CAF50",
  "East Antarctica" = "#FF9800",
  "King Haakon"     = "#9C27B0"
)
SECTORS <- names(SECTOR_LABELS)
INDEX_LABELS <- c(
  "SAM_annual"    = "SAM",
  "Nino34_annual" = "Nino34",
  "ASL_annual"    = "ASL",
  "ZW3R_annual"   = "ZW3R"
)
INDICES <- names(INDEX_LABELS)

# ── Load data ─────────────────────────────────────────────────────────────────
cat("Loading data...\n")
annual  <- read.csv(file.path(DATA_DIR,"annual_params.csv"))  %>% filter(Year>=YEAR_MIN,Year<=YEAR_MAX)
monthly <- read.csv(file.path(DATA_DIR,"monthly_params.csv")) %>% filter(Year>=YEAR_MIN,Year<=YEAR_MAX)
idx     <- read.csv(file.path(DATA_DIR,"master_index_detrended.csv"))
cat(sprintf("  Annual:%d  Monthly:%d  Index:%d\n",nrow(annual),nrow(monthly),nrow(idx)))

# ── Load monthly indices ──────────────────────────────────────────────────────
cat("Loading monthly indices...\n")

read_fixed_table <- function(path, skip_header=1) {
  lines <- readLines(path)
  lines <- lines[!is.na(lines) & nchar(trimws(lines))>0]
  lines <- lines[sapply(strsplit(trimws(lines),"\\s+"), length)==13]
  read.table(text=paste(lines,collapse="\n"), header=FALSE,
             col.names=c("year","Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"),
             na.strings=c("-99.99","999.9","99.9"))
}

sam_raw  <- read_fixed_table(file.path(INDEX_DIR,"marshall_sam_monthly.txt"))
nino_raw <- read_fixed_table(file.path(INDEX_DIR,"nina34.data"))

to_long <- function(df, val_name) {
  df %>%
    filter(suppressWarnings(as.numeric(as.character(year))) %in% 1900:2100) %>%
    mutate(year=as.integer(as.character(year))) %>%
    pivot_longer(-year, names_to="month_str", values_to=val_name) %>%
    mutate(Month=match(month_str, month.abb), Year=year) %>%
    select(Year, Month, all_of(val_name)) %>%
    filter(!is.na(.data[[val_name]]), !is.na(Month))
}

sam_long  <- to_long(sam_raw,  "SAM")
nino_long <- to_long(nino_raw, "Nino34")

asl_raw       <- read.csv(file.path(INDEX_DIR,"asli_era5_v3-latest.csv"), comment.char="#")
asl_raw$Year  <- as.integer(format(as.Date(asl_raw$time),"%Y"))
asl_raw$Month <- as.integer(format(as.Date(asl_raw$time),"%m"))
asl_long      <- asl_raw %>% select(Year, Month, ASL=RelCenPres)

zw3_raw  <- read.csv(file.path(INDEX_DIR,"ZW3_raphael_monthly.csv"))
zw3_long <- zw3_raw %>% rename(Year=year, Month=month, ZW3R=ZW3_index) %>% select(Year,Month,ZW3R)

idx_monthly <- sam_long %>%
  full_join(nino_long, by=c("Year","Month")) %>%
  full_join(asl_long,  by=c("Year","Month")) %>%
  full_join(zw3_long,  by=c("Year","Month")) %>%
  filter(Year>=YEAR_MIN, Year<=YEAR_MAX)
cat(sprintf("  Monthly index: %d rows\n", nrow(idx_monthly)))

# ── Helpers ───────────────────────────────────────────────────────────────────
detrend_vec <- function(years, values) {
  mask <- !is.na(values)
  if (sum(mask)<5) return(values)
  fit <- lm(values[mask]~years[mask])
  values - (coef(fit)[1] + coef(fit)[2]*years)
}

fit_gamlss_pair <- function(y, x, yrs, lbl) {
  ok <- complete.cases(y,x,yrs)
  y <- y[ok]; x <- x[ok]; yrs <- yrs[ok]
  if (length(y)<15) return(NULL)
  yrs_s <- as.numeric(scale(yrs))
  
  m0 <- tryCatch(gamlss(y~1,           sigma.formula=~1,          family=NO(),trace=FALSE),error=function(e)NULL)
  m1 <- tryCatch(gamlss(y~x,           sigma.formula=~1,          family=NO(),trace=FALSE),error=function(e)NULL)
  m2 <- tryCatch(gamlss(y~pb(yrs_s),   sigma.formula=~1,          family=NO(),trace=FALSE),error=function(e)NULL)
  m3 <- tryCatch(gamlss(y~x+pb(yrs_s), sigma.formula=~1,          family=NO(),trace=FALSE),error=function(e)NULL)
  m4 <- tryCatch(gamlss(y~x+pb(yrs_s), sigma.formula=~pb(yrs_s),  family=NO(),trace=FALSE),error=function(e)NULL)
  
  models      <- list(m0,m1,m2,m3,m4)
  model_names <- c("null","index_only","trend_only","index+trend","index+trend+vol")
  aic_vals    <- sapply(models, function(m) if(!is.null(m)) AIC(m) else NA_real_)
  best_idx    <- which.min(aic_vals)
  best_name   <- model_names[best_idx]
  cat(sprintf("  %-50s %s (AIC=%.1f)\n", lbl, best_name, min(aic_vals,na.rm=TRUE)))
  
  # Structural break
  break_year <- NA_real_
  break_p    <- NA_real_
  tryCatch({
    if (!is.null(m1)) {
      bp <- breakpoints(residuals(m1)~1)
      if (!is.na(bp$breakpoints[1])) {
        break_year <- as.numeric(yrs[bp$breakpoints[1]])
        sc <- sctest(residuals(m1)~1, type="Chow", point=bp$breakpoints[1])
        break_p <- round(sc$p.value,4)
      }
    }
  }, error=function(e) NULL)
  
  split_yr  <- if (!is.na(break_year)) break_year else 2016
  pre_sd    <- sd(y[yrs< split_yr], na.rm=TRUE)
  post_sd   <- sd(y[yrs>=split_yr], na.rm=TRUE)
  var_ratio <- post_sd/pre_sd
  
  best_mod  <- models[[best_idx]]
  mu_fit    <- if (!is.null(best_mod)) as.numeric(fitted(best_mod,"mu"))   else rep(mean(y),length(y))
  sigma_fit <- if (!is.null(best_mod)) as.numeric(exp(fitted(best_mod,"sigma"))) else rep(sd(y),length(y))
  sigma_fit <- pmin(sigma_fit, 3*sd(y,na.rm=TRUE))
  
  # Return scalars only — no nested data frames in summary
  list(
    n               = length(y),
    aic_null        = round(aic_vals[1],2),
    aic_index       = round(aic_vals[2],2),
    aic_trend       = round(aic_vals[3],2),
    aic_index_trend = round(aic_vals[4],2),
    aic_vol         = round(aic_vals[5],2),
    best_model      = best_name,
    break_year      = break_year,
    break_p         = break_p,
    pre_sd          = round(pre_sd,4),
    post_sd         = round(post_sd,4),
    var_ratio       = round(var_ratio,3),
    # plot data stored separately
    plot_Year       = list(as.numeric(yrs)),
    plot_y          = list(as.numeric(y)),
    plot_mu         = list(as.numeric(mu_fit)),
    plot_sigma      = list(as.numeric(sigma_fit))
  )
}

# ── PART 1: Annual GAMLSS ────────────────────────────────────────────────────
cat("\n=== PART 1: Annual scalar GAMLSS ===\n")

annual_rows <- list()

for (sec_col in SECTORS) {
  sec_label <- SECTOR_LABELS[sec_col]
  sec       <- annual %>% filter(sector==sec_col) %>% arrange(Year)
  
  for (ice_var in c("amplitude_raw_anom","amplitude_anom",
                    "max_doy_raw_anom","max_doy_anom")) {
    if (!ice_var %in% names(sec)) next
    var_label <- switch(ice_var,
                        amplitude_raw_anom="amplitude_raw", amplitude_anom="amplitude_apac",
                        max_doy_raw_anom="phase_raw",       max_doy_anom="phase_apac")
    
    for (idx_col in INDICES) {
      if (!idx_col %in% names(idx)) next
      idx_sub <- idx %>% select(Year, index_val=all_of(idx_col))
      merged  <- sec %>% select(Year) %>% mutate(y=sec[[ice_var]]) %>%
        left_join(idx_sub,by="Year") %>% filter(!is.na(y),!is.na(index_val))
      if (nrow(merged)<15) next
      
      y   <- detrend_vec(merged$Year, merged$y)
      x   <- merged$index_val
      lbl <- sprintf("%s|%s|%s", sec_label, var_label, INDEX_LABELS[idx_col])
      res <- fit_gamlss_pair(y, x, merged$Year, lbl)
      if (is.null(res)) next
      
      annual_rows[[lbl]] <- data.frame(
        sector     = sec_label,
        var_type   = var_label,
        index      = INDEX_LABELS[idx_col],
        data_type  = "annual",
        n          = res$n,
        best_model = res$best_model,
        break_year = res$break_year,
        break_p    = res$break_p,
        pre_sd     = res$pre_sd,
        post_sd    = res$post_sd,
        var_ratio  = res$var_ratio,
        aic_null   = res$aic_null,
        aic_index  = res$aic_index,
        aic_trend  = res$aic_trend,
        aic_index_trend = res$aic_index_trend,
        aic_vol    = res$aic_vol,
        stringsAsFactors = FALSE
      )
    }
  }
}

annual_df <- bind_rows(annual_rows)
write.csv(annual_df, file.path(DATA_DIR,"gamlss_expanded_annual.csv"), row.names=FALSE)
cat(sprintf("\nSaved → gamlss_expanded_annual.csv (%d rows)\n", nrow(annual_df)))

# ── PART 2: Monthly GAMLSS ───────────────────────────────────────────────────
cat("\n=== PART 2: Monthly amplitude GAMLSS ===\n")

monthly_rows <- list()

for (sec_col in SECTORS) {
  sec_label   <- SECTOR_LABELS[sec_col]
  sec_monthly <- monthly %>% filter(sector==sec_col) %>% arrange(Year,Month)
  
  for (idx_name in c("SAM","Nino34","ASL","ZW3R")) {
    if (!idx_name %in% names(idx_monthly)) next
    merged <- sec_monthly %>% select(Year,Month,monthly_amp_anom) %>%
      left_join(idx_monthly %>% select(Year,Month,all_of(idx_name)), by=c("Year","Month")) %>%
      filter(!is.na(monthly_amp_anom), !is.na(.data[[idx_name]])) %>% arrange(Year,Month)
    if (nrow(merged)<30) next
    
    t_axis   <- merged$Year + merged$Month/12
    merged$y <- detrend_vec(t_axis, merged$monthly_amp_anom)
    merged$x <- detrend_vec(t_axis, merged[[idx_name]])
    lbl      <- sprintf("%s|amplitude_monthly|%s", sec_label, idx_name)
    res      <- fit_gamlss_pair(merged$y, merged$x, t_axis, lbl)
    if (is.null(res)) next
    
    monthly_rows[[lbl]] <- data.frame(
      sector=sec_label, var_type="amplitude_monthly",
      index=idx_name, data_type="monthly",
      n=res$n, best_model=res$best_model,
      break_year=res$break_year, break_p=res$break_p,
      pre_sd=res$pre_sd, post_sd=res$post_sd, var_ratio=res$var_ratio,
      aic_null=res$aic_null, aic_index=res$aic_index,
      aic_trend=res$aic_trend, aic_index_trend=res$aic_index_trend,
      aic_vol=res$aic_vol, stringsAsFactors=FALSE
    )
  }
}

monthly_df <- bind_rows(monthly_rows)
write.csv(monthly_df, file.path(DATA_DIR,"gamlss_expanded_monthly.csv"), row.names=FALSE)
cat(sprintf("\nSaved → gamlss_expanded_monthly.csv (%d rows)\n", nrow(monthly_df)))

# ── PART 3: Line figures ─────────────────────────────────────────────────────
cat("\n=== PART 3: Rebuilding plot data for figures ===\n")

# Re-run key pairs to get plot data (stored separately to avoid list-column issues)
KEY_PAIRS <- list(
  list(var="amplitude_raw",  idx="Nino34", title="Amplitude (raw) ~ Niño3.4"),
  list(var="amplitude_apac", idx="Nino34", title="Amplitude (APAC) ~ Niño3.4"),
  list(var="phase_raw",      idx="SAM",    title="Phase (raw) ~ SAM"),
  list(var="phase_apac",     idx="SAM",    title="Phase (APAC) ~ SAM")
)

for (kp in KEY_PAIRS) {
  cat(sprintf("\nFigure: %s\n", kp$title))
  plot_list <- list()
  
  for (sec_col in SECTORS) {
    sec_label <- SECTOR_LABELS[sec_col]
    sec       <- annual %>% filter(sector==sec_col) %>% arrange(Year)
    
    ice_var <- switch(kp$var,
                      amplitude_raw  = "amplitude_raw_anom",
                      amplitude_apac = "amplitude_anom",
                      phase_raw      = "max_doy_raw_anom",
                      phase_apac     = "max_doy_anom")
    if (!ice_var %in% names(sec)) next
    
    idx_col <- paste0(kp$idx, "_annual")
    if (!idx_col %in% names(idx)) next
    idx_sub <- idx %>% select(Year, index_val=all_of(idx_col))
    merged  <- sec %>% select(Year) %>% mutate(y=sec[[ice_var]]) %>%
      left_join(idx_sub,by="Year") %>% filter(!is.na(y),!is.na(index_val))
    if (nrow(merged)<15) next
    
    y   <- detrend_vec(merged$Year, merged$y)
    x   <- merged$index_val
    lbl <- sprintf("%s|%s|%s", sec_label, kp$var, kp$idx)
    res <- fit_gamlss_pair(y, x, merged$Year, lbl)
    if (is.null(res)) next
    
    # Get break year from annual_df
    bk <- annual_df %>%
      filter(sector==sec_label, var_type==kp$var, index==kp$idx) %>%
      pull(break_year)
    bk <- if (length(bk)>0 && !is.na(bk[1])) as.numeric(bk[1]) else NA_real_
    
    plot_list[[sec_label]] <- data.frame(
      Year      = as.numeric(unlist(res$plot_Year)),
      y         = as.numeric(unlist(res$plot_y)),
      mu_fit    = as.numeric(unlist(res$plot_mu)),
      sigma_fit = as.numeric(unlist(res$plot_sigma)),
      sector    = sec_label,
      break_year= bk,
      stringsAsFactors=FALSE
    )
  }
  
  if (length(plot_list)==0) next
  sub <- bind_rows(plot_list)
  sub$sector <- factor(sub$sector, levels=names(SECTOR_COLORS))
  
  # Break year lines data
  bk_df <- sub %>%
    group_by(sector) %>%
    summarise(break_year=first(break_year), .groups="drop") %>%
    filter(!is.na(break_year)) %>%
    mutate(break_year=as.numeric(break_year))
  
  p <- ggplot(sub, aes(x=Year)) +
    geom_ribbon(aes(ymin=mu_fit-sigma_fit, ymax=mu_fit+sigma_fit,
                    fill=sector), alpha=0.15) +
    geom_line(aes(y=mu_fit, color=sector), linewidth=1.2) +
    geom_point(aes(y=y, color=sector), size=1.8, alpha=0.7) +
    { if (nrow(bk_df)>0)
      geom_vline(data=bk_df,
                 aes(xintercept=break_year, color=sector),
                 linewidth=0.8, linetype="dashed", alpha=0.7)
      else list() } +
    geom_vline(xintercept=2016, color="#2C2C2A",
               linewidth=0.6, linetype="dotted", alpha=0.5) +
    scale_color_manual(values=SECTOR_COLORS) +
    scale_fill_manual( values=SECTOR_COLORS) +
    facet_wrap(~sector, ncol=2, scales="free_y") +
    labs(title=kp$title,
         subtitle="Fitted mean ± 1σ | dashed=structural break | dotted=2016",
         x="Year", y="Anomaly") +
    theme_classic(base_size=10) +
    theme(strip.text=element_text(face="bold",size=10),
          plot.title=element_text(face="bold",size=12),
          plot.subtitle=element_text(size=8,color="#5F5E5A"),
          legend.position="none",
          panel.grid.major=element_line(color="#F5F5F5"))
  
  print(p)
  
  fname <- sprintf("fig_gamlss_%s_%s.png", gsub("\\+","_",kp$var), kp$idx)
  ggsave(file.path(FIG_DIR,fname), p, width=10, height=8, dpi=300)
  cat(sprintf("  Saved → %s\n", fname))
}

# ── PART 4: Summaries ─────────────────────────────────────────────────────────
cat("\n=== Break year summary ===\n")
print(annual_df %>%
        filter(!is.na(break_year)) %>%
        select(sector,var_type,index,best_model,break_year,break_p,var_ratio) %>%
        arrange(break_year))

cat("\n=== Volatility changes (var_ratio > 1.2) ===\n")
print(annual_df %>%
        filter(!is.na(var_ratio) & var_ratio>1.2) %>%
        select(sector,var_type,index,best_model,break_year,var_ratio) %>%
        arrange(desc(var_ratio)))

cat("\n=== Monthly GAMLSS summary ===\n")
print(monthly_df %>%
        select(sector,index,best_model,break_year,var_ratio) %>%
        arrange(sector,index))

cat("\nDone.\n")