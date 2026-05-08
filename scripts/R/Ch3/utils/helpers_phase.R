shift_vec <- function(x, lag_days) {
  n <- length(x)
  lag_days <- lag_days %% n
  if (lag_days == 0) return(x)
  c(tail(x, n - lag_days), head(x, lag_days))
}