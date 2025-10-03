# bass_foldables.R
# ---------------------------------------------------
# Script for data prep, Bass model fitting, forecasting, and plotting
# ---------------------------------------------------

# Data preparation
years <- 2019:2024
shipments <- c(1.0, 1.9, 8.1, 14.2, 18.1, 23.7)  # 2024 derived via China share calc

df <- data.frame(years, shipments)
df$cumulative <- cumsum(df$shipments)
df$lag_cum <- c(0, head(df$cumulative, -1))

# Fit Bass model
bass_model <- nls(
  shipments ~ M * (p + q * (lag_cum / M)) * (1 - lag_cum / M),
  data = df,
  start = list(p = 0.01, q = 0.3, M = 100)
)

coef_vals <- coef(bass_model)
p <- coef_vals["p"]
q <- coef_vals["q"]
M <- coef_vals["M"]

cat("Estimated Bass parameters:\n")
cat("p =", p, "\nq =", q, "\nM =", M, "\n")

# Simulation function
simulate_bass <- function(years, p, q, M) {
  n <- length(years)
  adopters <- numeric(n)
  cum <- numeric(n)
  
  for (t in 1:n) {
    cum_prev <- ifelse(t == 1, 0, cum[t-1])
    adopters[t] <- M * (p + q * (cum_prev/M)) * (1 - cum_prev/M)
    cum[t] <- cum_prev + adopters[t]
  }
  data.frame(years = years, shipments = adopters, cumulative = cum)
}

# Forecast
forecast_years <- 2019:2025
forecast <- simulate_bass(forecast_years, p, q, M)

# Plot
plot(df$years, df$shipments, type="b", pch=16, col="purple",
     xlab="Year", ylab="Units shipped (mln)",
     main="Bass Diffusion Model: Actual vs Forecast")
lines(forecast$years, forecast$shipments, col="orange", lwd=2)
legend("topleft", legend=c("Actual / IDC data","Bass Model"),
       col=c("purple","orange"), lty=c(1,1), pch=c(16, NA))