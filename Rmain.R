
library(cmdstanr)    
library(posterior)   
library(bayesplot)  
set.seed(42)

T        <- 250
sigma_x  <- 6  
sigma_y  <- 8  

mu       <- numeric(T)
y        <- numeric(T)

mu[1]    <- 50
y[1]     <- rnorm(1, mu[1], sigma_y)

for (t in 2:T) {
  mu[t] <- mu[t-1] + rnorm(1, 0, sigma_x)          
  y[t]  <- rnorm(1, mu[t], sigma_y)                
}

plot(as.ts(mu), type = "l", col = "blue", lwd = 2,
     ylab = "y", xlab = "t", main = "Simulated data")

mu <- mu[1:220]
y <- y[1:220]

summary(mu)

T <- 220

# introduce 15 % missing at random (OPTIONAL)
# miss_idx        <- sample(T, size = round(0.15 * T))
# y <- round(mu + rnorm(T, 0, 2), 0)
# y_obs           <- y
# y_obs[miss_idx] <- NA

y_obs <- round(y,0)

stan_data <- list(
  T    = T,
  y_raw = y_obs
)


mod <- cmdstan_model("StanModels/SimpleKalman.stan")

fit <- mod$sample(
  data   = stan_data,
  seed   = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup  = 1000,
  iter_sampling = 1000,
  refresh = 200
)

# ─── 4.  Quick look at results ───────────────────────────────────────────────
print(fit$summary(c("sigma_proc", "sigma_meas"), "median", "q5", "q95"))

# trace-plot for the two variance parameters
bayesplot::mcmc_trace(fit$draws(c("sigma_proc", "sigma_meas")))
