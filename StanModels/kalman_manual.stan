// Local Level Model with Marginalized State
data {
  int<lower=1> T;          // Number of time points
  vector[T] y;             // Observations
}

parameters {
  real<lower=0> sigma_eps; // Measurement noise std (Vε)
  real<lower=0> sigma_xi;  // Level disturbance std (Wξ)
  real mu_0;               // Initial level
}

model {
  // Priors
  sigma_eps ~ cauchy(0, 2.5);
  sigma_xi ~ cauchy(0, 2.5);
  mu_0 ~ normal(0, 10);
  
  // Define state space model components
  real mu = mu_0;         // State (level)
  real P = 10.0;          // State variance (high initial uncertainty)
  
  // Kalman filter loop - computes marginal likelihood
  for (t in 1:T) {
    // Prediction step
    real mu_pred = mu;                          // μt|t-1 = μt-1
    real P_pred = P + square(sigma_xi);         // Pt|t-1 = Pt-1 + Wξ
    
    // One-step forecast
    real y_pred = mu_pred;                      // ŷt = μt|t-1
    real F = 1.0;                               // Measurement equation coefficient
    real S = P_pred + square(sigma_eps);        // Forecast variance
    
    // Likelihood contribution
    target += normal_lpdf(y[t] | y_pred, sqrt(S));
    
    // Update step
    real K = P_pred / S;                        // Kalman gain
    mu = mu_pred + K * (y[t] - y_pred);         // μt = μt|t-1 + K(yt - ŷt)
    P = P_pred * (1 - K);                       // Pt = (1 - K)Pt|t-1
  }
}

generated quantities {
  vector[T] mu;          // Filtered level
  vector[T] y_pred;      // One-step ahead predictions
  
  {
    // Rerun Kalman filter to extract states
    real mu_state = mu_0;
    real P = 10.0;
    
    for (t in 1:T) {
      // Prediction step
      real mu_pred = mu_state;
      real P_pred = P + square(sigma_xi);
      
      // Store one-step ahead prediction
      y_pred[t] = mu_pred;
      
      // Update step
      real S = P_pred + square(sigma_eps);
      real K = P_pred / S;
      
      // Update state based on observation
      mu_state = mu_pred + K * (y[t] - mu_pred);
      P = P_pred * (1 - K);
      
      // Store filtered state
      mu[t] = mu_state;
    }
  }
}