import numpy as np
import pandas as pd

from create_experiment import run_simulations
from analyze_experiment import compute_MAPEs, gather_MAPE, plot_MAPE

# Set seed for reproducibility
np.random.seed(0)

# Choose values for mu, mu_i, sigma2_i
mu = 1.0
mu_i = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
sigma2_i = np.array([0.00001, 0.00002, 0.00004, 0.00008, 0.00016, 0.00032, 0.00064, 0.00128, 0.00256])

# Choose distribution
L = 'lognormal'

# Choose number of experiments
N = 500

# Run simulations
mu_tilde_i = run_simulations(mu_i, sigma2_i, mu, L, N)

# Compute MAPES
(error, error_mean) = compute_MAPEs(mu_i, mu_tilde_i)

# Create pandas dataframe to store and plot the results
df = gather_MAPE(mu_i, sigma2_i, error)

# Plot MAPE
plot_MAPE(df)

