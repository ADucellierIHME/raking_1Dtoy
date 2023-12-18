import numpy as np
import pandas as pd

from create_experiment import run_simulations
from analyze_experiment import compute_MAPEs, gather_MAPE, plot_MAPE

# Set seed for reproducibility
np.random.seed(0)

# Choose values for mu, mu_i, sigma2_i
mu = 1.0
mu_i = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
sigma_i = np.array([0.000390625, 0.00078125, 0.0015625, 0.003125, \
                    0.00625, 0.0125, 0.025, 0.05, 0.1])

# Choose distribution
L = 'lognormal'

# Choose number of experiments
N = 500

# Run simulations
mu_tilde_i = run_simulations(mu_i, sigma_i, mu, L, N)

# Compute MAPES
(error, error_mean) = compute_MAPEs(mu_i, mu_tilde_i)

# Create pandas dataframe to store and plot the results
df = gather_MAPE(mu_i, sigma_i, error)

# Plot MAPE
plot_MAPE(df)

