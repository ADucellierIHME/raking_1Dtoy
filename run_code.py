import numpy as np
import pandas as pd

from create_experiment import run_simulations, run_simulations_test_distances
from analyze_experiment import compute_MAPEs, compute_MAPEs_test_distances
from analyze_experiment import gather_MAPE, gather_MAPE_test_distances
from analyze_experiment import plot_MAPE, plot_MAPE_test_distances

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

# To test the methods with the different implmentations with or without sigma
# Run simulations
#mu_tilde_i = run_simulations(mu_i, sigma_i, mu, L, N)

# Compute MAPES
#(error, error_mean) = compute_MAPEs(mu_i, mu_tilde_i)

# Create pandas dataframe to store and plot the results
#df = gather_MAPE(mu_i, sigma_i, error)

# Plot MAPE
#plot_MAPE(df)

# To test the methods with the different distances
# Run simulations
mu_tilde_i = run_simulations_test_distances(mu_i, sigma_i, mu, L, N)

# Compute MAPES
(error, error_mean) = compute_MAPEs_test_distances(mu_i, mu_tilde_i)

# Create pandas dataframe to store and plot the results
df = gather_MAPE_test_distances(mu_i, sigma_i, error)

# Plot MAPE
plot_MAPE_test_distances(df)

