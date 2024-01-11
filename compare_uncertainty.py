"""
Python scripts to run experiment where all the random variables
have the same mean but different standard deviations. The objective function
depends on the standard deviation and we compare the error between
the different ways to take the uncertainty into account.
"""

import numpy as np
import pandas as pd

from analyze_experiment import compute_MAPEs, gather_MAPE, plot_MAPE
from generate_data import generate_data
from raking_methods import raking_without_sigma, raking_with_sigma, raking_with_sigma_complex

def single_simulation(mu_i, sigma_i, mu, L):
    """
    Function to make a single simulation
    and compute the mu_tilde_i with 3 methods.
    Input:
      - mu_i: 1D Numpy array, means of the RV
      - sigma_i: 1D Numpy array, standard deviations of the RV
      - mu: scalar, the known value of the mean of the sum
      - L: character, distribution to be used
           (lognomal, ...)
    Output:
      - mu_tilde_i: 2D Numpy array, estimated means
                    (one column per raking method)
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array.'
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array.'
    assert len(mu_i) == len(sigma_i), \
        'Means and standard deviations arrays should have the same size.'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented.'

    x_i = generate_data(mu_i, sigma_i, L)
    v_i = np.ones(len(x_i))
    mu_tilde_i = np.zeros((len(x_i), 3))
    mu_tilde_i[:, 0] = raking_without_sigma(x_i, mu)
    mu_tilde_i[:, 1] = raking_with_sigma(x_i, sigma_i, mu)
    mu_tilde_i[:, 2] = raking_with_sigma_complex(x_i, sigma_i, mu)
    return mu_tilde_i

def run_simulations(mu_i, sigma_i, mu, L, N):
    """
    Function to run N simulations and compute the MAPE
    for the 3 raking methods.
    Input:
      - mu_i: 1D Numpy array, means of the RV
      - sigma_i: 1D Numpy array, standard deviations of the RV
      - mu: scalar, the known value of the mean of the sum
      - L: character, distribution to be used
           (lognomal, ...)
      - N: number of simulations
    Output:
      - mu_tilde_i: 3D Numpy array, estimated means
                    (dim1: number of random variables
                     dim2: number of raking methods
                     dim3: number of simulations)
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array.'
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array.'
    assert (len(mu_i) == len(sigma_i)), \
        'Means and standard deviations arrays should have the same size.'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented.'

    mu_tilde_i = np.zeros((len(mu_i), 3, N))
    for i in range(0, N):
        mu_tilde_i[:, :, i] = single_simulation(mu_i, sigma_i, mu, L)
    return mu_tilde_i

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

# Define names of raking methods
names = ['without_sigma',
         'with_sigma',
         'with_sigma_complex']

# Run simulations
mu_tilde_i = run_simulations(mu_i, sigma_i, mu, L, N)

# Compute MAPES
(error, error_mean) = compute_MAPEs(mu_i, mu_tilde_i)

# Create pandas dataframe to store and plot the results
df = gather_MAPE(mu_i, sigma_i, error, names)

# Plot MAPE
plot_MAPE(df, 'uncertainty')

