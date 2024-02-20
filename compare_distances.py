"""
Python scripts to run experiment where all the random variables
have the same mean but different standard deviations.
We test several distances for the objective function:
  - chi-square distance
  - entropic distance
  - inverse entropic distance
  - inverse chi square distance
  - l2 distance
  - logit raking
"""

import numpy as np
import pandas as pd

from analyze_experiment import compute_MAPEs, gather_MAPE, plot_MAPE
from generate_data import generate_data
from raking_methods_1D import raking_chi2_distance
from raking_methods_1D import raking_entropic_distance
from raking_methods_1D import raking_inverse_entropic_distance
from raking_methods_1D import raking_inverse_chi2_distance
from raking_methods_1D import raking_l2_distance
from raking_methods_1D import raking_logit

def single_simulation(mu_i, sigma_i, mu, L):
    """
    Function to make a single simulation
    and compute the mu_tilde_i with different distances.
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
    assert (len(mu_i) == len(sigma_i)), \
        'Means and standard deviations arrays should have the same size.'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented.'

    x_i = generate_data(mu_i, sigma_i, L)
    q_i = np.ones(len(x_i))
    v_i = np.ones(len(x_i))
    l_i = 0.5 * x_i
    h_i = 2 * x_i
    mu_tilde_i = np.zeros((len(x_i), 6))
    mu_tilde_i[:, 0] = raking_chi2_distance(x_i, q_i, v_i, mu)
    mu_tilde_i[:, 1] = raking_entropic_distance(x_i, q_i, v_i, mu)
    mu_tilde_i[:, 2] = raking_inverse_entropic_distance(x_i, q_i, v_i, mu)
    mu_tilde_i[:, 3] = raking_inverse_chi2_distance(x_i, q_i, v_i, mu)
    mu_tilde_i[:, 4] = raking_l2_distance(x_i, q_i, v_i, mu)
    mu_tilde_i[:, 5] = raking_logit(x_i, l_i, h_i, q_i, v_i, mu)

    return mu_tilde_i

def run_simulations(mu_i, sigma_i, mu, L, N):
    """
    Function to run N simulations and compute the MAPE
    for the 6 distances.
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

    mu_tilde_i = np.zeros((len(mu_i), 6, N))
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
names = ['chi2_distance', \
         'entropic_distance', \
         'inverse_entropic_distance', \
         'inverse_chi2_distance', \
         'l2_distance', \
         'logit']

# Run simulations
mu_tilde_i = run_simulations(mu_i, sigma_i, mu, L, N)

# Compute MAPES
(error, error_mean) = compute_MAPEs(mu_i, mu_tilde_i)

# Create pandas dataframe to store and plot the results
df = gather_MAPE(mu_i, sigma_i, error, names)

# Plot MAPE
plot_MAPE(df, 'distances')

