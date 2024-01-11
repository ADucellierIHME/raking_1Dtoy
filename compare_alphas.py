"""
Python scripts to run experiment where all the random variables
have the same mean but different standard deviations.
We test a general distance function
using different values of the parameter alpha.
"""

import numpy as np
import pandas as pd

from analyze_experiment import compute_MAPEs, gather_MAPE, plot_MAPE
from generate_data import generate_data
from raking_methods import raking_general_distance

def single_simulation(mu_i, sigma_i, q_i, mu, L, alphas):
    """
    Function to make a single simulation
    for the len(alphas) raking distances.
    Input:
      - mu_i: 1D Numpy array, means of the RV
      - sigma_i: 1D Numpy array, standard deviations of the RV
      - q_i: 1D Numpy array, weights for the distance
      - mu: scalar, the known value of the mean of the sum
      - L: character, distribution to be used
           (lognomal, ...)
      - alphas: list, values of the distance parameter
    Output:
      - mu_tilde_i: 2D Numpy array, estimated means
                    (one column per value of alpha)
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array.'
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert (len(mu_i) == len(sigma_i)), \
        'Means and standard deviations arrays should have the same size.'
    assert (len(mu_i) == len(q_i)), \
        'Means and weights arrays should have the same size.'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented.'

    x_i = generate_data(mu_i, sigma_i, L)
    mu_tilde_i = np.zeros((len(x_i), len(alphas)))
    for index, alpha in enumerate(alphas):
        mu_tilde_i[:, index] = raking_general_distance(x_i, q_i, alpha, mu)
    return mu_tilde_i

def run_simulations(mu_i, sigma_i, q_i, mu, L, N, alphas):
    """
    Function to run N simulations for the len(alphas) raking distances.
    Input:
      - mu_i: 1D Numpy array, means of the RV
      - sigma_i: 1D Numpy array, standard deviations of the RV
      - q_i: 1D Numpy array, weights for the distance
      - mu: scalar, the known value of the mean of the sum
      - L: character, distribution to be used
           (lognomal, ...)
      - N: number of simulations
      - alphas: list, values of the distance parameter
    Output:
      - mu_tilde_i: 3D Numpy array, estimated means
                    (dim1: number of random variables
                     dim2: number of values for alpha
                     dim3: number of simulations)
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array'
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array'
    assert (len(mu_i) == len(sigma_i)), \
        'Means and standard deviations arrays should have the same size'
    assert (len(mu_i) == len(sigma_i)), \
        'Means and weights arrays should have the same size'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented'

    mu_tilde_i = np.zeros((len(mu_i), len(alphas), N))
    for i in range(0, N):
        mu_tilde_i[:, :, i] = single_simulation(mu_i, sigma_i, q_i, mu, L, alphas)
    return mu_tilde_i

# Set seed for reproducibility
np.random.seed(0)

# Number of variables
#n = 20

# Generating means uniformly between 4 and 6
#means = np.random.uniform(0.6, 0.8, n)

# Generating multipliers between 0.2 and 0.9
#multipliers = np.linspace(0.1, 0.8, n)

# Generating one set of observations as means multiplied by the multipliers
#observations = means * multipliers

# Raking to the true sum of means (no randomness)
#sum_means = np.sum(means)
#raked_observations = observations * (sum_means / np.sum(observations))

# Calculate weights w_i = 1/multiplier
#weights = (0.1+multipliers)

# Choose values for mu, mu_i, sigma2_i, q_i
mu = 1.0
mu_i = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
sigma_i = np.array([0.000390625, 0.00078125, 0.0015625, 0.003125, \
                    0.00625, 0.0125, 0.025, 0.05, 0.1])
q_i = np.ones(9)

# Choose distribution
L = 'lognormal'

# Choose number of experiments
N = 500

# Choose values for alpha
alphas = [1, 0, -0.5, -1, -2]

# Define names of raking methods
names = ['alpha = 1',
         'alpha = 0',
         'alpha = -1/2',
         'alpha = -1',
         'alpha = -2']

# To test the methods with the different values of alpha
# Run simulations
mu_tilde_i = run_simulations(mu_i, sigma_i, q_i, mu, L, N, alphas)

# Compute MAPES
(error, error_mean) = compute_MAPEs(mu_i, mu_tilde_i)

# Create pandas dataframe to store and plot the results
df = gather_MAPE(mu_i, sigma_i, error, names)

# Plot MAPE
plot_MAPE(df, 'alphas')

