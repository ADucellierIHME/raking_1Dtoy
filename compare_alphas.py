"""
Python scripts to run experiment where all the random variables
have the same mean but different standard deviations.
We test a general distance function
using different values of the parameter alpha.
"""

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_experiment import compute_MAPEs, gather_MAPE, gather_iters, plot_MAPE, plot_iters
from generate_data import generate_data
from raking_methods_1D import raking_general_distance

def single_simulation(mu_i, sigma_i, v_i, q_i, mu, L, alphas):
    """
    Function to make a single simulation
    for the len(alphas) raking distances.
    Input:
      - mu_i: 1D Numpy array, means of the RV
      - sigma_i: 1D Numpy array, standard deviations of the RV
      - v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      - q_i: 1D Numpy array, weights for the distance
      - mu: scalar, the known value of the mean of the sum
      - L: character, distribution to be used
           (lognomal, ...)
      - alphas: list, values of the distance parameter
    Output:
      - mu_tilde_i: 2D Numpy array, estimated means
                    (one column per value of alpha)
      - num_iters: 1D Numpy array, number of iterations
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array.'
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear weights should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert (len(mu_i) == len(sigma_i)), \
        'Means and standard deviations arrays should have the same size.'
    assert len(mu_i) == len(v_i), \
        'Means and linear weights arrays should have the same size.'
    assert (len(mu_i) == len(q_i)), \
        'Means and weights arrays should have the same size.'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented.'

    x_i = generate_data(mu_i, sigma_i, L)
    mu_tilde_i = np.zeros((len(x_i), len(alphas)))
    num_iters = np.zeros(len(alphas))
    for index, alpha in enumerate(alphas):
        (mu_tilde_i[:, index], lambda_k, num_iters[index]) = raking_general_distance(x_i, v_i, q_i, alpha, mu, 1.0, 500, True, True)
    return (mu_tilde_i, num_iters)

def run_simulations(mu_i, sigma_i, v_i, q_i, mu, L, N, alphas):
    """
    Function to run N simulations for the len(alphas) raking distances.
    Input:
      - mu_i: 1D Numpy array, means of the RV
      - sigma_i: 1D Numpy array, standard deviations of the RV
      - v_i: 1D Numpy array, weights for the linear constraint (usually 1)
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
      - num_iters: 2D Numpy array, number of iterations
                   (dim1: number of values for alpha
                    dim2: number of simulations)
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array'
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array'
    assert isinstance(v_i, np.ndarray), \
        'Linear weights should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array'
    assert (len(mu_i) == len(sigma_i)), \
        'Means and standard deviations arrays should have the same size'
    assert len(mu_i) == len(v_i), \
        'Means and linear weights arrays should have the same size.'
    assert (len(mu_i) == len(sigma_i)), \
        'Means and weights arrays should have the same size'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented'

    mu_tilde_i = np.zeros((len(mu_i), len(alphas), N))
    num_iters = np.zeros((len(alphas), N))
    for i in range(0, N):
        (mu_tilde_i[:, :, i], num_iters[:, i]) = single_simulation(mu_i, sigma_i, v_i, q_i, mu, L, alphas)
    return (mu_tilde_i, num_iters)

def plot_MAPE_mean(df_MAPE, filename):
    """
    Function to plot the MAPE results.
    Input:
      df_MAPE: pandas DataFrame with columns
               [variable, true_mean, standard_deviation, method, MAPE]
    Output: None
    """
    chart1 = alt.Chart(df_MAPE).mark_bar().encode(
        x=alt.X('method:O', axis=alt.Axis(title='Method')),
        y=alt.Y('MAPE:Q', axis=alt.Axis(title='Mean Average Percentage Error')),
        color=alt.Color('method:N', legend=alt.Legend(title='Raking method')),
        column=alt.Column('variable:O', header=alt.Header(title='Variable', titleFontSize=16))
    ).configure_axis(
        labelFontSize=16,
        titleFontSize=16
    ).configure_legend(
        labelFontSize=16,
        titleFontSize=16,
    ).configure_header(
        labelFontSize=16,
        titleFontSize=16
    )
    chart1.save('MAPE_variable_' + filename + '.html')

    chart2 = alt.Chart(df_MAPE).mark_line().encode(
        x=alt.X('true_mean:Q', axis=alt.Axis(title='Mean')),
        y=alt.Y('MAPE:Q', axis=alt.Axis(title='Mean Average Percentage Error')),
        color=alt.Color('method:N', legend=alt.Legend(title='Raking method'))
    ).configure_axis(
        labelFontSize=16,
        titleFontSize=16
    ).configure_legend(
        labelFontSize=16,
        titleFontSize=16,
    )
    chart2.save('MAPE_mean_' + filename + '.html')

# Set seed for reproducibility
np.random.seed(0)

# First simulation: Same means, different standard deviations
# Weight is equal to 1 + 50 sigma
# Choose values for mu, mu_i, sigma2_i, v_i, q_i
mu = 1.0
mu_i = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
sigma_i = np.array([0.000390625, 0.00078125, 0.0015625, 0.003125, \
                    0.00625, 0.0125, 0.025, 0.05, 0.1])
v_i = np.ones(9)
q_i = 1 + 50 * sigma_i

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

# Run simulations
(mu_tilde_i, num_iters) = run_simulations(mu_i, sigma_i, v_i, q_i, mu, L, N, alphas)

# Compute MAPES
(error, error_mean) = compute_MAPEs(mu_i, mu_tilde_i)

# Create pandas dataframe to store and plot the results
df = gather_MAPE(mu_i, sigma_i, error, names)
df_iters = gather_iters(num_iters, names)

# Plot MAPE
plot_MAPE(df, 'alphas_std')
plot_iters(df_iters, 'alphas_std')

# Second simulation: Different means, same standard deviations
# Weight is equal to mu
# Choose values for mu, mu_i, sigma2_i, v_i, q_i
mu = 5.0
mu_i = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
sigma_i = 0.025 * np.ones(9)
v_i = np.ones(9)
q_i = mu_i

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

# Run simulations
(mu_tilde_i, num_iters) = run_simulations(mu_i, sigma_i, v_i, q_i, mu, L, N, alphas)

# Compute MAPES
(error, error_mean) = compute_MAPEs(mu_i, mu_tilde_i)

# Create pandas dataframe to store and plot the results
df = gather_MAPE(mu_i, sigma_i, error, names)
df_iters = gather_iters(num_iters, names)

# Plot MAPE
plot_MAPE_mean(df, 'alphas_mean')
plot_iters(df_iters, 'alphas_mean')

# Third simulation: Generate voluntary wrong data
# and see how much raking decreases the error
# Number of variables
n = 20

# Generating means uniformly between 0.6 and 0.8
mu_i = np.random.uniform(0.6, 0.8, n)

# Generating multipliers between 0.1 and 0.8
multipliers = np.linspace(0.1, 0.8, n)

# Generating one set of observations as means multiplied by the multipliers
x_i = mu_i * multipliers

# Calculate weights
q_i = 1.0 / (0.1 + multipliers)

# Rake to the true sum of means
mu = np.sum(mu_i)
v_i = np.ones(n)

# Choose values for alpha
alphas = [1, 0, -0.5, -1, -2]

mu_tilde_i = np.zeros((n, len(alphas)))
for index, alpha in enumerate(alphas):
    (mu_tilde_i[:, index], lambda_k) = raking_general_distance(x_i, v_i, q_i, alpha, mu, 1.0, 500, False, True)

# Plotting the true means, original observations, and raked observations
plt.figure(figsize=(12, 6))

# True means in black
plt.scatter(np.arange(n), mu_i, color='black', marker='o', label='True Means')

# Original observations in red
plt.scatter(np.arange(n), x_i, color='black', marker='x', label='Original Observations')

# Raked observations
plt.scatter(np.arange(n), mu_tilde_i[:, 0], color='blue', marker='x', label='alpha = 1')
plt.scatter(np.arange(n), mu_tilde_i[:, 1], color='green', marker='x', label='alpha = 0')
plt.scatter(np.arange(n), mu_tilde_i[:, 2], color='yellow', marker='x', label='alpha = -1/2')
plt.scatter(np.arange(n), mu_tilde_i[:, 3], color='orange', marker='x', label='alpha = -1')
plt.scatter(np.arange(n), mu_tilde_i[:, 4], color='red', marker='x', label='alpha = -2')

plt.title('Comparison of True Means, Original Observations, and Raked Observations')
plt.xlabel('Variables')
plt.ylabel('Values')
plt.xticks(np.arange(n), labels=[f"{multiplier:.2f}" for multiplier in multipliers])
plt.legend()
plt.grid(True)
plt.savefig('compare_raked_observations.png')
