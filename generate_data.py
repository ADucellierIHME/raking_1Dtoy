"""
Module to generate synthetic data for all the experiments.
"""

import numpy as np

from math import exp, log, sqrt

def generate_data(mu_i, sigma_i, L):
    """
    Function to generate a sample dataset.
    For each value in (mu_i, sigma_i), we generate a
    single realization from the pdf L(mu_i , sigma_i).
    Input:
      - mu_i: 1D Numpy array, means of the RV
      - sigma_i: 1D Numpy array, stanard devations of the RV
      - L: character, distribution to be used
           (lognomal, ...)
    Output:
      - x_i: 1D Numpy array, observed values
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array.'
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array.'
    assert (len(mu_i) == len(sigma_i)), \
        'Means and standard deviations arrays should have the same size.'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented.'

    n = len(mu_i)
    x_i = np.zeros(n)
    for index, (mu, sigma) in enumerate(zip(mu_i, sigma_i)):
        if L == 'lognormal':
            v = log(1 + sigma**2 / mu**2)
            m = log(mu) - v / 2
            z = np.random.normal(0, 1, 1)[0]
            x_i[index] = exp(m + sqrt(v) * z)
    return x_i

