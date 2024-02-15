"""
This module implements different ways to introduce the uncertainty
in the raking method in 1D.
"""

import numpy as np

def raking_without_sigma(x_i, mu):
    """
    In the first version of raking, we do not use the
    values of the standard deviations sigma_i.
    Input:
      - x_i: 1D Numpy array, the observed values of the RV
      - mu: scalar, the known value of the mean of the sum
    Output:
      - mu_tilde_i: the estimated means
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'

    alpha = mu / np.sum(x_i)
    mu_tilde_i = alpha * x_i
    return mu_tilde_i

def raking_with_sigma(x_i, sigma_i, mu):
    """
    In the second version of raking, we use the
    values of the standard deviations sigma_i.
    Input:
      - x_i: 1D Numpy array, the observed values of the RV
      - sigma_i: 1D Numpy array, the known values of the standard deviations
      - mu: scalar, the known value of the mean of the sum
    Output:
      - mu_tilde_i: the estimated means
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array.'
    assert len(x_i) == len(sigma_i), \
        'Observations and standard deviations arrays should have the same size.'

    C = (mu - np.sum(x_i)) / np.sum(sigma_i * x_i)
    alpha_i = 1 + C * sigma_i
    mu_tilde_i = alpha_i * x_i
    return mu_tilde_i

def raking_with_sigma_complex(x_i, sigma_i, mu):
    """
    In the third version of raking, we use the
    values of the standard deviations sigma_i but we do not
    impose the condition on the ratios alpha_i-1/sigma_i.
    Input:
      - x_i: 1D Numpy array, the observed values of the RV
      - sigma_i: 1D numpy array, the know values of the standard deviations
      - mu: scalar, the known value of the mean of the sum
    Output:
      - mu_tilde_i: the estimated means
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array.'
    assert len(x_i) == len(sigma_i), \
        'Observations and standard deviations arrays should have the same size.'

    lambda_L = (mu - np.sum(x_i)) / np.sum(np.square(x_i) * sigma_i / 2)
    alpha_i = 1 + lambda_L * x_i * sigma_i / 2
    mu_tilde_i = alpha_i * x_i
    return mu_tilde_i

