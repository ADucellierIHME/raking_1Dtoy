"""
This module implements the different raking methods in 1D.
"""

import numpy as np

from scipy.optimize import root

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

def raking_chi2_distance(x_i, v_i, mu):
    """
    Raking using the chi2 distance (mu - x)^2 / 2x.
    Input:
      x_i: 1D Numpy array, observed values
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert len(x_i) == len(v_i), \
        'Observations and weights arrays should have the same size.'

    lambda_k = (np.sum(x_i) - mu) / np.sum(x_i * v_i)
    mu_i = x_i * (1 - v_i * lambda_k)
    return mu_i

def raking_entropic_distance(x_i, v_i, mu):
    """
    Raking using the entropic distance mu log(mu/x) + x - mu.
    Input:
      x_i: 1D Numpy array, observed values
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert len(x_i) == len(v_i), \
        'Observations and weights arrays should have the same size.'

    # Root finding problem to find lambda

#    def fun(lambda_k, mu, x_i, v_i):
#        return mu - np.sum(x_i * v_i * np.exp(- v_i * lambda_k))
#    def jac(lambda_k, mu, x_i, v_i):
#        return np.sum(x_i * np.square(v_i) * np.exp(- v_i * lambda_k))
#    lambda_k = root(fun=fun, jac=jac, x0=[0.1], args=(mu, x_i, v_i), method='lm')
        
    epsilon = 1
    lambda_k = -0.1
    while epsilon > 1e-5:
        f1 = mu - np.sum(x_i * v_i * np.exp(- v_i * lambda_k))
        f2 = np.sum(x_i * np.square(v_i) * np.exp(- v_i * lambda_k))
        lambda_k = lambda_k - f1 / f2
        epsilon = abs(f1 / f2)
    mu_i = x_i * np.exp(- v_i * lambda_k)
    return mu_i
    
def raking_inverse_entropic_distance(x_i, v_i, mu):
    """
    Raking using the entropic distance x log(x/mu) + mu - x
    Input:
      x_i: 1D Numpy array, observed values
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert len(x_i) == len(v_i), \
        'Observations and weights arrays should have the same size.'

    # Root finding problem to get lambda

#    def fun(lambda_k, mu, x_i, v_i):
#        return mu - np.sum(x_i * v_i /(1 + v_i * lambda_k))
#    def jac(lambda_k, mu, x_i, v_i):
#        return np.sum(x_i * np.square(v_i) / np.square(1 + v_i * lambda_k))
#    lambda_k = root(fun=fun, jac=jac, x0=[0.1], args=(mu, x_i, v_i))

    epsilon = 1
    lambda_k = -0.1
    while epsilon > 1.0e-5:
        f1 = mu - np.sum(x_i * v_i /(1 + v_i * lambda_k))
        f2 = np.sum(x_i * np.square(v_i) / np.square(1 + v_i * lambda_k))
        lambda_k = lambda_k - f1 / f2
        epsilon = abs(f1 / f2)

    mu_i = x_i / (1 + v_i * lambda_k)
    return mu_i

def raking_general_distance(x_i, v_i, q_i, alpha, mu, max_iter=500):
    """
    Raking using the general distance 1/alpha (x/alpha+1 (mu/x)^alpha+1 - mu + cte)
    Input:
      x_i: 1D Numpy array, observed values
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      q_i: 1D Numpy array, weights for the observations
      alpha: scalar, parameter for the distance
      mu: scalar, sum of raked observations
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear weights should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert len(x_i) == len(v_i), \
        'Observations and linear weights arrays should have the same size.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'

    # Root find problem to get lambda

    # Initialization with the value of lambda for alpha = 1
    lambda_k = (np.sum(v_i * x_i) - mu) / np.sum(x_i * np.square(v_i) * q_i)

    # If alpha = 1, we have a direct solution for lambda
    if alpha != 1:
        # Verify that we start with a valid value for lambda
        # This could be a problem only if alpha > 0.5 or alpha < -1
        if (alpha > 0.5) or (alpha < -1.0):
            gamma = 1.0
            iter_gam = 0
            while (np.any(1 - gamma * alpha * q_i * v_i * lambda_k <= 0.0)) & \
                  (iter_gam < max_iter):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
            lambda_k = gamma * lambda_k
        # Loop until there is no more changes in the value of lambda
        epsilon = 1
        iter_eps = 0
        while (epsilon > 1.0e-10) & (iter_eps < max_iter):
            if alpha == 0:
                f0 = mu - np.sum(v_i * x_i * np.exp(- q_i * v_i * lambda_k))
                f1 = np.sum(x_i * np.square(v_i) * q_i * np.exp(- q_i * v_i * lambda_k))
            else:
                f0 = mu - np.sum(v_i * x_i * np.power(1 - alpha * q_i * v_i * lambda_k, 1.0 / alpha))
                f1 = np.sum(x_i * np.square(v_i) * q_i * np.power(1 - alpha * q_i * v_i * lambda_k, 1.0 / alpha - 1.0))
            # Make sure that the new value of lambda is still valid
            # This could be a problem only if alpha > 0.5 or alpha < -1
            if (alpha > 0.5) or (alpha < -1.0):
                gamma = 1.0
                iter_gam = 0
                while (np.any(1 - alpha * q_i * v_i * (lambda_k - gamma * f0 / f1) <= 0.0)) & \
                      (iter_gam < max_iter):
                    gamma = gamma / 2.0
                    iter_gam = iter_gam + 1
            else:
                gamma = 1.0
            epsilon = abs(gamma * f0 / (f1 * lambda_k))
            iter_eps = iter_eps + 1
            lambda_k = lambda_k - gamma * f0 / f1

    # Compute the raked values from the value of lambda
    if alpha == 0:
        mu_i = x_i * np.exp(- q_i * v_i * lambda_k)
    else:
        mu_i = x_i * np.power(1 - alpha * q_i * v_i * lambda_k, 1.0 / alpha)
    # Check the margin
    if abs((mu - np.sum(mu_i)) / mu) > 1.0e-5:
        print('The raked values do not sum to the margin.')
    return mu_i

