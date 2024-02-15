"""
This module implements the different raking methods in 1D.
"""

import numpy as np

def raking_chi2_distance(x_i, q_i, v_i, mu):
    """
    Raking using the chi2 distance (mu - x)^2 / 2x.
    Input:
      x_i: 1D Numpy array, observed values
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    lambda_k = (np.sum(v_i * x_i) - mu) / np.sum(q_i * np.square(v_i) * x_i)
    mu_i = x_i * (1 - q_i * v_i * lambda_k)
    return mu_i

def raking_entropic_distance(x_i, q_i, v_i, mu, max_iter=500):
    """
    Raking using the entropic distance mu log(mu/x) + x - mu.
    Input:
      x_i: 1D Numpy array, observed values
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
      max_iter: integer, maximum number of iterations for Newton's method
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    epsilon = 1
    lambda_k = 0.0
    iter_eps = 0
    while (epsilon > 1.0e-5) & (iter_eps < max_iter):
        f1 = mu - np.sum(v_i * x_i * np.exp(- q_i * v_i * lambda_k))
        f2 = np.sum(q_i * np.square(v_i) * x_i * np.exp(- q_i * v_i * lambda_k))
        lambda_k = lambda_k - f1 / f2
        epsilon = abs(f1 / f2)
        iter_eps = iter_eps + 1
    mu_i = x_i * np.exp(- q_i * v_i * lambda_k)
    return mu_i
    
def raking_inverse_entropic_distance(x_i, q_i, v_i, mu, max_iter=500):
    """
    Raking using the inverse entropic distance x log(x/mu) + mu - x.
    Input:
      x_i: 1D Numpy array, observed values
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
      max_iter: integer, maximum number of iterations for Newton's method
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    epsilon = 1
    lambda_k = 0.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        f1 = mu - np.sum(v_i * x_i /(1 + q_i * v_i * lambda_k))
        f2 = np.sum(q_i * np.square(v_i) * x_i / np.square(1 + q_i * v_i * lambda_k))
        lambda_k = lambda_k - f1 / f2
        epsilon = abs(f1 / f2)
        iter_eps = iter_eps + 1
    mu_i = x_i / (1 + q_i * v_i * lambda_k)
    return mu_i

def raking_inverse_chi2_distance(x_i, q_i, v_i, mu, max_iter=500):
    """
    Raking using the inverse chi2 distance (x - mu)^2 / 2mu.
    Input:
      x_i: 1D Numpy array, observed values
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
      max_iter: integer, maximum number of iterations for Newton's method
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    epsilon = 1
    lambda_k = 0.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        f1 = mu - np.sum(v_i * x_i /np.sqrt(1 + 2 *q_i * v_i * lambda_k))
        f2 = np.sum(q_i * np.square(v_i) * x_i / np.power(1 + 2 * q_i * v_i * lambda_k, 1.5))
        gamma = 1.0
        iter_gam = 0
        while (np.any(1 + 2 * q_i * vi * (lambda_k - gamma * f1 / f2) <= 0.0)) & \
            (iter_gam < max_iter):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
        lambda_k = lambda_k - gamma * f1 / f2
        epsilon = gamma * abs(f1 / f2)
        iter_eps = iter_eps + 1
    mu_i = x_i / np.sqrt(1 + 2 * q_i * v_i * lambda_k)
    return mu_i

def raking_general_distance(x_i, q_i, v_i, alpha, mu, max_iter=500):
    """
    Input:
      x_i: 1D Numpy array, observed values
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
      max_iter: integer, maximum number of iterations for Newton's method
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if alpha == 1:
        mu_i = raking_chi2_distance(x_i, q_i, v_i, mu)
    else:
        epsilon = 1
        lambda_k = 0.0
        iter_eps = 0
        while (epsilon > 1.0e-10) & (iter_eps < max_iter):
            if alpha == 0:
                f1 = mu - np.sum(v_i * x_i * np.exp(- q_i * v_i * lambda_k))
                f2 = np.sum(q_i * np.square(v_i) * x_i * np.exp(- q_i * v_i * lambda_k))
            else:
                f1 = mu - np.sum(v_i * x_i * np.power(1 - alpha * q_i * v_i * lambda_k, 1.0 / alpha))
                f2 = np.sum(q_i * np.square(v_i) * x_i * np.power(1 - alpha * q_i * v_i * lambda_k, 1.0 / alpha - 1.0))
            if (alpha > 0.5) or (alpha < -1.0):
                gamma = 1.0
                iter_gam = 0
                while (np.any(1 - alpha * q_i * v_i * (lambda_k - gamma * f1 / f2) <= 0.0)) & \
                      (iter_gam < max_iter):
                    gamma = gamma / 2.0
                    iter_gam = iter_gam + 1
            else:
                gamma = 1.0
            lambda_k = lambda_k - gamma * f0 / f1
            epsilon = gamma * abs(f1 / f2)
            iter_eps = iter_eps + 1
    if alpha == 0:
        mu_i = x_i * np.exp(- q_i * v_i * lambda_k)
    else:
        mu_i = x_i * np.power(1 - alpha * q_i * v_i * lambda_k, 1.0 / alpha)
    return mu_i

def raking_l2_distance(x_i, q_i, v_i, mu):
    """
    Raking using the l2 distance (mu - x)^2 / 2.
    Input:
      x_i: 1D Numpy array, observed values
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    lambda_k = (np.sum(v_i * x_i) - mu) / np.sum(q_i * np.square(v_i))
    mu_i = x_i - q_i * v_i * lambda_k
    return mu_i

def raking_logit(x_i, l_i, h_i, q_i, v_i, mu, max_iter=500):
    """
    Logit raking ensuring that l_i < mu_i < h_i.
    Input:
      x_i: 1D Numpy array, observed values
      l_i: 1D Numpy array, lower bound for the observations
      h_i: 1D Numpy array, upper bound for the observations
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
      max_iter: integer, maximum number of iterations for Newton's method
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(l_i, np.ndarray), \
        'Lower bounds should be a Numpy array.'
    assert isinstance(h_i, np.ndarray), \
        'Upper bounds should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(l_i), \
        'Observations and lower bounds arrays should have the same size.'
    assert len(x_i) == len(h_i), \
        'Observations and upper bounds arrays should have the same size.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    epsilon = 1
    lambda_k = 0.0
    iter_eps = 0
    while (epsilon > 1.0e-5) & (iter_eps < max_iter):
        f1 = mu - np.sum(v_i * (l_i * (h_i - x_i) + \
            h_i * (x_i - l_i) * np.exp(- q_i * v_i * lambda_k)) / \
            ((h_i - x_i) + (x_i - l_i) * np.exp(- q_i * v_i * lambda_k)))
        f2 = np.sum(q_i * np.square(v_i) * ((x - l) * (h - x) * (h - l) * \
            np.exp(- q_i * v_i * lambda_k)) / \
            np.square((h - x) + (x - l) * np.exp(- q_i * v_i * lambda_k)))  
        lambda_k = lambda_k - f1 / f2
        epsilon = abs(f1 / f2)
        iter_eps = iter_eps + 1
    mu_i = v_i * (l_i * (h_i - x_i) + \
        h_i * (x_i - l_i) * np.exp(- q_i * v_i * lambda_k)) / \
        ((h_i - x_i) + (x_i - l_i) * np.exp(- q_i * v_i * lambda_k))
    return mu_i

