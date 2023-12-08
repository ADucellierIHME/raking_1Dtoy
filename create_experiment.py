import numpy as np

from math import exp, log, sqrt

def generate_data(mu_i, sigma2_i, L):
    """
    Function to generate a sample dataset.
    For each value in (mu_i, sigma2_i), we generate a
    single realization from the pdf L(mu_i , sigma2_i).
    Input:
      - mu_i: 1D numpy array, means of the RV
      - sigma2_i: 1D numpy array, variances of the RV
      - L: character, distribution to be used
           (lognomal, ...)
    Output:
      - x_i: 1D numpy array, observed values
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array'
    assert isinstance(sigma2_i, np.ndarray), \
        'Variances should be a Numpy array'
    assert (len(mu_i) == len(sigma2_i)), \
        'Means and variances arrays should have the same size'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented'

    n = len(mu_i)
    x_i = np.zeros(n)
    for index, (mu, sigma2) in enumerate(zip(mu_i, sigma2_i)):
        if L == 'lognormal':
            v = log(1 + sigma2 / mu**2)
            m = log(mu) - v / 2
            z = np.random.normal(0, 1, 1)[0]
            x_i[index] = exp(m + sqrt(v) * z)
    return x_i

def raking_without_sigma2(x_i, mu):
    """
    In the first version of raking, we do not use the
    values of the variances sigma2_i
    Input:
      - x_i: the observed values of the random variables
      - mu: scalar, the known value of the mean of the sum
    Output:
      - mu_tilde_i: the estimated means
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array'

    alpha = mu / np.sum(x_i)
    mu_tilde_i = alpha * x_i
    return mu_tilde_i

def raking_with_sigma2(x_i, sigma2_i, mu):
    """
    In the second version of raking, we use the
    values of the variances sigma2_i
    Input:
      - x_i: the observed values of the random variables
      - sigma2_i: 1D numpy array, the know values of the variances
      - mu: scalar, the known value of the mean of the sum
    Output:
      - mu_tilde_i: the estimated means
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array'

    C = (mu - np.sum(x_i)) / np.sum(sigma2_i * x_i)
    alpha_i = 1 + C * sigma2_i
    mu_tilde_i = alpha_i * x_i
    return mu_tilde_i

def raking_with_sigma2_complex(x_i, sigma2_i, mu):
    """
    In the third version of raking, we use the
    values of the variances sigma2_i but we do not
    impose the condition on the ratios alpha_i-1/sigma2_i
    Input:
      - x_i: the observed values of the random variables
      - sigma2_i: 1D numpy array, the know values of the variances
      - mu: scalar, the known value of the mean of the sum
    Output:
      - mu_tilde_i: the estimated means
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array'

    lambda_L = (mu - np.sum(x_i)) / np.sum(np.square(x_i) * sigma2_i / 2)
    alpha_i = 1 + lambda_L * x_i * sigma2_i / 2
    mu_tilde_i = alpha_i * x_i
    return mu_tilde_i

def single_simulation(mu_i, sigma2_i, mu, L):
    """
    Function to make a single simulation
    and compute the mu_tilde_i with 3 methods.
    Input:
      - mu_i: 1D numpy array, means of the RV
      - sigma2_i: 1D numpy array, variances of the RV
      - mu: scalar, the known value of the mean of the sum
      - L: character, distribution to be used
           (lognomal, ...)
    Output:
      - mu_tilde_i: 2D numpy array, estimated means
                    (one column per raking method)
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array'
    assert isinstance(sigma2_i, np.ndarray), \
        'Variances should be a Numpy array'
    assert (len(mu_i) == len(sigma2_i)), \
        'Means and variances arrays should have the same size'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented'

    x_i = generate_data(mu_i, sigma2_i, L)
    mu_tilde_i = np.zeros((len(x_i), 3))
    mu_tilde_i[:, 0] = raking_without_sigma2(x_i, mu)
    mu_tilde_i[:, 1] = raking_with_sigma2(x_i, sigma2_i, mu)
    mu_tilde_i[:, 2] = raking_with_sigma2_complex(x_i, sigma2_i, mu)
    return mu_tilde_i

def run_simulations(mu_i, sigma2_i, mu, L, N):
    """
    Function to run N simulations and compute the MAPE
    for the 3 raking methods.
    Input:
      - mu_i: 1D numpy array, means of the RV
      - sigma2_i: 1D numpy array, variances of the RV
      - mu: scalar, the known value of the mean of the sum
      - L: character, distribution to be used
           (lognomal, ...)
      - N: number of simulations
    Output:
      - mu_tilde_i: 3D numpy array, estimated means
                    (dim1: number of random variables
                     dim2: number of raking methods
                     dim3: number of simulations)
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array'
    assert isinstance(sigma2_i, np.ndarray), \
        'Variances should be a Numpy array'
    assert (len(mu_i) == len(sigma2_i)), \
        'Means and variances arrays should have the same size'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented'

    mu_tilde_i = np.zeros((len(mu_i), 3, N))
    for i in range(0, N):
        mu_tilde_i[:, :, i] = single_simulation(mu_i, sigma2_i, mu, L)
    return mu_tilde_i

