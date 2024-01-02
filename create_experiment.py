import numpy as np

from math import abs, exp, log, sqrt

def generate_data(mu_i, sigma_i, L):
    """
    Function to generate a sample dataset.
    For each value in (mu_i, sigma_i), we generate a
    single realization from the pdf L(mu_i , sigma_i).
    Input:
      - mu_i: 1D numpy array, means of the RV
      - sigma_i: 1D numpy array, stanard devations of the RV
      - L: character, distribution to be used
           (lognomal, ...)
    Output:
      - x_i: 1D numpy array, observed values
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array'
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array'
    assert (len(mu_i) == len(sigma_i)), \
        'Means and standard deviations arrays should have the same size'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented'

    n = len(mu_i)
    x_i = np.zeros(n)
    for index, (mu, sigma) in enumerate(zip(mu_i, sigma_i)):
        if L == 'lognormal':
            v = log(1 + sigma**2 / mu**2)
            m = log(mu) - v / 2
            z = np.random.normal(0, 1, 1)[0]
            x_i[index] = exp(m + sqrt(v) * z)
    return x_i

def raking_entropic_distance(p, v, T):
    """
    Raking using the entropic distance
    Input:
      p: 1D Numpy array, observed values
      v: 1D Numpy array, weights for the linear constraint (usually 1)
      T: scalar, total of the linear constraint
    Output:
      x: 1D Numpy array, raked values
    """
    epsilon = 1
    while epsilon > 1e-10
        f1 = T - np.sum(v * p * np.exp(- lambda_k * v))
        f2 = np.sum(np.square(v) * p * np.exp(- lambda_k * v))
        epsilon = abs(f1 / (f2 * lambda_k))
        lambda_k = lambda_k - f1 / f2
    x = p * np.exp(- lambda_k * v)
    return x
    
def raking_without_sigma(x_i, mu):
    """
    In the first version of raking, we do not use the
    values of the standard deviations sigma_i
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

def raking_with_sigma(x_i, sigma_i, mu):
    """
    In the second version of raking, we use the
    values of the standard deviations sigma_i
    Input:
      - x_i: the observed values of the random variables
      - sigma_i: 1D numpy array, the know values of the standard deviations
      - mu: scalar, the known value of the mean of the sum
    Output:
      - mu_tilde_i: the estimated means
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array'

    C = (mu - np.sum(x_i)) / np.sum(sigma_i * x_i)
    alpha_i = 1 + C * sigma_i
    mu_tilde_i = alpha_i * x_i
    return mu_tilde_i

def raking_with_sigma_complex(x_i, sigma_i, mu):
    """
    In the third version of raking, we use the
    values of the standard deviations sigma_i but we do not
    impose the condition on the ratios alpha_i-1/sigma_i
    Input:
      - x_i: the observed values of the random variables
      - sigma_i: 1D numpy array, the know values of the standard deviations
      - mu: scalar, the known value of the mean of the sum
    Output:
      - mu_tilde_i: the estimated means
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array'

    lambda_L = (mu - np.sum(x_i)) / np.sum(np.square(x_i) * sigma_i / 2)
    alpha_i = 1 + lambda_L * x_i * sigma_i / 2
    mu_tilde_i = alpha_i * x_i
    return mu_tilde_i

def single_simulation(mu_i, sigma_i, mu, L):
    """
    Function to make a single simulation
    and compute the mu_tilde_i with 3 methods.
    Input:
      - mu_i: 1D numpy array, means of the RV
      - sigma_i: 1D numpy array, standard deviations of the RV
      - mu: scalar, the known value of the mean of the sum
      - L: character, distribution to be used
           (lognomal, ...)
    Output:
      - mu_tilde_i: 2D numpy array, estimated means
                    (one column per raking method)
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array'
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array'
    assert (len(mu_i) == len(sigma_i)), \
        'Means and standard deviations arrays should have the same size'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented'

    x_i = generate_data(mu_i, sigma_i, L)
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
      - mu_i: 1D numpy array, means of the RV
      - sigma_i: 1D numpy array, standard deviations of the RV
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
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array'
    assert (len(mu_i) == len(sigma_i)), \
        'Means and standard deviations arrays should have the same size'
    assert L in ['lognormal'], \
        'Only lognormal distribution is implemented'

    mu_tilde_i = np.zeros((len(mu_i), 3, N))
    for i in range(0, N):
        mu_tilde_i[:, :, i] = single_simulation(mu_i, sigma_i, mu, L)
    return mu_tilde_i

