import altair as alt
import numpy as np
import pandas as pd

def MAPE(x, x_tilde):
    """
    Compute the Mean Average Percentage Error (MAPE).
    Input:
      - x: scalar, true value
      - x_tilde: 1D Numpy array, estimated values
    Output:
      - error: scalar, value of MAPE
    """
    assert isinstance(x_tilde, np.ndarray), \
        'Estimated values should be a Numpy array'

    error = np.mean(np.abs((x - x_tilde) / x))
    return error

def compute_MAPEs(mu_i, mu_tilde_i):
    """
    Compute MAPE for the 3 raking methods.
    We compute each for each random variables
    and also the average MAPE over all the random variables
    Input:
      - mu_i: 1D Numpy array, the true values of the mean
      - mu_tilde_i: 3D Numpy array, the estimated values
                    (dim1: number of random variables
                     dim2: number of raking methods
                     dim3: number of simulations)
    Output:
      - error: 2D Numpy array, MAPE for each RV and each method
      - error_mean: 1D Numpy array, MAPE for each method
    """
    assert isinstance(mu_i, np.ndarray), \
        'True values should be a Numpy array'
    assert isinstance(mu_tilde_i, np.ndarray), \
        'Estimated values should be a Numpy array'
    assert (len(mu_i) == np.shape(mu_tilde_i)[0]), \
        'True values and estimated values should have the same size'

    n = len(mu_i)
    num_methods = np.shape(mu_tilde_i)[1]
    error = np.zeros((n, num_methods))
    error_mean = np.zeros(num_methods)
    for j in range(0, num_methods):
        for i in range(0, n):
            error[i, j] = MAPE(mu_i[i], mu_tilde_i[i, j, :])
        error_mean[j] = np.mean(error[:, j])
    return (error, error_mean)

def gather_MAPE(mu_i, sigma2_i, error):
    """
    Function to gather the MAPE results into a pandas dataframe
    Input:
      - mu_i: 1D numpy array, means of the RV
      - sigma2_i: 1D numpy array, variances of the RV
      - error: 2D Numpy array, MAPE for each RV and each method
    Output:
      - df: pandas DataFrame with columns
               [variable, true_mean, variance, method]
    """
    df1 = pd.DataFrame({'variable': np.arange(1, len(mu_i) + 1),
                        'true_mean': mu_i,
                        'variance': sigma2_i,
                        'method': 'without_sigma2',
                        'MAPE': error[:, 0]})
    df2 = pd.DataFrame({'variable': np.arange(1, len(mu_i) + 1),
                        'true_mean': mu_i,
                        'variance': sigma2_i,
                        'method': 'with_sigma2',
                        'MAPE': error[:, 1]})
    df3 = pd.DataFrame({'variable': np.arange(1, len(mu_i) + 1),
                        'true_mean': mu_i,
                        'variance': sigma2_i,
                        'method': 'with_sigma2_complex',
                        'MAPE': error[:, 2]})
    df = pd.concat([df1, df2, df3])
    return df

def plot_MAPE(df_MAPE):
    """
    Function to plot the MAPE results
    Input:
      df_MAPE: pandas DataFrame with columns
               [variable, true_mean, variance, method]
    Output: None
    """
    chart1 = alt.Chart(df_MAPE).mark_bar().encode(
      x='method:O',
      y='MAPE:Q',
      color='method:N',
      column='variable:O'
    )
    chart1.save('MAPE_variable.html')

    chart2 = alt.Chart(df_MAPE).mark_point().encode(
      x='variance:Q',
      y='MAPE:Q',
      color='method:N'
    )
    chart2.save('MAPE_variance.html')

