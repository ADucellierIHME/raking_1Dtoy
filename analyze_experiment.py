"""
Module to compute and plot the MAPE for all the experiments.
"""

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
        'Estimated values should be a Numpy array.'

    error = np.mean(np.abs((x - x_tilde) / x))
    return error

def compute_MAPEs(mu_i, mu_tilde_i):
    """
    Compute MAPE for all the raking methods.
    We compute the MAPE for each random variable
    and also the average MAPE over all the random variables.
    Input:
      - mu_i: 1D Numpy array, the true values of the means
      - mu_tilde_i: 3D Numpy array, the estimated values
                    (dim1: number of random variables
                     dim2: number of raking methods
                     dim3: number of simulations)
    Output:
      - error: 2D Numpy array, MAPE for each RV and each method
      - error_mean: 1D Numpy array, MAPE for each method
    """
    assert isinstance(mu_i, np.ndarray), \
        'True values should be a Numpy array.'
    assert isinstance(mu_tilde_i, np.ndarray), \
        'Estimated values should be a Numpy array.'
    assert len(mu_i) == np.shape(mu_tilde_i)[0], \
        'True values and estimated values should have the same size.'

    n = len(mu_i)
    num_methods = np.shape(mu_tilde_i)[1]
    error = np.zeros((n, num_methods))
    error_mean = np.zeros(num_methods)
    for j in range(0, num_methods):
        for i in range(0, n):
            error[i, j] = MAPE(mu_i[i], mu_tilde_i[i, j, :])
        error_mean[j] = np.mean(error[:, j])
    return (error, error_mean)

def gather_MAPE(mu_i, sigma_i, error, names):
    """
    Function to gather the MAPE results into a pandas dataframe.
    Input:
      - mu_i: 1D Numpy array, means of the RV
      - sigma_i: 1D Numpy array, standard deviations of the RV
      - error: 2D Numpy array, MAPE for each RV and each method
      - names: list, names of the methods
    Output:
      - df: pandas DataFrame with columns
               [variable, true_mean, standard_deviation, method, MAPE]
    """
    assert isinstance(mu_i, np.ndarray), \
        'Means should be a Numpy array.'
    assert isinstance(sigma_i, np.ndarray), \
        'Standard deviations should be a Numpy array.'
    assert len(mu_i) == len(sigma_i), \
        'Means and standard deviations arrays should have the same size.'
    assert len(mu_i) == np.shape(error)[0], \
        'The number of variables must be the same as the number of rows in the erro matrix.'
    assert np.shape(error)[1] == len(names), \
        'The number of names must be the same as the number of columns in the error matrix.'

    dfs = []
    for index, name in enumerate(names):
        df = pd.DataFrame({'variable': np.arange(1, len(mu_i) + 1),
                           'true_mean': mu_i,
                           'standard_deviation': sigma_i,
                           'method': name,
                           'MAPE': error[:, index]})
        dfs.append(df)
    df = pd.concat(dfs)
    return df

def gather_iters(num_iters, names):
    """
    Function to gather the number of iterations into a pandas dataframe.
    Input:
      - num_iters: 2D Numpy array, number of iterations
      - names: list, names of the methods
    Output:
      - df: pandas DataFrame with columns iterations and method
    """
    assert isinstance(num_iters, np.ndarray), \
        'Means should be a Numpy array.'
    assert np.shape(num_iters)[0] == len(names), \
        'The number of names must be the same as the number of columns in the iterations matrix.'

    dfs = []
    for index, name in enumerate(names):
        df = pd.DataFrame({'iterations': num_iters[index, :],
                           'method': name})
        dfs.append(df)
    df = pd.concat(dfs)
    return df

def plot_MAPE(df_MAPE, filename):
    """
    Function to plot the MAPE results.
    Input:
      df_MAPE: pandas DataFrame with columns
               [variable, true_mean, standard_deviation, method, MAPE]
      filename: string, name of the output file to save the plots
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
        x=alt.X('standard_deviation:Q', axis=alt.Axis(title='Standard deviation')),
        y=alt.Y('MAPE:Q', axis=alt.Axis(title='Mean Average Percentage Error')),
        color=alt.Color('method:N', legend=alt.Legend(title='Raking method'))
    ).configure_axis(
        labelFontSize=16,
        titleFontSize=16
    ).configure_legend(
        labelFontSize=16,
        titleFontSize=16,
    )
    chart2.save('MAPE_standard_deviation_' + filename + '.html')

def plot_iters(df_iters, filename):
    """
    Function to plot the number of iterations.
    Input:
      df_MAPE: pandas DataFrame with columns
               [variable, true_mean, standard_deviation, method, MAPE]
      filename: string, name of the output file to save the plots
    Output: None
    """
    chart1 = alt.Chart(df_iters).mark_bar().encode(
        x=alt.X('iterations:Q', bin=True, axis=alt.Axis(title='Number of iterations')),
        y='count()',
        column=alt.Column('method:N', header=alt.Header(title='Raking method', titleFontSize=16))
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
    chart1.save('iterations_' + filename + '.html')

