"""
This Python script is to test the raking methods
"""

import numpy as np
import pandas as pd

from raking_methods_1D import raking_chi2_distance
from raking_methods_1D import raking_l2_distance
from raking_methods_1D import raking_entropic_distance
from raking_methods_1D import raking_inverse_entropic_distance
from raking_methods_1D import raking_inverse_chi2_distance
from raking_methods_1D import raking_general_distance
from raking_methods_1D import raking_logit

pd.options.mode.chained_assignment = None

# Read dataset
df = pd.read_excel('../test_raking_data/2D_raking_example.xlsx', nrows=16)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Format dataset
df.rename(columns={'acause': 'cause', \
                   'parent_value': 'all_cause_value', \
                   'mcnty_value': 'all_race_value'}, \
    inplace=True)
df['value'] = df['value'] * df['pop']
df['all_cause_value'] = df['all_cause_value'] * df['pop']
total_pop = np.sum(df['pop'].unique())
df['all_race_value'] = df['all_race_value'] * total_pop
df.drop(columns=['parent', 'wt', 'level', 'pop_weight'], inplace=True)

# Test chi2 distance
# ------------------

# For each race, rake by cause using both functions
races = df['race'].unique().tolist()
df_raked_cause = []
for race in races:
    df_sub = df.loc[df['race'] == race]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_cause_value'].unique()) != 1:
        print('The margin should be the same for all causes.')
    mu = df_sub['all_cause_value'].iloc[0]
    result_direct = raking_chi2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, True)
    df_sub['value_raked_direct'] = result_direct
    result_full = raking_chi2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, False)
    df_sub['value_raked_full'] = result_full
    df_raked_cause.append(df_sub)
df_raked_cause = pd.concat(df_raked_cause)
df_raked_cause.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('Chi2 distance, rake by cause, diff = ', \
    np.sum(np.abs(df_raked_cause['value_raked_direct'].to_numpy() - df_raked_cause['value_raked_full'].to_numpy())))

# For each cause, rake by race using both functions
causes = df['cause'].unique().tolist()
df_raked_race = []
for cause in causes:
    df_sub = df.loc[df['cause'] == cause]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_race_value'].unique()) != 1:
        print('The margin should be the same for all races.')
    mu = df_sub['all_race_value'].iloc[0]
    result_direct = raking_chi2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, True)
    df_sub['value_raked_direct'] = result_direct
    result_full = raking_chi2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, False)
    df_sub['value_raked_full'] = result_full
    df_raked_race.append(df_sub)
df_raked_race = pd.concat(df_raked_race)
df_raked_race.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('Chi2 distance, rake by race, diff = ', \
    np.sum(np.abs(df_raked_race['value_raked_direct'].to_numpy() - df_raked_race['value_raked_full'].to_numpy())))

# Test l2 distance
# ----------------

# For each race, rake by cause using both functions
races = df['race'].unique().tolist()
df_raked_cause = []
for race in races:
    df_sub = df.loc[df['race'] == race]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_cause_value'].unique()) != 1:
        print('The margin should be the same for all causes.')
    mu = df_sub['all_cause_value'].iloc[0]
    result_direct = raking_l2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, True)
    df_sub['value_raked_direct'] = result_direct
    result_full = raking_l2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, False)
    df_sub['value_raked_full'] = result_full
    df_raked_cause.append(df_sub)
df_raked_cause = pd.concat(df_raked_cause)
df_raked_cause.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('L2 distance, rake by cause, diff = ', \
    np.sum(np.abs(df_raked_cause['value_raked_direct'].to_numpy() - df_raked_cause['value_raked_full'].to_numpy())))

# For each cause, rake by race using both functions
causes = df['cause'].unique().tolist()
df_raked_race = []
for cause in causes:
    df_sub = df.loc[df['cause'] == cause]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_race_value'].unique()) != 1:
        print('The margin should be the same for all races.')
    mu = df_sub['all_race_value'].iloc[0]
    result_direct = raking_l2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, True)
    df_sub['value_raked_direct'] = result_direct
    result_full = raking_l2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, False)
    df_sub['value_raked_full'] = result_full
    df_raked_race.append(df_sub)
df_raked_race = pd.concat(df_raked_race)
df_raked_race.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('L2 distance, rake by race, diff = ', \
    np.sum(np.abs(df_raked_race['value_raked_direct'].to_numpy() - df_raked_race['value_raked_full'].to_numpy())))

# Test entropic distance
# ----------------------

# For each race, rake by cause using both functions
races = df['race'].unique().tolist()
df_raked_cause = []
for race in races:
    df_sub = df.loc[df['race'] == race]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_cause_value'].unique()) != 1:
        print('The margin should be the same for all causes.')
    mu = df_sub['all_cause_value'].iloc[0]
    (result_direct, num_iter) = raking_entropic_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, True)
    print('Entropic distance - race ', race, ' - direct: ', num_iter, 'iterations')
    df_sub['value_raked_direct'] = result_direct
    (result_full, num_iter) = raking_entropic_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, False)
    print('Entropic distance - race ', race, ' - full: ', num_iter, 'iterations')
    df_sub['value_raked_full'] = result_full
    df_raked_cause.append(df_sub)
df_raked_cause = pd.concat(df_raked_cause)
df_raked_cause.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('Entropic distance, rake by cause, diff = ', \
    np.sum(np.abs(df_raked_cause['value_raked_direct'].to_numpy() - df_raked_cause['value_raked_full'].to_numpy())))

# For each cause, rake by race using both functions
causes = df['cause'].unique().tolist()
df_raked_race = []
for cause in causes:
    df_sub = df.loc[df['cause'] == cause]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_race_value'].unique()) != 1:
        print('The margin should be the same for all races.')
    mu = df_sub['all_race_value'].iloc[0]
    (result_direct, num_iter) = raking_entropic_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, True)
    print('Entropic distance - cause ', cause, ' - direct: ', num_iter, 'iterations')
    df_sub['value_raked_direct'] = result_direct
    (result_full, num_iter) = raking_entropic_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, False)
    print('Entropic distance - cause ', cause, ' - full: ', num_iter, 'iterations')
    df_sub['value_raked_full'] = result_full
    df_raked_race.append(df_sub)
df_raked_race = pd.concat(df_raked_race)
df_raked_race.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('Entropic distance, rake by race, diff = ', \
    np.sum(np.abs(df_raked_race['value_raked_direct'].to_numpy() - df_raked_race['value_raked_full'].to_numpy())))

# Test inverse entropic distance
# ------------------------------

# For each race, rake by cause using both functions
races = df['race'].unique().tolist()
df_raked_cause = []
for race in races:
    df_sub = df.loc[df['race'] == race]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_cause_value'].unique()) != 1:
        print('The margin should be the same for all causes.')
    mu = df_sub['all_cause_value'].iloc[0]
    (result_direct, num_iter) = raking_inverse_entropic_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, True)
    print('Inverse entropic distance - race ', race, ' - direct: ', num_iter, 'iterations')
    df_sub['value_raked_direct'] = result_direct
    (result_full, num_iter) = raking_inverse_entropic_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, False)
    print('Inverse entropic distance - race ', race, ' - full: ', num_iter, 'iterations')
    df_sub['value_raked_full'] = result_full
    df_raked_cause.append(df_sub)
df_raked_cause = pd.concat(df_raked_cause)
df_raked_cause.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('Inverse entropic distance, rake by cause, diff = ', \
    np.sum(np.abs(df_raked_cause['value_raked_direct'].to_numpy() - df_raked_cause['value_raked_full'].to_numpy())))

# For each cause, rake by race using both functions
causes = df['cause'].unique().tolist()
df_raked_race = []
for cause in causes:
    df_sub = df.loc[df['cause'] == cause]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_race_value'].unique()) != 1:
        print('The margin should be the same for all races.')
    mu = df_sub['all_race_value'].iloc[0]
    (result_direct, num_iter) = raking_inverse_entropic_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, True)
    print('Inverse entropic distance - cause ', cause, ' - direct: ', num_iter, 'iterations')
    df_sub['value_raked_direct'] = result_direct
    (result_full, num_iter) = raking_inverse_entropic_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, False)
    print('Inverse entropic distance - cause ', cause, ' - full: ', num_iter, 'iterations')
    df_sub['value_raked_full'] = result_full
    df_raked_race.append(df_sub)
df_raked_race = pd.concat(df_raked_race)
df_raked_race.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('Inverse entropic distance, rake by race, diff = ', \
    np.sum(np.abs(df_raked_race['value_raked_direct'].to_numpy() - df_raked_race['value_raked_full'].to_numpy())))

# Test inverse chi2 distance
# --------------------------

# For each race, rake by cause using both functions
races = df['race'].unique().tolist()
df_raked_cause = []
for race in races:
    df_sub = df.loc[df['race'] == race]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_cause_value'].unique()) != 1:
        print('The margin should be the same for all causes.')
    mu = df_sub['all_cause_value'].iloc[0]
    (result_direct, num_iter) = raking_inverse_chi2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, True)
    print('Inverse chi2 distance - race ', race, ' - direct: ', num_iter, 'iterations')
    df_sub['value_raked_direct'] = result_direct
    (result_full, num_iter) = raking_inverse_chi2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, False)
    print('Inverse chi2 distance - race ', race, ' - full: ', num_iter, 'iterations')
    df_sub['value_raked_full'] = result_full
    df_raked_cause.append(df_sub)
df_raked_cause = pd.concat(df_raked_cause)
df_raked_cause.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('Inverse chi2 distance, rake by cause, diff = ', \
    np.sum(np.abs(df_raked_cause['value_raked_direct'].to_numpy() - df_raked_cause['value_raked_full'].to_numpy())))

# For each cause, rake by race using both functions
causes = df['cause'].unique().tolist()
df_raked_race = []
for cause in causes:
    df_sub = df.loc[df['cause'] == cause]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_race_value'].unique()) != 1:
        print('The margin should be the same for all races.')
    mu = df_sub['all_race_value'].iloc[0]
    (result_direct, num_iter) = raking_inverse_chi2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, True)
    print('Inverse chi2 distance - cause ', cause, ' - direct: ', num_iter, 'iterations')
    df_sub['value_raked_direct'] = result_direct
    (result_full, num_iter) = raking_inverse_chi2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, False)
    print('Inverse chi2 distance - cause ', cause, ' - full: ', num_iter, 'iterations')
    df_sub['value_raked_full'] = result_full
    df_raked_race.append(df_sub)
df_raked_race = pd.concat(df_raked_race)
df_raked_race.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('Inverse chi2 distance, rake by race, diff = ', \
    np.sum(np.abs(df_raked_race['value_raked_direct'].to_numpy() - df_raked_race['value_raked_full'].to_numpy())))

# Test general distance
# ---------------------

# For each race, rake by cause using both functions
races = df['race'].unique().tolist()
df_raked_cause = []
for race in races:
    df_sub = df.loc[df['race'] == race]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_cause_value'].unique()) != 1:
        print('The margin should be the same for all causes.')
    mu = df_sub['all_cause_value'].iloc[0]
    (result_direct, num_iter) = raking_general_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), -0.5, mu, 500, True, True)
    print('General distance - race ', race, ' - direct: ', num_iter, 'iterations')
    df_sub['value_raked_direct'] = result_direct
    (result_full, num_iter) = raking_general_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), -0.5, mu, 500, True, False)
    print('General distance - race ', race, ' - full: ', num_iter, 'iterations')
    df_sub['value_raked_full'] = result_full
    df_raked_cause.append(df_sub)
df_raked_cause = pd.concat(df_raked_cause)
df_raked_cause.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('General distance, rake by cause, diff = ', \
    np.sum(np.abs(df_raked_cause['value_raked_direct'].to_numpy() - df_raked_cause['value_raked_full'].to_numpy())))

# For each cause, rake by race using both functions
causes = df['cause'].unique().tolist()
df_raked_race = []
for cause in causes:
    df_sub = df.loc[df['cause'] == cause]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_race_value'].unique()) != 1:
        print('The margin should be the same for all races.')
    mu = df_sub['all_race_value'].iloc[0]
    (result_direct, num_iter) = raking_general_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), -0.5, mu, 500, True, True)
    print('General distance - cause ', cause, ' - direct: ', num_iter, 'iterations')
    df_sub['value_raked_direct'] = result_direct
    (result_full, num_iter) = raking_general_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), -0.5, mu, 500, True, False)
    print('General distance - cause ', cause, ' - full: ', num_iter, 'iterations')
    df_sub['value_raked_full'] = result_full
    df_raked_race.append(df_sub)
df_raked_race = pd.concat(df_raked_race)
df_raked_race.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('General distance, rake by race, diff = ', \
    np.sum(np.abs(df_raked_race['value_raked_direct'].to_numpy() - df_raked_race['value_raked_full'].to_numpy())))

# Test logit distance
# -------------------

# For each race, rake by cause using both functions
races = df['race'].unique().tolist()
df_raked_cause = []
for race in races:
    df_sub = df.loc[df['race'] == race]
    x_i = df_sub['value'].to_numpy()
    l_i = np.zeros(len(x_i))
    h_i = df_sub['pop'].to_numpy()
    if len(df_sub['all_cause_value'].unique()) != 1:
        print('The margin should be the same for all causes.')
    mu = df_sub['all_cause_value'].iloc[0]
    (result_direct, num_iter) = raking_logit(x_i, l_i, h_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, True)
    print('Logit - race ', race, ' - direct: ', num_iter, 'iterations')
    df_sub['value_raked_direct'] = result_direct
    (result_full, num_iter) = raking_logit(x_i, l_i, h_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, False)
    print('Logit - race ', race, ' - full: ', num_iter, 'iterations')
    df_sub['value_raked_full'] = result_full
    df_raked_cause.append(df_sub)
df_raked_cause = pd.concat(df_raked_cause)
df_raked_cause.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('Logit, rake by cause, diff = ', \
    np.sum(np.abs(df_raked_cause['value_raked_direct'].to_numpy() - df_raked_cause['value_raked_full'].to_numpy())))

# For each cause, rake by race using both functions
causes = df['cause'].unique().tolist()
df_raked_race = []
for cause in causes:
    df_sub = df.loc[df['cause'] == cause]
    x_i = df_sub['value'].to_numpy()
    l_i = np.zeros(len(x_i))
    h_i = df_sub['pop'].to_numpy()
    if len(df_sub['all_race_value'].unique()) != 1:
        print('The margin should be the same for all races.')
    mu = df_sub['all_race_value'].iloc[0]
    (result_direct, num_iter) = raking_logit(x_i, l_i, h_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, True)
    print('Logit - cause ', cause, ' - direct: ', num_iter, 'iterations')
    df_sub['value_raked_direct'] = result_direct
    (result_full, num_iter) = raking_logit(x_i, l_i, h_i, np.ones(len(x_i)), np.ones(len(x_i)), mu, 500, True, False)
    print('Logit - cause ', cause, ' - full: ', num_iter, 'iterations')
    df_sub['value_raked_full'] = result_full
    df_raked_race.append(df_sub)
df_raked_race = pd.concat(df_raked_race)
df_raked_race.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('Logit, rake by race, diff = ', \
    np.sum(np.abs(df_raked_race['value_raked_direct'].to_numpy() - df_raked_race['value_raked_full'].to_numpy())))

