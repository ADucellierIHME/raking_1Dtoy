"""
This Python script is to test the 
"""

import numpy as np
import pandas as pd

from raking_methods_1D import raking_entropic_distance, raking_vectorized_entropic_distance
from raking_methods_1D import raking_l2_distance, raking_vectorized_l2_distance

pd.options.mode.chained_assignment = None

# Read dataset
df = pd.read_excel('../data/2D_raking_example.xlsx', nrows=16)
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

# Test entropic distance

# For each race, rake by cause using initial function
races = df['race'].unique().tolist()
df_raked = []
for race in races:
    df_sub = df.loc[df['race'] == race]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_cause_value'].unique()) != 1:
        print('The margin should be the same for all causes.')
    mu = df_sub['all_cause_value'].iloc[0]
    result = raking_entropic_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu)
    df_sub['value_raked'] = result
    df_raked.append(df_sub)
df_raked = pd.concat(df_raked)
df_raked['value_raked'] = df_raked['value_raked'] / df_raked['pop']
df_raked.sort_values(by=['race', 'cause'], inplace=True)

# For each race, rake by cause using vectorized function
df_raked_cause = raking_vectorized_entropic_distance(df, 'cause', ['race', 'mcnty', 'year', 'age', 'sex', 'sim'])
df_raked_cause['value_raked'] = df_raked_cause['value_raked'] / df_raked_cause['pop']
df_raked_cause.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('Entropic distance, rake by cause, diff = ', \
    np.sum(np.abs(df_raked['value_raked'].to_numpy() - df_raked_cause['value_raked'].to_numpy())))

# For each cause, rake by race using initial function
causes = df['cause'].unique().tolist()
df_raked = []
for cause in causes:
    df_sub = df.loc[df['cause'] == cause]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_race_value'].unique()) != 1:
        print('The margin should be the same for all races.')
    mu = df_sub['all_race_value'].iloc[0]
    result = raking_entropic_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu)
    df_sub['value_raked'] = result
    df_raked.append(df_sub)
df_raked = pd.concat(df_raked)
df_raked['value_raked'] = df_raked['value_raked'] / df_raked['pop']
df_raked.sort_values(by=['race', 'cause'], inplace=True)

# For each cause, rake by race using vectorized function
df_raked_race = raking_vectorized_entropic_distance(df, 'race', ['cause', 'mcnty', 'year', 'age', 'sex', 'sim'])
df_raked_race['value_raked'] = df_raked_race['value_raked'] / df_raked_cause['pop']
df_raked_race.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('Entropic distance, rake by race, diff = ', \
    np.sum(np.abs(df_raked['value_raked'].to_numpy() - df_raked_race['value_raked'].to_numpy())))

# Test l2 distance

# For each race, rake by cause using initial function
races = df['race'].unique().tolist()
df_raked = []
for race in races:
    df_sub = df.loc[df['race'] == race]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_cause_value'].unique()) != 1:
        print('The margin should be the same for all causes.')
    mu = df_sub['all_cause_value'].iloc[0]
    result = raking_l2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu)
    df_sub['value_raked'] = result
    df_raked.append(df_sub)
df_raked = pd.concat(df_raked)
df_raked['value_raked'] = df_raked['value_raked'] / df_raked['pop']
df_raked.sort_values(by=['race', 'cause'], inplace=True)

# For each race, rake by cause using vectorized function
df_raked_cause = raking_vectorized_l2_distance(df, 'cause', ['race', 'mcnty', 'year', 'age', 'sex', 'sim'])
df_raked_cause['value_raked'] = df_raked_cause['value_raked'] / df_raked_cause['pop']
df_raked_cause.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('L2 distance, rake by cause, diff = ', \
    np.sum(np.abs(df_raked['value_raked'].to_numpy() - df_raked_cause['value_raked'].to_numpy())))

# For each cause, rake by race using initial function
causes = df['cause'].unique().tolist()
df_raked = []
for cause in causes:
    df_sub = df.loc[df['cause'] == cause]
    x_i = df_sub['value'].to_numpy()
    if len(df_sub['all_race_value'].unique()) != 1:
        print('The margin should be the same for all races.')
    mu = df_sub['all_race_value'].iloc[0]
    result = raking_l2_distance(x_i, np.ones(len(x_i)), np.ones(len(x_i)), mu)
    df_sub['value_raked'] = result
    df_raked.append(df_sub)
df_raked = pd.concat(df_raked)
df_raked['value_raked'] = df_raked['value_raked'] / df_raked['pop']
df_raked.sort_values(by=['race', 'cause'], inplace=True)

# For each cause, rake by race using vectorized function
df_raked_race = raking_vectorized_l2_distance(df, 'race', ['cause', 'mcnty', 'year', 'age', 'sex', 'sim'])
df_raked_race['value_raked'] = df_raked_race['value_raked'] / df_raked_cause['pop']
df_raked_race.sort_values(by=['race', 'cause'], inplace=True)

# Compare values between two raking methods
print('L2 distance, rake by race, diff = ', \
    np.sum(np.abs(df_raked['value_raked'].to_numpy() - df_raked_race['value_raked'].to_numpy())))

