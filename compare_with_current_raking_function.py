"""
This Python script is to compare the current raking function from
the LSAE Engineering team with our own raking code
"""

import numpy as np
import pandas as pd

from raking_uncertainty import raking_without_sigma

# Read dataset
df = pd.read_excel('../data/2D_raking_example.xlsx', nrows=16)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Get raked values by cause
df_cause = pd.read_csv('../data/raked_cause.csv')

# Get raked values by race
df_race = pd.read_csv('../data/raked_race.csv')

# Mutiply by population to get total number of cases
df['total_value'] = df['value'] * df['pop']
df['total_parent_value'] = df['parent_value'] * df['pop']
total_pop = np.sum(df['pop'].unique())
df['total_mcnty_value'] = df['mcnty_value'] * total_pop

# For each race, rake by cause
races = df['race'].unique().tolist()
df_raked = []
for race in races:
    df_sub = df.loc[df['race'] == race]
    x_i = df_sub['total_value'].to_numpy()
    if len(df_sub['total_parent_value'].unique()) != 1:
        print('The margin should be the same for all causes.')
    mu = df_sub['total_parent_value'].iloc[0]
    result = raking_without_sigma(x_i, mu)
    df_sub['raked_values'] = result
    df_raked.append(df_sub)
df_raked = pd.concat(df_raked)

# Divide by population to get prevalence
df_raked['raked_values'] = df_raked['raked_values'] / df_raked['pop']
df_raked.sort_values(by=['race', 'acause'], inplace=True)

# Check if the raked values add up to the margin
for race in races:
    df_sub = df_raked.loc[df_raked['race'] == race]
    print('race ', race, ' - difference = ', \
        abs(np.sum(df_sub['raked_values'].to_numpy()) - \
        df_sub['parent_value'].iloc[0]))

# Check if the difference between the current raking function
# and our own raking code is 0
diff = np.sum(np.abs(df_cause['value'].to_numpy() - \
                     df_raked['raked_values'].to_numpy()))
if diff < 1.0e-10:
    print('We get the same results when raking by cause.')
else:
    print('We get different results when raking by cause.')

# For each cause, rake by race
causes = df['acause'].unique().tolist()
df_raked = []
for cause in causes:
    df_sub = df.loc[df['acause'] == cause]
    x_i = df_sub['total_value'].to_numpy()
    if len(df_sub['total_mcnty_value'].unique()) != 1:
        print('The margin should be the same for all races.')
    mu = df_sub['total_mcnty_value'].iloc[0]
    result = raking_without_sigma(x_i, mu)
    df_sub['raked_values'] = result
    df_raked.append(df_sub)
df_raked = pd.concat(df_raked)

# Divide by population to get prevalence
df_raked['raked_values'] = df_raked['raked_values'] / df_raked['pop']
df_raked.sort_values(by=['race', 'acause'], inplace=True)

# Check if the raked values add up to the margin
for cause in causes:
    df_sub = df_raked.loc[df_raked['acause'] == cause]
    print('cause ', cause, ' - difference = ', \
        abs(np.sum(df_sub['raked_values'].to_numpy() * \
        df_sub['pop'].to_numpy()) - df_sub['total_mcnty_value'].iloc[0]))

# Check if the difference between the current raking function
# and our own raking code is 0
diff = np.sum(np.abs(df_race['value'].to_numpy() - \
                     df_raked['raked_values'].to_numpy()))
if diff < 1.0e-10:
    print('We get the same results when raking by race.')
else:
    print('We get different results when raking by race.')

