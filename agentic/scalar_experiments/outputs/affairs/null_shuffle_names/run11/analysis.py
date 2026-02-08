import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'affairs.csv'
df = pd.read_csv(csv_path)

# Identify columns based on metadata mismatch
# 'religiousness' column appears to be children (yes/no)
# 'age' column appears to be affairs frequency

children_col = 'religiousness'
affairs_col = 'age'

# Basic checks
print('Children value counts:')
print(df[children_col].value_counts(dropna=False))
print('\nAffairs summary:')
print(df[affairs_col].describe())

# Map children yes/no to binary
children_bin = df[children_col].map({'yes': 1, 'no': 0})

# Mean affairs by children
mean_affairs = df.groupby(children_col)[affairs_col].mean()
count_affairs = df.groupby(children_col)[affairs_col].count()
print('\nMean affairs by children:')
print(pd.concat([mean_affairs, count_affairs], axis=1, keys=['mean_affairs', 'n']))

# Use nonparametric Mann-Whitney U test if available? Use scipy
from scipy import stats

affairs_yes = df.loc[df[children_col] == 'yes', affairs_col]
affairs_no = df.loc[df[children_col] == 'no', affairs_col]

u_stat, p_value = stats.mannwhitneyu(affairs_yes, affairs_no, alternative='two-sided')
print(f"\nMann-Whitney U p-value: {p_value:.6f}")

# Also fit a simple OLS with robust SE for effect size (affairs is ordinal)
X = sm.add_constant(children_bin)
ols = sm.OLS(df[affairs_col], X).fit(cov_type='HC3')
print('\nOLS results:')
print(ols.summary())

# Also fit Poisson since outcome is count-like
poisson = sm.GLM(df[affairs_col], X, family=sm.families.Poisson()).fit()
print('\nPoisson results:')
print(poisson.summary())

# Effect size
mean_diff = mean_affairs.loc['yes'] - mean_affairs.loc['no']
print(f"\nMean difference (yes - no): {mean_diff:.6f}")
