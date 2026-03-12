import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Clean / compute efficiency
# Efficiency: nuts opened per second
# Guard against zero seconds (if any)
df = df.copy()

# Convert categorical columns to category
for col in ['sex', 'help', 'hammer']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Efficiency
# If seconds == 0, set to NaN to avoid inf (unlikely)
df['efficiency'] = df['nuts_opened'] / df['seconds'].replace(0, np.nan)

# Basic summaries
summary = df[['age', 'sex', 'help', 'nuts_opened', 'seconds', 'efficiency']].describe(include='all')
print('SUMMARY')
print(summary)

# OLS regression: efficiency ~ age + sex + help
# Use categorical encoding via formula
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()
print('\nOLS: efficiency ~ age + sex + help')
print(model.summary())

# Also run with hammer as control
model_hammer = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=df).fit()
print('\nOLS: efficiency ~ age + sex + help + hammer')
print(model_hammer.summary())

# Nonparametric correlation for age vs efficiency (Spearman)
from scipy.stats import spearmanr
rho, p = spearmanr(df['age'], df['efficiency'], nan_policy='omit')
print('\nSpearman age vs efficiency: rho=', rho, 'p=', p)

# Group comparisons for sex and help
# t-tests (Welch)
from scipy.stats import ttest_ind

# Sex
groups_sex = df.groupby('sex')['efficiency']
if len(groups_sex) == 2:
    vals = [g.dropna().values for _, g in groups_sex]
    t_sex = ttest_ind(vals[0], vals[1], equal_var=False)
    print('\nWelch t-test efficiency by sex:', t_sex)

# Help
groups_help = df.groupby('help')['efficiency']
if len(groups_help) == 2:
    vals = [g.dropna().values for _, g in groups_help]
    t_help = ttest_ind(vals[0], vals[1], equal_var=False)
    print('\nWelch t-test efficiency by help:', t_help)

