import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')

# Map columns based on observed values and metadata mismatches
# age -> age (years)
# nuts_opened (count) appears in 'help'
# seconds (duration) appears in 'chimpanzee'
# sex appears in 'nuts_opened'
# help (yes/no) appears in 'seconds'

df = df.rename(columns={
    'help': 'nuts_opened_count',
    'chimpanzee': 'seconds',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'seconds': 'help_received',
})

# Drop rows with missing or nonpositive seconds

df = df[df['seconds'] > 0].copy()

# Efficiency: nuts opened per second

df['efficiency'] = df['nuts_opened_count'] / df['seconds']

# For stability, also compute log efficiency (add small epsilon)

eps = 1e-6

df['log_efficiency'] = np.log(df['efficiency'] + eps)

# Ensure categorical types

df['sex'] = df['sex'].astype('category')

df['help_received'] = df['help_received'].astype('category')

# Model: log_efficiency ~ age + sex + help_received

model = smf.ols('log_efficiency ~ age + sex + help_received', data=df).fit()

# Also model on raw efficiency

model_raw = smf.ols('efficiency ~ age + sex + help_received', data=df).fit()

# Descriptives by groups

group_stats = df.groupby(['sex', 'help_received'])['efficiency'].agg(['mean','median','count'])

# Simple correlations

corr_age = df[['age','efficiency']].corr().iloc[0,1]

print('N:', len(df))
print('\nEfficiency summary:')
print(df['efficiency'].describe())
print('\nGroup stats (efficiency):')
print(group_stats)
print('\nCorrelation age-efficiency:', corr_age)
print('\nOLS log efficiency summary:')
print(model.summary())
print('\nOLS raw efficiency summary:')
print(model_raw.summary())
