import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Rename to avoid collisions
analysis_df = df.rename(
    columns={
        'hammer': 'age_years',
        'nuts_opened': 'sex',
        'sex': 'hammer_type',
        'help': 'nuts_opened',
        'chimpanzee': 'duration_seconds',
        'seconds': 'received_help',
    }
).copy()

# Clean help indicator
analysis_df['received_help'] = analysis_df['received_help'].astype(str).str.strip().str.lower()
analysis_df['received_help'] = analysis_df['received_help'].map({'y': 1, 'n': 0})

# Drop rows with missing mapping
analysis_df = analysis_df.dropna(subset=['received_help', 'duration_seconds', 'nuts_opened', 'age_years', 'sex'])

# Efficiency: nuts opened per second
analysis_df['efficiency'] = analysis_df['nuts_opened'] / analysis_df['duration_seconds']

print('rows', len(analysis_df))
print(analysis_df[['age_years', 'sex', 'received_help', 'nuts_opened', 'duration_seconds', 'efficiency']].head())

print('\nEfficiency summary')
print(analysis_df['efficiency'].describe())

# OLS regression
model = smf.ols('efficiency ~ age_years + C(sex) + received_help', data=analysis_df).fit()
print('\nOLS summary')
print(model.summary())

robust = model.get_robustcov_results(cov_type='HC3')
print('\nOLS with HC3 robust SEs')
print(robust.summary())

anova = sm.stats.anova_lm(model, typ=2)
print('\nANOVA (Type II)')
print(anova)

print('\nCorrelation age vs efficiency')
print(analysis_df[['age_years', 'efficiency']].corr())

print('\nGroup means by sex')
print(analysis_df.groupby('sex')['efficiency'].agg(['mean','median','count']))
print('\nGroup means by received_help')
print(analysis_df.groupby('received_help')['efficiency'].agg(['mean','median','count']))
