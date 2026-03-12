import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map columns based on info.json descriptions
# age column is ID; hammer column is age in years
# nuts_opened column is sex; help column is number of nuts opened
# chimpanzee column is duration in seconds; seconds column is help (yes/no)

# Rename for clarity
analysis_df = df.rename(
    columns={
        'hammer': 'age_years',
        'nuts_opened': 'sex',
        'help': 'nuts_opened',
        'chimpanzee': 'duration_seconds',
        'seconds': 'received_help',
    }
).copy()

# Clean help indicator
analysis_df['received_help'] = analysis_df['received_help'].str.strip().str.lower()
analysis_df['received_help'] = analysis_df['received_help'].map({'y': 1, 'n': 0})

# Drop rows with missing mapping
analysis_df = analysis_df.dropna(subset=['received_help'])

# Efficiency: nuts opened per second
analysis_df['efficiency'] = analysis_df['nuts_opened'] / analysis_df['duration_seconds']

# Basic checks
print('rows', len(analysis_df))
print(analysis_df[['age_years', 'sex', 'received_help', 'nuts_opened', 'duration_seconds', 'efficiency']].head())

# Summary statistics
print('\nEfficiency summary')
print(analysis_df['efficiency'].describe())

# OLS regression
# sex categorical, received_help binary, age_years continuous
model = smf.ols('efficiency ~ age_years + C(sex) + received_help', data=analysis_df).fit()
print('\nOLS summary')
print(model.summary())

# Robust (HC3) SEs for sensitivity
robust = model.get_robustcov_results(cov_type='HC3')
print('\nOLS with HC3 robust SEs')
print(robust.summary())

# Partial eta-squared via ANOVA
anova = sm.stats.anova_lm(model, typ=2)
print('\nANOVA (Type II)')
print(anova)

# Correlations (age vs efficiency)
print('\nCorrelation age vs efficiency')
print(analysis_df[['age_years', 'efficiency']].corr())

# Group means for sex and help
print('\nGroup means by sex')
print(analysis_df.groupby('sex')['efficiency'].agg(['mean','median','count']))
print('\nGroup means by received_help')
print(analysis_df.groupby('received_help')['efficiency'].agg(['mean','median','count']))
