import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Define efficiency as nuts opened per second
_df['rate'] = _df['nuts_opened'] / _df['seconds']

# Fit OLS with robust (HC3) standard errors
model = smf.ols('rate ~ age + C(sex) + C(help)', data=_df).fit(cov_type='HC3')

# Basic descriptive stats for context
means_by_sex = _df.groupby('sex')['rate'].mean()
means_by_help = _df.groupby('help')['rate'].mean()

print('Rate (nuts/sec) summary:')
print(_df['rate'].describe())
print('\nMean rate by sex:')
print(means_by_sex)
print('\nMean rate by help:')
print(means_by_help)
print('\nOLS (HC3 robust) results:')
print(model.summary())
