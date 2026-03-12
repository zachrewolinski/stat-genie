import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Rename columns for clarity
_df = _df.rename(columns={
    'feature1': 'id',
    'feature2': 'age',
    'feature3': 'sex',
    'feature4': 'hammer',
    'feature5': 'nuts_opened',
    'feature6': 'duration_sec',
    'feature7': 'helped'
})

# Compute efficiency: nuts opened per minute (more interpretable)
_df['efficiency'] = _df['nuts_opened'] / (_df['duration_sec'] / 60.0)

# Clean categories
_df['sex'] = _df['sex'].astype('category')
_df['helped'] = _df['helped'].astype('category')

# Basic summary
print('Rows:', len(_df))
print(_df[['age','sex','helped','nuts_opened','duration_sec','efficiency']].head())

# OLS regression with age, sex, helped
model = smf.ols('efficiency ~ age + C(sex) + C(helped)', data=_df).fit()
print(model.summary())

# Also check with robust (HC3) standard errors
robust = model.get_robustcov_results(cov_type='HC3')
print('\nRobust HC3 summary:')
print(robust.summary())

# Optional: include hammer type as control (sensitivity)
model2 = smf.ols('efficiency ~ age + C(sex) + C(helped) + C(hammer)', data=_df).fit()
print('\nWith hammer control:')
print(model2.summary())
robust2 = model2.get_robustcov_results(cov_type='HC3')
print('\nWith hammer control (HC3):')
print(robust2.summary())

# Save key stats to a small json-like print
coefs = pd.DataFrame({
    'coef': model.params,
    'pvalue': model.pvalues
})
print('\nCoefficients (main model):')
print(coefs)

coefs2 = pd.DataFrame({
    'coef': model2.params,
    'pvalue': model2.pvalues
})
print('\nCoefficients (hammer model):')
print(coefs2)
