import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Basic cleaning
# Ensure expected columns exist
cols = _df.columns.tolist()

# Create indicators
_df['children_yes'] = (_df['children'].astype(str).str.lower() == 'yes').astype(int)
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

# Descriptives by children
summary = _df.groupby('children_yes').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    any_affair_rate=('any_affair', 'mean')
).reset_index()

# Unadjusted difference
mean_diff = summary.loc[summary['children_yes'] == 1, 'mean_affairs'].iloc[0] - summary.loc[summary['children_yes'] == 0, 'mean_affairs'].iloc[0]
rate_diff = summary.loc[summary['children_yes'] == 1, 'any_affair_rate'].iloc[0] - summary.loc[summary['children_yes'] == 0, 'any_affair_rate'].iloc[0]

# Adjusted models
# Use controls commonly associated with affairs in this dataset
controls = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'gender']

# OLS on affairs (count-like, but ok for directional signal)
formula_ols = 'affairs ~ children_yes + ' + ' + '.join(controls)
model_ols = smf.ols(formula_ols, data=_df).fit()

# Logistic regression on any affair
formula_logit = 'any_affair ~ children_yes + ' + ' + '.join(controls)
model_logit = smf.logit(formula_logit, data=_df).fit(disp=False)

# Extract key stats
coef_ols = model_ols.params['children_yes']
p_ols = model_ols.pvalues['children_yes']

coef_logit = model_logit.params['children_yes']
p_logit = model_logit.pvalues['children_yes']

# Convert logit coef to odds ratio
odds_ratio = float(np.exp(coef_logit))

# Print results
print('Summary by children (0=no, 1=yes)')
print(summary.to_string(index=False))
print('\nUnadjusted differences (children_yes - no)')
print(f'mean affairs diff: {mean_diff:.4f}')
print(f'any affair rate diff: {rate_diff:.4f}')

print('\nOLS (affairs) coef for children_yes')
print(f'coef: {coef_ols:.4f}, p-value: {p_ols:.4g}')

print('\nLogit (any_affair) coef for children_yes')
print(f'coef: {coef_logit:.4f}, odds ratio: {odds_ratio:.4f}, p-value: {p_logit:.4g}')
