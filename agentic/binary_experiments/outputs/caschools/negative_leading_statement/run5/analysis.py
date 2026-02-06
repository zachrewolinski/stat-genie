import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('caschools.csv')

# Feature engineering
_df['str'] = _df['students'] / _df['teachers']
_df['avg_score'] = _df[['read', 'math']].mean(axis=1)

# Basic summaries
print('Rows:', len(_df))
print('\nStudent-teacher ratio (str) summary:')
print(_df['str'].describe())
print('\nAverage score summary:')
print(_df['avg_score'].describe())

# Correlation
corr = _df[['str', 'avg_score']].corr().iloc[0, 1]
print(f"\nCorrelation(str, avg_score): {corr:.4f}")

# Simple bivariate regression
X_simple = sm.add_constant(_df['str'])
model_simple = sm.OLS(_df['avg_score'], X_simple).fit()
print('\nBivariate OLS: avg_score ~ str')
print(model_simple.summary().tables[1])

# Multiple regression with key covariates
controls = ['income', 'lunch', 'english', 'expenditure']
X_multi = sm.add_constant(_df[['str'] + controls])
model_multi = sm.OLS(_df['avg_score'], X_multi).fit()
print('\nMultiple OLS: avg_score ~ str + income + lunch + english + expenditure')
print(model_multi.summary().tables[1])

# Save key results for inspection
results = {
    'corr_str_avg_score': corr,
    'simple_coef_str': model_simple.params['str'],
    'simple_pvalue_str': model_simple.pvalues['str'],
    'multi_coef_str': model_multi.params['str'],
    'multi_pvalue_str': model_multi.pvalues['str'],
}

print('\nKey results:')
for k, v in results.items():
    print(f"{k}: {v}")
