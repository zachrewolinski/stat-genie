import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

# Load data
_df = pd.read_csv('boxes.csv')
print('Rows', len(_df))
print('Columns', _df.columns.tolist())
print('Head')
print(_df.head())

# Inspect unique values
for col in _df.columns:
    print('\n', col, 'unique', sorted(_df[col].dropna().unique().tolist())[:20], 'nunique', _df[col].nunique())

# Define majority choice as 1 if majority option chosen (value 2)
_df = _df.copy()
_df['majority_choice'] = (_df['majority_first'] == 2).astype(int)

# Age centered
_df['age_c'] = _df['age'] - _df['age'].mean()

# Use site id as culture if available
# y is site id per metadata; treat as categorical
_df['site'] = _df['y'].astype('category')

# Also consider culture column if binary
_df['culture_bin'] = _df['culture']

print('\nMajority choice rate overall', _df['majority_choice'].mean())

# Model 1: age only
m1 = smf.glm('majority_choice ~ age_c', data=_df, family=sm.families.Binomial()).fit()
# Model 2: age + site
m2 = smf.glm('majority_choice ~ age_c + site', data=_df, family=sm.families.Binomial()).fit()
# Model 3: age + site + interaction
m3 = smf.glm('majority_choice ~ age_c * site', data=_df, family=sm.families.Binomial()).fit()

# Likelihood ratio tests
lr_12 = 2*(m2.llf - m1.llf)
lr_23 = 2*(m3.llf - m2.llf)

p_12 = chi2.sf(lr_12, m2.df_model - m1.df_model)
p_23 = chi2.sf(lr_23, m3.df_model - m2.df_model)

print('\nModel summaries')
print('m1 age only: llf', m1.llf, 'df', m1.df_model)
print('m2 age+site: llf', m2.llf, 'df', m2.df_model)
print('m3 age*site: llf', m3.llf, 'df', m3.df_model)
print('LR test m2 vs m1 (site effect):', lr_12, 'p', p_12)
print('LR test m3 vs m2 (age*site interaction):', lr_23, 'p', p_23)

# Age effect
age_coef = m1.params['age_c']
age_p = m1.pvalues['age_c']
print('Age coef', age_coef, 'p', age_p)

# Site-level majority rates and age trends
site_rates = _df.groupby('site')['majority_choice'].mean().sort_index()
print('\nSite majority rates')
print(site_rates)

# Age group analysis by bins
bins = [3.5,5.5,7.5,9.5,11.5,13.5,15.5]
labels = ['4-5','6-7','8-9','10-11','12-13','14-15']
_df['age_group'] = pd.cut(_df['age'], bins=bins, labels=labels)

age_rates = _df.groupby('age_group')['majority_choice'].mean()
print('\nAge group majority rates')
print(age_rates)

# Two-way table by site and age group
pivot = _df.pivot_table(values='majority_choice', index='site', columns='age_group', aggfunc='mean')
print('\nSite x age_group majority rates')
print(pivot)

# Save key outputs to json-like dict for later
import json
results = {
    'n': int(len(_df)),
    'majority_rate': float(_df['majority_choice'].mean()),
    'age_coef': float(age_coef),
    'age_p': float(age_p),
    'site_lr_p': float(p_12),
    'interaction_lr_p': float(p_23),
    'site_rates': site_rates.to_dict(),
    'age_rates': age_rates.to_dict(),
}
with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

