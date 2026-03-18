import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Map feature names to meaningful vars based on info.json
# feature2: affairs count/frequency; feature6: children yes/no

# Basic cleaning

df = _df.copy()

# ensure children categorical
# normalize strings
if df['feature6'].dtype != object:
    df['feature6'] = df['feature6'].astype(str)

# lower/strip

df['feature6'] = df['feature6'].str.strip().str.lower()

# compute basic group stats

# affairs frequency variable
# feature2 numeric

# Summary by children

group_stats = df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])

# Proportion any affair (feature2 > 0)

df['any_affair'] = (df['feature2'] > 0).astype(int)
any_stats = df.groupby('feature6')['any_affair'].agg(['mean','count'])

# t-test difference in means? we can use statsmodels to test difference

# simple OLS for mean difference
model_ols = smf.ols('feature2 ~ C(feature6)', data=df).fit()

# logistic regression for any_affair
model_logit = smf.logit('any_affair ~ C(feature6)', data=df).fit(disp=False)

# also control for confounders: age (feature4), years married (feature5), religiosity (feature7), education (feature8), occupation (feature9), marriage rating (feature10), gender (feature3)

# ensure feature3 categorical
if df['feature3'].dtype != object:
    df['feature3'] = df['feature3'].astype(str)

df['feature3'] = df['feature3'].str.strip().str.lower()

# OLS with controls
model_ols_ctrl = smf.ols('feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=df).fit()

# logistic with controls
model_logit_ctrl = smf.logit('any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=df).fit(disp=False)

# Extract key stats

results = {
    'group_stats': group_stats,
    'any_stats': any_stats,
    'ols_coef': model_ols.params.to_dict(),
    'ols_pvalues': model_ols.pvalues.to_dict(),
    'ols_ctrl_coef': model_ols_ctrl.params.to_dict(),
    'ols_ctrl_pvalues': model_ols_ctrl.pvalues.to_dict(),
    'logit_coef': model_logit.params.to_dict(),
    'logit_pvalues': model_logit.pvalues.to_dict(),
    'logit_ctrl_coef': model_logit_ctrl.params.to_dict(),
    'logit_ctrl_pvalues': model_logit_ctrl.pvalues.to_dict(),
    'logit_odds_ratio': np.exp(model_logit.params).to_dict(),
    'logit_ctrl_odds_ratio': np.exp(model_logit_ctrl.params).to_dict(),
}

# Save summary to stdout
print('GROUP_STATS')
print(group_stats)
print('\nANY_STATS')
print(any_stats)
print('\nOLS_COEF_PVALUES')
for k in model_ols.params.index:
    print(k, model_ols.params[k], model_ols.pvalues[k])
print('\nOLS_CTRL_COEF_PVALUES')
for k in model_ols_ctrl.params.index:
    print(k, model_ols_ctrl.params[k], model_ols_ctrl.pvalues[k])
print('\nLOGIT_COEF_PVALUES')
for k in model_logit.params.index:
    print(k, model_logit.params[k], model_logit.pvalues[k], 'OR', np.exp(model_logit.params[k]))
print('\nLOGIT_CTRL_COEF_PVALUES')
for k in model_logit_ctrl.params.index:
    print(k, model_logit_ctrl.params[k], model_logit_ctrl.pvalues[k], 'OR', np.exp(model_logit_ctrl.params[k]))
