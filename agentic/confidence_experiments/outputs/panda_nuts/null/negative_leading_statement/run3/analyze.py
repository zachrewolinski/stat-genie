import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Compute efficiency: nuts opened per second
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Basic checks
print('Rows:', len(_df))
print('Efficiency summary:')
print(_df['efficiency'].describe())

# Encode categorical variables
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')

# OLS model: efficiency ~ age + sex + help
model = smf.ols('efficiency ~ age + sex + help', data=_df).fit()
print('\nOLS model (classic SE)')
print(model.summary())

# Cluster-robust SE by chimpanzee ID (accounts for repeated observations)
cluster_model = model.get_robustcov_results(cov_type='cluster', groups=_df['chimpanzee'])
print('\nOLS model (cluster-robust SE by chimpanzee)')
print(cluster_model.summary())

# Secondary model controlling for hammer type
_df['hammer'] = _df['hammer'].astype('category')
model_hammer = smf.ols('efficiency ~ age + sex + help + hammer', data=_df).fit()
cluster_model_hammer = model_hammer.get_robustcov_results(cov_type='cluster', groups=_df['chimpanzee'])
print('\nOLS model with hammer (cluster-robust SE by chimpanzee)')
print(cluster_model_hammer.summary())

# Simple bivariate checks
print('\nBivariate checks:')
# Age correlation
r, p = stats.pearsonr(_df['age'], _df['efficiency'])
print(f'Age vs efficiency Pearson r={r:.3f}, p={p:.4f}')

# Sex group comparison
sex_groups = [_df.loc[_df['sex']==lvl, 'efficiency'] for lvl in _df['sex'].cat.categories]
if len(sex_groups) == 2:
    tstat, pval = stats.ttest_ind(sex_groups[0], sex_groups[1], equal_var=False)
    print(f'Sex t-test (Welch): t={tstat:.3f}, p={pval:.4f}')

# Help group comparison
help_groups = [_df.loc[_df['help']==lvl, 'efficiency'] for lvl in _df['help'].cat.categories]
if len(help_groups) == 2:
    tstat, pval = stats.ttest_ind(help_groups[0], help_groups[1], equal_var=False)
    print(f'Help t-test (Welch): t={tstat:.3f}, p={pval:.4f}')

# Group means for interpretability
print('\nGroup means:')
print(_df.groupby('sex')['efficiency'].mean())
print(_df.groupby('help')['efficiency'].mean())
