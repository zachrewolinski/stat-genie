import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Compute efficiency: nuts opened per second
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Basic info
print('rows', len(_df))
print(_df[['efficiency','age','sex','help','chimpanzee']].head())

# Check for missing
print('missing', _df[['efficiency','age','sex','help']].isna().sum())

# OLS with categorical sex, help
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=_df).fit()
print(model.summary())

# Cluster-robust SE by chimpanzee
robust = model.get_robustcov_results(cov_type='cluster', groups=_df['chimpanzee'])
print('\nCluster-robust SE by chimpanzee')
print(robust.summary())

# MixedLM with random intercept per chimpanzee (if possible)
try:
    md = smf.mixedlm('efficiency ~ age + C(sex) + C(help)', data=_df, groups=_df['chimpanzee'])
    mdf = md.fit(reml=False)
    print('\nMixedLM')
    print(mdf.summary())
except Exception as e:
    print('\nMixedLM failed:', e)

# Nonparametric comparisons for sex and help (Mann-Whitney)
from scipy import stats

sex_groups = [g['efficiency'].values for _, g in _df.groupby('sex')]
if len(sex_groups) == 2:
    u, p = stats.mannwhitneyu(sex_groups[0], sex_groups[1], alternative='two-sided')
    print('\nMann-Whitney sex: U=', u, 'p=', p)

help_groups = [g['efficiency'].values for _, g in _df.groupby('help')]
if len(help_groups) == 2:
    u, p = stats.mannwhitneyu(help_groups[0], help_groups[1], alternative='two-sided')
    print('Mann-Whitney help: U=', u, 'p=', p)

# Correlation with age
r, p = stats.pearsonr(_df['age'], _df['efficiency'])
print('\nPearson age-efficiency r=', r, 'p=', p)

# Spearman
r_s, p_s = stats.spearmanr(_df['age'], _df['efficiency'])
print('Spearman age-efficiency r=', r_s, 'p=', p_s)
