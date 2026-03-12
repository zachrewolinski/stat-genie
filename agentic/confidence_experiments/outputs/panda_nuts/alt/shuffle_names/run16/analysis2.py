import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Map columns to semantic meaning based on metadata/descriptions
# According to info.json, these appear shuffled in this dataset.
# - 'help' column seems to be number of nuts opened
# - 'chimpanzee' column seems to be session duration in seconds
# - 'nuts_opened' column seems to be sex (m/f)
# - 'seconds' column seems to be help received (y/N)

_df = _df.rename(columns={
    'help': 'nuts_count',
    'chimpanzee': 'seconds_duration',
    'nuts_opened': 'sex_mf',
    'seconds': 'help_received'
})

# Efficiency: nuts opened per second
_df['efficiency'] = _df['nuts_count'] / _df['seconds_duration']

# Clean/encode predictors
_df['sex_mf'] = _df['sex_mf'].astype('category')
_df['help_received'] = _df['help_received'].astype('category')

# Basic summaries
print('Rows:', len(_df))
print(_df[['age','sex_mf','help_received','nuts_count','seconds_duration','efficiency']].head())

# OLS regression with robust SEs
model = smf.ols('efficiency ~ age + C(sex_mf) + C(help_received)', data=_df).fit(cov_type='HC3')
print('\nOLS HC3 summary:')
print(model.summary())

# Non-parametric checks
# Age vs efficiency: Spearman correlation
spearman = stats.spearmanr(_df['age'], _df['efficiency'])
print('\nSpearman age-efficiency:', spearman)

# Sex difference (m vs f): Mann-Whitney U
male = _df[_df['sex_mf']=='m']['efficiency']
female = _df[_df['sex_mf']=='f']['efficiency']
if len(male) > 0 and len(female) > 0:
    mw_sex = stats.mannwhitneyu(male, female, alternative='two-sided')
    print('\nMann-Whitney sex m vs f:', mw_sex)

# Help received (y vs N): Mann-Whitney U
help_yes = _df[_df['help_received']=='y']['efficiency']
help_no = _df[_df['help_received']=='N']['efficiency']
if len(help_yes) > 0 and len(help_no) > 0:
    mw_help = stats.mannwhitneyu(help_yes, help_no, alternative='two-sided')
    print('\nMann-Whitney help y vs N:', mw_help)

# Effect sizes: group means
print('\nGroup means:')
print('Overall efficiency mean:', _df['efficiency'].mean())
print('Sex m mean:', male.mean(), 'Sex f mean:', female.mean())
print('Help y mean:', help_yes.mean(), 'Help N mean:', help_no.mean())

# Save results for interpretation
