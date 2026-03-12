import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
df = pd.read_csv('panda_nuts.csv')

# Basic cleaning: standardize help and sex to lowercase
for col in ['sex', 'help', 'hammer']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# Compute efficiency: nuts opened per second
# Avoid division by zero (though seconds min 2.5 per metadata)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Drop rows with missing values in key fields
analysis_df = df[['efficiency', 'age', 'sex', 'help']].dropna().copy()

# OLS regression
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=analysis_df).fit(cov_type='HC3')

# ANOVA-style comparisons (nonparametric) for sex and help
# Mann-Whitney U for sex/help (if both groups present)
results = {}

# Age correlation (Spearman)
results['age_spearman'] = stats.spearmanr(analysis_df['age'], analysis_df['efficiency'])

# sex test
if analysis_df['sex'].nunique() == 2:
    groups = [g['efficiency'].values for _, g in analysis_df.groupby('sex')]
    # Two-sided Mann-Whitney U
    u_stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
    results['sex_mwu'] = (u_stat, p_val)
else:
    results['sex_mwu'] = None

# help test
if analysis_df['help'].nunique() == 2:
    groups = [g['efficiency'].values for _, g in analysis_df.groupby('help')]
    u_stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
    results['help_mwu'] = (u_stat, p_val)
else:
    results['help_mwu'] = None

# Print key outputs
print('N:', len(analysis_df))
print('Efficiency summary:', analysis_df['efficiency'].describe())
print('\nOLS (HC3) summary:')
print(model.summary())

print('\nSpearman age-efficiency:', results['age_spearman'])
print('Mann-Whitney sex:', results['sex_mwu'])
print('Mann-Whitney help:', results['help_mwu'])

