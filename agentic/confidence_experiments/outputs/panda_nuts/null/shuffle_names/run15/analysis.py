import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = 'panda_nuts.csv'

df = pd.read_csv(DATA_PATH)

# Map shuffled columns to semantic meaning based on observed values
age = df['age']
sex = df['nuts_opened']  # 'm'/'f'
help = df['seconds']     # 'y'/'N'
nuts_opened = df['help']
seconds = df['chimpanzee']

# Efficiency: nuts opened per second
# Guard against division by zero
seconds = seconds.replace(0, np.nan)

df = df.copy()
df['age_years'] = age

df['sex_mf'] = sex

df['help_yes'] = help

df['nuts_opened'] = nuts_opened

df['seconds'] = seconds

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Drop rows with missing efficiency
analysis_df = df.dropna(subset=['efficiency', 'age_years', 'sex_mf', 'help_yes'])

# Basic group summaries
summary = {
    'n': len(analysis_df),
    'efficiency_mean': analysis_df['efficiency'].mean(),
    'efficiency_std': analysis_df['efficiency'].std(ddof=1),
    'by_sex': analysis_df.groupby('sex_mf')['efficiency'].agg(['mean','std','count']).to_dict(),
    'by_help': analysis_df.groupby('help_yes')['efficiency'].agg(['mean','std','count']).to_dict(),
}

# Correlation age vs efficiency
spearman = stats.spearmanr(analysis_df['age_years'], analysis_df['efficiency'])
pearson = stats.pearsonr(analysis_df['age_years'], analysis_df['efficiency'])

# Two-sample tests for sex and help
sex_groups = [analysis_df.loc[analysis_df['sex_mf'] == g, 'efficiency'] for g in analysis_df['sex_mf'].unique()]
help_groups = [analysis_df.loc[analysis_df['help_yes'] == g, 'efficiency'] for g in analysis_df['help_yes'].unique()]

# Use t-test (Welch) for binary groups
sex_vals = analysis_df['sex_mf'].unique()
help_vals = analysis_df['help_yes'].unique()

sex_test = None
if len(sex_vals) == 2:
    g1 = analysis_df.loc[analysis_df['sex_mf'] == sex_vals[0], 'efficiency']
    g2 = analysis_df.loc[analysis_df['sex_mf'] == sex_vals[1], 'efficiency']
    sex_test = stats.ttest_ind(g1, g2, equal_var=False)

help_test = None
if len(help_vals) == 2:
    g1 = analysis_df.loc[analysis_df['help_yes'] == help_vals[0], 'efficiency']
    g2 = analysis_df.loc[analysis_df['help_yes'] == help_vals[1], 'efficiency']
    help_test = stats.ttest_ind(g1, g2, equal_var=False)

# Multiple regression
model = smf.ols('efficiency ~ age_years + C(sex_mf) + C(help_yes)', data=analysis_df).fit()

print('SUMMARY', summary)
print('SPEARMAN', spearman)
print('PEARSON', pearson)
print('SEX_TEST', sex_vals, sex_test)
print('HELP_TEST', help_vals, help_test)
print(model.summary())
