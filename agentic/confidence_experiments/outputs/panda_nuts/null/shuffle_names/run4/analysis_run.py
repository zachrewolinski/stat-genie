import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map variables based on info.json descriptions
# age in years -> column 'hammer'
# sex -> column 'nuts_opened' (m/f)
# help received -> column 'seconds' (y/N)
# nuts opened count -> column 'help'
# session duration seconds -> column 'chimpanzee'

age = df['hammer'].astype(float)
sex = df['nuts_opened'].astype(str)
help = df['seconds'].astype(str)

nuts_opened = df['help'].astype(float)
duration_sec = df['chimpanzee'].astype(float)

efficiency = nuts_opened / duration_sec

# Basic summaries
summary = pd.DataFrame({
    'age_years': age,
    'sex': sex,
    'help': help,
    'nuts_opened': nuts_opened,
    'duration_sec': duration_sec,
    'efficiency': efficiency,
})

print('efficiency summary')
print(summary['efficiency'].describe())
print('\nsex counts')
print(summary['sex'].value_counts())
print('\nhelp counts')
print(summary['help'].value_counts())

# Group means
print('\nmean efficiency by sex')
print(summary.groupby('sex')['efficiency'].agg(['mean','median','std','count']))
print('\nmean efficiency by help')
print(summary.groupby('help')['efficiency'].agg(['mean','median','std','count']))

# Correlation with age (Pearson and Spearman)
pearson_r, pearson_p = stats.pearsonr(summary['age_years'], summary['efficiency'])
spearman_r, spearman_p = stats.spearmanr(summary['age_years'], summary['efficiency'])
print(f"\nPearson r(age, efficiency) = {pearson_r:.3f}, p = {pearson_p:.4f}")
print(f"Spearman r(age, efficiency) = {spearman_r:.3f}, p = {spearman_p:.4f}")

# t-tests for sex/help groups (Welch)
sex_groups = summary['sex'].unique()
if len(sex_groups) == 2:
    g1 = summary.loc[summary['sex'] == sex_groups[0], 'efficiency']
    g2 = summary.loc[summary['sex'] == sex_groups[1], 'efficiency']
    t_stat, t_p = stats.ttest_ind(g1, g2, equal_var=False)
    print(f"\nWelch t-test sex groups ({sex_groups[0]} vs {sex_groups[1]}): t = {t_stat:.3f}, p = {t_p:.4f}")

help_groups = summary['help'].unique()
if len(help_groups) == 2:
    h1 = summary.loc[summary['help'] == help_groups[0], 'efficiency']
    h2 = summary.loc[summary['help'] == help_groups[1], 'efficiency']
    t_stat, t_p = stats.ttest_ind(h1, h2, equal_var=False)
    print(f"Welch t-test help groups ({help_groups[0]} vs {help_groups[1]}): t = {t_stat:.3f}, p = {t_p:.4f}")

# OLS regression
model_df = summary.copy()
model = smf.ols('efficiency ~ age_years + C(sex) + C(help)', data=model_df).fit()
print('\nOLS summary (coefficients)')
print(model.summary())

# Effect sizes: difference in means
if len(sex_groups) == 2:
    mean_diff_sex = g1.mean() - g2.mean()
    print(f"\nMean efficiency difference (sex {sex_groups[0]} - {sex_groups[1]}): {mean_diff_sex:.4f}")

if len(help_groups) == 2:
    mean_diff_help = h1.mean() - h2.mean()
    print(f"Mean efficiency difference (help {help_groups[0]} - {help_groups[1]}): {mean_diff_help:.4f}")
