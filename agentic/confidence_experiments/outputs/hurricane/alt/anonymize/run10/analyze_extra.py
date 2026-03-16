import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('hurricane.csv')
cols = {f'feature{i}': f'f{i}' for i in range(1, 15)}
df = df.rename(columns=cols)

# Spearman correlation
rho, p = stats.spearmanr(df['f4'], df['f8'])
print('Spearman f4 vs deaths:', rho, p)

# Compare deaths by gender indicator
male = df[df['f6']==0]['f8']
female = df[df['f6']==1]['f8']
print('Deaths male mean/median:', male.mean(), male.median(), 'n', len(male))
print('Deaths female mean/median:', female.mean(), female.median(), 'n', len(female))

# Mann-Whitney U test (nonparametric)
stat, p = stats.mannwhitneyu(female, male, alternative='two-sided')
print('Mann-Whitney U:', stat, p)

# t-test on log1p deaths
stat_t, p_t = stats.ttest_ind(np.log1p(female), np.log1p(male), equal_var=False)
print('t-test log1p deaths:', stat_t, p_t)

