import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('hurricane.csv')

# group means by gender_mf
for outcome in ['alldeaths', 'log_deaths']:
    pass

# add log deaths

df['log_deaths'] = np.log1p(df['alldeaths'])

print('Mean deaths by gender_mf (0=male,1=female):')
print(df.groupby('gender_mf')['alldeaths'].mean())
print('Median deaths:')
print(df.groupby('gender_mf')['alldeaths'].median())

# t-test on log deaths
male = df[df['gender_mf']==0]['log_deaths']
female = df[df['gender_mf']==1]['log_deaths']
tstat, pval = stats.ttest_ind(male, female, equal_var=False)
print(f'T-test log_deaths male vs female: t={tstat:.3f}, p={pval:.4f}')

# Nonparametric
u, p = stats.mannwhitneyu(male, female, alternative='two-sided')
print(f'Mann-Whitney log_deaths: U={u:.1f}, p={p:.4f}')

# Regression with interaction? But simple OLS
model = smf.ols('log_deaths ~ gender_mf', data=df).fit(cov_type='HC3')
print(model.summary().tables[1])

# Also with masfem continuous
model2 = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')
print(model2.summary().tables[1])

