import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('hurricane.csv')

df['log_deaths'] = np.log1p(df['name'])

# Pearson and Spearman correlations
for col in ['category','ind']:
    pearson = stats.pearsonr(df[col].dropna(), df['log_deaths'].loc[df[col].notna()])
    spearman = stats.spearmanr(df[col], df['log_deaths'], nan_policy='omit')
    print(col, 'pearson r,p', pearson, 'spearman rho,p', spearman)

# binary gender difference
male = df[df['masfem_mturk']==0]['log_deaths']
female = df[df['masfem_mturk']==1]['log_deaths']
ttest = stats.ttest_ind(male, female, equal_var=False, nan_policy='omit')
print('binary gender t-test', ttest)
print('means male,female', male.mean(), female.mean())

# effect size (Cohen d)
# compute Cohen d for unequal sample sizes
n1, n2 = len(male), len(female)
var1, var2 = male.var(ddof=1), female.var(ddof=1)
pooled = ((n1-1)*var1 + (n2-1)*var2)/(n1+n2-2)
cohen_d = (male.mean() - female.mean())/np.sqrt(pooled)
print('cohen d (male-female)', cohen_d)
