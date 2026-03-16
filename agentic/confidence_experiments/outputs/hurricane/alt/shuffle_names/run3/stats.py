import pandas as pd
import numpy as np
import scipy.stats as stats

df = pd.read_csv('hurricane.csv')

# Derived columns
log_deaths = np.log1p(df['name'])

# Group means by binary female indicator
for col in ['masfem_mturk']:
    grp = df.groupby(col)['name']
    print('Deaths mean by', col, grp.mean().to_dict())
    print('Deaths median by', col, grp.median().to_dict())
    log_grp = pd.DataFrame({'log_deaths': log_deaths, col: df[col]}).groupby(col)['log_deaths']
    print('Log deaths mean by', col, log_grp.mean().to_dict())

# t-test on log deaths
male = log_deaths[df['masfem_mturk']==0]
female = log_deaths[df['masfem_mturk']==1]
print('t-test log deaths female vs male', stats.ttest_ind(female, male, equal_var=False))

# Spearman correlation between fem index and deaths
print('Spearman category vs log deaths', stats.spearmanr(df['category'], log_deaths))
print('Spearman ind vs log deaths', stats.spearmanr(df['ind'], log_deaths))

