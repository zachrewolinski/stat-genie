import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

df = pd.read_csv('affairs.csv')

summary = df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])

any_affair = (df['feature2'] > 0).astype(int)
rate = df.groupby('feature6')[any_affair.name if any_affair.name else None].mean() if False else df.groupby('feature6').apply(lambda g: (g['feature2']>0).mean())

mean_diff = summary.loc['yes','mean'] - summary.loc['no','mean']
rate_diff = rate.loc['yes'] - rate.loc['no']

print('summary')
print(summary)
print('\nany_affair_rate')
print(rate)
print('\nmean_diff_yes_minus_no', mean_diff)
print('rate_diff_yes_minus_no', rate_diff)

x = df.loc[df['feature6']=='yes','feature2']
y = df.loc[df['feature6']=='no','feature2']

tstat, pval = stats.ttest_ind(x, y, equal_var=False)
print('\nttest mean frequency (yes vs no) tstat', tstat, 'pval', pval)

x1 = (df.loc[df['feature6']=='yes','feature2']>0)
x2 = (df.loc[df['feature6']=='no','feature2']>0)
count = np.array([x1.sum(), x2.sum()])
obs = np.array([x1.size, x2.size])
stat, pval2 = proportions_ztest(count, obs)
print('prop ztest any_affair yes vs no z', stat, 'pval', pval2)
