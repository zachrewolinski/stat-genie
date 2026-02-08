import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('affairs.csv')

# Clean: ensure children is categorical yes/no
# Compute means
summary = df.groupby('children')['affairs'].agg(['count','mean','std'])

# Difference: yes - no
mean_yes = summary.loc['yes','mean']
mean_no = summary.loc['no','mean']
std_yes = summary.loc['yes','std']
std_no = summary.loc['no','std']
count_yes = summary.loc['yes','count']
count_no = summary.loc['no','count']

# t-test
# Use Welch t-test

t_stat, p_val = stats.ttest_ind(
    df.loc[df['children']=='yes','affairs'],
    df.loc[df['children']=='no','affairs'],
    equal_var=False,
    nan_policy='omit'
)

# Cohen's d (pooled sd)
# Using Hedges g? just d

pooled_sd = np.sqrt(((count_yes-1)*std_yes**2 + (count_no-1)*std_no**2) / (count_yes+count_no-2))
cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd>0 else np.nan

# Any affairs indicator
df['any_affair'] = (df['affairs'] > 0).astype(int)
any_summary = df.groupby('children')['any_affair'].mean()
prop_yes = any_summary.loc['yes']
prop_no = any_summary.loc['no']

# Proportion test
# Use two-proportion z-test
n_yes = count_yes
n_no = count_no
x_yes = df.loc[df['children']=='yes','any_affair'].sum()
x_no = df.loc[df['children']=='no','any_affair'].sum()

p_pool = (x_yes + x_no) / (n_yes + n_no)
se = np.sqrt(p_pool*(1-p_pool)*(1/n_yes + 1/n_no))
z_stat = (x_yes/n_yes - x_no/n_no) / se if se>0 else np.nan
p_prop = 2*(1-stats.norm.cdf(abs(z_stat))) if se>0 else np.nan

print('summary')
print(summary)
print('mean_yes',mean_yes,'mean_no',mean_no,'diff',mean_yes-mean_no)
print('t_stat',t_stat,'p_val',p_val)
print('cohen_d',cohen_d)
print('any_summary')
print(any_summary)
print('prop_yes',prop_yes,'prop_no',prop_no,'diff',prop_yes-prop_no)
print('z_stat',z_stat,'p_prop',p_prop)
